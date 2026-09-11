"""Stepping electrophysiology, contraction and mechanics together.

Three pieces, each doing one thing:

:class:`ActiveTensionDriver`
    Advances a contraction model from electrophysiology samples.  It is the
    callback ``ReactionEikonal.solve(on_step=...)`` expects, and knows nothing
    about meshes.
:class:`HeldStretch`
    The stretch a contraction model sees between two mechanics solves, as a
    consistent ``(lambda, dlambda/dt)`` pair.
:class:`ActiveStressCoupling`
    One mechanics solve: nodal tension in, nodal fibre stretch out, owning the
    transfers, the warm start and the load continuation.

The coupling is one-way from the electrophysiology: calcium drives contraction,
contraction drives deformation, deformation returns a stretch.  Nothing here
feeds mechanics back into the electrophysiology.

A contraction model is anything with ``step(calcium, stretch, stretch_rate,
dt) -> tension``, ``tension(stretch)``, ``checkpoint()`` and ``restore(state)``;
:class:`~torchcor.electromechanics.land.Land2017` is one, and nothing in this
module is specific to it.

**The coupling scheme is explicit and its stability is not established.**  The
stretch applied over an interval comes from the tension at the interval's
start, which is the ordinary staggered scheme, first order in the interval.  On
a homogeneous verification it is accurate and first-order convergent while the
deformation is small, and loses stability once the stretch moves quickly --
refining the interval makes that worse, not better.  See
``benchmark.verification``.  An iterated predictor-corrector, for which
``checkpoint``/``restore`` exist, is the usual remedy and is not implemented.
"""

import torch


class HeldStretch:
    """The fibre stretch a contraction model sees between mechanics solves.

    Mechanics is solved every ``interval`` milliseconds while contraction steps
    at the much smaller electrophysiology ``dt``.  The stretch has to arrive as
    a consistent history over that interval, not as a jump on one step:
    differencing successive samples over ``dt`` turns a 0.1 change spread over
    5 ms into a rate of -10/ms instead of -0.02/ms, five hundred times too
    large, which drives the distortion states to their limit and the tension
    negative.

    So the interval is interpolated.  :meth:`advance` records where the stretch
    is going and how long it has to get there; :meth:`at` returns the stretch on
    that straight line together with the constant rate that generates it, so
    ``lambda`` and ``dlambda/dt`` always agree with one another.
    """

    def __init__(self, n, device=None, dtype=torch.float64):
        self.value = torch.ones(n, device=device, dtype=dtype)
        self.rate = torch.zeros(n, device=device, dtype=dtype)
        self._from = self.value.clone()
        self._t0 = 0.0
        self._t1 = 0.0

    def advance(self, target, now, interval):
        """Head for ``target`` over ``interval`` milliseconds, starting at ``now``."""
        self._from = self.value.clone()
        self.rate = (target - self._from) / interval
        self._t0, self._t1 = now, now + interval

    def at(self, t):
        """``(lambda, dlambda/dt)`` at time ``t``."""
        if t >= self._t1:
            self.value = self._from + self.rate * (self._t1 - self._t0)
            return self.value, torch.zeros_like(self.rate)
        self.value = self._from + self.rate * max(t - self._t0, 0.0)
        return self.value, self.rate


class ActiveTensionDriver:
    """Advance a contraction model from electrophysiology samples.

    Pass an instance as ``ReactionEikonal.solve(on_step=...)``.  The solver
    hands over ``(t, vm, calcium)`` and the time step comes from consecutive
    timestamps, so it cannot disagree with the solver about dt.

    ``stretch`` supplies the fibre stretch and its rate together, because the
    two have to be consistent:

    * ``None`` -- isometric, ``lambda = 1`` and ``dlambda/dt = 0``;
    * a number -- that fixed stretch, zero rate;
    * anything with ``at(t)`` returning ``(lambda, dlambda/dt)``, which is what
      :class:`HeldStretch` provides for a mechanics coupling.

    A bare callable is deliberately refused: it would leave the driver to
    difference stretches itself, over the wrong interval.

    ``record_every`` in milliseconds keeps tension and calcium on the device;
    ``None`` records nothing, which is the point of a callback.  The solver
    reuses its buffers, so anything kept here is cloned.
    """

    def __init__(self, model, stretch=None, record_every=None):
        self.model = model
        self.stretch = stretch
        self.record_every = record_every
        self.tension = None
        self.times, self.tension_history, self.calcium_history = [], [], []
        self._t = None
        self._next_record = 0.0

    @property
    def _shape_like(self):
        return self.model.state[0]

    def _stretch_at(self, t):
        if self.stretch is None:
            ones = torch.ones_like(self._shape_like)
            return ones, torch.zeros_like(ones)
        if hasattr(self.stretch, "at"):
            return self.stretch.at(t)
        value = torch.as_tensor(self.stretch, device=self.model.device,
                                dtype=self.model.dtype).expand_as(self._shape_like)
        return value, torch.zeros_like(value)

    def __call__(self, t, vm, calcium):
        lam, rate = self._stretch_at(t)
        if self._t is None:                       # the initial sample: no step yet
            self._t = t
            self.tension = self.model.tension(lam)
            self._next_record = t
            self._record(t, calcium)
            return
        self.tension = self.model.step(calcium, lam, rate, t - self._t)
        self._t = t
        self._record(t, calcium)

    def _record(self, t, calcium):
        if self.record_every is None or t + 1e-12 < self._next_record:
            return
        self._next_record = t + self.record_every
        self.times.append(t)
        self.tension_history.append(self.tension.clone())
        self.calcium_history.append(calcium.clone())

    def history(self):
        """``(times, tension, calcium)`` as stacked tensors, or empty."""
        if not self.times:
            return (torch.empty(0), torch.empty(0), torch.empty(0))
        return (torch.tensor(self.times, device=self.model.device, dtype=self.model.dtype),
                torch.stack(self.tension_history), torch.stack(self.calcium_history))


class ActiveStressCoupling:
    """One mechanics solve: nodal active tension in, nodal fibre stretch out.

    Owns the transfers, the warm start and the load continuation, so a coupled
    run is a loop over :meth:`solve` rather than a re-assembled problem each
    time.

    Parameters
    ----------
    problem:
        A built mechanics problem, e.g. ``Mechanics(...)._build()``.
    mesh, fibre:
        The mechanics mesh and its per-cell fibre direction, ``(n_cells, 3)``.
    tension:
        The ``(n_cells, 1)`` buffer the material was constructed with.  It is
        written in place, which is how the tension reaches the material without
        rebuilding it.
    connectivity:
        Which nodes the contraction model occupies, as a slice of
        ``mesh.cells``.  ``None`` means all of them -- correct when contraction
        runs on the mechanics mesh.  When it runs on the cell vertices, e.g. an
        electrophysiology mesh, pass ``mesh.cells[:, :4]``.
    stress_scale:
        Multiplies the tension on its way into the material, for the common
        case of a contraction model in kilopascal and a material in pascal.
        The tension enters as the fibre-fibre component of the *second
        Piola-Kirchhoff* stress, which is a modelling convention: reading it as
        a Cauchy tension instead would need ``J lambda^-2`` as well.
    load_steps:
        Continuation attempts, tried in order until one converges.
    """

    def __init__(self, problem, mesh, fibre, tension, *, connectivity=None,
                 n_nodes=None, stress_scale=1.0, load_steps=(4, 16, 64),
                 newton_atol=1e-9, stiffness=None, reference_stretch=None):
        from torchcor.electromechanics.transfer import (
            CellDeformation, CellToNode, NodeToCell, fibre_stretch)
        self._fibre_stretch = fibre_stretch
        connectivity = mesh.cells if connectivity is None else connectivity
        self.problem = problem
        self.mesh = mesh
        self.fibre = fibre
        self.tension = tension
        self.stress_scale = float(stress_scale)
        self.load_steps = tuple(load_steps)
        self.newton_atol = float(newton_atol)
        self.deformation = CellDeformation(mesh)
        self.to_cells = NodeToCell(connectivity)
        self.to_nodes = CellToNode(connectivity,
                                   mesh.points.shape[0] if n_nodes is None else n_nodes)
        # Optional buffers of a stabilized active stress: the active stiffness
        # and the stretch the contraction model was evaluated at.  Present only
        # when the material is a StabilizedActiveStressMaterial.
        self.stiffness = stiffness
        self.reference_stretch = reference_stretch
        self.displacement = None
        self.report = None

    def solve(self, tension_nodal, clamp_negative=False, stiffness_nodal=None,
              reference_stretch_nodal=None):
        """Solve for the deformation and return the nodal fibre stretch.

        Raises when no continuation converges: a coupled trajectory that
        carries on from a failed mechanics step is not a solution of anything.
        """
        from torchcor.mechanics.solver import QuasiStaticSolver
        cell = self.to_cells(tension_nodal)
        self.tension.copy_((cell.clamp_min(0.0) if clamp_negative else cell)
                           * self.stress_scale)
        if stiffness_nodal is not None:
            self.stiffness.copy_(self.to_cells(stiffness_nodal) * self.stress_scale)
        if reference_stretch_nodal is not None:
            self.reference_stretch.copy_(self.to_cells(reference_stretch_nodal))
        for steps in self.load_steps:
            u, report = QuasiStaticSolver(
                self.problem, load_steps=steps, verbose=False,
                newton_atol=self.newton_atol).solve(u=self.displacement)
            if report.converged:
                break
        self.report = report
        if not report.converged:
            raise RuntimeError(
                f"mechanics did not converge with load steps {self.load_steps}: "
                f"{report}")
        self.displacement = u.clone()
        gradient = self.deformation(u.reshape(-1, 3))
        cell_stretch = self._fibre_stretch(gradient, self.fibre.unsqueeze(1)).mean(dim=1)
        return self.to_nodes(cell_stretch), cell_stretch
