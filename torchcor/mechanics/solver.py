"""Nonlinear driver for :mod:`torchcor.mechanics`.

Coordinates the three nested loops of a quasi-static finite-strain solve --
load continuation, augmented Lagrangian updates, and Newton-Raphson -- together
with the convergence tests and the recovery paths when any of them fails.

All numerical work is delegated: :mod:`torchcor.mechanics.assembly` builds the
residual and tangent, :mod:`torchcor.mechanics.linear` solves the linear
systems. This module contains no geometry or benchmark-specific decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from torchcor.mechanics.assembly import FiniteStrainProblem
from torchcor.mechanics.linear import (
    BiCGStab, BlockJacobiPreconditioner, ConjugateGradient, KrylovSolver,
)


__all__ = ["SolveReport", "QuasiStaticSolver", "DynamicSolver"]


@dataclass
class SolveReport:
    """Outcome of a :class:`QuasiStaticSolver` run."""

    converged: bool
    load_factor: float
    load_steps: int
    newton_iterations: int
    linear_iterations: int
    volume_error: float            #: max projected volume-constraint error
    #: Smallest J of the *end* state.  A beat that deforms hard and springs
    #: back ends near one, so this does not describe the loaded state; see
    #: :attr:`worst_jacobian`.
    min_jacobian: float
    residual: float
    wall_time: float
    #: Full-quadrature volume diagnostics; see
    #: :meth:`FiniteStrainProblem.volume_diagnostics`.
    volume: Dict[str, float] = field(default_factory=dict)
    #: Peak CUDA memory allocated during the solve, in MiB (0 on CPU).
    peak_memory_mib: float = 0.0
    #: Smallest J over the whole interval, including the initial state, and
    #: when it happened.  This is the one to report: the end state is not the
    #: worst state, and neither is the first step.
    worst_jacobian: float = float("nan")
    worst_jacobian_time: float = float("nan")
    #: Why load steps were rejected, and how often.  Reaching full load after
    #: many cutbacks is not the same as never stumbling, and the cause says
    #: which part of the solve to look at.  The categories follow PETSc's
    #: ``SNESConvergedReason``: function-domain error, linear-solve failure,
    #: line-search failure, and iteration-limit.
    cutbacks: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        state = "converged" if self.converged else "FAILED"
        extra = ""
        if self.volume:
            extra += (f", full-quad max|J-1|={self.volume['max_error']:.2e}"
                      f", rms={self.volume['rms_error']:.2e}"
                      f", dV/V={self.volume['volume_change']:+.2e}")
        if self.peak_memory_mib:
            extra += f", peak {self.peak_memory_mib:.0f} MiB"
        if self.cutbacks:
            extra += ", cutbacks: " + "; ".join(
                f"{n}x {why}" for why, n in sorted(self.cutbacks.items(),
                                                   key=lambda kv: -kv[1]))
        return (f"{state}: lambda={self.load_factor:.6f}, steps={self.load_steps}, "
                f"newton={self.newton_iterations}, krylov={self.linear_iterations}, "
                f"mixed constraint={self.volume_error:.2e}, "
                f"min J={self.min_jacobian:.6f}, "
                f"|R|={self.residual:.3e}{extra}, {self.wall_time:.2f} s")


class QuasiStaticSolver:
    """Load continuation, mixed volume constraints, and Newton's method.

    Newton uses the consistent tangent with the line search that structural
    finite-element codes use.  ``line_search="critical-point"`` -- the default,
    and the only search implemented -- is a secant search for the point where
    the energy slope ``G(alpha) = du . R(u + alpha du)`` vanishes (Crisfield,
    *Non-linear Finite Element Analysis of Solids and Structures*, vol. 1,
    section 9.5; the same merit as PETSc's ``SNESLINESEARCHCP``, and as Ratel's
    solver).  It is *not* residual-norm backtracking, and is named for what it
    does.  The residual *norm* is
    deliberately not the merit function.  On an unloaded body ``|R(0)|`` is
    just the applied load and is tiny, while a correct Newton step raises the
    internal forces, so a sufficient-decrease test on ``|R|`` rejects the very
    step that solves the problem and backtracks to a step length of order
    1e-3, taking a residual reduction of 0.04% per iteration for ever.  The
    energy slope has no such defect: it is the derivative of the potential
    along the correction, and it is happy for the residual to rise on the way
    to equilibrium.  A trial that leaves the admissible region -- an inverted
    element or a non-finite residual -- is cut back instead, as ``SNES`` does
    with a domain error, since the merit does not exist there.  A solve is
    abandoned when the residual grows past ``divergence_tol`` times the
    smallest it has reached (``SNESSetDivergenceTolerance``), or when it
    exhausts ``max_newton``, which defaults to PETSc's ``SNES`` limit of 50.
    That budget is generous on purpose: with an adaptive forcing term the early
    corrections are cheap, and abandoning an increment costs a whole retry.
    Nothing here tries to detect "converging too slowly" -- see :meth:`_newton`
    for why every such test misfires on this problem.

    The load increment is chosen from how hard the last one was.  The
    controller is FEBio's automatic time stepper (``FETimeStepController::
    AutoTimeStep``): the increment moves toward ``max_load_step`` by at most
    20% of the gap when the last one converged in fewer than
    ``optimal_newton`` corrections, and eases off in proportion when it took
    more.  The ceiling is the whole load, Abaqus's default; because the
    controller only ever closes a fifth of the remaining gap, it approaches
    a large ceiling without overshooting it, and a lower one (FEBio's usual
    three times the initial increment) measured 20-40% slower on both
    benchmarks for no gain in reliability.  Growing by a fixed factor after every success instead guarantees a
    periodic overshoot, a wasted failed increment and a cutback.  Cutbacks
    follow Abaqus's ``*CONTROLS`` table: a quarter for divergence or a domain
    failure, a half otherwise, and at most ``max_attempts`` = 5 tries at one
    load level, the number FEBio's ``max_retries`` also uses.  Each increment
    starts from a linear extrapolation of the previous one -- Abaqus's default
    increment extrapolation, abandoned when the increment has shrunk below a
    tenth of the last accepted one (Abaqus ``D_E``) -- which keeps Newton near
    its quadratic regime rather than restarting it from a state the new load
    has already left.

    The forcing term for the inner solve is Eisenstat-Walker's choice 2 at
    PETSc's constants, with the tolerance held non-increasing within a solve.
    The GPU references (Brown et al. 2022; ExaDG) use a fixed 1e-3 instead;
    measured on the ventricle that took 23.4 s with three cutbacks where the
    adaptive term took 11.5 s with none, because it lets the early, badly
    modelled corrections be cheap and only the last ones exact.

    ``line_search="none"`` selects full Newton corrections. It is not
    recommended: without a search an inadmissible trial has nowhere to go but a
    cutback, so a single overshoot throws away the whole increment.
    Useful inexact linear solves must satisfy a true-residual
    forcing bound below one. Load-step failure restores displacement and all
    pressure multipliers before retrying a smaller increment.

    ``newton_rtol`` scales by the larger internal/external force norm;
    ``newton_atol`` supplies an absolute floor in the chosen force units.
    ``al_tol`` bounds the projected mixed volume constraint. Local volume
    errors must be checked separately with independent quadrature.

    The Krylov method is chosen from the assembled tangent rather than assumed.
    A follower pressure gives a nonsymmetric load stiffness in general, but the
    nonsymmetric part is an edge term on the boundary of the loaded surface
    (Bonet & Wood, *Nonlinear Continuum Mechanics for Finite Element Analysis*,
    section 6.5; Simo, Taylor & Wriggers 1991), so a pressure whose surface is
    closed or whose edge is constrained -- an inflated ventricle held at the
    base -- assembles a symmetric tangent. Assuming otherwise is expensive:
    BiCGStab can break down, and was observed diverging to a relative residual
    of 27 after 20,000 iterations on a system that conjugate gradients solve in
    530. So the symmetry is measured on the assembled values, and CG is used
    when it holds. A caller may supply a Krylov solver explicitly.
    """

    def __init__(
        self,
        problem: FiniteStrainProblem,
        linear_solver: Optional[KrylovSolver] = None,
        newton_rtol: float = 1e-9,
        newton_atol: float = 1e-10,
        linear_forcing: float = 0.5,
        max_newton: int = 50,
        line_search: str = "critical-point",
        divergence_tol: float = 1e4,
        optimal_newton: int = 8,
        max_load_step: float = 1.0,
        max_attempts: int = 5,
        extrapolate: bool = True,
        load_steps: int = 10,
        min_load_step: float = 1e-4,
        al_tol: float = 1e-5,
        max_al: int = 10,
        verbose: bool = True,
    ) -> None:
        self.problem = problem
        # Chosen at each Newton solve, once the tangent exists to look at,
        # unless the caller supplied a method explicitly.
        self.linear_solver = linear_solver
        self._user_solver = linear_solver is not None
        self._symmetric = None
        self.newton_rtol = float(newton_rtol)
        self.newton_atol = float(newton_atol)
        self.linear_forcing = float(linear_forcing)
        self.max_newton = int(max_newton)
        self.line_search = line_search
        self.divergence_tol = float(divergence_tol)
        self.optimal_newton = int(optimal_newton)
        self.max_load_step = float(max_load_step)
        self.max_attempts = int(max_attempts)
        self.extrapolate = bool(extrapolate)
        #: Eisenstat-Walker choice 2 at PETSc's constants, except the start:
        #: PETSc's 0.3 stalls a weak-penalty problem (kappa/mu = 5) that 0.1
        #: solves, and 0.1 costs 5% on the ventricle.
        self.forcing_initial = 0.1
        self.forcing_max = 0.9
        self.forcing_gamma = 1.0
        self.forcing_alpha = 0.5*(1.0 + 5.0**0.5)
        self.forcing_threshold = 0.1
        self._eta = self.forcing_initial
        self._previous_norm = None
        self._indefinite = None
        #: Accept once ``|G(alpha)|`` has fallen to this fraction of ``|G(0)|``;
        #: Crisfield recommends a slack value so the full step is usually taken.
        self.line_search_beta = 0.9      # FEBio lstol
        self.min_alpha = 1e-10
        self.max_alpha = 1.0             # PETSc CP maxlambda, Abaqus s_max, FEBio
        self.max_backtrack = 8
        self.load_steps = int(load_steps)
        self.min_load_step = float(min_load_step)
        self.al_tol = float(al_tol)
        self.max_al = int(max_al)
        self.verbose = bool(verbose)

        self._validate()
        self._newton_count = 0
        self._krylov_count = 0

    def _validate(self) -> None:
        """Reject settings a run could not terminate under.

        ``min_load_step = 0`` is the dangerous one: cutbacks halve the step
        until it underflows, and ``dlam < 0`` never becomes true, so the
        continuation loop retries for ever.
        """
        if self.line_search not in ("critical-point", "none"):
            raise ValueError("line_search must be 'critical-point' or 'none'")
        positive = dict(min_load_step=self.min_load_step, al_tol=self.al_tol,
                        linear_forcing=self.linear_forcing)
        for name, value in positive.items():
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        for name, value in dict(newton_rtol=self.newton_rtol,
                                newton_atol=self.newton_atol).items():
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative, "
                                 f"got {value}")
        if self.newton_rtol == 0.0 and self.newton_atol == 0.0:
            raise ValueError("newton_rtol and newton_atol cannot both be zero")
        for name, value in dict(load_steps=self.load_steps, max_newton=self.max_newton,
                                max_al=self.max_al).items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if not 0.0 < self.linear_forcing < 1.0:
            raise ValueError("linear_forcing must lie strictly between 0 and 1")
        if not (np.isfinite(self.divergence_tol) and self.divergence_tol > 1.0):
            raise ValueError("divergence_tol must exceed 1, got "
                             f"{self.divergence_tol}")
        if not self.min_load_step < self.max_load_step <= 1.0:
            raise ValueError("need min_load_step < max_load_step <= 1, got "
                             f"{self.min_load_step} and {self.max_load_step}")
        for name, value in dict(optimal_newton=self.optimal_newton,
                                max_attempts=self.max_attempts).items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

    # ------------------------------------------------------------------ utils
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _tangent(self, values: torch.Tensor) -> torch.Tensor:
        """Apply Dirichlet elimination to the consistent tangent."""
        asm = self.problem.assembler
        asm.apply_constraints(values)
        return values

    def _residual_norm(self, R: torch.Tensor) -> torch.Tensor:
        """``|R|`` over the free DOFs, on device."""
        return self.problem.constraints.free_norm(R)

    # ----------------------------------------------------------------- newton
    def _system(self, u, load_factor, tangent):
        """Residual and tangent of the system Newton solves.

        Quasi-statics solves the problem's own force balance; a time
        integrator overrides this to add inertia and evaluate at its own
        intermediate state, and inherits everything else -- line search,
        forcing term, Krylov choice, cutbacks -- unchanged.
        """
        return self.problem.evaluate(u, load_factor, tangent)

    def _probe(self, u, load_factor):
        """Evaluate an admissible trial, transferring scalar diagnostics once."""
        try:
            R, _, raw = self._system(u, load_factor, tangent=False)
        except torch.linalg.LinAlgError:
            return None
        info = raw.resolve()
        if not (np.isfinite(info["min_J"]) and info["min_J"] > 0.0
                and np.isfinite(info["residual_norm"])):
            return None
        return R, info

    def _threshold(self, info: dict) -> float:
        """The residual this solve has to reach, in force units.

        Progress is measured against this rather than against the residual the
        solve started from.  Starting from an unloaded state ``|R(0)|`` is only
        the applied load, so a step that correctly raises the internal forces
        looks like divergence when compared with it.
        """
        scales = (info["fint_norm"], info["fext_norm"])
        if not all(np.isfinite(s) and s >= 0.0 for s in scales):
            return float("nan")
        return max(self.newton_atol, self.newton_rtol * max(*scales, 0.0))

    def _converged(self, norm: float, info: dict) -> bool:
        """Force-residual test: absolute *or* relative to the natural force scale.

        The relative scale ``max(|f_int|, |f_ext|)`` vanishes together with the
        residual on an unloaded body, so a relative test alone can never be
        satisfied there.  The absolute tolerance is what makes the trivial case
        terminate; it is in the model's force units and is user-visible.
        """
        threshold = self._threshold(info)
        return bool(np.isfinite(norm) and np.isfinite(threshold)
                    and norm <= threshold)

    def _forcing(self, norm: float, history, threshold: float) -> float:
        """Eisenstat-Walker adaptive forcing term (PETSc's ``-snes_ksp_ew``).

        Solving the first corrections of an increment to a tight tolerance is
        wasted work: the tangent they are built from is a poor model of a
        state far from equilibrium.  Choice 2 of Eisenstat & Walker (1996)
        sets the tolerance from how well the last correction actually
        predicted the residual drop, so early solves are cheap and only the
        final ones are exact.  A fixed tolerance instead pays the tightest
        price at every iteration, which on a weakly preconditioned system is
        where most of the run goes.  Constants are PETSc's defaults.
        """
        if len(history) < 2 or not np.isfinite(norm):
            self._eta = self.forcing_initial
            return self._eta
        previous = self._previous_norm
        if not (previous and np.isfinite(previous) and previous > 0.0):
            return self.forcing_initial
        eta = self.forcing_gamma * (norm/previous) ** self.forcing_alpha
        safeguard = self.forcing_gamma * self._eta ** self.forcing_alpha
        if safeguard > self.forcing_threshold:
            eta = max(eta, safeguard)
        eta = min(eta, self.forcing_max)
        # Eisenstat-Walker assumes the residual decreases from one correction
        # to the next; ours rises over the first few (see :meth:`_newton`), and
        # the raw ratio answers that by *loosening* to the 0.9 cap, which
        # returns a direction so inexact it stops being a descent direction at
        # all.  Requiring the tolerance to be non-increasing within a solve
        # keeps the adaptivity without that failure mode.
        eta = min(eta, self._eta)
        # ...and the paper's other safeguard, against *oversolving*: a linear
        # solve accurate beyond what the nonlinear tolerance can use is pure
        # cost (Eisenstat & Walker 1996, section 3; PETSc SNESKSPEW).  Without
        # it the ratio drives the tolerance to 1e-8 on the finest beam, and a
        # weakly preconditioned BiCGStab cannot reach that in 10,000
        # iterations, so the increment is thrown away instead.
        if np.isfinite(threshold) and threshold > 0.0 and norm > 0.0:
            eta = max(eta, 0.5 * threshold / norm)
        return float(min(max(eta, 1e-12), self.forcing_max))

    def _ensure_solver(self, values: torch.Tensor) -> None:
        """Keep the Krylov method matched to the tangent as the tangent changes.

        Symmetry is a property of the current matrix, not of the problem: a
        follower load that switches on later, or a rate-dependent stiffness,
        turns a symmetric tangent nonsymmetric part-way through a run.
        Choosing once and never looking again leaves conjugate gradients
        running on a matrix that no longer meets their assumptions, which the
        negative-curvature test does not detect.  The test is one comparison
        over the stored values, so it is repeated whenever a Newton solve
        starts.
        """
        if self._user_solver:
            return
        perm = self.problem.assembler.transpose_perm
        scale = float(values.abs().max())
        asymmetry = float((values - values[perm]).abs().max())
        symmetric = scale == 0.0 or asymmetry <= 1e-10*scale
        if symmetric is self._symmetric and self.linear_solver is not None:
            return
        self._symmetric = symmetric
        self._log(f"    tangent is {'symmetric' if symmetric else 'nonsymmetric'} "
                  f"(relative asymmetry {asymmetry/max(scale, 1e-300):.2e}); using "
                  f"{'CG' if symmetric else 'BiCGStab'}")
        self.linear_solver = (ConjugateGradient if symmetric else BiCGStab)(
            BlockJacobiPreconditioner(), rtol=self.forcing_initial, max_iter=10000)

    def _newton(self, u: torch.Tensor, load_factor: float) -> Tuple[bool, torch.Tensor, dict]:
        problem = self.problem
        R, values, raw = self._system(u, load_factor, tangent=True)
        info = raw.resolve()
        # How far from converged, not how far from where we started.
        def distance(state):
            threshold = self._threshold(state)
            if not (np.isfinite(threshold) and threshold > 0.0):
                return float("inf")
            return state["residual_norm"] / threshold

        history = [distance(info)]

        for _ in range(self.max_newton):
            norm = info["residual_norm"]
            if not np.isfinite(norm) or not info["min_J"] > 0.0:
                info["reason"] = ("non-finite residual" if not np.isfinite(norm)
                                  else "inverted element")
                return False, u, info
            if self._converged(norm, info):
                return True, u, info
            # PETSc's SNESSetDivergenceTolerance.  Nothing here tries to
            # detect "converging too slowly": the residual legitimately *rises*
            # over the first several corrections of a load increment, because
            # a correct step develops internal forces far larger than the
            # applied load it is balancing -- on the beam it climbs from
            # 1.3e-3 to 1.4e-1 before falling to 1e-11.  Every monotone-progress
            # test mistakes that for failure.  The iteration ceiling bounds the
            # waste instead, and automatic incrementation keeps healthy
            # increments well inside it.
            if history[-1] > self.divergence_tol * history[0]:
                info["reason"] = "residual diverged"
                return False, u, info

            values = self._tangent(values)
            # Every matrix, not only the first of the solve: the tangent can
            # lose symmetry between corrections -- a follower load or a
            # rate-dependent term switching on part-way through -- and a method
            # chosen for the first matrix is not justified for the rest.
            self._ensure_solver(values)
            A = problem.assembler.to_csr(values)
            rhs = problem.constraints.zero_constrained(R.clone())

            eta = self._forcing(norm, history, self._threshold(info))
            self._eta, self._previous_norm = eta, norm
            self.linear_solver.rtol = eta
            self.linear_solver.preconditioner.update(problem.assembler, values)
            res = self.linear_solver.solve(A, -rhs)
            self._krylov_count += res.iterations
            self._newton_count += 1

            # Conjugate gradients is only valid while the tangent is positive
            # definite, and active stress or a pressure load can take Newton
            # through iterates where it is not.  The rule the reference codes
            # use (Ratel: "GMRES for systems that are not SPD") is to solve
            # such a system with a method that does not need definiteness --
            # the exact Newton direction is what converges here, and a
            # truncated or steepest-descent direction is not.  BiCGStab is
            # that method for us.  CG stays the default for the next solve,
            # since the tangent is usually definite again once equilibrium is
            # near.
            if res.truncated:
                if self._indefinite is None:
                    self._indefinite = BiCGStab(BlockJacobiPreconditioner(),
                                                rtol=eta, max_iter=20000)
                self._indefinite.rtol = eta
                self._indefinite.preconditioner.update(problem.assembler, values)
                res = self._indefinite.solve(A, -rhs)
                self._krylov_count += res.iterations

            # An *inexact* correction is worth taking, but only if it is
            # actually useful.  Inexact-Newton methods bound the true relative
            # linear residual below one; a solve that did no better than the
            # zero vector is a breakdown, and retrying it just burns assemblies
            # at an unchanged state.
            relative = res.residual / norm if norm > 0.0 else float("inf")
            if not res.converged and not (np.isfinite(relative)
                                          and relative < self.linear_forcing):
                info["reason"] = "linear solve gave no useful correction"
                return False, u, info

            du = problem.constraints.zero_constrained(res.x)
            searching = self.line_search == "critical-point"
            # G(0) = du . R.  Crisfield's search looks for the root of G and
            # does not need G(0) < 0: on an indefinite tangent the exact
            # Newton direction can have either sign and still be the step
            # that converges.  A vanishing G(0) means the full step already
            # sits at the energy stationary point along du, so there is
            # nothing to search for.
            slope = float(torch.dot(du, rhs)) if searching else 0.0
            if not np.isfinite(slope):
                info["reason"] = "non-finite residual"
                return False, u, info

            alpha, trial = self._line_search(u, du, load_factor, norm, slope,
                                             searching)
            if trial is None:
                info["reason"] = ("line search could not reduce the residual"
                                  if searching else "invalid Newton trial")
                return False, u, info
            u = u + alpha*du
            R, values, raw = self._system(u, load_factor, tangent=True)
            info = raw.resolve()
            history.append(distance(info))

        done = self._converged(info["residual_norm"], info)
        if not done:
            info["reason"] = f"no convergence in {self.max_newton} Newton steps"
        return done, u, info

    def _line_search(self, u, du, load_factor, norm, slope, searching):
        """Crisfield's line search on the energy slope; returns ``(alpha, trial)``.

        ``slope`` is ``G(0) = du . R(u)``, negative for a descent direction.
        The search looks for ``G(alpha) = 0`` by secant iteration, accepting as
        soon as ``|G|`` has fallen to ``line_search_beta`` of its initial value
        -- usually at the full step, which is what keeps Newton quadratic.

        Using ``|R|`` as the merit instead would reject the full step whenever
        the internal forces grow faster than the applied load, which is the
        normal situation early in a load step.  See the class docstring.
        """
        if not searching or slope == 0.0:
            return 1.0, self._probe(u + du, load_factor)

        alpha, best = 1.0, None
        for _ in range(self.max_backtrack):
            trial = self._probe(u + alpha*du, load_factor)
            if trial is None:                       # outside the domain
                alpha *= 0.5
                if alpha < self.min_alpha:
                    break
                continue
            g = float(torch.dot(du, trial[0]))
            if not np.isfinite(g):
                alpha *= 0.5
                if alpha < self.min_alpha:
                    break
                continue
            if best is None or abs(g) < best[0]:
                best = (abs(g), alpha, trial)
            if abs(g) <= self.line_search_beta * abs(slope):
                return alpha, trial
            # Secant through (0, G(0)) and (alpha, G(alpha)), safeguarded.
            denom = slope - g
            nxt = alpha*slope/denom if denom != 0.0 else 0.5*alpha
            alpha = min(max(nxt, 0.1*alpha), self.max_alpha)
            if alpha < self.min_alpha:
                break
        # No root found within the budget: take the best admissible trial we
        # saw.  Crisfield treats the search as an improvement, not a
        # requirement -- refusing to move at all is the worse failure.
        if best is not None:
            return best[1], best[2]
        return None, None

    # ---------------------------------------------------------- load stepping
    def _solve_step(self, u: torch.Tensor, load_factor: float
                    ) -> Tuple[bool, torch.Tensor, dict]:
        """One load level: Uzawa updates around Newton, with full rollback.

        A step succeeds only if Newton converged *and* the incompressibility
        constraint holds at that converged state -- the constraint is therefore
        rechecked after the last Newton solve, not merely before it.  Any other
        outcome restores both the displacement and the multiplier field, so the
        continuation loop can retry a smaller step from the state it accepted.
        """
        problem = self.problem
        u_backup = u.clone()
        p_bar = problem.p_bar
        p_bar_backup = None if p_bar is None else p_bar.clone()

        u = problem.apply_dirichlet(u, load_factor)
        info: dict = {"min_J": float("nan")}
        first_round = None

        for _ in range(self.max_al):
            before = self._newton_count
            ok, u, info = self._newton(u, load_factor)
            if first_round is None:
                # The first multiplier round measures how nonlinear the load
                # increment itself is; later rounds only re-solve for an
                # updated multiplier and say nothing about the increment.
                first_round = self._newton_count - before
                info["step_newton"] = first_round
            if not ok:
                break
            err = problem.volume_error(u)
            info["volume_error"] = err
            info["step_newton"] = first_round
            if err <= self.al_tol:
                return True, u, info
            problem.update_multipliers(u)
        else:
            info["reason"] = ("mixed volume constraint iteration limit")

        if p_bar_backup is not None:
            problem.p_bar.copy_(p_bar_backup)
        return False, u_backup, info

    def _predict(self, u, increment, dlam, dlam_done, load_factor):
        """Start the increment from a linear extrapolation of the last one.

        Abaqus extrapolates the previous increment into the next one by
        default, and for the same reason: restarting Newton from the old
        converged state throws away everything the previous increment learned,
        and on a stiffening material that costs the corrections that keep the
        iteration inside its quadratic regime.  The extrapolation is scaled by
        the ratio of load increments, and is discarded if it lands outside the
        admissible region -- a predictor is only ever a guess.
        """
        if not (self.extrapolate and increment is not None and dlam_done):
            return u.clone()
        if dlam < 0.1 * dlam_done:       # Abaqus D_E: too far from the last step
            return u.clone()
        guess = u + (dlam / dlam_done) * increment
        if self._probe(guess, load_factor) is None:
            return u.clone()
        return guess

    def solve(self, u: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, SolveReport]:
        """Ramp the load from zero to full and return ``(u, report)``."""
        problem = self.problem
        # Scoped to this solve, so a refinement study reports per-mesh peaks.
        cuda = problem.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(problem.device)
            torch.cuda.synchronize(problem.device)
        t0 = time.time()
        self._newton_count = 0
        self._krylov_count = 0
        cutbacks: Dict[str, int] = {}

        u = torch.zeros(problem.n_dofs, dtype=problem.dtype,
                        device=problem.device) if u is None else u.clone()

        lam = 0.0
        dlam = min(1.0 / max(self.load_steps, 1), self.max_load_step)
        n_steps = 0
        dlam_prev = dlam
        attempts = 0
        increment = None                 # last accepted du, for extrapolation
        dlam_done = None
        info: dict = {"min_J": float("nan")}

        while lam < 1.0 - 1e-12:
            target = min(1.0, lam + dlam)
            if target <= lam:            # the step can no longer advance the load
                self._log(f"  cannot advance past lambda = {lam:.6f}")
                break
            start = self._predict(u, increment, target - lam, dlam_done, target)
            ok, u_try, info = self._solve_step(start, target)
            if ok:
                increment = u_try - u
                dlam_done = target - lam
                u = u_try
                lam = target
                n_steps += 1
                used = info.get("step_newton", self.max_newton)
                self._log(f"  load step {n_steps:2d}: lambda = {lam:.6f}  "
                          f"newton = {used:3d}  "
                          f"mixed constraint = {info.get('volume_error', float('nan')):.2e}  "
                          f"min J = {info['min_J']:.6f}")
                # FEBio's automatic time stepper (FETimeStepController::
                # AutoTimeStep), verbatim: the increment moves toward its
                # ceiling by at most 20% of the remaining gap when the last one
                # was easier than ``optimal_newton`` corrections, and eases off
                # in proportion when it was harder.  It is self-limiting -- an
                # increment that took exactly the optimum leaves the size alone
                # -- so it cannot produce the periodic overshoot that growing
                # after every success does.
                scale = (self.optimal_newton / max(used, 1)) ** 0.5
                if scale >= 1.0:
                    dlam = dlam + (self.max_load_step - dlam) * min(0.20, scale - 1.0)
                    dlam = min(dlam, 5.0 * dlam_prev, self.max_load_step)
                else:
                    dlam = dlam - (dlam - self.min_load_step) * (1.0 - scale)
                dlam = max(dlam, self.min_load_step)
                dlam_prev = dlam
                attempts = 0
            else:
                why = info.get("reason", "unknown")
                cutbacks[why] = cutbacks.get(why, 0) + 1
                attempts += 1
                # Abaqus *CONTROLS: a quarter for a domain failure or
                # divergence (D_f), a half for an iteration limit or a line
                # search that ran out (D_c); at most I_A = 5 attempts at one
                # load level, which FEBio's max_retries agrees on.
                domain = why in ("residual diverged", "non-finite residual",
                                 "inverted element", "invalid Newton trial")
                dlam *= 0.25 if domain else 0.5
                self._log(f"  cutback ({why}): dlambda -> {dlam:.3e}")
                if attempts >= self.max_attempts or dlam < self.min_load_step:
                    break

        _, _, raw = self._system(u, lam, tangent=False)
        final = raw.resolve()
        volume = problem.volume_diagnostics(u)
        if cuda:
            torch.cuda.synchronize(problem.device)
        peak = (torch.cuda.max_memory_allocated(problem.device) / 2 ** 20
                if cuda else 0.0)
        report = SolveReport(
            converged=(lam >= 1.0 - 1e-12
                       and np.isfinite(final["residual_norm"])
                       and final["min_J"] > 0.0
                       and self._converged(final["residual_norm"], final)
                       and volume["constraint_error"] <= self.al_tol),
            load_factor=lam,
            load_steps=n_steps,
            newton_iterations=self._newton_count,
            linear_iterations=self._krylov_count,
            volume_error=volume["constraint_error"],
            min_jacobian=final["min_J"],
            residual=final["residual_norm"],
            wall_time=time.time() - t0,
            volume=volume,
            peak_memory_mib=peak,
            cutbacks=cutbacks,
        )
        return u, report


class DynamicSolver(QuasiStaticSolver):
    r"""Elastodynamics by the generalized-alpha method.

    Chung & Hulbert (1993).  The unknown of each step is the end-of-step
    displacement; acceleration and velocity follow from the Newmark relations,
    and the balance is imposed at the intermediate states
    :math:`t_{n+1-\alpha_f}`, which is what lets the scheme damp the
    unresolved high frequencies of a stiff mesh without damping the physical
    motion.  ``rho_infinity`` is the spectral radius at infinite frequency:
    ``1`` is the undamped trapezoidal rule, ``0.5`` -- the default, and what
    the cardiac benchmarks use -- damps aggressively, ``0`` maximally.

    Everything except the residual comes from :class:`QuasiStaticSolver`: the
    energy-slope line search, the adaptive forcing term, the Krylov choice and
    the cutback rules all apply per time step.  A step that fails is retried as
    two half steps rather than abandoned, so a difficult moment costs
    resolution there and nowhere else.

    Loads that vary are given as schedules -- a callable pressure or active
    tension -- and are evaluated at the intermediate time of each step.  A
    constant load stays constant: physical time is not a continuation factor,
    so nothing here ramps with the clock.  Prescribed displacements are applied
    at their full value.

    Parameters
    ----------
    dt:
        Time step.
    rho_infinity:
        Numerical damping of the highest resolved frequency, in ``[0, 1]``.
    """

    def __init__(self, problem, dt: float, rho_infinity: float = 0.5,
                 **options) -> None:
        if problem.mass is None:
            raise ValueError("dynamics needs the mass matrix: build the problem "
                             "with a nonzero density")
        super().__init__(problem, **options)
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError(f"dt must be finite and positive, got {dt}")
        if not 0.0 <= rho_infinity <= 1.0:
            raise ValueError(f"rho_infinity must lie in [0, 1], got {rho_infinity}")
        self.dt = float(dt)
        rho = float(rho_infinity)
        self.alpha_m = (2.0*rho - 1.0)/(rho + 1.0)
        self.alpha_f = rho/(rho + 1.0)
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        self.beta = 0.25*(1.0 - self.alpha_m + self.alpha_f)**2
        self.mass = problem.assembler.to_csr(problem.mass)

    def _states(self, u: torch.Tensor):
        """Newmark acceleration and velocity implied by an end-of-step ``u``."""
        dt, beta, gamma = self._dt, self.beta, self.gamma
        a = (u - self._u - dt*self._v - dt*dt*(0.5 - beta)*self._a)/(beta*dt*dt)
        v = self._v + dt*((1.0 - gamma)*self._a + gamma*a)
        return v, a

    def _system(self, u, load_factor, tangent):
        v, a = self._states(u)
        af, am = self.alpha_f, self.alpha_m
        R, values, info = self.problem.evaluate(
            (1.0 - af)*u + af*self._u, 1.0, tangent,
            velocity=(1.0 - af)*v + af*self._v,
            velocity_scale=self.gamma/(self.beta*self._dt), time=self._time)

        inertia = self.mass @ ((1.0 - am)*a + am*self._a)
        R = R + inertia
        if tangent:
            values = ((1.0 - af)*values
                      + ((1.0 - am)/(self.beta*self._dt**2))*self.problem.mass)
        # The residual and its scale must both count the inertia, or the
        # convergence test measures a force the step is not balancing.
        info = replace(
            info, residual_norm=self.problem.constraints.free_norm(R),
            internal_norm=torch.maximum(info.internal_norm,
                                        torch.linalg.vector_norm(inertia)))
        return R, values, info

    def solve(self, t_end: float, u: Optional[torch.Tensor] = None,
              velocity: Optional[torch.Tensor] = None,
              observer=None) -> Tuple[torch.Tensor, SolveReport]:
        """March from ``t = 0`` to ``t_end``.

        ``u`` and ``velocity`` are the initial conditions; the initial
        acceleration is not assumed but solved from ``M a = -R(u, v)``, since
        a body released away from equilibrium accelerates from the first
        instant and starting it at rest would misplace the whole history.

        ``observer(t, u, v, a)`` is called after each accepted step, which is
        how a time history is recorded without the solver knowing what is
        being measured.
        """
        problem = self.problem
        cuda = problem.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(problem.device)
            torch.cuda.synchronize(problem.device)
        start = time.time()
        self._newton_count = self._krylov_count = 0
        cutbacks: Dict[str, int] = {}

        self._u, self._v = self._initial_state(u, velocity)
        self._a = self._initial_acceleration()
        if observer is not None:
            observer(0.0, self._u, self._v, self._a)

        t, steps, info = 0.0, 0, {"min_J": float("nan")}
        # The initial state counts: a body released from a compressed
        # configuration is at its worst before the first step is taken.
        worst, worst_at = float(problem.jacobians(self._u).amin()), 0.0
        while t < t_end - 1e-12:
            dt = min(self.dt, t_end - t)
            ok, info = self._advance(t, dt)
            while not ok and dt > self.dt*self.min_load_step:
                why = info.get("reason", "unknown")
                cutbacks[why] = cutbacks.get(why, 0) + 1
                dt *= 0.5
                self._log(f"  cutback ({why}): dt -> {dt:.3e}")
                ok, info = self._advance(t, dt)
            if not ok:
                break
            t += dt
            steps += 1
            step_min = info.get("end_min_J", float("nan"))
            if step_min == step_min and step_min < worst:      # skips NaN
                worst, worst_at = step_min, t
            self._log(f"  t = {t:.6f}  newton = {info.get('step_newton', 0):3d}  "
                      f"min J = {step_min:.6f}")
            if observer is not None:
                observer(t, self._u, self._v, self._a)

        # The residual of the last accepted step, not a fresh evaluation: the
        # state has already advanced, so recomputing here would difference the
        # converged displacement against itself and report a spurious
        # acceleration -- and with it a residual the solve never had.
        volume = problem.volume_diagnostics(self._u)
        if cuda:
            torch.cuda.synchronize(problem.device)
        return self._u, SolveReport(
            converged=t >= t_end - 1e-12, load_factor=t, load_steps=steps,
            newton_iterations=self._newton_count,
            linear_iterations=self._krylov_count,
            volume_error=volume["constraint_error"],
            min_jacobian=info.get("end_min_J", volume["min_jacobian"]),
            residual=info.get("residual_norm", float("nan")),
            wall_time=time.time() - start,
            volume=volume, cutbacks=cutbacks,
            peak_memory_mib=(torch.cuda.max_memory_allocated(problem.device)/2**20
                             if cuda else 0.0),
            worst_jacobian=worst if worst < float("inf") else float("nan"),
            worst_jacobian_time=worst_at)

    def _initial_state(self, u, velocity):
        """Initial displacement and velocity, checked against the boundary.

        A degree of freedom held fixed for all time has a prescribed
        displacement and therefore no velocity.  Initial data saying otherwise
        describes a different problem from the one the boundary conditions
        state, and every later step inherits the contradiction, so it is
        rejected here rather than quietly overwritten.  Omitted data is
        initialised to satisfy the boundary.
        """
        problem = self.problem
        zero = torch.zeros(problem.n_dofs, dtype=problem.dtype,
                           device=problem.device)
        fixed = problem.constraints.mask
        atol = problem.constraints.atol
        u = problem.apply_dirichlet(zero.clone(), 1.0) if u is None else u.clone()
        v = zero.clone() if velocity is None else velocity.clone()
        for name, field in (("displacement", u), ("velocity", v)):
            if field.shape != zero.shape:
                raise ValueError(f"initial {name} must have {zero.numel()} "
                                 f"entries, got {tuple(field.shape)}")
            if not bool(torch.isfinite(field).all()):
                raise ValueError(f"initial {name} must be finite")
        if float((fixed*(u - problem.apply_dirichlet(u.clone(), 1.0))).abs().max()) > atol:
            raise ValueError("initial displacement disagrees with the prescribed "
                             "Dirichlet values on constrained degrees of freedom")
        if float((fixed*v).abs().max()) > atol:
            raise ValueError("initial velocity is nonzero on degrees of freedom "
                             "held fixed for all time")
        return u, v

    def _initial_acceleration(self) -> torch.Tensor:
        """Solve ``M a = -R(u, v)`` at ``t = 0``, on the free degrees of freedom.

        The elimination has to happen *before* the solve.  Solving the whole
        mass system and zeroing the constrained accelerations afterwards is not
        the same problem: the constrained columns still couple into the free
        equations, so every free acceleration comes out wrong.
        """
        problem = self.problem
        R = problem.evaluate(self._u, 1.0, tangent=False, velocity=self._v,
                             time=0.0)[0]
        rhs = problem.constraints.zero_constrained(-R)
        if not bool(rhs.any()):
            return torch.zeros_like(rhs)

        # A copy: the physical mass matrix is still needed, unconstrained, by
        # the inertia term of every step.
        mass = problem.mass.clone()
        problem.assembler.apply_constraints(mass)
        solver = ConjugateGradient(BlockJacobiPreconditioner(), rtol=1e-10,
                                   max_iter=2000)
        solver.preconditioner.update(problem.assembler, mass)
        result = solver.solve(problem.assembler.to_csr(mass), rhs)
        if not result.converged:
            raise RuntimeError("could not solve for the initial acceleration "
                               f"(relative residual {result.residual:.2e})")
        return problem.constraints.zero_constrained(result.x)

    def _advance(self, t: float, dt: float) -> Tuple[bool, dict]:
        """One step of size ``dt``; on success the state is advanced.

        Generalized-alpha balances forces at an *intermediate* state, so that
        state being admissible says nothing about the end state that actually
        gets committed -- a step can converge on an intermediate configuration
        with positive Jacobians and still end on an inverted one.  The end
        state is therefore checked on its own before it is accepted.
        """
        problem = self.problem
        self._dt = dt
        self._time = t + (1.0 - self.alpha_f)*dt
        p_bar = problem.p_bar
        p_bar_backup = None if p_bar is None else p_bar.clone()

        ok, u, info = self._solve_step(self._u.clone(), 1.0)
        if not ok:
            return False, info

        v, a = self._states(u)
        end_min_J = float(problem.jacobians(u).amin())
        finite = all(bool(torch.isfinite(x).all()) for x in (u, v, a))
        info = dict(info, end_min_J=end_min_J)
        if not (finite and end_min_J > 0.0):
            info["reason"] = ("non-finite end state" if not finite
                              else f"inverted end state (min J = {end_min_J:.4f})")
            if p_bar_backup is not None:
                p_bar.copy_(p_bar_backup)
            return False, info

        self._u, self._v, self._a = u, v, a
        return True, info
