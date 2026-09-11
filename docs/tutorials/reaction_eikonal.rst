Reaction-Eikonal model
======================

The reaction-eikonal model recovers the transmembrane voltage at a fraction of
monodomain cost.  The **eikonal** solver computes the wavefront arrival time
``t_a(x)`` from prescribed conduction velocities; a **reaction** step then
triggers each cell's action potential at its own ``t_a``.

Two variants are selected with the ``diffusion`` flag:

* ``diffusion=False`` (R-E\ :sup:`-`) -- cells are electrically independent;
  there is no linear solve, so it is fast.  Only ``add_velocity`` is needed.
* ``diffusion=True`` (R-E\ :sup:`+`) -- adds the monodomain diffusion term
  (a conjugate-gradient solve every step) for electrotonic coupling; this also
  needs ``add_conductivity``.

Activation map only (the fast standalone step)
----------------------------------------------

The eikonal field *is* the activation map -- no reaction needed:

.. code-block:: python

   from torchcor.electrophysiology import ReactionEikonal
   from torchcor.ionic import ModifiedMS2v

   im = ModifiedMS2v(dt=0.01)
   sim = ReactionEikonal([im], T=500, dt=0.01, diffusion=False)

   sim.load_mesh("/path/to/atrium/Case_1")
   sim.add_velocity([1, 2, 3, 4, 5, 6], vel_l=0.6, vel_t=0.3)   # m/s
   sim.add_stimulus("/path/to/atrium/Case_1/Case_1.vtx",
                    start=0.0, duration=2.0, intensity=50)

   AT = sim.eikonal_activation_times()   # wavefront arrival time at every node

Full transmembrane voltage (e.g. as the ECG source)
---------------------------------------------------

.. code-block:: python

   Vm = sim.solve(snapshot_interval=1, result_path="./out")   # reuses the eikonal AT
   sim.save_vm(Vm)                                            # source for the lead-field ECG

.. note::
   ``add_velocity`` (conduction velocity, m/s) drives the eikonal.  For
   ``diffusion=True`` the ``add_conductivity`` values must imply the *same*
   conduction velocity, otherwise the diffusion-driven front and the eikonal
   arrival times diverge.  ``diffusion=False`` uses only the velocity.

How it works
------------

**Eikonal stage.**
``add_velocity`` builds a per-element metric ``M = V⁻¹`` from the conduction
velocities, where ``V`` holds the squared velocities along and across the fibre.
The travel time of a step ``d`` through an element is then ``√(dᵀ M d)`` =
distance ÷ velocity.  The **Fast Iterative Method** relaxes the arrival-time
field: starting from the stimulus seeds, each node's time is repeatedly updated
from its neighbouring simplices — a vertex is updated from its opposite edge
(triangles) or face (tetrahedra) as

.. math::

   t_a(\text{node}) = \min_{\text{neighbours}}\;\big(t_a(\text{nbr}) + \text{travel time}\big),

swept over the whole mesh until the field stops changing.  The result is
``t_a(x)`` — the activation map — with **no reaction and no linear solve**.

**Reaction stage.**
A foot current switches on in a short window around each node's ``t_a`` and ramps
the membrane up to the threshold ``V_th``; the cell's own ionic model then fires
the upstroke, reproducing the full action potential.  The foot is gated off once
``Vm`` crosses ``V_th``, so the intrinsic upstroke takes over and the front fires
at ``t_a``.  The same foot also triggers the seed nodes, so no separate stimulus
current is injected.

The ``diffusion`` flag then chooses how Vm is advanced each step:

* ``diffusion=False`` integrates ``Cm dVm/dt = I_foot − I_ion`` independently per
  node — no coupling, no linear solve (just ``u = b / Cm``).
* ``diffusion=True`` additionally assembles and solves the monodomain diffusion
  term with conjugate gradient (see :doc:`monodomain`), so neighbouring cells
  exchange current.

.. seealso::
   The trigger current can be switched from the default action-potential foot to
   a fitted triple-Gaussian via ``ReactionEikonal.set_diffusion_current``.
