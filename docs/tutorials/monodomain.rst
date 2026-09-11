Monodomain simulation
=====================

The monodomain model is the full reaction-diffusion reference: it solves

.. math::

   \beta C_m \frac{\partial V_m}{\partial t}
       = \nabla \cdot (\sigma_m \nabla V_m) - \beta I_\mathrm{ion}(V_m, \eta) + I_\mathrm{stim}

on the heart mesh, with an implicit (conjugate-gradient) diffusion solve and an
explicit ionic update each time step.

1. Ionic model
--------------

Pick a cell model and (optionally) tune its parameters:

.. code-block:: python

   import torchcor as tc
   from torchcor.electrophysiology import Monodomain
   from torchcor.ionic import TenTusscherPanfilov

   tc.set_device("cuda:0")
   dtype = tc.float64

   im = TenTusscherPanfilov(cell_type="ENDO", dt=0.01, dtype=dtype)

For multiple tissue types, pass a **list** of models; assign each to regions with
its ``region_ids`` attribute (unset models cover the remaining regions).

2. Build the simulator and load the mesh
----------------------------------------

.. code-block:: python

   sim = Monodomain(ionic_models=[im], T=500, dt=0.01, dtype=dtype,
                    mass_lumping=True)
   sim.load_mesh(path="/path/to/heart", unit_conversion=1000)

* ``T`` is the total duration (ms), ``dt`` the time step (ms).
* ``load_mesh`` reads CARP ``.pts`` / ``.elem`` / ``.lon`` files; ``unit_conversion``
  divides the node coordinates (CARP stores micrometres, so ``1000`` gives mm).
* ``mass_lumping=True`` lumps the FEM mass matrix — this matches openCARP's
  default and gives the correct conduction velocity (the consistent mass matrix
  conducts ~35% too fast).

3. Conductivities
-----------------

Anisotropic bidomain conductivities (S/m) per region.  ``il``/``it`` are
intracellular longitudinal/transverse, ``el``/``et`` extracellular; the
monodomain effective conductivity is their harmonic mean.

.. code-block:: python

   sim.add_conductivity([24, 25],     il=0.5272, it=0.2076, el=1.0732, et=0.4227)
   sim.add_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

(Passing only ``il``/``it`` uses them directly as the monodomain conductivity.)

4. Stimuli
----------

Each stimulus depolarises the nodes listed in a ``.vtx`` file:

.. code-block:: python

   for name, start in [("LV_sf", 0.0), ("LV_pf", 0.0), ("LV_af", 0.0),
                       ("RV_sf", 5.0), ("RV_mod", 5.0)]:
       sim.add_stimulus(f"/path/to/heart/pacing/{name}.vtx",
                        start=start, duration=1.0, intensity=100)

For pacing, pass ``period`` and ``count`` to repeat a stimulus.

5. Solve
--------

.. code-block:: python

   Vm = sim.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100,
                  snapshot_interval=1, verbose=True, result_path="./out")

``Vm`` is a ``(T, N_nodes)`` tensor sampled every ``snapshot_interval`` ms.
``a_tol`` / ``r_tol`` / ``max_iter`` control the per-step CG solve.

6. Post-processing
------------------

.. code-block:: python

   ATs = sim.compute_activation_map(Vm, snapshot_interval=1, threshold=0)
   RTs = sim.compute_repolarization_map(Vm, search_after=ATs,
                                        snapshot_interval=1, threshold=-40)

   sim.save_vm(Vm)                  # -> result_path/Vm.pt  (e.g. for the ECG solver)
   sim.vm_to_vtk(Vm=Vm, step=10)    # VTK frames + ATs/RTs for ParaView

* ``compute_activation_map`` returns the local activation time (first up-crossing
  of ``threshold``) per node, ``NaN`` where the node never activated.
* ``compute_repolarization_map`` finds the down-crossing after ``search_after``
  (the activation map), so it returns proper repolarisation, not the upstroke.

How the solve works
-------------------

Internally, ``solve()`` does the following.

**1. Assembly (once).**
The mesh is turned into finite-element **mass** ``M`` and **stiffness** ``K``
matrices, where ``K`` carries the conductivity tensor ``sigma_m``.  The system
matrix

.. math::

   A = C_m\,M + \theta\,\Delta t\,K

is assembled and a Jacobi preconditioner is built for the conjugate-gradient
(CG) solver.  With ``mass_lumping=True`` the mass matrix is diagonalised, which
speeds up the solve and corrects the conduction velocity.

**2. Time stepping (operator splitting).**
Each ``dt`` is split into a reaction half-step and a diffusion half-step:

* **Reaction (explicit).**  Every node advances its ionic model one step
  (``differentiate``) — an embarrassingly parallel per-node ODE that fills the
  right-hand side ``b`` together with any stimulus current.
* **Diffusion (implicit).**  ``b`` is multiplied by ``M``, the explicit part of
  the diffusion is subtracted, and ``A u = b`` is solved with preconditioned CG.
  Because the diffusion is implicit it is unconditionally stable, so ``dt`` is
  limited only by the stiffness of the ionic model.

**3. Snapshots.**
``Vm`` is recorded every ``snapshot_interval`` ms and returned as a
``(T, N_nodes)`` tensor.

The per-step cost is dominated by the CG diffusion solve — which is exactly what
the :doc:`reaction_eikonal` model avoids when you don't need the full physics.

.. tip::
   If you only need activation timing, the :doc:`reaction_eikonal` model gives
   it far faster.  Use the monodomain when you need the full diffusion physics
   (e.g. as the most accurate source for ECGs).
