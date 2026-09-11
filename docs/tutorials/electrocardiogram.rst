ECGs and electrograms (lead field)
==================================

:class:`~torchcor.electrocardiogram.leadfield.LeadField` turns a cardiac transmembrane voltage
``Vm`` into body-surface potentials using the **reciprocal lead-field method** on
a heart-torso finite-element mesh.  It solves the elliptic problem

.. math::

   -\nabla \cdot (G \nabla u) = \nabla \cdot (\sigma_i \nabla V_m)

once per electrode (a Right-Leg-grounded adjoint solve), so each subsequent ECG
is just a dense matrix-vector product ``Vm(t) . q_electrode`` — cheap for the
whole time trace.

Mesh requirement
----------------

The heart mesh must be the **same sub-domain of the torso mesh** (shared node
ordering), so that the ``Vm`` you computed on the heart maps onto the torso
nodes.  Extract the heart with ``torchcor.electrocardiogram.torso_heart.TorsoHeartMesh``;
``LeadField.build()`` verifies the node correspondence.

Step 1 -- get the source ``Vm``
-------------------------------

Run a heart EP simulation (monodomain for full fidelity, or reaction-eikonal for
a cheap source) and keep ``Vm`` on the heart mesh:

.. code-block:: python

   # monodomain (most accurate) -- see the Monodomain tutorial
   Vm = sim.solve(..., result_path="./out")          # (T, N_heart)
   # ... or the cheap reaction-eikonal source:
   #   Vm = reaction_eikonal_sim.solve(...)

Step 2 -- assemble the lead field
---------------------------------

.. code-block:: python

   import torch
   from torchcor.electrocardiogram import LeadField

   lf = LeadField(torso_mesh_dir="/path/to/torso",
                  heart_mesh_dir="/path/to/heart",
                  device=torch.device("cuda:0"), dtype=torch.float64)

   # torso conductivities (S/m, isotropic) for every non-heart region tag
   lf.add_torso_conductivity([10, 11, 12], g=0.6667)
   lf.add_torso_conductivity([3], g=0.05)
   # ... one call per torso tag ...

   # heart conductivities (S/m, anisotropic bidomain) -- same as the EP run
   lf.add_heart_conductivity([24, 25],     il=0.5272, it=0.2076, el=1.0732, et=0.4227)
   lf.add_heart_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

   lf.build()

The solve runs in **float64** (lead-field accuracy degrades badly in float32).

Step 3 -- electrodes and lead-field precompute
----------------------------------------------

.. code-block:: python

   lf.load_electrodes("/path/to/electrodes/lf_src.vtx")   # V1..V6, RA, LA, RL, LL
   lf.precompute_all(a_tol=1e-8, r_tol=1e-8, max_iter=20000)

This is one CG solve per electrode (nine, with RL as the ground reference).

Step 4 -- compute and plot the 12-lead ECG
------------------------------------------

.. code-block:: python

   ecg = lf.compute_12lead(Vm.cpu())     # dict: I, II, III, aVR, aVL, aVF, V1..V6
   lf.plot_ecg(ecg, "ecg_12lead.png", dt=1.0)

``compute_12lead`` forms the Einthoven / Goldberger limb leads and the Wilson
central-terminal precordial leads, and prints an Einthoven consistency check
(``II == I + III``).

.. note::
   To sanity-check the whole pipeline independently of the reciprocity
   assumption, ``lf.validate_direct(Vm, frames=[...])`` solves the full forward
   problem at a few frames and compares it to the lead-field result — they must
   agree to solver tolerance.  If the ECG still looks wrong after that passes,
   the cause is the ``Vm`` source, not the lead field.

How it works
------------

``build()`` assembles two stiffness matrices on the **torso** mesh:

* ``K_bulk`` with the bulk conductivity ``G`` (``sigma_i + sigma_e`` inside the
  heart, ``sigma_T`` in the surrounding torso), and
* ``K_i`` with the intracellular conductivity ``sigma_i`` on the **heart elements
  only** — the cardiac current source.

The trick is **reciprocity**.  Instead of solving the elliptic problem for every
time frame, ``precompute_all()`` solves the grounded *adjoint* problem **once per
electrode**,

.. math::

   K_\mathrm{bulk}\, Z_e = e_e, \qquad Z_e(\text{ground}) = 0,

and stores the **lead-field vector** ``q_e = −K_i Z_e`` restricted to the heart
nodes.  The potential at electrode ``e`` is then just a dense product,

.. math::

   V_e(t) = V_m(t) \cdot q_e,

so the entire ECG time trace costs **one matrix-vector product per electrode** —
not one large elliptic solve per frame.  This is why a 500 ms, 1 kHz ECG is
essentially free once the nine lead fields are precomputed.
