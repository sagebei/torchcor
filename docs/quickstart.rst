Quickstart
==========

A minimal **monodomain** simulation on an atrial surface mesh:

.. code-block:: python

   import torchcor as tc
   from torchcor.electrophysiology import Monodomain
   from torchcor.ionic import ModifiedMS2v

   tc.set_device("cuda:0")
   dtype = tc.float32

   im = ModifiedMS2v(dt=0.01, dtype=dtype)
   sim = Monodomain(ionic_models=[im], T=500, dt=0.01, dtype=dtype)

   sim.load_mesh(path="/path/to/atrium/Case_1", unit_conversion=1000)
   sim.add_conductivity(region_ids=[1, 2, 3, 4, 5, 6], il=0.3, it=0.06)
   sim.add_stimulus("/path/to/atrium/Case_1/Case_1.vtx",
                    start=0.0, duration=2.0, intensity=50)

   Vm = sim.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100,
                  snapshot_interval=1, result_path="./out")

``Vm`` is a ``(T, N_nodes)`` tensor of the transmembrane voltage.  Post-process
it into activation / repolarisation maps, export to VTK, or feed it to the ECG
lead-field solver.

For the faster eikonal-based workflow and for ECGs, see :doc:`tutorials/index`.
