Ionic models
============

Biophysical cell models live in :mod:`torchcor.ionic`.  Each exposes the same
minimal interface the simulators use:

* ``initialize(n_nodes)`` -- allocate per-node state, return the resting Vm;
* ``differentiate(Vm)``   -- one ionic-current update step.

Pass an instance (or a list, one per region) to
:class:`~torchcor.simulator.monodomain.Monodomain` or ``ReactionEikonal``.

Available models
----------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Class
     - Description
   * - ``ModifiedMS2v``
     - Modified two-variable Mitchell-Schaeffer model (fast, phenomenological).
   * - ``MitchellSchaeffer``
     - Mitchell-Schaeffer phenomenological model.
   * - ``CourtemancheRamirezNattel``
     - Human **atrial** myocyte model.
   * - ``TenTusscherPanfilov``
     - Human **ventricular** myocyte model
       (``cell_type`` = ``ENDO`` / ``MID`` / ``EPI``).

Example
-------

.. code-block:: python

   from torchcor.ionic import ModifiedMS2v, TenTusscherPanfilov

   atrial      = ModifiedMS2v(dt=0.01)                       # phenomenological
   ventricular = TenTusscherPanfilov(cell_type="ENDO", dt=0.01)

Compiled ionic updates
----------------------

For ordinary Python simulations, the tensor-backed ionic models automatically
wrap ``differentiate`` at construction with ``torch.compile(fullgraph=True)``
and ``options={"triton.cudagraphs": False}``. This covers ``ModifiedMS2v``,
``AlievPanfilov``, ``CourtemancheRamirezNattel``, ``TenTusscherPanfilov`` and both
``MitchellSchaeffer`` variants. The Mitchell--Schaeffer variants inherit the
wrapper from ``BaseCellModel``; ``BaseCellModelRL`` also installs it for its
subclasses. No additional constructor flag or manual wrapper is needed:

.. code-block:: python

   ventricular = TenTusscherPanfilov(
       cell_type="ENDO", dt=0.01,
   )

Compilation happens on the first update, after ``initialize`` has
allocated the states and tables on the final device. No warmup steps are performed
by the constructor. The equations, tables, dtype and timestep are unchanged.
The guarded wrapper is skipped during TorchScript compilation, preserving the
scripted path for classes registered with ``@torch.jit.script``. NumPy reference
implementations under ``ionic/cellml`` are not GPU tensor updates and are unchanged.

For an eager reference run, restore the original bound method on that instance
before stepping (including for inherited implementations):

.. code-block:: python

   ventricular.differentiate = type(ventricular).differentiate.__get__(ventricular)

For the conservative CUDA arithmetic tested with PyTorch 2.7.1 and its bundled
Triton, launch a fresh Python process with ``TRITON_DEFAULT_FP_FUSION=0``. Use a
dedicated compiler cache when switching this setting, for example:

.. code-block:: bash

   TRITON_DEFAULT_FP_FUSION=0 \
   TORCHINDUCTOR_CACHE_DIR=/tmp/torchcor-nofma-v1 \
   python -m demo.monodomain_ventricle

This disables multiply-add contraction while retaining GPU kernel fusion; the
library does not change process-wide compiler settings. Full-trajectory numerical
validation is still necessary for the cell types and parameters used in a study.
Do not also wrap ``differentiate`` manually.

.. note::
   Several cell models are registered with ``@torch.jit.script`` and are not
   introspected by autodoc -- the table above is the reference. The constructor
   wrapper separately enables ``torch.compile`` for Python callers.
