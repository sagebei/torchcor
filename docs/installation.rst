Installation
============

torchcor runs on **Python 3.10+** and is built on PyTorch, so it uses your GPU
when one is available and falls back to CPU otherwise.

Install
-------

.. tab-set::

   .. tab-item:: pip (recommended)

      .. code-block:: bash

         pip install torchcor

      Installs the latest release from `PyPI <https://pypi.org/project/torchcor/>`_
      together with its runtime dependencies (PyTorch, NumPy, SciPy, PyVista, …).

   .. tab-item:: From source (development)

      Clone the repository and install in editable mode so your changes take
      effect immediately:

      .. code-block:: bash

         git clone https://github.com/sagebei/torchcor.git
         cd torchcor
         pip install -e .

.. tip::

   For GPU runs, install the PyTorch build that matches your CUDA toolkit first
   (see `pytorch.org/get-started <https://pytorch.org/get-started/locally/>`_),
   then ``pip install torchcor``.  On CPU-only machines the default wheel is fine.

Requirements
------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Requirement
   * - Python
     - ≥ 3.10
   * - Core
     - PyTorch ≥ 2.0, NumPy, SciPy
   * - Meshing / visualisation
     - PyVista, pygmsh
   * - ECG / signal analysis
     - pandas, wfdb, scikit-learn, seaborn, matplotlib
   * - GPU (optional)
     - a CUDA-capable device + a matching PyTorch build

Verify the installation
-----------------------

.. code-block:: python

   import torchcor as tc
   print(tc.get_device())          # cuda:0  (or cpu)

   from torchcor.electrophysiology import Monodomain, ReactionEikonal
   from torchcor.ionic import TenTusscherPanfilov
   print("torchcor is ready")

Select the device once at the top of a script:

.. code-block:: python

   import torchcor as tc
   tc.set_device("cuda:0")         # or "cpu"

Next steps
----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⚡ Quickstart
      :link: quickstart
      :link-type: doc

      Run your first monodomain simulation in a dozen lines.

   .. grid-item-card:: 📚 Tutorials
      :link: tutorials/index
      :link-type: doc

      Monodomain, reaction-eikonal, and body-surface ECGs.
