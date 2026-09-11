torchcor
========

.. rst-class:: lead

   **GPU-accelerated cardiac electrophysiology simulation in PyTorch.**

torchcor solves large-scale cardiac electrophysiology on the GPU — from the full
monodomain reaction-diffusion model to the fast reaction-eikonal model and
body-surface ECGs — with a library of biophysical ionic cell models and a
finite-element core that runs on CUDA or CPU.

.. code-block:: bash

   pip install torchcor

What's inside
-------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🫀 Monodomain
      :link: tutorials/monodomain
      :link-type: doc

      The full reaction-diffusion reference model: implicit FEM diffusion +
      explicit ionic update, on millions of nodes.

   .. grid-item-card:: ⚡ Reaction-Eikonal
      :link: tutorials/reaction_eikonal
      :link-type: doc

      Fast activation times from an eikonal solver, with the action potentials
      filled in — a fraction of monodomain cost.

   .. grid-item-card:: 📈 ECGs & electrograms
      :link: tutorials/electrocardiogram
      :link-type: doc

      Body-surface 12-lead ECGs from the transmembrane voltage via a
      reciprocal lead-field solver.

   .. grid-item-card:: 🧬 Ionic models
      :link: api/ionic
      :link-type: doc

      Phenomenological and biophysical human atrial / ventricular cell models.

Why torchcor
------------

- **GPU-native.** Everything — FEM assembly, the conjugate-gradient solves, the
  ionic ODEs, the eikonal sweeps — runs as vectorised PyTorch on the GPU.
- **Whole-heart scale.** Designed for million-node meshes (atria, ventricles,
  heart-torso).
- **From timing to ECG.** Activation maps, full transmembrane voltage, and
  body-surface potentials, in one library.

.. toctree::
   :hidden:
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :caption: API reference

   api/index
