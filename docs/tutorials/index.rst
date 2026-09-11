Tutorials
=========

.. toctree::
   :maxdepth: 1

   monodomain
   reaction_eikonal
   electrocardiogram

Choosing a model
----------------

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - You want…
     - Use
     - Cost
   * - Activation / isochrone map
     - ``ReactionEikonal.eikonal_activation_times()`` (no reaction)
     - cheapest (seconds)
   * - Full Vm cheaply (e.g. ECG source)
     - ``ReactionEikonal`` with ``diffusion=False``
     - cheap (no linear solve)
   * - Full diffusion physics on a coarse mesh
     - ``ReactionEikonal`` with ``diffusion=True``
     - moderate (CG per step)
   * - Most accurate reference / fine mesh
     - ``Monodomain``
     - full reaction-diffusion
   * - Body-surface ECG / electrograms
     - any of the above → ``LeadField``
     - + one solve per electrode

On a **fine** mesh, ``diffusion=True`` costs the same as ``Monodomain`` (the
diffusion dominates), so prefer ``Monodomain`` there; the reaction-eikonal pays
off on **coarse** meshes and for activation maps.
