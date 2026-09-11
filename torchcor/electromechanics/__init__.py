"""Electromechanics for torchcor: calcium to tension, tension to motion.

The package is organised by responsibility, and things are imported from the
module they belong to, so an import says where a class comes from:

======================================  ======================================
``torchcor.electromechanics.land``      Land (2017) active contraction model
``torchcor.electromechanics.coupling``  Stepping EP, contraction and mechanics
``torchcor.electromechanics.transfer``  Fields between nodes, cells and
                                        quadrature points
``torchcor.electromechanics.benchmark`` Validation, verification and examples
======================================  ======================================

A contraction model is anything with ``step(calcium, stretch, stretch_rate,
dt)``, ``tension(stretch)``, ``checkpoint()`` and ``restore(state)``.  Land is
one; the coupling is written against that contract, not against Land.

Isometric, with a prescribed calcium::

    from torchcor.electromechanics import Land2017
    land = Land2017(n_nodes, device="cuda")
    land.steady_state(calcium=0.1, stretch=1.0)
    tension = land.step(calcium, stretch=1.0, stretch_rate=0.0, dt=0.01)

Driven by electrophysiology, contracting a mesh::

    from torchcor.electromechanics import (
        ActiveStressCoupling, ActiveTensionDriver, HeldStretch, Land2017)

    land = Land2017(ep.n_nodes, device=ep.device, dtype=ep.dtype)
    land.steady_state(calcium=ep.calcium(), stretch=1.0)
    held = HeldStretch(ep.n_nodes, device=ep.device, dtype=ep.dtype)
    driver = ActiveTensionDriver(land, stretch=held)
    coupling = ActiveStressCoupling(problem, mesh, fibre, tension_buffer,
                                    connectivity=mesh.cells[:, :4],
                                    n_nodes=ep.n_nodes, stress_scale=1e3)

    def on_step(t, vm, calcium):
        driver(t, vm, calcium)
        if due(t):
            stretch, _ = coupling.solve(driver.tension)
            held.advance(stretch, now=t, interval=interval)

    ep.solve(on_step=on_step, record=False)

The tension reaches mechanics through the existing
:class:`~torchcor.mechanics.material.ActiveStressMaterial`; nothing in
``torchcor.mechanics`` is changed or wrapped.

The coupling is one-way from the electrophysiology, and its stepping scheme is
explicit -- see :mod:`torchcor.electromechanics.coupling` for what that has
been shown to do and what it has not.
"""

from torchcor.electromechanics import coupling, land, material, transfer
from torchcor.electromechanics.coupling import (
    ActiveStressCoupling,
    ActiveTensionDriver,
    HeldStretch,
)
from torchcor.electromechanics.land import Land2017, LandParameters
from torchcor.electromechanics.material import StabilizedActiveStressMaterial
from torchcor.electromechanics.transfer import (
    CellDeformation,
    CellToNode,
    NodeToCell,
    fibre_stretch,
)

__all__ = [
    "Land2017", "LandParameters",
    "ActiveTensionDriver", "HeldStretch", "ActiveStressCoupling",
    "NodeToCell", "CellToNode", "CellDeformation", "fibre_stretch",
    "StabilizedActiveStressMaterial",
    "coupling", "land", "material", "transfer",
]
