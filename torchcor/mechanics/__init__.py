"""Cardiac mechanics for torchcor: GPU finite-strain hyperelasticity.

The package is organised by responsibility, and things are imported from the
module they belong to rather than from one flat namespace -- so an import says
where a class comes from, and each module can be read on its own:

=================================  ============================================
``torchcor.mechanics.mesh``        Mesh geometry, connectivity, surface sets,
                                   point interpolation
``torchcor.mechanics.elements``    Shape functions and integration rules
``torchcor.mechanics.material``    Constitutive laws, fibre frames, active stress
``torchcor.mechanics.boundary``    Prescribed displacements and applied loads
``torchcor.mechanics.assembly``    Global residual and tangent
``torchcor.mechanics.linear``      Krylov solvers and preconditioners
``torchcor.mechanics.solver``      Load stepping, Newton, convergence, recovery
``torchcor.mechanics.visualisation``  VTK output and figures
``torchcor.mechanics.simulator``   :class:`Mechanics`, the high-level entry point
=================================  ============================================

Only the entry point is re-exported here, mirroring how the electrophysiology
side is used::

    from torchcor.mechanics import Mechanics
    from torchcor.mechanics.mesh import StructuredBoxMesh
    from torchcor.mechanics.material import GuccioneMaterial, IsochoricMaterial
    from torchcor.mechanics.boundary import DirichletBC, FollowerPressure

    mesh = StructuredBoxMesh((0, 0, 0), (10, 1, 1), (40, 4, 4),
                             order=2, device="cuda")
    material = IsochoricMaterial(GuccioneMaterial(C=2, bf=8, bt=2, bfs=4))
    sim = Mechanics(mesh, material, boundary=[
        DirichletBC.on_surface(mesh, "x-"),
        FollowerPressure.on_surface(mesh, "z-", 0.004)])
    u = sim.solve()

The submodules are imported here too, so ``torchcor.mechanics.mesh`` and
friends are available after a plain ``import torchcor.mechanics``.
"""

from torchcor.mechanics import (
    assembly,
    boundary,
    elements,
    linear,
    material,
    mesh,
    simulator,
    solver,
    visualisation,
)
from torchcor.mechanics.simulator import Mechanics

__all__ = [
    "Mechanics",
    "assembly",
    "boundary",
    "elements",
    "linear",
    "material",
    "mesh",
    "simulator",
    "solver",
    "visualisation",
]
