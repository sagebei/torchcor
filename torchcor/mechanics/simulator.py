"""The mechanics simulation, assembled from its parts.

Each module builds its own objects; :class:`Mechanics` takes them and solves.
Read a script top to bottom and it goes geometry, material, boundary, solve::

    from torchcor.mechanics import Mechanics
    from torchcor.mechanics.mesh import StructuredBoxMesh
    from torchcor.mechanics.material import GuccioneMaterial, IsochoricMaterial, MaterialAxes
    from torchcor.mechanics.boundary import DirichletBC, RobinBC, FollowerPressure

    # geometry, and the device everything else follows
    mesh = StructuredBoxMesh((0, 0, 0), (10, 1, 1), (40, 4, 4), order=2,
                             device="cuda")

    # how the tissue behaves
    material = IsochoricMaterial(GuccioneMaterial(C=2, bf=8, bt=2, bfs=4))
    fibres = MaterialAxes(f=(1, 0, 0))

    # what holds it, and what pushes it
    fixed = DirichletBC.on_surface(mesh, "x-")
    load = FollowerPressure.on_surface(mesh, "z-", 0.004)

    # put it together and solve
    sim = Mechanics(mesh, material, axes=fibres, boundary=[fixed, load])
    u = sim.solve()
    sim.to_vtk("beam.vtu")
    sim.to_png("beam.png")

``Mechanics`` owns no numerics of its own.  It holds the parts, hands them to
:mod:`torchcor.mechanics.assembly` and :mod:`torchcor.mechanics.solver`, and
gives back the displacement -- so any part can be replaced without touching it.
Problems 2 and 3 are the same script with a ventricular mesh, a pressure on
``"endo"``, and for problem 3 an :class:`ActiveStressMaterial` wrapped around
the passive law.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch

from torchcor.mechanics.assembly import FiniteStrainProblem
from torchcor.mechanics.boundary import DirichletBC, RobinBC
from torchcor.mechanics.material import HyperelasticMaterial, MaterialAxes
from torchcor.mechanics.mesh import Mesh
from torchcor.mechanics.solver import (
    DynamicSolver, QuasiStaticSolver, SolveReport,
)
from torchcor.mechanics.visualisation import render_deformation, write_mesh

__all__ = ["Mechanics"]


class Mechanics:
    """A finite-strain simulation, assembled from its parts.

    Parameters
    ----------
    mesh:
        The geometry, with its surfaces named.  It also fixes the device and
        dtype of the whole simulation -- there is one place that decides, and
        this is it.
    material:
        Passive constitutive law, e.g. :class:`GuccioneMaterial`.  Wrap it in
        :class:`ActiveStressMaterial` to add contraction.
    axes:
        Local fibre frame; ``None`` means the material's own axes are the
        global ones.
    boundary:
        Prescribed displacements (:class:`DirichletBC`), surface loads
        (:class:`FollowerPressure`) and elastic/viscous support
        (:class:`RobinBC`), in any order.  Anything that can compute its own
        ``contribution`` counts as a load, so a new kind of load needs no
        change here.
    bulk_modulus:
        Augmented Lagrangian penalty enforcing incompressibility.  It affects
        conditioning, not the converged answer.  ``None`` leaves volume to the
        material, for a law that carries its own volumetric term.
    density:
        Mass per unit reference volume.  Required by :meth:`solve_dynamic`
        and ignored by the quasi-static :meth:`solve`.
    viscosity:
        Kelvin-Voigt coefficient of a viscous stress ``eta * dE/dt``, which
        damps the motion.  Only acts in a dynamic solve.
    quadrature_order:
        Gauss points per axis. The default follows the element's polynomial
        degree; increase this for an integration-convergence check.
    chunk_size:
        Elements processed together during assembly, controlling peak memory.
    """

    def __init__(
        self,
        mesh: Mesh,
        material: HyperelasticMaterial,
        axes: Optional[MaterialAxes] = None,
        boundary: Sequence[object] = (),
        bulk_modulus: Optional[float] = 1.0e2,
        *,
        density: float = 0.0,
        viscosity: float = 0.0,
        quadrature_order: Optional[int] = None,
        surface_quadrature_order: Optional[int] = None,
        chunk_size: int = 2048,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.axes = axes
        self.bulk_modulus = None if bulk_modulus is None else float(bulk_modulus)
        self.density = float(density)
        self.viscosity = float(viscosity)
        self.quadrature_order = quadrature_order
        self.surface_quadrature_order = surface_quadrature_order
        self.chunk_size = chunk_size
        self.dtype = mesh.dtype
        self.device = mesh.device

        self.dirichlet: List[DirichletBC] = []
        self.support: List[RobinBC] = []
        self.loads: List[object] = []
        for bc in boundary:
            if isinstance(bc, DirichletBC):
                self.dirichlet.append(bc)
            elif isinstance(bc, RobinBC):
                self.support.append(bc)
            elif hasattr(bc, "contribution"):
                self.loads.append(bc)
            else:
                raise TypeError(
                    f"{type(bc).__name__} is not a boundary condition: expected a "
                    "DirichletBC, or a load that can compute its contribution")

        self.problem: Optional[FiniteStrainProblem] = None
        self.report: Optional[SolveReport] = None
        self.u: Optional[torch.Tensor] = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Mechanics({self.mesh.n_cells} cells, {self.mesh.n_dofs} DOF, "
                f"{self.material!r}, {len(self.dirichlet)} constrained surface(s), "
                f"{len(self.loads)} load(s), device={self.device}, "
                f"dtype={self.dtype})")

    # ----------------------------------------------------------------- solve
    def _build(self, restrained: bool = True) -> FiniteStrainProblem:
        """The assembled problem.

        ``restrained`` asks for the static well-posedness check: with nothing
        holding it a static body has a rigid-body nullspace and no unique
        equilibrium.  A *dynamic* body needs no holding -- mass and initial
        conditions define the motion -- so the check does not apply there.
        """
        if restrained and not self.dirichlet and not self.support:
            raise RuntimeError("nothing restrains the body: add a DirichletBC "
                               "or a RobinBC")
        self.problem = FiniteStrainProblem(
            mesh=self.mesh, material=self.material, axes=self.axes,
            bulk_modulus=self.bulk_modulus, density=self.density,
            viscosity=self.viscosity, dirichlet=self.dirichlet,
            pressures=self.loads, robin=self.support,
            quadrature_order=self.quadrature_order,
            surface_quadrature_order=self.surface_quadrature_order,
            chunk_size=self.chunk_size,
        )
        return self.problem

    def _finish(self, verbose: bool, raise_on_failure: bool) -> torch.Tensor:
        if verbose:
            print(f"  {self.report}", flush=True)
        if raise_on_failure and not self.report.converged:
            raise RuntimeError(f"mechanics solve failed: {self.report}")
        return self.u

    def solve(self, load_steps: int = 10, verbose: bool = True,
              raise_on_failure: bool = True, **options) -> torch.Tensor:
        """Ramp the loads from zero to full and return the displacement.

        Failure raises by default. Use ``raise_on_failure=False`` to inspect
        a partial solution and ``sim.report``. Extra keyword arguments go to
        :class:`QuasiStaticSolver`.
        """
        solver = QuasiStaticSolver(self._build(), load_steps=load_steps,
                                   verbose=verbose, **options)
        self.u, self.report = solver.solve()
        return self._finish(verbose, raise_on_failure)

    def solve_dynamic(self, t_end: float, dt: float, verbose: bool = True,
                      raise_on_failure: bool = True, observer=None,
                      u0: Optional[torch.Tensor] = None,
                      v0: Optional[torch.Tensor] = None,
                      **options) -> torch.Tensor:
        """March the loads through time and return the final displacement.

        Loads that vary in time are given as schedules -- a callable pressure
        or active tension -- which the integrator evaluates at each step.
        ``observer(t, u, v, a)`` sees every accepted step, which is how a time
        history is recorded.  Extra keyword arguments go to
        :class:`DynamicSolver`.
        """
        solver = DynamicSolver(self._build(restrained=False), dt=dt,
                               verbose=verbose, **options)
        self.u, self.report = solver.solve(t_end, u=u0, velocity=v0,
                                           observer=observer)
        return self._finish(verbose, raise_on_failure)

    # ------------------------------------------------------- postprocessing
    @property
    def material_model(self):
        """The oriented material the assembler evaluates (after :meth:`solve`)."""
        return None if self.problem is None else self.problem.oriented

    @property
    def incompressibility(self):
        """The incompressibility model and its multiplier (after :meth:`solve`)."""
        return None if self.problem is None else self.problem.volumetric

    def volume_report(self, n_gauss: int = 6) -> dict:
        """How well incompressibility actually holds, measured independently.

        ``n_gauss`` points per axis sample the local error left by the weak
        mixed constraint. Using a denser rule than assembly checks between
        the integration points as well.
        """
        self._solved()
        return self.problem.volume_diagnostics(self.u, n_gauss=n_gauss)

    def volume_error_per_cell(self, n_gauss: int = 6) -> torch.Tensor:
        """Volume-weighted RMS ``|J - 1|`` in each element, same rule."""
        self._solved()
        return self.problem.volume_error_per_cell(self.u, n_gauss=n_gauss)

    @property
    def displacement(self) -> torch.Tensor:
        """``(n_points, 3)`` nodal displacement."""
        return self._solved().reshape(-1, 3)

    def deformed(self) -> torch.Tensor:
        """``(n_points, 3)`` deformed nodal coordinates."""
        return self.mesh.points + self.displacement

    def probe(self, points, **locate) -> torch.Tensor:
        """Where undeformed ``points`` end up.

        Keyword arguments reach the mesh locator; see ``mesh.locate`` for the
        residual tolerance and the policy for points outside the mesh.
        """
        self._solved()
        pts = torch.as_tensor(points, dtype=self.dtype, device=self.device)
        return self.mesh.interpolate(self.deformed(), pts, **locate)

    def to_vtk(self, path: Union[str, Path]) -> Path:
        """Write one ``.vtu``: reference mesh plus the displacement field.

        Both shapes come from the single file -- open it for the undeformed
        body, then apply ParaView's "Warp By Vector" on ``displacement``.
        """
        u = self.displacement
        return write_mesh(path, self.mesh, point_data={
            "displacement": u, "displacement_magnitude": u.norm(dim=1)})

    def to_png(self, path: Union[str, Path], points=None, curves=None,
               title: Optional[str] = None, **options) -> Path:
        """Render the reference and deformed shapes to a PNG."""
        return render_deformation(path, self.mesh, self.displacement,
                                  points=points, curves=curves, title=title,
                                  **options)

    def _solved(self) -> torch.Tensor:
        if self.u is None:
            raise RuntimeError("nothing to post-process: call solve() first")
        return self.u
