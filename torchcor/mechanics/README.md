# TorchCor mechanics

This module solves quasi-static and dynamic finite-strain mechanics with
PyTorch CUDA tensors. The mesh sets the device and precision, and it also sets
the element family: [mesh.py](mesh.py) provides tensor-product hexahedra of any
order (`HexMesh`) and tetrahedra of order one or two (`TetMesh`), which is what
stored meshes contain. Everything above the mesh is written against whichever
it holds. The benchmark examples use quadratic elements and `torch.float64`.
CPU compatibility is not a development requirement.

The CUDA regression suite is in [tests/](tests/); run it with
`TORCHCOR_TEST_DEVICE=cuda:0 python -m unittest discover -s torchcor/mechanics/tests -t .`.
Benchmark results, with the reference data they are scored against, live beside
each runner under [benchmark/](benchmark/).
The [P3 report](benchmark/land/p3.py) covers active contraction,
signed twist, and the additional checks on spatial fibre directions.

## Set up a simulation

```python
import torch
from torchcor.mechanics import Mechanics
from torchcor.mechanics.mesh import StructuredBoxMesh
from torchcor.mechanics.material import GuccioneMaterial, IsochoricMaterial, MaterialAxes
from torchcor.mechanics.boundary import DirichletBC, FollowerPressure

mesh = StructuredBoxMesh(
    origin=(0, 0, 0), lengths=(10, 1, 1), divisions=(20, 2, 2),
    order=2, device="cuda:0", dtype=torch.float64,
)
material = IsochoricMaterial(GuccioneMaterial(C=2, bf=8, bt=2, bfs=4))
sim = Mechanics(
    mesh, material, axes=MaterialAxes(f=(1, 0, 0)),
    boundary=[DirichletBC.on_surface(mesh, "x-"),
              FollowerPressure.on_surface(mesh, "z-", pressure=0.004)],
)
u = sim.solve()                         # raises if the solve fails
print(sim.probe([[10, 0.5, 1]]))        # deformed position, not displacement
print(sim.volume_report(n_gauss=6))
```

Use consistent units: this example uses millimetres and kPa. `DirichletBC`
fixes displacement components; its default fixes all three directions.
`components=(0,)` fixes only movement in x. Positive `FollowerPressure` pushes
inward, following the surface's changing normal and area. Name surfaces in the
mesh so the same setup works with different geometries. The ventricle uses
`"base"`, `"endo"`, and `"epi"`.

## Numerical method

The solver increases pressure and prescribed displacements in load increments.
Prescribed active stress can optionally follow the same load factor.
Within each increment, Newton iterations balance forces and augmented
Lagrangian updates enforce incompressibility in a discrete pressure space.
An unsuccessful increment restores the accepted displacement and multiplier
state before retrying at a smaller increment.

Quadratic elements use four discontinuous pressure modes per element. That
mixed path is exercised and verified on hexahedra; on tetrahedra it is
constructed the same way but is not covered by the tests, and the cardiac
benchmarks do not reach it because their law carries its own volumetric
penalty (`bulk_modulus=None`). Treat it as untested there.
Pressure and dilatation are eliminated locally at each augmentation iteration,
leaving a displacement system for the sparse linear solver. Physical-volume
weights and the derivative of this elimination appear in both residual and
tangent. [The formulation note](FORMULATION.md) gives the
equations, established references, and independent checks. This is a PyTorch
implementation of those equations, not the same numerical solver as deal.II.

The general solver uses a critical-point line search on the energy slope and
a preconditioned Krylov method chosen from the tangent's measured symmetry:
conjugate gradients when it is symmetric, BiCGStab when it is not.
`sim.solve(line_search="none")` explicitly selects full Newton steps. The
P1 runner selects that option for its measured performance baseline; P2 uses
Neither option guarantees convergence for every load or material.
The nonlinear driver's default linear relative tolerance is `1e-6`; force
and mixed-volume convergence retain their separate, stricter tolerances.
`IsochoricMaterial` removes the passive law's volume-changing response. For
prescribed active stress, use
`ActiveStressMaterial(IsochoricMaterial(passive), tension=...)` so the passive
split does not alter the specified activation. Activation is currently a
prescribed material value; add `ramp=True` to scale it with the load factor.
The default is constant activation. This is load continuation, not a model of
time-dependent muscle activation.

`MaterialAxes(f=...)` accepts a uniform direction, directions per cell, or
directions per quadrature point. It also accepts a callable receiving reference
coordinates of shape `(n_cells, n_quadrature, 3)` and returning directions.
The callable runs once during setup on the mesh device. P3 uses this interface
to sample the paper's rotating fibre field throughout the ventricular wall.

Element calculations, sparse matrices, Krylov vectors, and preconditioner data
stay on the GPU. Python controls convergence and load increments through scalar
checks. Mesh setup and plotting/output can use the host. BiCGStab uses CUDA
graphs for suitable built-in preconditioners to reduce repeated kernel-launch
overhead; custom preconditioners use the ordinary PyTorch execution path.

## Read the results

`sim.report.converged` means the force and discrete constraint checks passed at
full load. To inspect a failed partial solution deliberately, pass
`raise_on_failure=False` and inspect the report before using the displacement.

The small `constraint_error` measures the **projected** volume violation. It
does not mean volume is preserved that accurately everywhere. Inspect the
separate RMS and maximum `J-1` errors in `sim.volume_report()`: `J=1` means
unchanged material volume. Expansion and compression can cancel in the total
volume, and finite sampling cannot bound every point inside an element.

The structured ventricular mesh has a collapsed apex. Positive sampled
Jacobians and agreement at the apex do not prove stability of these elements
under every deformation. Check directional refinement and local errors,
especially before claiming accuracy near the apex of the twisting active
problem. The supported scope is 3D hexahedral and tetrahedral mechanics,
quasi-static or dynamic; passing benchmark position checks does not establish
uniformly accurate local volume preservation.

## Run the benchmarks

From the repository root, run one mesh or omit `--mesh` for a refinement study:

```bash
python -m torchcor.mechanics.benchmark.land.p1 --device cuda:0 --mesh 20 2 2
python -m torchcor.mechanics.benchmark.land.p2 --device cuda:0 --mesh 2 12 24
python -m torchcor.mechanics.benchmark.land.p3 --device cuda:1 --mesh 3 16 32
```

Each runner writes its JSON and figures into its own folder beside the scripts
(`benchmark/land/p1`, `p2`, `p3`); `--out DIR` overrides that. Once all three
have run, `python torchcor/mechanics/benchmark/land/export_csv.py` collects
every plotted quantity into `benchmark/land/torchcor_land.csv`, one row per
point, tagged with the figure of Land et al. it belongs to.

One benchmark lives per folder under `benchmark/`, so a second suite is added
alongside `land/` rather than into it; `benchmark/report.py` holds the
argument, JSON and figure helpers they share.

The benchmark runners write numeric JSON and comparison figures. Their 1%
position criterion is a project check; the [Land et al. paper](resources/Land.pdf)
does not prescribe an executable pass rule. The strain guides are approximate
figure readings for P1/P2. P3 uses a published participant's apex table and
also checks the sign of twist; no numerical strain reference was available
for its plots. Compare the full refinement curves and volume diagnostics,
not only the printed position check. Reported displacement DOFs are three per
node before eliminating prescribed movements; local multiplier coefficients
are additional state, not global linear-system unknowns.
