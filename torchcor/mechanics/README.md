# TorchCor mechanics

GPU finite-strain mechanics for cardiac tissue, built with PyTorch.
Supports passive and prescribed active stress, quasi-static and dynamic
simulations, and hexahedral and tetrahedral meshes.

**Benchmarks:** [Aróstica B1/B2](#aróstica-b1-and-b2) · [Land P1/P2/P3](#land-p1-p2-and-p3)

## Quick start

Install [TorchCor](../../README.md) with CUDA-enabled PyTorch. This example
fixes one end of a beam and applies pressure to its lower face.
Units are **mm and kPa**; the mesh sets the GPU and precision.

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
fixed = DirichletBC.on_surface(mesh, "x-")
pressure = FollowerPressure.on_surface(mesh, "z-", pressure=0.004)

sim = Mechanics(
    mesh, material, axes=MaterialAxes(f=(1, 0, 0)),
    boundary=[fixed, pressure],
)
u = sim.solve()
print(sim.report)
sim.to_vtk("beam.vtu")
```

Open `beam.vtu` in ParaView and **Warp By Vector** using `displacement`.
Use `sim.to_png("beam.png")` for a quick image,
`sim.probe(points)` for deformed positions of reference points, and
`sim.volume_report()` for local and total volume changes.

## Benchmarks

Run commands from the repository root. Each runner writes results and
comparison figures beside its script; `--out DIR` selects another destination.

### Aróstica B1 and B2

The [Aróstica benchmark](resources/Arostica.pdf) tests time-dependent ventricular
mechanics, including pressure, active contraction, inertia and tissue support.

| Case | Geometry and loading | Comparison |
| --- | --- | --- |
| [B1](benchmark/arostica/b1/b1.py) | Single ventricle; passive, active, combined and Step 2 cases A/B/C. | Displacement histories at the prescribed probes against published participants. |
| [B2](benchmark/arostica/b2/b2.py) | Two ventricles with separate cavity pressures; coarse and fine meshes. | Each mesh compared with its own participant dataset using the RED displacement metric. |

```bash
# B1: all six cases; use --case passive to run just one.
python -m torchcor.mechanics.benchmark.arostica.b1.b1 --device cuda:0

# B2: both published mesh resolutions.
python -m torchcor.mechanics.benchmark.arostica.b2.b2 --device cuda:0 --mesh coarse
python -m torchcor.mechanics.benchmark.arostica.b2.b2 --device cuda:0 --mesh fine
```

These runners require the supplied meshes, fibres and participant data in
their `reference/` directories. The `--dt` argument is in seconds.
Agreement with participant ranges establishes benchmark agreement; mesh and
time-step convergence must be assessed separately.

### Land P1, P2 and P3

The [Land benchmark](resources/Land.pdf) tests quasi-static deformation:

| Case | Problem |
| --- | --- |
| [P1](benchmark/land/p1.py) | Pressure-loaded cantilever beam. |
| [P2](benchmark/land/p2.py) | Passive ventricular inflation. |
| [P3](benchmark/land/p3.py) | Ventricular inflation and active contraction with rotating fibres. |

```bash
python -m torchcor.mechanics.benchmark.land.p1 --device cuda:0 --mesh 20 2 2
python -m torchcor.mechanics.benchmark.land.p2 --device cuda:0 --mesh 2 12 24
python -m torchcor.mechanics.benchmark.land.p3 --device cuda:0 --mesh 3 16 32
```

Omit `--mesh` for a refinement study. The runners' 1% position threshold is a
project check, not an acceptance rule specified by the paper.

## Building other simulations

- **Materials:** Neo-Hookean, Guccione and Holzapfel–Ogden laws.
  `MaterialAxes` supplies uniform or spatially varying fibre directions;
  `ActiveStressMaterial` adds prescribed second Piola–Kirchhoff fibre stress.
- **Boundaries:** `DirichletBC` prescribes displacement, `FollowerPressure`
  follows the deforming surface, and `RobinBC` adds elastic or viscous support.
- **Dynamics:** use `sim.solve_dynamic(t_end=..., dt=...)` with `density`
  and optional `viscosity`. B1/B2 show time-dependent loads and history recording.
- **Incompressibility:** the default uses an augmented Lagrangian constraint.
  For a material with its own volumetric penalty, use `bulk_modulus=None`,
  as in Aróstica. The mixed tetrahedral path remains unvalidated.

Numerical assembly and linear solves use GPU tensors; Python controls stepping,
and preprocessing and output may use the CPU. Failed solves raise by default.
Check local volume errors as well as displacement agreement, particularly near
the structured ventricular mesh's collapsed apex.

## Code and tests

| Files | Responsibility |
| --- | --- |
| [simulator.py](simulator.py) | Public `Mechanics` interface. |
| [mesh.py](mesh.py), [elements.py](elements.py) | Geometry, element bases and quadrature. |
| [material.py](material.py), [boundary.py](boundary.py) | Tissue laws, constraints and loads. |
| [assembly.py](assembly.py), [solver.py](solver.py), [linear.py](linear.py) | Assembly, nonlinear stepping and GPU linear solves. |
| [visualisation.py](visualisation.py) | Export and plotting. |
| [benchmark/](benchmark/) | Benchmark-specific inputs and comparisons. |

Run the CUDA regression suite from the repository root:

```bash
TORCHCOR_TEST_DEVICE=cuda:0 python -m unittest discover -s torchcor/mechanics/tests -t .
```
