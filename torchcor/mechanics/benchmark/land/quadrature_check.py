"""Check four- versus six-point integration on small P1/P2 meshes (CUDA)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from torchcor.mechanics import Mechanics
from torchcor.mechanics.boundary import DirichletBC, FollowerPressure
from torchcor.mechanics.material import GuccioneMaterial, IsochoricMaterial, MaterialAxes
from torchcor.mechanics.mesh import StructuredBoxMesh, TruncatedEllipsoidMesh
from torchcor.mechanics.benchmark.land import p1, p2


def solve(problem, n_gauss, device):
    beam = problem == "p1"
    reference = p1 if beam else p2
    if beam:
        mesh = StructuredBoxMesh(p1.ORIGIN, p1.LENGTHS, (20, 2, 2), order=2, device=device)
        fixed, loaded = p1.CLAMPED_FACE, p1.LOADED_FACE
        axes = MaterialAxes(f=p1.FIBRE)
        probes = [[10., .5, 1.]]
    else:
        mesh = TruncatedEllipsoidMesh(divisions=(1, 8, 16), order=2, device=device,
                                      endo=p2.ENDO, epi=p2.EPI, base_z=p2.BASE_Z)
        fixed, loaded = p2.FIXED_SURFACE, p2.LOADED_SURFACE
        axes = None
        probes = [[0., 0., -p2.ENDO[1]], [0., 0., -p2.EPI[1]]]
    sim = Mechanics(mesh, IsochoricMaterial(GuccioneMaterial(**reference.MATERIAL)),
        axes=axes, bulk_modulus=100. if beam else 1000., quadrature_order=n_gauss,
        boundary=[DirichletBC.on_surface(mesh, fixed),
                  FollowerPressure.on_surface(mesh, loaded, reference.PRESSURE)])
    sim.solve(line_search="none" if beam else "critical-point")
    return sim, dict(problem=problem, n_gauss=n_gauss, dofs=mesh.n_dofs,
                     positions=sim.probe(probes)[:, 2].tolist(),
                     volume=sim.volume_report(n_gauss=8), solver=asdict(sim.report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for problem in ("p1", "p2"):
        coarse, first = solve(problem, 4, args.device)
        dense, second = solve(problem, 6, args.device)
        records.append(dict(problem=problem, rules=[first, second],
            position_change_mm=[b-a for a, b in zip(first["positions"], second["positions"])],
            relative_displacement_change=float((dense.u-coarse.u).norm()/dense.u.norm())))
        args.out.write_text(json.dumps(records, indent=2)+"\n")
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    main()
