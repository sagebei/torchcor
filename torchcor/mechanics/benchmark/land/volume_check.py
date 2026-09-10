"""Locate P2's independently sampled volume errors; no solver modifications."""
import argparse
import json
from pathlib import Path

import torch
from torchcor.mechanics.benchmark.p2 import solve_ventricle
from torchcor.mechanics.elements import LagrangeHex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mesh", nargs=3, type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    solution = solve_ventricle(tuple(args.mesh), device=args.device)
    if not solution.report.converged:
        raise RuntimeError("Cannot assess a failed solve")
    mesh, problem = solution.mesh, solution.simulator.problem
    rule = LagrangeHex(mesh.order, 6, mesh.dtype, mesh.device)
    J, weights = problem.sample_jacobians(solution.u, 6)
    points = torch.einsum("qa,eai->eqi", rule.N, mesh.points[mesh.cells])
    error = (J-1).abs()
    worst = int(error.flatten().argmax())
    bands = []
    for low, high in ((-21., -15.), (-15., -10.), (-10., -5.), (-5., 0.), (0., 4.), (4., 5.)):
        mask = (points[..., 2] >= low) & (points[..., 2] < high)
        volume = weights[mask].sum()
        if bool(mask.any()):
            bands.append(dict(z_interval_mm=[low, high], volume_fraction=float(volume/weights.sum()),
                rms_error=float((torch.sum(weights[mask]*error[mask]**2)/volume).sqrt()),
                max_error=float(error[mask].max())))
    result = dict(mesh=args.mesh, volume=solution.volume,
        worst_reference_point_mm=points.reshape(-1, 3)[worst].tolist(),
        worst_J=float(J.flatten()[worst]),
        volume_fractions_above_error={str(limit):float(weights[error>limit].sum()/weights.sum())
                                     for limit in (.01, .05, .1)}, z_bands=bands)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
