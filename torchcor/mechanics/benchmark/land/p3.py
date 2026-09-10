"""Land et al. problem 3: ventricular inflation, contraction, and twist.

    python -m torchcor.mechanics.benchmark.p3 --device cuda:1 --mesh 2 12 24 --out results/p3

The geometry and measurements are shared with P2. The existing mechanics
solver receives an anisotropic passive law, the paper's reference fibre field,
60 kPa active second Piola stress, and 15 kPa endocardial follower pressure.
Pressure and activation increase proportionally during continuation.
"""
from pathlib import Path

import numpy as np
import torch

from torchcor.mechanics import Mechanics
from torchcor.mechanics.boundary import DirichletBC, FollowerPressure
from torchcor.mechanics.material import (
    ActiveStressMaterial, GuccioneMaterial, IsochoricMaterial, MaterialAxes,
)
from torchcor.mechanics.mesh import TruncatedEllipsoidMesh
from torchcor.mechanics.benchmark import report as reporting
from torchcor.mechanics.benchmark.ventricle import (
    BASE_Z, ENDO, EPI, LAYERS, STATIONS, VentricleSolution,
)


MATERIAL = dict(C=2., bf=8., bt=2., bfs=4.)
PRESSURE = 15.                                  # kPa, at full load
ACTIVE_TENSION = 60.                            # prescribed second Piola stress, kPa
MESHES = [(1, 6, 12), (2, 12, 24), (3, 16, 32), (4, 20, 40)]

# Refined FEniCS participant results, not an exact solution or all-code consensus.
# Pinned source: validation/takeover/reference/peppu_summary_results.txt.
APEX = dict(endo=-12.347, epi=-15.452)
TOLERANCE = .01                                 # our position comparison criterion
TWIST_GUIDE = (1.3, 1.8)                        # approximate main cluster, figure 11


def fibre_directions(points: torch.Tensor, mesh: TruncatedEllipsoidMesh) -> torch.Tensor:
    """Paper equations 2.7–2.12, sampled at physical reference positions.

    The analytic mesh-coordinate inverse supplies t,u,v. Use the paper's
    derivatives explicitly: the mesh's third grid axis runs in decreasing v,
    so taking that grid tangent would reverse one part of the helix.
    The exact apex has no unique fibre; interior Gauss points avoid it.
    """
    shape = points.shape
    q = points.reshape(-1, 3)
    if bool((q[:, :2].norm(dim=-1) <= 64*torch.finfo(q.dtype).eps*max(mesh.epi)).any()):
        raise ValueError("the paper's fibre field is undefined on the apical axis")
    t, s, v = mesh.to_parametric(q)
    rs, rl = mesh.radii(t)
    u = -torch.pi + s*(-torch.arccos(mesh.base_z/rl) + torch.pi)
    du = torch.stack((rs*torch.cos(u)*torch.cos(v),
                      rs*torch.cos(u)*torch.sin(v), -rl*torch.sin(u)), dim=-1)
    dv = torch.stack((-rs*torch.sin(u)*torch.sin(v),
                      rs*torch.sin(u)*torch.cos(v), torch.zeros_like(v)), dim=-1)
    alpha = torch.pi*(.5-t)
    f = (du/du.norm(dim=-1, keepdim=True)*torch.sin(alpha)[:, None]
         + dv/dv.norm(dim=-1, keepdim=True)*torch.cos(alpha)[:, None])
    return f.reshape(shape)


def solve_ventricle(divisions=(2, 12, 24), order: int = 2, device=None,
                    dtype: torch.dtype = torch.float64, verbose: bool = True,
                    bulk_modulus: float = 1000., **solver_options) -> VentricleSolution:
    """Set up P3 with the same four steps as P1/P2 and solve to full load."""
    # 1. The same reference ventricle and named surfaces as P2.
    mesh = TruncatedEllipsoidMesh(divisions=divisions, order=order, endo=ENDO,
                                  epi=EPI, base_z=BASE_Z, dtype=dtype, device=device)

    # 2. Split the passive law first, then add the paper's prescribed active stress.
    material = ActiveStressMaterial(IsochoricMaterial(GuccioneMaterial(**MATERIAL)),
                                    tension=ACTIVE_TENSION, ramp=True)
    axes = MaterialAxes(f=lambda points: fibre_directions(points, mesh))

    # 3. Hold the base and apply pressure to the deforming inner surface.
    fixed = DirichletBC.on_surface(mesh, "base")
    load = FollowerPressure.on_surface(mesh, "endo", PRESSURE)

    # 4. Reuse the nonlinear solver, at its defaults.  Starting from a 0.4%
    #    increment, as one benchmark participant did, was a workaround for an
    #    earlier solver; it now only costs 30% for the same answer.
    sim = Mechanics(mesh, material, axes=axes, boundary=[fixed, load],
                    bulk_modulus=bulk_modulus)
    if verbose:
        print(f"mesh {tuple(divisions)}, order {order}: {mesh.n_cells} cells, "
              f"{mesh.n_points} nodes, {mesh.n_dofs} DOF on {mesh.device}", flush=True)
    sim.solve(verbose=verbose, raise_on_failure=False, **solver_options)
    return VentricleSolution(mesh, sim.u, sim.report, tuple(divisions), order, sim,
                             volume=sim.volume_report(n_gauss=6))


def numeric_results(solutions):
    """Store signed twist and the same material-point strains measured in P2."""
    records = []
    for solution in solutions:
        valid = reporting.solve_is_valid(solution)
        measurements = {f"{k}_apex_z": value for k, value in solution.apex_z().items()}
        if valid[0] and not all(np.isfinite(value) for value in measurements.values()):
            valid = False, "non-finite apex coordinate"
        if valid[0]:
            line = solution.midline().cpu().numpy()
            strains = {"/".join(k): value for k, value in solution.strains().items()}
            if not np.isfinite(line).all() or not all(np.isfinite(v).all() for v in strains.values()):
                valid = False, "non-finite midline or strain measurement"
            else:
                dominant = line[np.argmax(np.abs(line[:, 1])), 1]
                measurements.update(midline_mm=line, peak_y_mm=float(line[:, 1].max()),
                    dominant_signed_y_mm=float(dominant),
                    twist_direction_correct=bool(dominant > 0),
                    strain_stations_s=STATIONS, strain_layers_t=LAYERS, strains_percent=strains)
        records.append(reporting.result_record(solution, measurements, valid))
    reporting.add_refinement(records, ["endo_apex_z", "epi_apex_z"])
    return records


def report_benchmark(solutions, records):
    print("\nPROBLEM 3: final apex positions and signed midwall twist")
    print(f"{'mesh':>14} {'DOF':>9} {'endo z':>10} {'epi z':>10} {'peak y':>10} {'valid':>7}")
    for record in records:
        print(f"{str(tuple(record['divisions'])):>14} {record['displacement_dofs']:>9} "
              f"{record['endo_apex_z']:>10.4f} {record['epi_apex_z']:>10.4f} "
              f"{record.get('peak_y_mm', float('nan')):>10.4f} {str(record['valid']):>7}")
        if not record["valid"]:
            print(f"  Not comparable: {record['invalid_reason']}")
    valid = [record for record in records if record["valid"]]
    if not valid:
        print("VERDICT: FAIL — no valid full-load solution")
        return False
    finest = max(valid, key=lambda record: record["displacement_dofs"])
    errors = {name: abs(finest[f"{name}_apex_z"]-target)/abs(target)
              for name, target in APEX.items()}
    passed = (len(valid) == len(records) and max(errors.values()) <= TOLERANCE
              and finest["twist_direction_correct"])
    print(f"\nParticipant apex reference: {APEX}; relative errors: "
          + ", ".join(f"{name} {100*error:.2f}%" for name, error in errors.items()))
    print(f"Figure 11: positive y twist, approximate peak-y guide {TWIST_GUIDE} mm.")
    print("The guide is a visual comparison, not an official acceptance threshold.")
    print("\nDistance-change strain (%), stations from apex to base; LONG has nine pairs:")
    for key, values in finest["strains_percent"].items():
        print(f"{key:>12}: " + " ".join(f"{value:7.2f}" for value in values))
    change = finest.get("change_from_previous_mesh")
    if change:
        drift = max(abs(value) for row in change["strains_percentage_points"].values() for value in row)
        print(f"Largest strain change between the two finest meshes: {drift:.3f} percentage points.")
    else:
        print("One mesh only: refinement convergence has not been assessed.")
    reporting.volume_table(solutions)
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'} — 1% participant-position comparison, "
          "correct twist direction, and all requested solves valid.")
    print("Strain, twist magnitude, and local-volume accuracy remain separate diagnostics.")
    return bool(passed)


def write_midline_figure(solutions, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for solution in solutions:
        if not reporting.solve_is_valid(solution)[0]:
            continue
        line = solution.midline().cpu().numpy()
        for ax, component, label in zip(axes, (2, 1), ("z", "y")):
            ax.plot(line[:, 0], line[:, component], label=f"{solution.n_dofs} DOFs")
            ax.set(xlabel="x (mm)", ylabel=f"{label} (mm)")
            ax.grid(alpha=.2)
    axes[0].set_title("Midwall shortening: paper figure 10")
    axes[1].set_title("Signed midwall twist: paper figure 11")
    axes[1].axhline(0, color="0.5", linestyle="--", linewidth=.8)
    axes[0].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main(argv=None):
    args = reporting.arguments(__doc__.splitlines()[0], argv)
    meshes = [tuple(args.mesh)] if args.mesh else MESHES
    solutions = [solve_ventricle(mesh, device=args.device) for mesh in meshes]
    records = numeric_results(solutions)
    passed = report_benchmark(solutions, records)
    reference = dict(source="Land et al. 2015, figures 9–12; pinned FEniCS participant table",
        apex_z_mm=APEX, relative_position_tolerance=TOLERANCE,
        twist_direction="positive y on the t=0.5, v=0 midwall line",
        approximate_peak_y_mm=TWIST_GUIDE, strain_acceptance_threshold=None)
    paths = [reporting.write_json(args.out/"p3_results.json", 3, records, passed, reference,
        verdict_scope="Finest participant-position agreement, positive signed twist, and validity "
                      "of every requested solve; strain, twist magnitude and local-volume "
                      "accuracy remain separate diagnostics.")]
    paths += reporting.comparison_figures(records,
        {f"{key}_apex_z": (value, (value-TOLERANCE*abs(value), value+TOLERANCE*abs(value)))
         for key, value in APEX.items()}, None,
        [[f"{layer}/{kind}" for kind in ("circ", "long", "trans")]
         for layer in ("endo", "epi", "mid")], args.out/"p3",
        reference_description="Dashed positions: published participant; shaded bands: our 1% criterion")
    valid = [solution for solution in solutions if reporting.solve_is_valid(solution)[0]]
    if valid:
        paths.append(write_midline_figure(valid, args.out/"p3_midline.png"))
        finest = max(valid, key=lambda solution: solution.n_dofs)
        paths.append(finest.simulator.to_png(args.out/"p3.png", elev=12, azim=-60, alpha=.55,
            curves={"deformed midwall": finest.midline()},
            title=f"Problem 3: 15 kPa pressure, 60 kPa active stress; {finest.n_dofs} DOFs"))
    for path in paths:
        print(f"wrote {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
