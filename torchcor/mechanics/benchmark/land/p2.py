"""Benchmark problem 2: inflation of a ventricle (Land et al., 2015).

Run it from the repository root::

    python -m torchcor.mechanics.benchmark.p2
    python -m torchcor.mechanics.benchmark.p2 --device cuda:1 --mesh 2 12 24 --out results/p2

It writes numeric results, a deformation figure, and refinement/strain plots.

The problem, from section 2c(ii) of the paper:

=========================  ====================================================
Geometry                   truncated ellipsoid: endocardium ``rs=7, rl=17``,
                           epicardium ``rs=10, rl=20``, base plane ``z=5`` mm
Constitutive law           Guccione, isotropic
Parameters                 ``C = 10`` kPa, ``bf = bt = bfs = 1``
Fixed                      the base plane ``z = 5``, in all directions
Pressure                   ``10`` kPa on the endocardial surface
Incompressibility          fully incompressible, ``J = 1``
=========================  ====================================================

The benchmark quantities are the deformed ``z`` of the endocardial and
epicardial apex (figure 6), the deformed midwall line (figure 7) and the
longitudinal, circumferential and radial strains (figure 8).

Reference: S. Land *et al.*, *Proc. R. Soc. A* **471**: 20150641 (2015),
``../resources/benchmark.pdf``.
"""

from pathlib import Path

import numpy as np
import torch

from torchcor.mechanics import Mechanics
from torchcor.mechanics.boundary import DirichletBC, FollowerPressure
from torchcor.mechanics.material import GuccioneMaterial, IsochoricMaterial
from torchcor.mechanics.mesh import TruncatedEllipsoidMesh
from torchcor.mechanics.benchmark import report as reporting
from torchcor.mechanics.benchmark.ventricle import (
    BASE_Z, CIRC_ANGLE, ENDO, EPI, LAYERS, STATIONS, VentricleSolution,
)


# =============================================================================
#  The problem, exactly as the paper specifies it
# =============================================================================
MATERIAL = dict(C=10.0, bf=1.0, bt=1.0, bfs=1.0)  # C in kPa, isotropic
FIXED_SURFACE = "base"
LOADED_SURFACE = "endo"
PRESSURE = 10.0                                   # kPa

#: Meshes to solve, as (transmural, apex-to-base, circumferential) elements.
MESHES = [(1, 6, 12), (1, 8, 16), (2, 12, 24), (3, 16, 32), (4, 20, 40)]


# =============================================================================
#  What the paper says the answer is
# =============================================================================
#: Figure 6, deformed apex z (mm).  Consensus and the spread of the submitted
#: solutions, with the LifeV outliers excluded.
APEX = dict(endo=(-26.6, (-26.9, -26.4)), epi=(-28.3, (-28.5, -28.1)))
#: Figure 7, deformed z of the apex end of the midwall line (mm).
MIDLINE_APEX = (-27.3, (-27.6, -27.1))
#: Figure 8: rounded visual guides through the main cluster of submissions.
#: These are approximate readings, not supplied numerical reference data, and
#: cannot establish a rigorous strain pass. Panels are CIRC, LONG, TRANS.
#: LONG has nine pairs, plotted at their FIRST station (p1 through p9).
STRAIN_GUIDE = {
    ("endo", "circ"): [62, 67, 72, 75, 77, 78, 78, 75, 63, 28],
    ("endo", "long"): [58, 57, 55, 54, 53, 52, 52, 53, 61],
    ("endo", "trans"): [-50, -43, -36, -31, -30, -34, -39, -45, -46, -26],
    ("epi", "circ"): [19, 23, 28, 34, 39, 42, 43, 43, 35, 15],
    ("epi", "long"): [22, 28, 34, 40, 43, 46, 48, 50, 47],
    ("epi", "trans"): [-35, -35, -34, -32, -31, -33, -36, -39, -42, -23],
    ("mid", "circ"): [35, 40, 46, 51, 54, 57, 58, 57, 47, 20],
    ("mid", "long"): [36, 40, 44, 47, 48, 49, 50, 52, 53],
    ("mid", "trans"): [-43, -40, -34, -31, -31, -33, -38, -42, -44, -24],
}
#: Our apex agreement criterion. The paper specifies no executable pass rule.
TOLERANCE = 0.01


# =============================================================================
#  Solving it
# =============================================================================
def solve_ventricle(divisions=(2, 12, 24), order: int = 2, device=None,
                    dtype: torch.dtype = torch.float64, verbose: bool = True,
                    bulk_modulus: float = 1000.0,
                    **solver_options) -> "VentricleSolution":
    """Solve problem 2 on one mesh.

    Same four steps as problem 1: geometry, material, boundary, solve.  Only
    the mesh and the surface names differ -- the material is isotropic here, so
    no fibre frame is needed.
    """
    # 1. The geometry, with surfaces named "endo", "epi" and "base".
    mesh = TruncatedEllipsoidMesh(divisions=divisions, order=order, endo=ENDO,
                                  epi=EPI, base_z=BASE_Z, device=device, dtype=dtype)

    # 2. How the tissue behaves.  Isotropic, so the fibre frame is irrelevant.
    material = IsochoricMaterial(GuccioneMaterial(**MATERIAL))

    # 3. What holds it, and what pushes it.
    fixed = DirichletBC.on_surface(mesh, FIXED_SURFACE)
    load = FollowerPressure.on_surface(mesh, LOADED_SURFACE, PRESSURE)

    # 4. Put it together and inflate.
    sim = Mechanics(mesh, material, boundary=[fixed, load], bulk_modulus=bulk_modulus)
    if verbose:
        print(f"mesh {tuple(divisions)}, order {order}: {mesh.n_cells} cells, "
              f"{mesh.n_points} nodes, {mesh.n_dofs} DOF on {mesh.device}", flush=True)
    sim.solve(verbose=verbose, raise_on_failure=False, **solver_options)

    return VentricleSolution(mesh, sim.u, sim.report, tuple(divisions), order, sim,
                             volume=sim.volume_report(n_gauss=6))


# =============================================================================
#  Comparing the shared measurements with problem 2 references
# =============================================================================


def solve_is_valid(solution: VentricleSolution) -> tuple[bool, str]:
    """Whether a solution may be compared against the benchmark at all."""
    valid, reason = reporting.solve_is_valid(solution)
    if not valid:
        return valid, reason
    apex = solution.apex_z()
    if not all(np.isfinite(v) for v in apex.values()):
        return False, f"non-finite apex coordinate {apex}"
    return True, ""


def report_benchmark(solutions: list[VentricleSolution]) -> bool:
    """Return apex agreement; print strain and volume diagnostics separately."""
    rule = "=" * 78
    print(f"\n{rule}\nBENCHMARK PROBLEM 2 -- deformed apex z (mm)\n{rule}")
    print(f"{'mesh':>14} {'DOF':>8} {'endo':>9} {'epi':>9} "
          f"{'endo err':>9} {'epi err':>9} {'valid':>6}")

    valid = []
    for s in solutions:
        ok, why = solve_is_valid(s)
        apex = s.apex_z()
        if ok:
            valid.append(s)
        print(f"{str(s.divisions):>14} {s.n_dofs:>8} {apex['endo']:>9.3f} "
              f"{apex['epi']:>9.3f} "
              f"{100*(apex['endo']-APEX['endo'][0])/abs(APEX['endo'][0]):>8.2f}% "
              f"{100*(apex['epi']-APEX['epi'][0])/abs(APEX['epi'][0]):>8.2f}% "
              f"{('yes' if ok else 'NO'):>6}")
        if not ok:
            print(f"{'':>14}   ^ not comparable: {why}")

    if not valid:
        print(f"\n{rule}\nVERDICT: FAIL   nothing solved.\n{rule}")
        return False

    finest = max(valid, key=lambda s: s.n_dofs)
    apex = finest.apex_z()
    errors = {k: abs(apex[k] - APEX[k][0])/abs(APEX[k][0]) for k in APEX}
    missing = len(solutions) - len(valid)
    # Reduce explicitly: Python's max() would happily step over a later NaN.
    worst = max(errors.values()) if all(np.isfinite(e) for e in errors.values()) \
        else float("nan")
    passed = np.isfinite(worst) and worst <= TOLERANCE and not missing

    print(f"\nconsensus  endo {APEX['endo'][0]:.1f} mm (spread {APEX['endo'][1]}), "
          f"epi {APEX['epi'][0]:.1f} mm (spread {APEX['epi'][1]})  (fig. 6)")
    end = finest.midline()[0]
    print(f"midwall line apex end   {float(end[2]):.2f} mm   vs reference "
          f"{MIDLINE_APEX[0]:.1f} mm (spread {MIDLINE_APEX[1]})  (fig. 7)")

    # Station-by-station, so a curve with the right extremes but the wrong
    # shape is visible.  Compare these against the panels of figure 8 directly;
    # the printed ranges alone cannot detect a spatially wrong curve.
    values = finest.strains()
    print(f"\nstrain, % at each station, apex (p1) to base (fig. 8)")
    print("LONG is labelled by the first point of each pair: p1 through p9.")
    print("Paper guides are rounded visual readings; differences are diagnostic only.")
    print(f"{'layer':>6} {'kind':>6} " + "".join(f"{'p'+str(i+1):>7}"
                                                 for i in range(10))
          + f" {'max guide diff':>15}")
    for key in sorted(values, key=lambda k: (k[0], k[1])):
        row = values[key]
        difference = float(np.abs(row - STRAIN_GUIDE[key]).max())
        pad = "" if len(row) == 10 else f"{'':>7}"
        print(f"{key[0]:>6} {key[1]:>6} "
              + "".join(f"{v:>7.1f}" for v in row) + pad + f" {difference:>12.2f} pp")

    # A converged curve should barely move between the two finest meshes.
    if len(valid) > 1:
        previous = sorted(valid, key=lambda s: s.n_dofs)[-2]
        before = previous.strains()
        drift = max(float(np.abs(values[k] - before[k]).max()) for k in values)
        moved = {k: abs(apex[k] - previous.apex_z()[k]) for k in apex}
        print(f"\nrefinement between {previous.divisions} and {finest.divisions}: "
              f"apex moved {max(moved.values()):.3f} mm, strain by {drift:.2f} "
              f"percentage points")
    else:
        print("One mesh only: refinement convergence was not assessed.")

    reporting.volume_table(valid)

    print(f"\n{rule}")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}   finest valid mesh "
          f"{finest.divisions} ({finest.n_dofs} DOF)")
    print(f"         endo {apex['endo']:.3f} mm ({100*errors['endo']:.2f}%), "
          f"epi {apex['epi']:.3f} mm ({100*errors['epi']:.2f}%), "
          f"tolerance {100*TOLERANCE:.0f}%")
    print("         Position agreement only; the 1% threshold is our criterion.")
    print("         Strain and local-volume convergence require separate assessment.")
    if missing:
        print(f"         but {missing} of {len(solutions)} meshes did not solve, "
              f"so convergence is not demonstrated")
    print(rule)
    return passed


# =============================================================================
#  Looking at it
# =============================================================================
def write_figure(solution: VentricleSolution, path: str = "ventricle.png") -> Path:
    """Render the ventricle: reference shape inside the inflated one.

    The deformed surface is translucent because it encloses the reference one,
    and both midwall lines are drawn so the inflation is visible.
    """
    sim = solution.simulator
    apex = solution.apex_z()
    return sim.to_png(
        path, elev=12, azim=-60, alpha=.55,
        curves={"midwall line, deformed": solution.midline(),
                "midwall line, reference": solution.reference_at(
                    np.full(101, .5), np.linspace(0, 1, 101), np.zeros(101))},
        points={"apex (endo, epi)": torch.stack(
            [solution.at(0., 0., 0.)[0], solution.at(1., 0., 0.)[0]])},
        title=(f"Benchmark problem 2: ventricle inflated to {PRESSURE:.0f} kPa\n"
               f"{solution.divisions} order-{solution.order} mesh, "
               f"{solution.n_dofs} DOF, apex z = {apex['endo']:.2f} (endo), "
               f"{apex['epi']:.2f} (epi) mm"),
    )


# =============================================================================
#  Running it
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    args = reporting.arguments(__doc__.splitlines()[0], argv)
    meshes = [tuple(args.mesh)] if args.mesh else MESHES
    solutions = [solve_ventricle(m, device=args.device) for m in meshes]
    passed = report_benchmark(solutions)
    records = numeric_results(solutions)
    guide = {"/".join(k): v for k, v in STRAIN_GUIDE.items()}
    reference = {"source": "Land et al. 2015, figures 6–8; approximate figure readings",
                 "apex_z_mm": {k: v[0] for k, v in APEX.items()},
                 "apex_spread_mm": {k: v[1] for k, v in APEX.items()},
                 "relative_position_tolerance": TOLERANCE,
                 "strain_guide_percent": guide, "strain_acceptance_threshold": None}
    paths = [reporting.write_json(args.out/"ventricle_results.json", 2, records, passed, reference)]
    paths += reporting.comparison_figures(
        records, {f"{k}_apex_z": v for k, v in APEX.items()}, guide,
        [[f"{layer}/{kind}" for kind in ("circ", "long", "trans")]
         for layer in ("endo", "epi", "mid")], args.out/"ventricle")
    valid = [s for s in solutions if solve_is_valid(s)[0]]
    if valid:
        paths.append(write_figure(max(valid, key=lambda s: s.n_dofs), str(args.out/"ventricle.png")))
    for path in paths:
        print(f"wrote {path}")
    return 0 if passed else 1


def numeric_results(solutions):
    """Numbers and matched-station differences for reproducible comparisons."""
    records = []
    for s in solutions:
        valid = solve_is_valid(s)
        measurements = {f"{k}_apex_z": v for k, v in s.apex_z().items()}
        if valid[0]:
            strains = s.strains()
            measurements.update(
                midline_mm=s.midline().cpu().numpy(),
                strain_stations_s=STATIONS, strain_layers_t=LAYERS,
                strains_percent={"/".join(k): v for k, v in strains.items()},
                strain_reference_difference_pp={
                    "/".join(k): strains[k] - STRAIN_GUIDE[k] for k in STRAIN_GUIDE})
        records.append(reporting.result_record(s, measurements, valid))
    reporting.add_refinement(records, ["endo_apex_z", "epi_apex_z"])
    return records


if __name__ == "__main__":
    raise SystemExit(main())
