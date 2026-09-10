"""Benchmark problem 1: deformation of a beam (Land et al., 2015).

Run it from the repository root to solve the problem, check the answer against
the published consensus, and get a picture::

    python -m torchcor.mechanics.benchmark.p1
    python -m torchcor.mechanics.benchmark.p1 --device cuda:1 --mesh 20 2 2 --out results/p1

It writes numeric results, a deformation figure, and refinement/strain plots.

The problem, from section 2c(i) of the paper:

=========================  ====================================================
Geometry                   ``x in [0, 10], y in [0, 1], z in [0, 1]`` mm
Constitutive law           Guccione, transversely isotropic
Parameters                 ``C = 2`` kPa, ``bf = 8``, ``bt = 2``, ``bfs = 4``
Fibre direction            constant, ``(1, 0, 0)``
Fixed                      left face ``x = 0``, in all directions
Pressure                   ``0.004`` kPa on the whole bottom face ``z = 0``
Incompressibility          fully incompressible, ``J = 1``
=========================  ====================================================

The pressure is a *follower* load: it turns with the deformed surface and
scales with its area.  The benchmark quantity is the deformed ``z`` of the
point ``(10, 0.5, 1)`` -- figure 3 of the paper.

Reference: S. Land *et al.*, *Proc. R. Soc. A* **471**: 20150641 (2015),
``../resources/benchmark.pdf``.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from torchcor.mechanics import Mechanics
from torchcor.mechanics.boundary import DirichletBC, FollowerPressure
from torchcor.mechanics.material import GuccioneMaterial, IsochoricMaterial, MaterialAxes
from torchcor.mechanics.mesh import StructuredBoxMesh
from torchcor.mechanics.solver import SolveReport
from torchcor.mechanics.benchmark import report as reporting


# =============================================================================
#  The problem, exactly as the paper specifies it
# =============================================================================
ORIGIN = (0.0, 0.0, 0.0)
LENGTHS = (10.0, 1.0, 1.0)                        # mm
MATERIAL = dict(C=2.0, bf=8.0, bt=2.0, bfs=4.0)   # C in kPa
FIBRE = (1.0, 0.0, 0.0)
CLAMPED_FACE = "x-"                               # the face x = 0
LOADED_FACE = "z-"                                # the face z = 0
PRESSURE = 0.004                                  # kPa
PROBE = (10.0, 0.5, 1.0)                          # the point figure 3 reports

#: Meshes to solve, coarse to fine, to show the answer converging.
MESHES = [(10, 1, 1), (20, 2, 2), (30, 3, 3), (40, 4, 4), (60, 6, 6)]


# =============================================================================
#  What the paper says the answer is
# =============================================================================
CONSENSUS_TIP_Z = 4.17                    # figure 3, mm
SUBMISSION_SPREAD = (4.151, 4.185)        # converged submissions, figure 3
MIDLINE_END = (9.385, 3.690)              # figure 4b, deformed end of the line
#: Figure 5, consensus strain (%) along the mid-line, LifeV outlier excluded.
#: Digitised from the printed figure, so indicative only.
STRAIN = {
    "x": [-0.155, -0.100, -0.072, -0.036, -0.017, -0.006, 0.000, 0.003, 0.005],
    "y": [0.000, 0.380, 0.245, 0.155, 0.093, 0.058, 0.038, 0.027, 0.022, 0.020],
    "z": [0.000, 0.905, 0.720, 0.550, 0.415, 0.300, 0.200, 0.115, 0.050, 0.005],
}
#: Our pass mark, matching the 0.5-1% inter-code agreement the paper reports.
#: The paper itself prescribes no executable pass/fail rule.
TOLERANCE = 0.01

#: The mid-line points the strains are measured between.
STRAIN_POINTS = np.stack([np.arange(10.0), np.full(10, 0.5), np.full(10, 0.5)], axis=1)


# =============================================================================
#  Solving it
# =============================================================================
def solve_beam(divisions=(40, 4, 4), order: int = 2, device=None,
               dtype: torch.dtype = torch.float64, verbose: bool = True,
               bulk_modulus: float = 100.0,
               **solver_options) -> "BeamSolution":
    """Solve problem 1 on one mesh.

    Geometry, then material, then boundary, then solve -- each built by the
    module it belongs to.  Problems 2 and 3 are the same four steps: a
    ventricular mesh in place of the box, a pressure on ``"endo"`` instead of
    ``"z-"``, and for problem 3 the passive law wrapped in
    ``ActiveStressMaterial(IsochoricMaterial(passive), tension=...)`` -- in that
    order, so that the prescribed active stress survives the volumetric split
    instead of being projected by it.
    """
    # kappa = 100 is measured, not derived.  FEBio's rule for this benchmark
    # (kappa = 100 x C x b_max = 1600 here) assumes a direct linear solver,
    # where the penalty costs nothing; with an iterative solve it sets the
    # condition number, and on this beam kappa = 1600 cannot be solved at
    # all by block-Jacobi CG, kappa = 400 needs twelve cutbacks, kappa = 100
    # needs none.  The converged answer does not depend on it.
    # 1. The geometry.  Its six sides are named "x-", "x+", "y-", "y+", "z-", "z+",
    #    and it fixes the device and precision everything else follows.
    mesh = StructuredBoxMesh(ORIGIN, LENGTHS, divisions, order=order,
                             device=device, dtype=dtype)

    # 2. How the tissue behaves: the benchmark's law, fibres along the long axis.
    material = IsochoricMaterial(GuccioneMaterial(**MATERIAL))
    fibres = MaterialAxes(f=FIBRE)

    # 3. What holds it, and what pushes it.
    fixed = DirichletBC.on_surface(mesh, CLAMPED_FACE)
    load = FollowerPressure.on_surface(mesh, LOADED_FACE, PRESSURE)

    # 4. Put it together and solve.
    sim = Mechanics(mesh, material, axes=fibres, boundary=[fixed, load],
                    bulk_modulus=bulk_modulus)
    if verbose:
        print(f"mesh {tuple(divisions)}, order {order}: {mesh.n_cells} cells, "
              f"{mesh.n_points} nodes, {mesh.n_dofs} DOF on {mesh.device}",
              flush=True)
    sim.solve(verbose=verbose, raise_on_failure=False, **solver_options)

    return BeamSolution(mesh, sim.u, sim.report, tuple(divisions), order, sim,
                        volume=sim.volume_report(n_gauss=6))


# =============================================================================
#  Measuring it, the way the paper does
# =============================================================================
@dataclass
class BeamSolution:
    """One solved mesh, and the four quantities the paper compares."""

    mesh: StructuredBoxMesh
    u: torch.Tensor
    report: SolveReport
    divisions: tuple[int, int, int]
    order: int
    simulator: Mechanics | None = None
    volume: dict = field(default_factory=dict)

    @property
    def n_dofs(self) -> int:
        return self.mesh.n_dofs

    def probe(self, points) -> torch.Tensor:
        """Where undeformed ``points`` end up."""
        if self.simulator is not None:
            return self.simulator.probe(points)
        # Solutions rebuilt from stored fields have no simulator behind them.
        pts = torch.as_tensor(points, dtype=self.mesh.dtype, device=self.mesh.device)
        return self.mesh.interpolate(self.mesh.points + self.u.reshape(-1, 3), pts)

    def tip_z(self) -> float:
        """Deformed z of the probe point -- the headline number (figure 3)."""
        return float(self.probe([PROBE])[0, 2])

    def midline(self, n: int = 101) -> torch.Tensor:
        """Deformed image of the line ``(x, 0.5, 0.5)`` (figure 4)."""
        x = torch.linspace(0.0, LENGTHS[0], n, dtype=self.mesh.dtype,
                           device=self.mesh.device)
        half = torch.full_like(x, 0.5)
        return self.probe(torch.stack([x, half, half], dim=1))

    def strains(self) -> dict[str, np.ndarray]:
        """Strain along the mid-line in each direction (figure 5).

        Land et al. equation (3.1): the percentage change in distance between
        pairs 1 mm apart in x and 0.4 mm apart in y or z.
        """
        def between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            xa, xb = self.probe(a).cpu().numpy(), self.probe(b).cpu().numpy()
            return (np.linalg.norm(xa - xb, axis=1)
                    / np.linalg.norm(a - b, axis=1) - 1.0) * 100.0

        along = STRAIN_POINTS[:9]
        return {"x": between(along, along + [1.0, 0.0, 0.0]),
                "y": between(STRAIN_POINTS, STRAIN_POINTS + [0.0, 0.4, 0.0]),
                "z": between(STRAIN_POINTS, STRAIN_POINTS + [0.0, 0.0, 0.4])}


def solve_is_valid(solution: BeamSolution) -> tuple[bool, str]:
    """Require a completed valid solve before comparing the tip position."""
    valid, reason = reporting.solve_is_valid(solution)
    if not valid:
        return valid, reason
    if not np.isfinite(solution.tip_z()):
        return False, "non-finite probe coordinate"
    return True, ""


def report_benchmark(solutions: list[BeamSolution]) -> bool:
    """Return position agreement; print strain and volume diagnostics separately."""
    rule = "=" * 74
    lo, hi = SUBMISSION_SPREAD
    print(f"\n{rule}\nBENCHMARK PROBLEM 1 -- deformed z of the point {PROBE}\n{rule}")
    print(f"{'mesh':>12} {'order':>6} {'DOF':>8} {'z (mm)':>9} {'error':>8} "
          f"{'in spread':>10} {'valid':>6}")

    valid = []
    for s in solutions:
        ok, why = solve_is_valid(s)
        z = s.tip_z()
        if ok:
            valid.append(s)
        print(f"{str(s.divisions):>12} {s.order:>6} {s.n_dofs:>8} {z:>9.4f} "
              f"{(z - CONSENSUS_TIP_Z) / CONSENSUS_TIP_Z * 100:>7.2f}% "
              f"{('yes' if ok and lo <= z <= hi else 'no'):>10} "
              f"{('yes' if ok else 'NO'):>6}")
        if not ok:
            print(f"{'':>12}   ^ not comparable: {why}")

    if not valid:
        print(f"\n{rule}\nVERDICT: FAIL   nothing solved, so accuracy was not "
              f"assessed.\n{rule}")
        return False

    # Judge the finest mesh that solved -- but a study whose finer meshes failed
    # has not shown convergence, whatever the coarse ones say.
    finest = max(valid, key=lambda s: s.n_dofs)
    z = finest.tip_z()
    error = abs(z - CONSENSUS_TIP_Z) / CONSENSUS_TIP_Z
    missing = len(solutions) - len(valid)
    passed = error <= TOLERANCE and not missing

    end = finest.midline()[-1].cpu().numpy()
    strains = finest.strains()
    strain = {k: float(np.abs(strains[k] - STRAIN[k]).max()) for k in STRAIN}
    print(f"\nconsensus {CONSENSUS_TIP_Z:.2f} mm; converged submissions span "
          f"[{lo:.3f}, {hi:.3f}] mm  (fig. 3)")
    print(f"mid-line end   {end[0]:.3f}, {end[2]:.3f} mm   vs reference "
          f"{MIDLINE_END[0]:.3f}, {MIDLINE_END[1]:.3f} mm  (fig. 4b)")
    print(f"strain         max deviation {strain['x']:.3f} / {strain['y']:.3f} / "
          f"{strain['z']:.3f} pp in x / y / z  (fig. 5, digitised)")
    print("Strain reference values are approximate figure readings, with no pass threshold.")
    for direction, values in strains.items():
        print(f"  {direction}: " + " ".join(f"{value:+.4f}" for value in values) + " %")
    if len(valid) > 1:
        previous = sorted(valid, key=lambda s: s.n_dofs)[-2]
        before = previous.strains()
        drift = max(float(np.abs(strains[k] - before[k]).max()) for k in strains)
        print(f"refinement     {previous.divisions} -> {finest.divisions} moves "
              f"the tip by {abs(z - previous.tip_z()):.4f} mm and strain by "
              f"{drift:.4f} percentage points")
    else:
        print("One mesh only: refinement convergence was not assessed.")

    reporting.volume_table(valid)

    print(f"\n{rule}")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}   finest valid mesh "
          f"{finest.divisions} ({finest.n_dofs} DOF): z = {z:.4f} mm")
    print(f"         {error * 100:.2f}% from consensus, tolerance "
          f"{TOLERANCE * 100:.0f}%{'' if error <= TOLERANCE else '   -- outside it'}")
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
def write_figure(solution: BeamSolution, path: str = "beam.png") -> Path:
    """Render the beam, laid out like figure 1 of the paper.

    Reference shape in grey, deformed coloured by displacement magnitude, with
    the probe point and mid-line marked.
    """
    sim = solution.simulator
    return sim.to_png(
        path,
        points={f"probe {PROBE}": sim.probe([PROBE])},
        curves={"mid-line (x, 0.5, 0.5)": solution.midline()},
        title=(f"Benchmark problem 1: beam under {PRESSURE} kPa follower "
               f"pressure\n{solution.divisions} order-{solution.order} mesh, "
               f"{solution.n_dofs} DOF, tip z = {solution.tip_z():.4f} mm"),
    )


# =============================================================================
#  Running it
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    args = reporting.arguments(__doc__.splitlines()[0], argv,
                               default_out=Path(__file__).with_name("p1"))
    meshes = [tuple(args.mesh)] if args.mesh else MESHES
    solutions = [solve_beam(m, device=args.device) for m in meshes]
    passed = report_benchmark(solutions)
    records = numeric_results(solutions)
    reference = {"source": "Land et al. 2015, figures 3–5; approximate figure readings",
                 "tip_z_mm": CONSENSUS_TIP_Z, "tip_spread_mm": SUBMISSION_SPREAD,
                 "relative_position_tolerance": TOLERANCE,
                 "strain_guide_percent": STRAIN, "strain_acceptance_threshold": None}
    paths = [reporting.write_json(args.out/"beam_results.json", 1, records, passed, reference)]
    paths += reporting.comparison_figures(
        records, {"tip_z": (CONSENSUS_TIP_Z, SUBMISSION_SPREAD)}, STRAIN,
        [["x", "y", "z"]], args.out/"beam")
    valid = [s for s in solutions if solve_is_valid(s)[0]]
    if valid:
        paths.append(write_figure(max(valid, key=lambda s: s.n_dofs), str(args.out/"beam.png")))
    for path in paths:
        print(f"wrote {path}")
    return 0 if passed else 1


def numeric_results(solutions):
    """Numbers and matched-station differences for reproducible comparisons."""
    records = []
    for s in solutions:
        valid = solve_is_valid(s)
        measurements = {"tip_z": s.tip_z()}
        if valid[0]:
            strains = s.strains()
            measurements.update(
                midline_mm=s.midline().cpu().numpy(), strains_percent=strains,
                strain_reference_difference_pp={k: strains[k] - STRAIN[k] for k in STRAIN})
        records.append(reporting.result_record(s, measurements, valid))
    reporting.add_refinement(records, ["tip_z"])
    return records


if __name__ == "__main__":
    raise SystemExit(main())
