"""Benchmark 2 of Arostica et al. (2025): biventricular cardiac elastodynamics.

Reference
---------
R. Arostica et al., "A software benchmark for cardiac elastodynamics",
Comput. Methods Appl. Mech. Engrg. 435 (2025) 117485.  Section 4.

The same physics as benchmark 1 on a biventricular geometry: two cavities with
their own pressures, a stiffer base support, and three tracked points instead
of two.  The mechanics -- Holzapfel-Ogden law, Robin support, Kelvin-Voigt
viscosity, generalized-alpha time integration -- is the library's, unchanged.

    python -m torchcor.mechanics.benchmark.arostica.b2.b2
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from torchcor.mechanics import Mechanics
from torchcor.mechanics.assembly import volume_quadrature_order
from torchcor.mechanics.boundary import FollowerPressure, RobinBC
from torchcor.mechanics.material import ActiveStressMaterial, HolzapfelOgdenMaterial
from torchcor.mechanics.benchmark.arostica.b2 import reference
from torchcor.mechanics.benchmark.arostica.b2.geometry import (
    RESOLUTIONS, BiventricleMesh)

# ---------------------------------------------------------------- parameters
#: Density, viscosity, penalty and switch sharpness, as in benchmark 1.
DENSITY, VISCOSITY, BULK, SWITCH = 1.0e3, 1.0e2, 1.0e6, 100.0
#: The Robin supports.  The base is held ten times as stiffly as in benchmark
#: 1 (1e6 against 1e5); the epicardium is unchanged and still acts on the
#: normal component alone.
SUPPORT = dict(epi=dict(stiffness=1.0e8, damping=5.0e3, normal_only=True),
               base=dict(stiffness=1.0e6, damping=5.0e3, normal_only=False))
#: Holzapfel-Ogden moduli [Pa] and exponents, as in benchmark 1.
MATERIAL = dict(a=59.0, b=8.023, a_f=18472.0, b_f=16.026,
                a_s=2481.0, b_s=11.12, a_fs=216.0, b_fs=11.436)
#: Table 8: the active stress schedule.  The timings are *not* benchmark 1's
#: -- contraction starts at 0.163 s and relaxation at 0.5 s, against 0.16 and
#: 0.484 there -- and the reference implementation's shared defaults are
#: benchmark 1's, so the paper governs.  Fig. 6 gives the peak these produce.
ACTIVE = dict(sigma_0=1.5e5, gamma=0.005, a_min=-30.0, a_max=5.0,
              t_sys=0.163, t_dias=0.5)
#: The peak of the activation function in Fig. 6 [Pa], which the schedule
#: above must reproduce.
ACTIVE_PEAK = 120775.56
#: The two cavity pressures.  Same equations (7) and (8) as benchmark 1, with
#: the constants section 4 gives for each ventricle.
LV_PRESSURE = dict(a_min=-30.0, a_max=5.0, a_pre=5.0, a_mid=15.0,
                   sigma_pre=12000.0, sigma_mid=16000.0,
                   t_sys=0.17, t_dias=0.484, gamma=0.005)
RV_PRESSURE = dict(a_min=-30.0, a_max=5.0, a_pre=1.0, a_mid=10.0,
                   sigma_pre=3000.0, sigma_mid=4000.0,
                   t_sys=0.17, t_dias=0.484, gamma=0.005)
#: Section 4: the three tracked material points [m].  ``p2`` is on the right
#: ventricle and has no counterpart in benchmark 1.
PROBES = {"p0": (0.025, 0.03, 0.0), "p1": (0.0, 0.03, 0.0),
          "p2": (0.025, 0.0, 0.072)}
DURATION = 1.0
#: Generalized-alpha parameters of the reference implementation: alpha_m = 0.2
#: and alpha_f = 0.4, which is a spectral radius of 2/3.
RHO_INFINITY = 2.0/3.0


def _smooth(delta, gamma, rising):
    half = 0.5*(1.0 + np.tanh(delta/gamma))
    return half if rising else 1.0 - half


class Schedule:
    """A driver defined by an ODE, tabulated once and read back by time.

    Both the active stress and the cavity pressures are relaxation equations
    driven by a smooth activation window.  They do not depend on the
    deformation, so they are integrated once on a fine grid and interpolated;
    the solver then only ever asks for a value at a time.
    """

    def __init__(self, rate, duration=DURATION, steps=100_000):
        self.t = np.linspace(0.0, duration, steps + 1)
        y = np.zeros_like(self.t)
        dt = self.t[1] - self.t[0]
        for i in range(1, len(self.t)):
            decay, source = rate(self.t[i])
            y[i] = (y[i - 1] + dt*source)/(1.0 + dt*decay)   # implicit, stable
        self.y = y

    def __call__(self, time):
        return float(np.interp(np.clip(time, 0.0, self.t[-1]), self.t, self.y))

    @property
    def peak(self):
        return float(self.y.max())


def active_schedule():
    """Active fibre stress, equations (5) and (6)."""
    p = ACTIVE
    def rate(t):
        f = (_smooth(t - p["t_sys"], p["gamma"], True)
             * _smooth(t - p["t_dias"], p["gamma"], False))
        a = p["a_max"]*f + p["a_min"]*(1.0 - f)
        return abs(a), p["sigma_0"]*max(a, 0.0)
    return Schedule(rate)


def pressure_schedule(p):
    """Cavity pressure, equations (7) and (8), for one ventricle."""
    def rate(t):
        f = (_smooth(t - p["t_sys"], p["gamma"], True)
             * _smooth(t - p["t_dias"], p["gamma"], False))
        g = _smooth(t - p["t_dias"], p["gamma"], False)
        b = (p["a_max"]*f + p["a_min"]*(1.0 - f)) + p["a_pre"]*g + p["a_mid"]
        return abs(b), p["sigma_mid"]*max(b, 0.0) + p["sigma_pre"]*max(g, 0.0)
    return Schedule(rate)


# --------------------------------------------------------------------- solve
def solve_b2(resolution="coarse", dt=1.0e-3, device=None, verbose=True):
    """Solve benchmark 2 and return the tracked displacement histories.

    ``resolution`` picks one of the two refinement levels section 4.4 asks
    for; it selects the mesh, its fibres, the result filenames and the
    participant population the run is scored against, so the three cannot
    disagree.
    """
    mesh = BiventricleMesh.load(resolution, device=device)
    material = ActiveStressMaterial(
        HolzapfelOgdenMaterial(**MATERIAL, bulk_modulus=BULK,
                               compression_switch=SWITCH),
        active_schedule())
    # The frame is sampled at the same rule the assembly integrates with, so
    # the order is chosen here rather than left to each of them separately.
    quadrature = volume_quadrature_order(mesh.order, mesh.cell_element)
    axes = mesh.fibre_axes(quadrature)

    boundary = [FollowerPressure.on_surface(mesh, "lv", pressure_schedule(LV_PRESSURE)),
                FollowerPressure.on_surface(mesh, "rv", pressure_schedule(RV_PRESSURE))]
    boundary += [RobinBC.on_surface(mesh, name, **spec)
                 for name, spec in SUPPORT.items()]

    sim = Mechanics(mesh, material, axes=axes, boundary=boundary,
                    bulk_modulus=None, density=DENSITY, viscosity=VISCOSITY,
                    quadrature_order=quadrature)
    if verbose:
        print(f"mesh {mesh.label}: {mesh.n_cells} cells, {mesh.n_dofs} DOF "
              f"on {mesh.device}", flush=True)

    probes = torch.tensor(list(PROBES.values()), dtype=mesh.dtype, device=mesh.device)
    cells, xi = mesh.locate(probes)
    history = {"time": [], **{k: [] for k in PROBES}}

    def record(t, u, v, a):
        moved = mesh.interpolate_local(u.reshape(-1, 3), cells, xi)
        history["time"].append(t)
        for name, row in zip(PROBES, moved.tolist()):
            history[name].append(row)

    sim.solve_dynamic(DURATION, dt=dt, verbose=verbose, observer=record,
                      rho_infinity=RHO_INFINITY, raise_on_failure=False)
    return sim, {k: np.asarray(v) for k, v in history.items()}


# -------------------------------------------------------------------- report
def report(sim, history, scored=None):
    print(f"\n{'='*78}\nAROSTICA BENCHMARK 2 -- biventricular\n{'='*78}")
    print(f"\n{'point':>6} " + " ".join(f"{c:>27}" for c in "xyz")
          + "     (ours, then participant mean +- spread) [mm]")
    for name in PROBES:
        u = history[name]
        print(f"{name:>6} " + " ".join(
            f"{1e3*u[:, i].min():+7.2f}..{1e3*u[:, i].max():+7.2f}   " for i in range(3)))
        if scored:
            m, sd = scored[name]["mean"], scored[name]["std"]
            print(f"{'ref':>6} " + " ".join(
                f"{1e3*m[:, i].min():+7.2f}..{1e3*m[:, i].max():+7.2f} "
                f"+-{1e3*sd[:, i].max():4.2f}" for i in range(3)))

    # Two separate questions: did the solver finish, and does the answer agree
    # with the published participants?  A converged solve can still disagree.
    r = sim.report
    print(f"\n{'-'*78}\nSOLVER     {'converged' if r.converged else 'DID NOT CONVERGE'}"
          f" -- {r.load_steps} steps, {r.newton_iterations} newton, "
          f"{r.linear_iterations} krylov, {r.wall_time:.0f} s, "
          f"peak {r.peak_memory_mib:.0f} MiB, |R| {r.residual:.2e}")
    print(f"           min J {r.worst_jacobian:.4f} at t = {r.worst_jacobian_time:.3f} s"
          f" (end state {r.min_jacobian:.4f})"
          + (f", cutbacks {dict(r.cutbacks)}" if r.cutbacks else ""))
    if not scored:
        print(f"BENCHMARK  NOT SCORED -- the run must converge and cover the full "
              f"{DURATION:g} s interval\n{'-'*78}")
        return
    print(f"\nagreement with {scored['n_participants']} published participants, "
          f"by the paper's RED (equation 21):")
    print(reference.summary(scored))
    where = ", ".join(f"{p} {'inside' if agrees(scored, p) else 'OUTSIDE'}"
                      for p in PROBES)
    print(f"BENCHMARK  {'PASS' if agrees(scored) else 'FAIL'} -- {where} "
          f"the published participant range\n{'-'*78}")


def agrees(scored, probe=None) -> bool:
    """Whether a scored run agrees with the published participants.

    The paper sets no threshold, so this is the project's stated criterion:
    RED inside the published participant range, at every tracked point unless
    one is named.  Defined once, so the printed verdict and the exit code
    cannot disagree about what a pass is.
    """
    probes = PROBES if probe is None else (probe,)
    return all(scored[p]["red"] <= max(scored[p]["participant_red"].values())
               for p in probes)


def write_outputs(history, sim, dt, out: Path, scored=None):
    out.mkdir(parents=True, exist_ok=True)
    stem = f"b2_{sim.mesh.label}_dt{float(dt):g}"
    payload = {"case": sim.mesh.label, "torch": torch.__version__,
               "dofs": sim.mesh.n_dofs, "mesh": sim.mesh.label, "dt_s": dt,
               "solver": {"converged": sim.report.converged,
                          "steps": sim.report.load_steps,
                          "newton": sim.report.newton_iterations,
                          "krylov": sim.report.linear_iterations,
                          "wall_time_s": sim.report.wall_time,
                          "peak_mib": sim.report.peak_memory_mib,
                          "min_jacobian_end": sim.report.min_jacobian,
                          "worst_jacobian": sim.report.worst_jacobian,
                          "worst_jacobian_time_s": sim.report.worst_jacobian_time,
                          "residual": sim.report.residual,
                          "cutbacks": dict(sim.report.cutbacks)},
               "time_s": history["time"].tolist(),
               "displacement_m": {k: history[k].tolist() for k in PROBES}}
    if scored:
        payload["reference"] = {
            "source": "Zenodo 14260459, results_time_curves",
            "participants": scored["participants"],
            "red": {p: scored[p]["red"] for p in PROBES},
            "participant_red": {p: scored[p]["participant_red"] for p in PROBES}}
    (out/f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(PROBES), figsize=(13, 8), sharex=True,
                             layout="constrained")
    for col, name in enumerate(PROBES):
        u = history[name]
        for row, comp in enumerate("xyz"):
            ax = axes[row, col]
            if scored:
                g, m, sd = scored["time"], scored[name]["mean"], scored[name]["std"]
                ax.fill_between(g, 1e3*(m[:, row] - sd[:, row]),
                                1e3*(m[:, row] + sd[:, row]), color="0.8",
                                label="participants" if row == col == 0 else None)
                ax.plot(g, 1e3*m[:, row], color="0.45", lw=1,
                        label="their mean" if row == col == 0 else None)
            ax.plot(history["time"], 1e3*u[:, row], color="C3", lw=1.4,
                    label="TorchCor" if row == col == 0 else None)
            ax.set_ylabel(f"$u_{comp}$ [mm]" if col == 0 else "")
            ax.grid(alpha=.3)
        axes[0, col].set_title(f"particle {name}"
                               + (f"   RED {scored[name]['red']:.3f}" if scored else ""))
        axes[-1, col].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Arostica benchmark 2, biventricular: {sim.mesh.n_dofs} DOF")
    fig.savefig(out/f"{stem}.png", dpi=140)
    plt.close(fig)

    return [out/f"{stem}.json", out/f"{stem}.png", write_csv(out)]


def write_csv(out: Path) -> Path:
    """Every result in ``out`` as one table, for comparison with other solvers.

    Rebuilt from the result files rather than from the run that just finished,
    so a second refinement level adds rows instead of replacing the first
    level's -- the resolution is a column here, not part of the filename.
    """
    path = out/"torchcor_arostica_b2.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mesh", "dofs", "dt_s", "time_s"]
                        + [f"{p}_u{c}_m" for p in PROBES for c in "xyz"])
        for result in sorted(out.glob("b2_*.json")):
            run = json.loads(result.read_text())
            u = {p: np.asarray(run["displacement_m"][p]) for p in PROBES}
            for i, t in enumerate(run["time_s"]):
                writer.writerow([run["mesh"], run["dofs"], run["dt_s"], f"{t:.6f}"]
                                + [f"{u[p][i, c]:.9e}" for p in PROBES
                                   for c in range(3)])
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", default="coarse", choices=RESOLUTIONS,
                        help="which of the two supplied refinement levels")
    parser.add_argument("--dt", type=float, default=2.0e-3, metavar="SECONDS")
    parser.add_argument("--device", default="cuda", metavar="DEV")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)

    sim, history = solve_b2(args.mesh, dt=args.dt, device=args.device)
    scored = reference.compare(args.mesh, history["time"],
                               {p: history[p] for p in PROBES},
                               complete=sim.report.converged)
    report(sim, history, scored)
    for path in write_outputs(history, sim, args.dt, args.out, scored):
        print(f"wrote {path}")
    return 0 if scored is not None and agrees(scored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
