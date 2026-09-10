"""Benchmark 1 of Arostica et al. (2025): monoventricular cardiac elastodynamics.

Reference
---------
R. Arostica et al., "A software benchmark for cardiac elastodynamics",
Comput. Methods Appl. Mech. Engrg. 435 (2025) 117485.  Sections 2 and 3.

Everything here is specific to that paper: its parameters, its activation and
pressure schedules, and the two material points it tracks.  The mechanics --
Holzapfel-Ogden law, Robin support, Kelvin-Voigt viscosity, generalized-alpha
time integration -- is the library's.

    python b1.py --device cuda:0                 # the standard step 1 case
    python b1.py --case active --mesh 2 12 24
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from torchcor.mechanics import Mechanics
from torchcor.mechanics.boundary import FollowerPressure, RobinBC
from torchcor.mechanics.assembly import volume_quadrature_order
from torchcor.mechanics.material import ActiveStressMaterial, HolzapfelOgdenMaterial
from torchcor.mechanics.benchmark.arostica.b1 import reference
from torchcor.mechanics.benchmark.arostica.b1.geometry import (
    MonoventricleMesh, ReferenceMesh)

# ---------------------------------------------------------------- parameters
#: Table 1: density, viscosity, penalty, switch sharpness and the Robin supports.
DENSITY, VISCOSITY, BULK, SWITCH = 1.0e3, 1.0e2, 1.0e6, 100.0
SUPPORT = dict(epi=dict(stiffness=1.0e8, damping=5.0e3, normal_only=True),
               base=dict(stiffness=1.0e5, damping=5.0e3, normal_only=False))
#: Table 2: Holzapfel-Ogden moduli [Pa] and exponents.
MATERIAL = dict(a=59.0, b=8.023, a_f=18472.0, b_f=16.026,
                a_s=2481.0, b_s=11.12, a_fs=216.0, b_fs=11.436)
#: Table 3: active stress schedule.
ACTIVE = dict(sigma_0=1.5e5, gamma=0.005, a_min=-30.0, a_max=5.0,
              t_sys=0.16, t_dias=0.484)
#: Table 4: pressure schedule.
PRESSURE = dict(a_min=-30.0, a_max=5.0, a_pre=5.0, a_mid=1.0, sigma_pre=7000.0,
                sigma_mid=16000.0, t_sys=0.17, t_dias=0.484, gamma=0.005)
#: Table 5: the blinded step-2 stiffness and contractility variations.
STEP2 = {"A": dict(a=177., a_f=55416., a_fs=648., a_s=7443., sigma_0=2e5),
         "B": dict(a=295., a_f=92360., a_fs=1080., a_s=12405., sigma_0=1e5),
         "C": dict(a=19., a_f=6157., a_fs=72., a_s=827., sigma_0=2e5)}
#: The six cases the paper defines, in its order: the three of section 3.3
#: and the three table-5 parameter sets of section 3.5.
CASES = ("passive", "active", "both", "A", "B", "C")
#: Section 3.3.1: the two tracked material points [m].
PROBES = {"p0": (0.025, 0.03, 0.0), "p1": (0.0, 0.03, 0.0)}
DURATION = 1.0
#: Generalized-alpha parameters of the reference implementation.  Simula's
#: ``problem.py`` sets ``alpha_m = 0.2`` and ``alpha_f = 0.4`` directly, which
#: is a spectral radius of 2/3, and the paper's table 10 lists the same pair
#: for that group.  ``DynamicSolver`` derives both from the radius, so this is
#: the value that reproduces them.
RHO_INFINITY = 2.0/3.0


def _smooth(delta, gamma, rising):
    half = 0.5*(1.0 + np.tanh(delta/gamma))
    return half if rising else 1.0 - half


class Schedule:
    """A driver defined by an ODE, tabulated once and read back by time.

    Both the active stress (5) and the cavity pressure (7) are relaxation
    equations driven by a smooth activation window.  They do not depend on the
    deformation, so they are integrated once on a fine grid and interpolated;
    the solver then only ever asks for a value at a time.
    """

    def __init__(self, rate, duration=DURATION, steps=100_000):
        self.t = np.linspace(0.0, duration, steps + 1)
        y = np.zeros_like(self.t)
        dt = self.t[1] - self.t[0]
        for i in range(1, len(self.t)):
            decay, source = rate(self.t[i])
            y[i] = (y[i - 1] + dt*source)/(1.0 + dt*decay)   # implicit, unconditionally stable
        self.y = y

    def __call__(self, time):
        return float(np.interp(np.clip(time, 0.0, self.t[-1]), self.t, self.y))

    @property
    def peak(self):
        return float(self.y.max())


def active_schedule(sigma_0=None, **_):
    """Active fibre stress, equations (5) and (6)."""
    p = dict(ACTIVE, **({} if sigma_0 is None else {"sigma_0": sigma_0}))
    def rate(t):
        f = (_smooth(t - p["t_sys"], p["gamma"], True)
             * _smooth(t - p["t_dias"], p["gamma"], False))
        a = p["a_max"]*f + p["a_min"]*(1.0 - f)
        return abs(a), p["sigma_0"]*max(a, 0.0)
    return Schedule(rate)


def pressure_schedule():
    """Endocardial cavity pressure, equations (7) and (8).

    The three terms of ``b`` are summed, and ``alpha_pre`` multiplies the
    relaxation gate ``g_pre``.  Reading them as a product instead leaves
    ``|b| = 1`` after diastole, so the cavity never empties -- it still holds
    14.7 kPa at one second.  Summing them gives ``|b| = 29`` there, and the
    published peaks follow: 16074 Pa against the stated 16117 for table 4,
    16491 against 16491 for the biventricular left ventricle, 4163 against
    4167 for the right, and 0.005 Pa remaining at one second.
    """
    p = PRESSURE
    def rate(t):
        f = (_smooth(t - p["t_sys"], p["gamma"], True)
             * _smooth(t - p["t_dias"], p["gamma"], False))
        g = _smooth(t - p["t_dias"], p["gamma"], False)
        b = (p["a_max"]*f + p["a_min"]*(1.0 - f)) + p["a_pre"]*g + p["a_mid"]
        return abs(b), p["sigma_mid"]*max(b, 0.0) + p["sigma_pre"]*max(g, 0.0)
    return Schedule(rate)


# --------------------------------------------------------------------- solve
def solve_b1(case="both", step2=None, dt=1.0e-3, device=None, verbose=True,
             structured=None):
    """Solve one case and return the tracked displacement histories.

    ``case`` selects the split of section 3.3: ``"active"`` drops the cavity
    pressure, ``"passive"`` drops the contraction, ``"both"`` is step 1.
    ``step2`` names a parameter set of table 5, which the paper defines only
    for the combined loading, so it cannot be paired with a split.

    The benchmark is solved on the tetrahedral mesh the participants were
    given, which is what the published comparison is against.  ``structured``
    takes ``(nt, nu, nv)`` divisions and builds a hexahedral mesh of the same
    domain instead; that is a second discretisation for comparison, and it
    does not reproduce the published results (see BENCHMARK_STATUS.md §4.2).
    """
    if step2 and case != "both":
        raise ValueError(
            f"step-2 set {step2} is defined for the combined loading only; "
            f"case={case!r} would solve different physics from the published "
            f"case it is scored against")
    mesh = (ReferenceMesh.load(device=device) if structured is None else
            MonoventricleMesh(structured, device=device))
    changed = STEP2[step2] if step2 else {}
    passive = HolzapfelOgdenMaterial(
        **{k: changed.get(k, v) for k, v in MATERIAL.items()},
        bulk_modulus=BULK, compression_switch=SWITCH)

    tension = (0.0 if case == "passive"
               else active_schedule(changed.get("sigma_0")))
    material = ActiveStressMaterial(passive, tension)
    # The frame is sampled at the same rule the assembly integrates with, so
    # the order is chosen here rather than left to each of them separately.
    quadrature = volume_quadrature_order(mesh.order, mesh.cell_element)
    axes = mesh.fibre_axes(quadrature)

    pressure = 0.0 if case == "active" else pressure_schedule()
    boundary = [FollowerPressure.on_surface(mesh, "endo", pressure)]
    boundary += [RobinBC.on_surface(mesh, name, **spec)
                 for name, spec in SUPPORT.items()]

    sim = Mechanics(mesh, material, axes=axes, boundary=boundary,
                    bulk_modulus=None, density=DENSITY, viscosity=VISCOSITY,
                    quadrature_order=quadrature)
    if verbose:
        print(f"mesh {mesh.label}: {mesh.n_cells} cells, "
              f"{mesh.n_dofs} DOF on {mesh.device}; case {case}"
              + (f", step-2 set {step2}" if step2 else ""), flush=True)

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
def report(sim, history, case, tension, pressure, scored=None):
    print(f"\n{'='*78}\nAROSTICA BENCHMARK 1 -- monoventricular, case {case}\n{'='*78}")
    if tension is not None:
        print(f"active stress peak   {tension.peak:10.2f} Pa   paper figure 2: 118817.07"
              f"   at 1 s {tension(1.0):8.4f} Pa")
    if pressure is not None:
        print(f"cavity pressure peak {pressure.peak:10.2f} Pa   paper figure 3:  16117.52"
              f"   at 1 s {pressure(1.0):8.4f} Pa")

    print(f"\n{'point':>6} " + " ".join(f"{c:>27}" for c in "xyz")
          + "     (ours, then participant mean +- spread) [mm]")
    for name in PROBES:
        u = history[name]
        ours = " ".join(f"{1e3*u[:, i].min():+7.2f}..{1e3*u[:, i].max():+7.2f}   "
                        for i in range(3))
        print(f"{name:>6} {ours}")
        if scored:
            m, sd = scored[name]["mean"], scored[name]["std"]
            ref_ = " ".join(f"{1e3*m[:, i].min():+7.2f}..{1e3*m[:, i].max():+7.2f} "
                            f"+-{1e3*sd[:, i].max():4.2f}" for i in range(3))
            print(f"{'ref':>6} {ref_}")

    # Two separate questions: did the solver finish, and does the answer agree
    # with the published participants?  A converged solve can still disagree,
    # so neither verdict is allowed to stand in for the other.
    r = sim.report
    print(f"\n{'-'*78}\nSOLVER     {'converged' if r.converged else 'DID NOT CONVERGE'}"
          f" -- {r.load_steps} steps, {r.newton_iterations} newton, "
          f"{r.linear_iterations} krylov, {r.wall_time:.0f} s, "
          f"peak {r.peak_memory_mib:.0f} MiB, |R| {r.residual:.2e}")
    # The end state is not the worst state: this beat loads hard and springs
    # back, so a final J near one says nothing about the loaded configuration.
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
          f"the published participant range")
    print("-"*78)


def result_stem(case, mesh_label, dt) -> str:
    """Name of the result files for one run.

    Case, mesh and time step together identify a run, so two runs that differ
    in any of them cannot overwrite each other's results.
    """
    return f"b1_{case}_{mesh_label}_dt{float(dt):g}"


def write_outputs(history, sim, case, dt, out: Path, scored=None, settings=None):
    out.mkdir(parents=True, exist_ok=True)
    stem = result_stem(case, sim.mesh.label, dt)
    payload = {"case": case, "torch": torch.__version__,
               "dofs": sim.mesh.n_dofs, "mesh": sim.mesh.label, "dt_s": dt,
               "settings": {k: str(v) for k, v in (settings or {}).items()},
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
            "source": "Reidmen/cardiac_benchmark_toolkit results/data",
            "participants": scored["participants"],
            "red": {p: scored[p]["red"] for p in PROBES},
            "participant_red": {p: scored[p]["participant_red"] for p in PROBES}}
    (out/f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(PROBES), figsize=(10, 8), sharex=True,
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
    fig.suptitle(f"Arostica benchmark 1, case {case}: {sim.mesh.n_dofs} DOF")
    fig.savefig(out/f"{stem}.png", dpi=140)
    plt.close(fig)
    return [out/f"{stem}.json", out/f"{stem}.png"]


def agrees(scored, probe=None) -> bool:
    """Whether a scored case agrees with the published participants.

    The paper sets no threshold, so this is the project's stated criterion:
    RED inside the published participant range, at both tracked points unless
    one is named.  Defined once, because the table, the per-case verdict and
    the exit code must not be able to disagree about what a pass is.
    """
    probes = PROBES if probe is None else (probe,)
    return all(scored[p]["red"] <= max(scored[p]["participant_red"].values())
               for p in probes)


def score_table(scores) -> str:
    """The benchmark verdict for every case that was run, as one table."""
    lines = [f"{'case':>8} {'RED p0':>8} {'participants p0':>17} "
             f"{'RED p1':>8} {'participants p1':>17}  verdict"]
    for case, scored in scores.items():
        red = [scored[p]["red"] for p in PROBES]
        band = [sorted(scored[p]["participant_red"].values()) for p in PROBES]
        lines.append(
            f"{case:>8} " + " ".join(
                f"{r:>8.4f} {b[0]:>7.3f}-{b[-1]:<9.3f}" for r, b in zip(red, band))
            + f"  {'PASS' if agrees(scored) else 'FAIL'}")
    passed = sum(agrees(scored) for scored in scores.values())
    lines.append(f"\n{passed} of {len(scores)} cases inside the published "
                 f"participant range at both tracked points.")
    return "\n".join(lines)


def write_csv(histories, path: Path) -> Path:
    """Every tracked history, for comparison against the other solvers."""
    with path.open("w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["case", "mesh", "dofs", "dt_s", "time_s"]
                     + [f"{probe}_u{c}_m" for probe in PROBES for c in "xyz"])
        for case, (mesh, dofs, dt, history) in histories.items():
            for i, t in enumerate(history["time"]):
                out.writerow([case, mesh, dofs, dt, f"{t:.6f}"]
                             + [f"{history[p][i, c]:.9e}"
                                for p in PROBES for c in range(3)])
    return path


def run_case(case, dt, device, out, structured, verbose):
    """Solve one case, report it, write its files, and return its score."""
    step2 = case if case in STEP2 else None
    sim, history = solve_b1(case="both" if step2 else case, step2=step2, dt=dt,
                            device=device, structured=structured, verbose=verbose)
    tension = None if case == "passive" else active_schedule(
        (STEP2[step2] if step2 else {}).get("sigma_0"))
    pressure = None if case == "active" else pressure_schedule()
    scored = reference.compare(case, history["time"],
                               {p: history[p] for p in PROBES},
                               complete=sim.report.converged)
    report(sim, history, case, tension, pressure, scored)
    write_outputs(history, sim, case, dt, out, scored,
                  dict(dt=dt, device=device, structured=structured))
    return scored, (sim.mesh.label, sim.mesh.n_dofs, dt, history)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default=None, choices=CASES,
                        help="one case; by default all six are run")
    parser.add_argument("--dt", type=float, default=2.0e-3, metavar="SECONDS")
    parser.add_argument("--device", default="cuda", metavar="DEV")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("results"))
    parser.add_argument("--structured", type=int, nargs=3, default=None,
                        metavar=("NT", "NU", "NV"),
                        help="solve on a structured hexahedral mesh of the same "
                             "domain instead of the mesh the participants were "
                             "given; a comparison, not the benchmark")
    args = parser.parse_args(argv)

    structured = None if args.structured is None else tuple(args.structured)
    cases = (args.case,) if args.case else CASES
    scores, histories = {}, {}
    for case in cases:
        scored, history = run_case(case, args.dt, args.device, args.out,
                                   structured, verbose=True)
        histories[case] = history
        if scored is not None:
            scores[case] = scored

    print(f"\n{'='*78}\nAROSTICA BENCHMARK 1 -- agreement with the "
          f"{reference.TEAMS.__len__()} published participants\n{'='*78}")
    if not scores:
        print("no case was scored: a case must converge and cover the full "
              f"{DURATION:g} s interval")
        return 1
    print(score_table(scores))
    print(f"\nwrote {write_csv(histories, args.out/'torchcor_arostica_b1.csv')}")

    # Completion and agreement are separate failures, and either one is a
    # failure: a case that solved cleanly but disagrees is not a pass.
    unscored = [c for c in cases if c not in scores]
    disagreed = [c for c, scored in scores.items() if not agrees(scored)]
    if unscored:
        print(f"NOT SCORED: {', '.join(unscored)} -- a case must converge and "
              f"cover the full {DURATION:g} s interval")
    if disagreed:
        print(f"DISAGREES:  {', '.join(disagreed)} -- outside the published "
              f"participant range")
    return 1 if unscored or disagreed else 0


if __name__ == "__main__":
    raise SystemExit(main())
