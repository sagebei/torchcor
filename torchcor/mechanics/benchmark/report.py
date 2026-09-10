"""Small reporting and output helpers shared by the mechanics benchmarks."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch


def arguments(description, argv=None, default_out="."):
    """Parse the runner's options.  Outputs land in ``default_out`` unless
    ``--out`` says otherwise; each benchmark passes its own folder."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", default="cuda", metavar="DEV",
                        help="CUDA device, for example cuda:0 or cuda:1")
    parser.add_argument("--mesh", type=int, nargs=3, metavar=("N1", "N2", "N3"),
                        help="solve one mesh instead of the default refinement study")
    parser.add_argument("--out", type=Path, default=Path(default_out), metavar="DIR",
                        help=f"directory for numeric JSON and figures (default: {default_out})")
    args = parser.parse_args(argv)
    if torch.device(args.device).type != "cuda":
        parser.error("these benchmark runners require a CUDA device")
    if args.mesh is not None and min(args.mesh) < 1:
        parser.error("mesh counts must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def solve_is_valid(solution):
    """Require a completed solve and finite, positive independent samples."""
    r = solution.report
    if not r.converged:
        return False, "solver reported failure"
    if not np.isfinite(r.load_factor) or abs(r.load_factor - 1.) > 1e-9:
        return False, f"full load was not reached (lambda={r.load_factor})"
    if not np.isfinite(r.residual) or r.residual < 0:
        return False, "invalid force residual"
    if not np.isfinite(r.volume_error) or r.volume_error < 0:
        return False, "invalid mixed constraint residual"
    if not np.isfinite(r.min_jacobian) or r.min_jacobian <= 0:
        return False, "invalid Jacobian at the solver's quadrature points"
    if not bool(torch.isfinite(solution.u).all()):
        return False, "non-finite displacement"
    volume = solution.volume
    if not volume:
        return False, "independent volume diagnostics are unavailable"
    if not all(np.isfinite(value) for value in volume.values()):
        return False, "non-finite independent volume diagnostics"
    if volume["min_jacobian"] <= 0:
        return False, "inverted element at an independent quadrature point"
    return True, ""


def volume_table(solutions):
    print("\nVolume diagnostics: independent 6-point Gauss rule per axis.")
    print("The mixed constraint measures pressure moments; it does not force J=1 at every point.")
    print(f"{'mesh':>14} {'disp. DOFs':>11} {'max|J-1|':>10} {'RMS|J-1|':>10} "
          f"{'dV/V':>11} {'min J':>9}")
    for s in solutions:
        if not s.volume:
            continue
        v = s.volume
        print(f"{str(s.divisions):>14} {s.n_dofs:>11} {v['max_error']:>9.3%} "
              f"{v['rms_error']:>9.3%} {v['volume_change']:>+11.3e} "
              f"{v['min_jacobian']:>9.5f}")
    print("Maxima and minima are sampled values, not bounds throughout every element.")


def result_record(solution, measurements, valid):
    simulator = getattr(solution, "simulator", None)
    device = solution.mesh.device
    return {
        "divisions": list(solution.divisions), "order": solution.order,
        "cells": solution.mesh.n_cells, "nodes": solution.mesh.n_points,
        "displacement_dofs": solution.n_dofs,
        "device": str(solution.mesh.device), "dtype": str(solution.mesh.dtype),
        "device_name": (torch.cuda.get_device_name(device)
                        if device.type == "cuda" and torch.cuda.is_available() else None),
        "augmentation_parameter": None if simulator is None else simulator.bulk_modulus,
        "integration_points_per_axis": (None if simulator is None or simulator.problem is None
                                        else simulator.problem.elem_full.n_gauss),
        "solver": asdict(solution.report),
        "independent_volume": solution.volume, "volume_gauss_points_per_axis": 6,
        "valid": bool(valid[0]), "invalid_reason": valid[1],
        **measurements,
    }


def add_refinement(records, position_keys):
    """Attach signed changes from the previous valid mesh at matching stations."""
    previous = None
    for record in sorted(records, key=lambda r: r["displacement_dofs"]):
        if not record["valid"]:
            continue
        if previous is not None:
            record["change_from_previous_mesh"] = {
                "previous_displacement_dofs": previous["displacement_dofs"],
                "positions_mm": {key: record[key] - previous[key] for key in position_keys},
                "strains_percentage_points": {
                    key: np.asarray(value) - previous["strains_percent"][key]
                    for key, value in record["strains_percent"].items()},
                "volume_rms": record["independent_volume"]["rms_error"]
                              - previous["independent_volume"]["rms_error"],
            }
        previous = record


def write_json(path, problem, records, passed, reference, *,
               verdict_scope=("Finest position agreement and validity of every requested solve; "
                              "strain and volume refinement remain separate diagnostics.")):
    """Write portable JSON; non-finite failed-run values are explicit nulls."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    payload = {
        "problem": problem, "pytorch": torch.__version__,
        "position_agreement_passed": bool(passed),
        "verdict_scope": verdict_scope,
        "reference": reference, "results": records,
    }
    path = Path(path)
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")
    return path


def comparison_figures(records, positions, strain_reference, layout, stem, *,
                       reference_description=("Dashed lines and shaded ranges: "
                                              "approximate readings from the paper")):
    """Plot refinement and station strains, optionally adding reference guides."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = sorted((r for r in records if r["valid"]), key=lambda r: r["displacement_dofs"])
    if not valid:
        return []
    dofs = [r["displacement_dofs"] for r in valid]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, (target, spread) in positions.items():
        axes[0].plot(dofs, [r[name] for r in valid], "o-", label=name.replace("_", " "))
        axes[0].axhspan(*spread, color="0.8", alpha=.35)
        axes[0].axhline(target, color="0.4", linestyle="--", linewidth=.8)
    axes[0].set(ylabel="Final position z (mm)", title="Position agreement")
    axes[0].legend()
    for key, label in (("rms_error", "Volume-weighted RMS"), ("max_error", "Sampled maximum")):
        axes[1].plot(dofs, [100*r["independent_volume"][key] for r in valid], "o-", label=label)
    axes[1].set(ylabel="Local volume error (%)", title="Independent volume diagnostics")
    axes[1].legend()
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Displacement DOFs before boundary constraints")
        ax.grid(alpha=.2)
    fig.suptitle(reference_description)
    convergence = Path(f"{stem}_convergence.png")
    fig.savefig(convergence, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(layout), len(layout[0]), squeeze=False,
                             figsize=(10, 3*len(layout)), layout="constrained")
    colors = plt.cm.viridis(np.linspace(.15, .85, len(valid)))
    for i, row in enumerate(layout):
        for j, key in enumerate(row):
            ax = axes[i, j]
            for record, color in zip(valid, colors):
                values = record["strains_percent"][key]
                ax.plot(np.arange(1, len(values) + 1), values, "o-", markersize=3,
                        color=color, label=f"{record['displacement_dofs']} DOFs")
            if strain_reference is not None:
                reference = strain_reference[key]
                ax.plot(np.arange(1, len(reference) + 1), reference, "x--", color="0.35",
                        linewidth=.8, label="Paper: approximate visual guide")
            ax.set(title=key.replace("/", " "), xlabel="Station / first point of pair",
                   ylabel="Distance-change strain (%)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Reference guides are figure readings; strain agreement has no pass threshold"
                 if strain_reference is not None else
                 "Computed strain curves; no numerical reference curve supplied")
    strains = Path(f"{stem}_strains.png")
    fig.savefig(strains, dpi=180)
    plt.close(fig)
    return [convergence, strains]
