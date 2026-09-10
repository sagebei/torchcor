"""Compare TorchCor's reaction-eikonal run against openCARP's, node by node.

Both sides must have been run first:

    ./run_case_AF.slurm                 # openCARP  -> biv/act.igb, biv/vm.igb
    python igb.py                       #           -> results/*.npy
    python ../../../demo/eikonal_ventricle.py   # TorchCor -> demo/biventricle_eikonal/*.pt

The two describe the same simulation; see the header of ``parameters.par``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

RE_PLUS = "--re-plus" in sys.argv
HERE = Path(__file__).resolve().parent
OURS = HERE.parents[2]/"demo"/("biventricle_eikonal_rep" if RE_PLUS
                               else "biventricle_eikonal")
THEIRS = HERE/("results_rep" if RE_PLUS else "results")


def activation_from_voltage(vm, snapshot_interval, threshold=-30.0):
    """First time each node's voltage crosses ``threshold`` upwards.

    This is a measured activation time, which is not the same quantity as the
    eikonal model's prescribed arrival field: it lags it by the time the foot
    current takes to drive the cell to ``threshold``.  Nodes that never cross
    are ``nan``.
    """
    crossed = vm >= threshold
    ever = crossed.any(axis=0)
    first = crossed.argmax(axis=0).astype(float) * snapshot_interval
    return np.where(ever, first, np.nan)


def stats(name, a, b, unit):
    """Difference summary over the nodes where both sides have a value."""
    both = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[both] - b[both])
    row = {"quantity": name, "unit": unit, "compared": int(both.sum()),
           "only_ours": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
           "only_theirs": int((~np.isfinite(a) & np.isfinite(b)).sum()),
           "max": float(d.max()), "mean": float(d.mean()),
           "p99": float(np.percentile(d, 99)),
           "ours_range": [float(np.nanmin(a)), float(np.nanmax(a))],
           "theirs_range": [float(np.nanmin(b)), float(np.nanmax(b))]}
    print(f"  {name:26s} max {row['max']:9.4f} {unit}   mean {row['mean']:8.4f}   "
          f"p99 {row['p99']:8.4f}   n {row['compared']}")
    return row


def main():
    snapshot_interval = 1.0
    theirs_at = np.load(THEIRS/"act.npy")[0]
    theirs_vm = np.load(THEIRS/"vm.npy")
    ours_at = torch.load(OURS/"act.pt").numpy()
    ours_vm = torch.load(OURS/"vm.pt").numpy()

    print(f"mode     : {'RE+ (dream.solve = 1)' if RE_PLUS else 'RE- (dream.solve = 2)'}")
    print(f"openCARP : AT {theirs_at.shape}  Vm {theirs_vm.shape}")
    print(f"TorchCor : AT {ours_at.shape}  Vm {ours_vm.shape}\n")

    n = min(ours_vm.shape[0], theirs_vm.shape[0])
    report = [
        stats("prescribed arrival", ours_at, theirs_at, "ms"),
        stats("measured LAT (-30 mV)",
              activation_from_voltage(ours_vm[:n], snapshot_interval),
              activation_from_voltage(theirs_vm[:n], snapshot_interval), "ms"),
        stats("Vm, all frames", ours_vm[:n].ravel(), theirs_vm[:n].ravel(), "mV"),
        stats("Vm, final frame", ours_vm[n - 1], theirs_vm[n - 1], "mV"),
    ]
    (THEIRS/"comparison.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {THEIRS.name}/comparison.json")


if __name__ == "__main__":
    main()
