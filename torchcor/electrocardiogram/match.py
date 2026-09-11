"""
Overlay torchcor's 12-lead ECG on the openCARP ground truth, in the openCARP
HC2 plot style (3x4 layout, [mV] vs [ms]).

Inputs (derived from base_dir):
  - torchcor : <base>/torchcor_results/torchcor_ecg.npz   (written by run_ecg.py)
  - openCARP : <base>/opencarp_results/SIM_EP/ECG_lead_field.dat
Output:
  - match.png   (+ per-lead correlation / RMSE printed to stdout)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Base data directory -- all other paths are derived from it.
base_dir     = Path("/data/Bei/Torso/HC2")
torchcor_ecg = base_dir / "torchcor_results" / "torchcor_ecg.npz"
opencarp_ecg = base_dir / "opencarp_results" / "SIM_EP" / "ECG_lead_field.dat"
out_png      = Path(__file__).resolve().parent / "match.png"

# Lead column order in ECG_lead_field.dat (after the time column).
order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# Figure layout matching the openCARP HC2 plot: (display label, data key).
grid = [("LI", "I"),    ("aVR", "aVR"), ("V1", "V1"), ("V4", "V4"),
        ("LII", "II"),  ("aVL", "aVL"), ("V2", "V2"), ("V5", "V5"),
        ("LIII", "III"),("aVF", "aVF"), ("V3", "V3"), ("V6", "V6")]

# ---- load openCARP ground truth ----
oc = np.loadtxt(opencarp_ecg)
t_oc = oc[:, 0]
oc_lead = {n: oc[:, 1 + i] for i, n in enumerate(order)}

# ---- load torchcor result ----
if not torchcor_ecg.exists():
    raise FileNotFoundError(
        f"{torchcor_ecg} not found -- run `python run_ecg.py` first to generate it."
    )
data = np.load(torchcor_ecg)
dt = float(data["dt_ms"]) if "dt_ms" in data.files else 1.0
tc_lead = {n: data[n] for n in order}
t_tc = np.arange(len(tc_lead["I"])) * dt

# ---- overlay plot (openCARP vs torchcor) ----
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
fig.suptitle("HC2", fontsize=16)
for ax, (disp, key) in zip(axes.flat, grid):
    ax.plot(t_oc, oc_lead[key], color="crimson", lw=1.6, label="openCARP")
    ax.plot(t_tc, tc_lead[key], color="navy", lw=1.2, ls="--", label="torchcor")
    ax.set_title(disp, fontsize=13)
    ax.set_xlim(0, 500); ax.set_ylim(-5, 4.2)
    ax.set_xticks([0, 150, 300, 450]); ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("[ms]"); ax.set_ylabel("[mV]")
axes.flat[0].legend(fontsize=9, loc="upper right")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(out_png, dpi=140)
print(f"saved {out_png}")

# ---- per-lead metrics ----
T = min(len(t_oc), len(t_tc))
print("\n lead    corr     rmse(mV)")
corrs = []
for disp, key in grid:
    a = tc_lead[key][:T]; b = oc_lead[key][:T]
    c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    corrs.append(c)
    print(f" {disp:>4s}   {c:+.4f}   {rmse:.4f}")
print(f"\n mean correlation = {np.nanmean(corrs):+.4f}")
