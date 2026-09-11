"""
Render a publication-quality 12-lead ECG (clinical graph-paper style) from the
`torchcor_ecg.npz` written by run_ecg.py.  No comparison -- just torchcor's ECG.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

base_dir = Path("/data/Bei/Torso/HC2")
ecg_file = base_dir / "torchcor_results" / "torchcor_ecg.npz"
out_png  = Path("/home/bzhou6/torchcor/docs/ecg_leadfield.png")

data = np.load(ecg_file)
dt = float(data["dt_ms"]) if "dt_ms" in data.files else 1.0
order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
lead = {k: data[k] for k in order}
n = len(lead["I"])
t = np.arange(n) * dt / 1000.0            # seconds

layout = [["I", "aVR", "V1", "V4"],
          ["II", "aVL", "V2", "V5"],
          ["III", "aVF", "V3", "V6"]]

# ---- ECG graph-paper palette ----
PAPER = "#fff5f5"
MINOR = "#f7cccc"
MAJOR = "#e89096"
TRACE = "#111418"
TEXT  = "#1d1d1f"

plt.rcParams.update({"font.family": "DejaVu Sans"})

YMIN, YMAX = -5.3, 3.3

fig = plt.figure(figsize=(16, 8), facecolor="white")
gs = fig.add_gridspec(3, 4, hspace=0.16, wspace=0.06,
                      left=0.025, right=0.99, top=0.975, bottom=0.05)


def paper(ax):
    ax.set_facecolor(PAPER)
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.xaxis.set_major_locator(MultipleLocator(0.20))
    ax.yaxis.set_minor_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_locator(MultipleLocator(0.50))
    ax.grid(which="minor", color=MINOR, lw=0.5, zorder=0)
    ax.grid(which="major", color=MAJOR, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_color(MAJOR); s.set_linewidth(1.1)


for r in range(3):
    for c in range(4):
        ax = fig.add_subplot(gs[r, c])
        paper(ax)
        L = layout[r][c]
        ax.plot(t, lead[L], color=TRACE, lw=1.6, solid_capstyle="round",
                solid_joinstyle="round", zorder=3)
        ax.set_xlim(0, t[-1]); ax.set_ylim(YMIN, YMAX)
        ax.text(0.015, 0.94, L, transform=ax.transAxes, fontsize=15,
                fontweight="bold", color=TEXT, va="top", ha="left")

fig.savefig(out_png, dpi=200, facecolor="white")
print(f"saved {out_png}")
