import matplotlib.pyplot as plt

# Raw data as lines
raw = """
dt=0.05 dx=0.0125 0.001519944770561091
dt=0.05 dx=0.00625 0.0017595372395498268
dt=0.05 dx=0.003125 0.0018415684175346727
dt=0.025 dx=0.0125 0.0007338005163641288
dt=0.025 dx=0.00625 0.0008220034548861548
dt=0.025 dx=0.003125 0.0008955753342198574
dt=0.0125 dx=0.0125 0.0005624632393706483
dt=0.0125 dx=0.00625 0.0003743833794815817
dt=0.0125 dx=0.003125 0.00043235328004849775
dt=0.00625 dx=0.0125 0.0006036108706814442
dt=0.00625 dx=0.00625 0.00018248354101135618
dt=0.00625 dx=0.003125 0.0002036571536216315
dt=0.003125 dx=0.0125 0.0006540670684079802
dt=0.003125 dx=0.00625 0.00013991726508409374
dt=0.003125 dx=0.003125 9.278453081289178e-05 
"""

# Parse data
data = {}
for line in raw.strip().split("\n"):
    parts = line.split()
    dt = float(parts[0].split("=")[1])
    dx = float(parts[1].split("=")[1])
    val = float(parts[2])
    data.setdefault(dx, []).append((dt, val))

# Plot
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,   # Editable text in Illustrator
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

label = [r"$1.25 \times 10^{-2}$",  r"$6.25 \times 10^{-3}$", r"$3.125 \times 10^{-3}$", r"$1.5625 \times 10^{-3}$"]
colors = ["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f", "#9467bd"]
markers = ["o", "s", "^", "D", "v"] 
d = [10, 20, 40, 80, 160]
for i, (dx, pairs) in enumerate(list(data.items())):
    dt_vals = [p[0] for p in pairs if p[1] != 0]
    y_vals = [p[1] for p in pairs if p[1] != 0]
    print(dx, dt_vals, y_vals)
    ax.plot(dt_vals, y_vals, marker=markers[i], markersize=8, linewidth=2, label=f"$\\Delta x$: {label[i]}", color=colors[i])

r = 0.0017595372395498268
references = [(1/20, r), (1/40, r/2), (1/80, r/4), (1/160, r/8), (1/320, r/16)]
dx_vals = [p[0] for p in references]
y_vals = [p[1] for p in references]
ax.plot(dx_vals, y_vals, linestyle=':', linewidth=2, color="#7f7f7f", label=f"reference")

# Log axes
ax.set_xscale("log")
ax.set_yscale("log")

# Labels
ax.set_xlabel("Time step size $\\Delta t$")
ax.set_ylabel("Relative $L^2$ error")
# ax.set_title("Convergence behaviour for (dt, dx)")

# Legend
ax.legend(frameon=False, fontsize=15)

# Spines: remove top/right, thicken bottom/left
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# Ticks: outward direction, minor ticks on
ax.tick_params(direction="out", length=6, width=1, which="major")
ax.tick_params(direction="out", length=3, width=1, which="minor")
ax.minorticks_on()

# ymin, ymax = ax.get_ylim()
ax.set_yticks([1e-4, 1e-3])
xmin, xmax = ax.get_xlim()
print(xmin, xmax)
ax.set_xticks([3e-3, 5e-2])

ax.set_yticklabels([r"$10^{-4}$", r"$10^{-3}$"])
ax.set_xticklabels([r"$3 \times 10^{-3}$", r"$5 \times 10^{-2}$"])


plt.show()

# Subtle grid (optional)
ax.grid(True, which="major", linestyle="--", alpha=0.3)

# Layout + save
plt.tight_layout()
plt.savefig("convergence.pdf", bbox_inches="tight")
plt.show()