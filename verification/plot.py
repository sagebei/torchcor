import matplotlib.pyplot as plt

# Raw data as lines
raw = """
dt=0.025 dx=0.025 0.0024159011949221377
dt=0.025 dx=0.0125 0.0007101343969379604
dt=0.025 dx=0.00625 0.0007902901130795642
dt=0.025 dx=0.003125 0.0008491522630426874
dt=0.025 dx=0.0015625 0.0008379051589441343
dt=0.0125 dx=0.025 0.002625144411288577
dt=0.0125 dx=0.0125 0.0005511118973465636
dt=0.0125 dx=0.00625 0.0003442834850681267
dt=0.0125 dx=0.003125 0.0003753878356383474
dt=0.0125 dx=0.0015625 0.0003420096488547506
dt=0.00625 dx=0.025 0.002756615456977155
dt=0.00625 dx=0.0125 0.000605941835497956
dt=0.00625 dx=0.00625 0.00016001012116617988
dt=0.00625 dx=0.003125 0.0001399712215495667
dt=0.00625 dx=0.0015625 9.706209493672261e-05
dt=0.003125 dx=0.025 0.0028270593653493777
dt=0.003125 dx=0.0125 0.0006609518896433334
dt=0.003125 dx=0.00625 0.00014176779354547843
dt=0.003125 dx=0.003125 3.741514890424498e-05
dt=0.003125 dx=0.0015625 0.00013663463076446297
dt=0.0015625 dx=0.025 0.0028628791309734435
dt=0.0015625 dx=0.0125 0.0006942681178818433
dt=0.0015625 dx=0.00625 0.00017011064757265664
dt=0.0015625 dx=0.003125 0.00010638235982503358
dt=0.0015625 dx=0.0015625 0.0002560240983572698
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

label = [r"$2.5 \times 10^{-2}$", r"$1.25 \times 10^{-2}$",  r"$6.25 \times 10^{-3}$", r"$3.125 \times 10^{-3}$", r"$1.5625 \times 10^{-3}$"]
colors = ["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f", "#9467bd"]
markers = ["o", "s", "^", "D", "v"] 
d = [10, 20, 40, 80, 160]
for i, (dx, pairs) in enumerate(list(data.items())):
    dt_vals = [p[0] for p in pairs if p[1] != 0]
    y_vals = [p[1] for p in pairs if p[1] != 0]
    print(dx, dt_vals, y_vals)
    ax.plot(dt_vals, y_vals, marker=markers[i], markersize=8, linewidth=2, label=f"$\\Delta x$: {dx}", color=colors[i])

# r = 0.0034286942375709585
# references = [(1/10, r), (1/20, r/2), (1/40, r/4), (1/80, r/8), (1/160, r/16)]
# dx_vals = [p[0] for p in references]
# y_vals = [p[1] for p in references]
# ax.plot(dx_vals, y_vals, linestyle=':', linewidth=2, color="#7f7f7f")

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
# ax.set_yticks([1e-4, 1e-3])
# xmin, xmax = ax.get_xlim()
# print(xmin, xmax)
# ax.set_xticks([3e-3, 5e-2])

# Optional: format in scientific notation
# ax.set_yticklabels([r"$10^{-4}$", r"$10^{-3}$"])
# ax.set_xticklabels([r"$3 \times 10^{-3}$", r"$5 \times 10^{-2}$"])


plt.show()

# Subtle grid (optional)
ax.grid(True, which="major", linestyle="--", alpha=0.3)

# Layout + save
plt.tight_layout()
plt.savefig("convergence.pdf", bbox_inches="tight")
plt.show()