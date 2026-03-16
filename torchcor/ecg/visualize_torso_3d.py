"""
3-D rotating torso visualisation
=================================
Shows the synthetic torso ellipsoid (semi-transparent), the heart ellipsoid
(red), and the 10 standard 12-lead electrode positions with IEC 60601-2-25
colour coding.

Run standalone:
    python visualize_torso_3d.py

Produces:
    torso_3d_rotating.gif   — animated 360° rotation (requires Pillow)
    torso_3d.html           — interactive Plotly figure  (requires plotly)

If neither extra package is available the script falls back to a static
matplotlib 3-D figure saved as  torso_3d_static.png.
"""

from __future__ import annotations

import importlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (side-effect import)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Geometry constants (same as synthetic_torso.py) ──────────────────────────
_TA, _TB, _TC = 22.0, 12.0, 30.0          # torso semi-axes: lateral, AP, SI (cm)
_HA, _HB, _HC = 5.0,  4.0,  7.0           # heart semi-axes (cm)
_HCX, _HCY, _HCZ = -2.0, -2.0, 3.0       # heart centre offset from torso centre

# Electrode 3-D positions (x=left/right, y=anterior/posterior, z=superior/inferior)
ELECTRODES: dict[str, tuple[float, float, float]] = {
    "V1": ( 2.5,  11.5,  9.0),
    "V2": (-2.5,  11.5,  9.0),
    "V3": (-6.5,  11.0,  6.0),
    "V4": (-11.0, 10.0,  3.0),
    "V5": (-15.0,  8.5,  3.0),
    "V6": (-18.5,  6.0,  3.0),
    "RA": ( 19.0,  5.0, 22.0),
    "LA": (-19.0,  5.0, 22.0),
    "RL": ( 11.0,  2.0,-26.0),
    "LL": (-11.0,  2.0,-26.0),
}

COLORS: dict[str, str] = {
    "RA": "#FF0000", "LA": "#FFFF00",
    "LL": "#111111", "RL": "#00AA00",
    "V1": "#7B0000", "V2": "#FF4500",
    "V3": "#FF8C00", "V4": "#DAA520",
    "V5": "#228B22", "V6": "#1E5FA8",
}


def _ellipsoid_mesh(cx, cy, cz, ax, ay, az, n=40):
    """Return (X, Y, Z) surface mesh for an ellipsoid."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    X = cx + ax * np.outer(np.cos(u), np.sin(v))
    Y = cy + ay * np.outer(np.sin(u), np.sin(v))
    Z = cz + az * np.outer(np.ones_like(u), np.cos(v))
    return X, Y, Z


# ─────────────────────────────────────────────────────────────────────────────
# Option A: interactive HTML via Plotly
# ─────────────────────────────────────────────────────────────────────────────

def _plotly_figure(out_html="torso_3d.html"):
    import plotly.graph_objects as go

    traces = []

    # ── Torso surface ─────────────────────────────────────────────────────────
    X, Y, Z = _ellipsoid_mesh(0, 0, 0, _TA, _TB, _TC, n=60)
    traces.append(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, "rgba(210,180,140,0.18)"], [1, "rgba(210,180,140,0.18)"]],
        showscale=False,
        name="Torso",
        opacity=0.18,
        hoverinfo="skip",
        lighting=dict(ambient=0.7),
    ))

    # ── Heart surface ─────────────────────────────────────────────────────────
    Xh, Yh, Zh = _ellipsoid_mesh(_HCX, _HCY, _HCZ, _HA, _HB, _HC, n=40)
    traces.append(go.Surface(
        x=Xh, y=Yh, z=Zh,
        colorscale=[[0, "rgba(200,30,30,0.70)"], [1, "rgba(200,30,30,0.70)"]],
        showscale=False,
        name="Heart",
        opacity=0.70,
        hoverinfo="skip",
        lighting=dict(ambient=0.6),
    ))

    # ── Electrodes ────────────────────────────────────────────────────────────
    for name, (ex, ey, ez) in ELECTRODES.items():
        color = COLORS[name]
        traces.append(go.Scatter3d(
            x=[ex], y=[ey], z=[ez],
            mode="markers+text",
            marker=dict(size=10, color=color,
                        line=dict(color="black", width=1)),
            text=[name],
            textposition="top center",
            textfont=dict(size=11, color=color),
            name=name,
            hovertemplate=f"<b>{name}</b><br>x={ex:.1f} y={ey:.1f} z={ez:.1f}<extra></extra>",
        ))

    fig = go.Figure(data=traces)

    # Rotating camera animation frames
    n_frames = 72
    frames = []
    for i in range(n_frames):
        angle = i * 360 / n_frames
        eye_x = 80 * np.cos(np.radians(angle))
        eye_y = 80 * np.sin(np.radians(angle))
        frames.append(go.Frame(
            layout=dict(scene_camera=dict(
                eye=dict(x=eye_x / 80, y=eye_y / 80, z=0.6),
                up=dict(x=0, y=0, z=1),
            )),
            name=str(i),
        ))
    fig.frames = frames

    fig.update_layout(
        title=dict(
            text="3-D Synthetic Torso — 12-Lead ECG Electrodes",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="Lateral (cm) — right +",
            yaxis_title="Anterior–Posterior (cm) — anterior +",
            zaxis_title="Superior–Inferior (cm) — superior +",
            aspectmode="data",
            bgcolor="rgb(245,248,255)",
        ),
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.6),
            up=dict(x=0, y=0, z=1),
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.05, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Rotate",
                     method="animate",
                     args=[None, dict(
                         frame=dict(duration=50, redraw=True),
                         fromcurrent=True,
                         transition=dict(duration=0),
                         mode="immediate",
                     )]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode="immediate",
                         transition=dict(duration=0),
                     )]),
            ],
        )],
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(itemsizing="constant", font=dict(size=11)),
        height=700,
    )

    # Add anatomical orientation labels as annotations
    fig.add_annotation(
        x=0.01, y=0.5, xref="paper", yref="paper",
        text="<b>Patient's<br>Right →</b>",
        showarrow=False, font=dict(size=10, color="gray"),
    )

    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print(f"Saved interactive HTML: {out_html}")
    return out_html


# ─────────────────────────────────────────────────────────────────────────────
# Option B: animated GIF via matplotlib + Pillow
# ─────────────────────────────────────────────────────────────────────────────

def _matplotlib_gif(out_gif="torso_3d_rotating.gif", n_frames=72):
    from PIL import Image as PILImage
    import io

    frames_pil = []

    for i in range(n_frames):
        azim = i * 360 / n_frames

        fig = plt.figure(figsize=(7, 8), facecolor="#f5f8ff")
        ax = fig.add_subplot(111, projection="3d", facecolor="#f5f8ff")

        # ── Torso ─────────────────────────────────────────────────────────────
        X, Y, Z = _ellipsoid_mesh(0, 0, 0, _TA, _TB, _TC, n=30)
        ax.plot_surface(X, Y, Z, color="#d2b48c", alpha=0.15, linewidth=0,
                        antialiased=True, shade=True)
        # Wireframe outline
        ax.plot_wireframe(X, Y, Z, color="#a0856a", linewidth=0.3,
                          alpha=0.25, rstride=3, cstride=3)

        # ── Heart ─────────────────────────────────────────────────────────────
        Xh, Yh, Zh = _ellipsoid_mesh(_HCX, _HCY, _HCZ, _HA, _HB, _HC, n=25)
        ax.plot_surface(Xh, Yh, Zh, color="#c81e1e", alpha=0.75,
                        linewidth=0, antialiased=True, shade=True)

        # ── Electrodes ────────────────────────────────────────────────────────
        for name, (ex, ey, ez) in ELECTRODES.items():
            c = COLORS[name]
            ec = "gray" if c == "#111111" else "black"
            ax.scatter(ex, ey, ez, s=120, c=c, edgecolors=ec,
                       linewidths=1.2, zorder=5, depthshade=False)
            ax.text(ex, ey, ez + 2.2, name, fontsize=7, ha="center",
                    color=c, fontweight="bold")

        # ── Labels / decoration ───────────────────────────────────────────────
        ax.set_xlabel("Lateral (cm)", fontsize=8, labelpad=4)
        ax.set_ylabel("AP (cm)", fontsize=8, labelpad=4)
        ax.set_zlabel("SI (cm)", fontsize=8, labelpad=4)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-32, 32)
        ax.view_init(elev=15, azim=azim)
        ax.set_title(
            "3-D Synthetic Torso\n12-Lead ECG Electrode Positions",
            fontsize=10, fontweight="bold", pad=8,
        )
        ax.tick_params(labelsize=6)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        frames_pil.append(PILImage.open(buf).copy())

    frames_pil[0].save(
        out_gif,
        save_all=True,
        append_images=frames_pil[1:],
        loop=0,
        duration=55,   # ms per frame  ≈ 18 fps
        optimize=False,
    )
    print(f"Saved rotating GIF ({n_frames} frames): {out_gif}")
    return out_gif


# ─────────────────────────────────────────────────────────────────────────────
# Option C: static fallback
# ─────────────────────────────────────────────────────────────────────────────

def _matplotlib_static(out_png="torso_3d_static.png"):
    fig = plt.figure(figsize=(9, 10), facecolor="#f5f8ff")
    ax = fig.add_subplot(111, projection="3d", facecolor="#f5f8ff")

    X, Y, Z = _ellipsoid_mesh(0, 0, 0, _TA, _TB, _TC, n=30)
    ax.plot_surface(X, Y, Z, color="#d2b48c", alpha=0.15, linewidth=0,
                    antialiased=True, shade=True)
    ax.plot_wireframe(X, Y, Z, color="#a0856a", linewidth=0.3,
                      alpha=0.30, rstride=3, cstride=3)

    Xh, Yh, Zh = _ellipsoid_mesh(_HCX, _HCY, _HCZ, _HA, _HB, _HC, n=25)
    ax.plot_surface(Xh, Yh, Zh, color="#c81e1e", alpha=0.75,
                    linewidth=0, antialiased=True, shade=True)

    for name, (ex, ey, ez) in ELECTRODES.items():
        c = COLORS[name]
        ec = "gray" if c == "#111111" else "black"
        ax.scatter(ex, ey, ez, s=140, c=c, edgecolors=ec,
                   linewidths=1.2, zorder=5, depthshade=False)
        ax.text(ex, ey, ez + 2.5, name, fontsize=8, ha="center",
                color=c, fontweight="bold")

    ax.set_xlabel("Lateral (cm)")
    ax.set_ylabel("AP (cm)")
    ax.set_zlabel("SI (cm)")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-32, 32)
    ax.view_init(elev=20, azim=-60)
    ax.set_title(
        "3-D Synthetic Torso — 12-Lead ECG Electrode Positions\n"
        "(anterior-oblique view)",
        fontsize=11, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved static 3-D figure: {out_png}")
    return out_png


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plotly_ok  = importlib.util.find_spec("plotly")  is not None
    pillow_ok  = importlib.util.find_spec("PIL")     is not None

    print("Package availability:")
    print(f"  plotly : {'yes' if plotly_ok  else 'no'}")
    print(f"  Pillow : {'yes' if pillow_ok  else 'no'}")

    if plotly_ok:
        print("\n[1/1] Generating interactive HTML (Plotly) …")
        _plotly_figure("torso_3d.html")
    elif pillow_ok:
        print("\n[1/1] Generating rotating GIF (matplotlib + Pillow) …")
        _matplotlib_gif("torso_3d_rotating.gif", n_frames=72)
    else:
        print("\n[1/1] Generating static 3-D PNG (matplotlib) …")
        _matplotlib_static("torso_3d_static.png")

    print("Done.")
