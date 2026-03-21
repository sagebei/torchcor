"""
NIfTI Segmentation Viewer
==========================
Shows axial / coronal / sagittal slices of a TCIA label map with each
tissue coloured anatomically.  A slice slider lets you scroll through
the volume interactively.

Usage
-----
    python torchcor/ecg/nifti_viewer.py [path_to_nii_gz]

    If no path is given the script auto-finds the first NIfTI in the
    TCIA zip under Downloads.

Requires: nibabel, matplotlib  (both already installed)
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, RadioButtons

# ── Colour map: TCIA label → RGB (same palette as viewer_3d.py) ──────────────
# Index 0 = background (transparent / black)
LABEL_COLORS: dict[int, tuple] = {
    0:  (0.05, 0.05, 0.05),   # background — near black
    1:  (0.88, 0.72, 0.65),   # adrenal glands → inner body flesh
    2:  (0.58, 0.06, 0.06),   # aorta — dark arterial red
    3:  (0.72, 0.88, 0.96),   # bladder — ice blue
    4:  (0.80, 0.78, 0.90),   # brain — pale lavender
    5:  (0.82, 0.06, 0.06),   # heart — vivid crimson
    6:  (0.68, 0.28, 0.14),   # kidneys — reddish brown
    7:  (0.50, 0.16, 0.08),   # liver — mahogany
    8:  (0.85, 0.70, 0.40),   # pancreas — golden tan
    9:  (0.42, 0.12, 0.22),   # spleen — deep plum
    10: (0.75, 0.85, 0.80),   # thyroid — pale teal
    11: (0.45, 0.08, 0.28),   # IVC — dark purple-red
    12: (0.93, 0.68, 0.62),   # lung — pale salmon pink
    13: (0.96, 0.94, 0.84),   # carpal — ivory
    14: (0.96, 0.94, 0.84),   # clavicle — ivory
    15: (0.96, 0.94, 0.84),   # femur — ivory
    16: (0.96, 0.94, 0.84),   # fibula — ivory
    17: (0.96, 0.94, 0.84),   # humerus — ivory
    18: (0.96, 0.94, 0.84),   # metacarpal — ivory
    19: (0.96, 0.94, 0.84),   # metatarsal — ivory
    20: (0.96, 0.94, 0.84),   # patella — ivory
    21: (0.94, 0.92, 0.82),   # pelvis — slightly warmer ivory
    22: (0.96, 0.94, 0.84),   # fingers — ivory
    23: (0.96, 0.94, 0.84),   # radius — ivory
    24: (0.94, 0.92, 0.80),   # ribcage — ivory (slightly distinct)
    25: (0.96, 0.94, 0.84),   # scapula — ivory
    26: (0.96, 0.94, 0.84),   # skull — ivory
    27: (0.92, 0.90, 0.78),   # spine — warm ivory
    28: (0.96, 0.94, 0.84),   # sternum — ivory
    29: (0.96, 0.94, 0.84),   # tarsal — ivory
    30: (0.96, 0.94, 0.84),   # tibia — ivory
    31: (0.96, 0.94, 0.84),   # toes — ivory
    32: (0.96, 0.94, 0.84),   # ulna — ivory
    33: (0.72, 0.18, 0.16),   # skeletal muscle — raw red
    34: (0.98, 0.93, 0.60),   # subcutaneous fat — pale yellow
    35: (0.95, 0.86, 0.32),   # visceral fat — deeper yellow
    36: (0.68, 0.16, 0.14),   # psoas — slightly darker red
}

LABEL_NAMES: dict[int, str] = {
    0: "Background", 1: "Adrenal glands", 2: "Aorta", 3: "Bladder",
    4: "Brain", 5: "Heart", 6: "Kidneys", 7: "Liver", 8: "Pancreas",
    9: "Spleen", 10: "Thyroid", 11: "IVC", 12: "Lung",
    13: "Carpal", 14: "Clavicle", 15: "Femur", 16: "Fibula",
    17: "Humerus", 18: "Metacarpal", 19: "Metatarsal", 20: "Patella",
    21: "Pelvis", 22: "Fingers", 23: "Radius", 24: "Ribcage",
    25: "Scapula", 26: "Skull", 27: "Spine", 28: "Sternum",
    29: "Tarsal", 30: "Tibia", 31: "Toes", 32: "Ulna",
    33: "Skeletal muscle", 34: "Subcutaneous fat", 35: "Visceral fat",
    36: "Psoas",
}


def _build_colormap(max_label: int = 36):
    """Build a ListedColormap from LABEL_COLORS (index = label value)."""
    colors = []
    for i in range(max_label + 1):
        colors.append(LABEL_COLORS.get(i, (0.5, 0.5, 0.5)))
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, max_label + 1.5), max_label + 1)
    return cmap, norm


def _find_nifti() -> Path | None:
    downloads = Path("C:/Users/bulli/Downloads")
    # Already extracted?
    extracted = list(downloads.rglob("Healthy-Total-Body-CTs-???.nii.gz"))
    if extracted:
        return extracted[0]
    # Extract from zip
    zips = sorted(downloads.glob("*Healthy*Body*.zip"))
    if not zips:
        return None
    extract_dir = downloads / "_tcia_nifti"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zips[0]) as z:
        niftis = [f for f in z.namelist()
                  if f.endswith(".nii.gz") and "__MACOSX" not in f]
        if not niftis:
            return None
        z.extract(niftis[0], extract_dir)
        return extract_dir / niftis[0]


def launch(nifti_path: Path) -> None:
    import nibabel as nib

    print(f"Loading: {nifti_path.name} …")
    img  = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj).astype(np.int16)
    vox  = img.header.get_zooms()

    # Ensure RAS orientation for intuitive slice views
    data = nib.as_closest_canonical(img).get_fdata().astype(np.int16)

    print(f"  Shape : {data.shape}  voxel {vox[0]:.2f}×{vox[1]:.2f}×{vox[2]:.2f} mm")

    nx, ny, nz = data.shape
    cmap, norm = _build_colormap()

    # ── Initial slice positions ───────────────────────────────────────────────
    iz0 = nz // 2
    iy0 = ny // 2
    ix0 = nx // 2

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor="#1a1a2e")
    fig.suptitle(
        f"TCIA Label Map — {nifti_path.stem}",
        color="white", fontsize=13, fontweight="bold",
    )

    # Three slice panels + legend column
    ax_ax  = fig.add_axes([0.02, 0.25, 0.28, 0.65])   # axial
    ax_cor = fig.add_axes([0.35, 0.25, 0.28, 0.65])   # coronal
    ax_sag = fig.add_axes([0.68, 0.25, 0.28, 0.65])   # sagittal

    # Sliders
    ax_sl_ax  = fig.add_axes([0.02, 0.15, 0.28, 0.04], facecolor="#2a2a4a")
    ax_sl_cor = fig.add_axes([0.35, 0.15, 0.28, 0.04], facecolor="#2a2a4a")
    ax_sl_sag = fig.add_axes([0.68, 0.15, 0.28, 0.04], facecolor="#2a2a4a")

    sl_ax  = Slider(ax_sl_ax,  "Axial z",   0, nz - 1, valinit=iz0, valstep=1,
                    color="#4a4aaa")
    sl_cor = Slider(ax_sl_cor, "Coronal y", 0, ny - 1, valinit=iy0, valstep=1,
                    color="#4a4aaa")
    sl_sag = Slider(ax_sl_sag, "Sagittal x",0, nx - 1, valinit=ix0, valstep=1,
                    color="#4a4aaa")
    for sl in (sl_ax, sl_cor, sl_sag):
        sl.label.set_color("white")
        sl.valtext.set_color("white")

    def _slice(ax, image_slice, title):
        ax.clear()
        ax.imshow(image_slice.T, origin="lower", cmap=cmap, norm=norm,
                  interpolation="nearest", aspect="equal")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_facecolor("black")
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    def update(_):
        iz = int(sl_ax.val)
        iy = int(sl_cor.val)
        ix = int(sl_sag.val)
        _slice(ax_ax,  data[:, :, iz],  f"Axial  z={iz}")
        _slice(ax_cor, data[:, iy, :],  f"Coronal  y={iy}")
        _slice(ax_sag, data[ix, :, :],  f"Sagittal  x={ix}")

        # Draw crosshairs
        for ax, hv, pos in [
            (ax_ax,  (ix, iy),  "cross"),
            (ax_cor, (ix, iz),  "cross"),
            (ax_sag, (iy, iz),  "cross"),
        ]:
            ax.axvline(hv[0], color="#00ffcc", lw=0.5, alpha=0.6)
            ax.axhline(hv[1], color="#00ffcc", lw=0.5, alpha=0.6)

        fig.canvas.draw_idle()

    sl_ax.on_changed(update)
    sl_cor.on_changed(update)
    sl_sag.on_changed(update)

    # ── Tissue legend (bottom strip) ─────────────────────────────────────────
    present_labels = sorted(int(v) for v in np.unique(data) if v != 0)
    n_leg = len(present_labels)
    legend_ax = fig.add_axes([0.02, 0.02, 0.96, 0.10], facecolor="#1a1a2e")
    legend_ax.set_xlim(0, n_leg)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    for j, lbl in enumerate(present_labels):
        col = LABEL_COLORS.get(lbl, (0.5, 0.5, 0.5))
        name = LABEL_NAMES.get(lbl, f"Tag {lbl}")
        rect = plt.Rectangle((j, 0.4), 0.8, 0.5, color=col,
                              transform=legend_ax.transData)
        legend_ax.add_patch(rect)
        legend_ax.text(j + 0.4, 0.1, f"{lbl}\n{name}",
                       ha="center", va="bottom", fontsize=4.5,
                       color="white", transform=legend_ax.transData)

    # Initial draw
    update(None)

    print("\nControls:")
    print("  Drag sliders to scroll through axial / coronal / sagittal slices")
    print("  Cyan crosshairs show the current intersection point")
    print("  Colour legend at bottom shows all present tissue labels")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        nifti_path = Path(sys.argv[1])
    else:
        nifti_path = _find_nifti()
        if nifti_path is None:
            print("No NIfTI found. Pass path explicitly:")
            print("  python nifti_viewer.py path/to/seg.nii.gz")
            sys.exit(1)

    launch(nifti_path)
