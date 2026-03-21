"""
Interactive 3-D Tissue Viewer
==============================
Loads any VTU mesh with a RegionTag cell array and shows an interactive
3-D viewer with per-tissue opacity controls.

Each tissue has three opacity buttons in the render window:
  [■] Solid  (opacity 1.0)
  [◐] Ghost  (opacity 0.2)
  [□] Hide   (opacity 0.0)

Click a button to toggle that tissue to the corresponding opacity.

Usage
-----
    python torchcor/ecg/viewer_3d.py [path_to_vtu]

    If no path is given the script searches common output locations.

Requires: pyvista  (pip install pyvista)
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pyvista as pv

from torchcor.ecg.torso_tags import TAG_INFO, SKIN_TAG, BONE_TAG, LV_WALL

# ── Per-tag defaults: name + colour + starting opacity ───────────────────────
# Works for both KCL torso (tags 1–35) and TCIA torso (tags 1–12).
# Colour is RGB 0–1.

TAG_STYLE: dict[int, dict] = {
    # ── KCL whole-torso tags (official, from KCL_Info.pdf) ──────────────────
    #  Conductivities from the KCL dataset documentation (Qian et al. 2022).
    #  Colors approximate real in-vivo tissue appearance.
    #
    #  1  Inner body (bath)  0.247 S/m  — pale pink flesh / bulk soft tissue
    1:  dict(name="Inner body",              color=(0.88, 0.72, 0.65), opacity=0.0),
    #  2  Skin               0.117 S/m  — light warm tan
    2:  dict(name="Skin",                    color=(0.91, 0.76, 0.60), opacity=0.2),
    #  3  Bones              0.050 S/m  — ivory / cream white
    3:  dict(name="Bones",                   color=(0.96, 0.94, 0.84), opacity=1.0),
    #  4  Kidneys            0.167 S/m  — reddish-brown bean
    4:  dict(name="Kidneys",                 color=(0.68, 0.28, 0.14), opacity=1.0),
    #  5  Liver              0.167 S/m  — dark mahogany brown
    5:  dict(name="Liver",                   color=(0.50, 0.16, 0.08), opacity=1.0),
    #  6  Stomach            0.100 S/m  — muted yellow-pink
    6:  dict(name="Stomach",                 color=(0.84, 0.76, 0.60), opacity=1.0),
    #  7  Spleen             0.100 S/m  — deep plum / purple-red
    7:  dict(name="Spleen",                  color=(0.42, 0.12, 0.22), opacity=1.0),
    #  8  Lungs              0.071 S/m  — pale salmon pink (air-filled)
    8:  dict(name="Lungs",                   color=(0.93, 0.68, 0.62), opacity=0.5),
    #  9  SVC blood pool     0.667 S/m  — dark venous blood red
    9:  dict(name="SVC blood pool",          color=(0.52, 0.04, 0.04), opacity=0.8),
    # 10  Superior vena cava 0.667 S/m  — venous red vessel wall
    10: dict(name="Superior vena cava",      color=(0.58, 0.06, 0.18), opacity=0.8),
    # 11  Pulm. artery pool   0.667 S/m  — bright arterial red
    11: dict(name="Pulm. artery pool",       color=(0.72, 0.08, 0.08), opacity=0.8),
    # 12  Pulmonary artery   0.667 S/m  — arterial red vessel
    12: dict(name="Pulmonary artery",        color=(0.76, 0.12, 0.12), opacity=0.8),
    # 13  Aorta              0.667 S/m  — bright arterial red wall
    13: dict(name="Aorta",                   color=(0.75, 0.18, 0.12), opacity=0.8),
    # 14  Aorta blood pool   0.667 S/m  — dark arterial blood red
    14: dict(name="Aorta blood pool",        color=(0.58, 0.06, 0.06), opacity=0.8),
    # 15  SVC plane — merged into tag 9 in anatomy model; kept for ICD model
    15: dict(name="SVC plane",               color=(0.48, 0.16, 0.42), opacity=0.3),
    # 16  Mitral valve       0.667 S/m  — fibrous off-white ivory
    16: dict(name="Mitral valve",            color=(0.92, 0.88, 0.76), opacity=1.0),
    # 17  Pulmonary valve    0.667 S/m  — fibrous off-white
    17: dict(name="Pulmonary valve",         color=(0.90, 0.86, 0.74), opacity=1.0),
    # 18  Aortic valve       0.667 S/m  — fibrous off-white
    18: dict(name="Aortic valve",            color=(0.94, 0.90, 0.78), opacity=1.0),
    # 19  Tricuspid valve    0.667 S/m  — fibrous off-white
    19: dict(name="Tricuspid valve",         color=(0.91, 0.87, 0.75), opacity=1.0),
    # 20  L. atrial wall     0.250 S/m  — medium myocardial red
    20: dict(name="L. atrial wall",          color=(0.76, 0.14, 0.14), opacity=1.0),
    # 21  L. atrial blood    0.667 S/m  — dark oxygenated blood
    21: dict(name="L. atrial blood pool",    color=(0.54, 0.06, 0.06), opacity=0.5),
    # 22  R. atrial blood    0.667 S/m  — dark deoxygenated blood
    22: dict(name="R. atrial blood pool",    color=(0.50, 0.04, 0.04), opacity=0.5),
    # 23  R. atrial wall     0.250 S/m  — myocardial red
    23: dict(name="R. atrial wall",          color=(0.72, 0.14, 0.14), opacity=1.0),
    # 24  RV blood pool      0.667 S/m  — dark deoxygenated blood
    24: dict(name="RV blood pool",           color=(0.48, 0.04, 0.04), opacity=0.5),
    # 25  LV blood pool      0.667 S/m  — slightly brighter oxygenated
    25: dict(name="LV blood pool",           color=(0.56, 0.05, 0.05), opacity=0.5),
    # 26  RV wall  bidomain  0.136/0.018 S/m — bright myocardial red
    # (KCL anatomy VTU: renumbered from CARP tag 34)
    26: dict(name="RV wall",                 color=(0.88, 0.10, 0.10), opacity=1.0),
    # 27  LV wall  bidomain  0.136/0.018 S/m — vivid crimson (thicker)
    # (KCL anatomy VTU: renumbered from CARP tag 35)
    27: dict(name="LV wall",                 color=(0.82, 0.06, 0.06), opacity=1.0),

    # ── ICD hardware tags (present in KCL_torso3_ICD.vtu only) ─────────────
    # In the ICD VTU these use original CARP tag numbers
    28: dict(name="Coronary sinus coil",     color=(0.72, 0.74, 0.80), opacity=0.6),
    29: dict(name="SVC coil",                color=(0.68, 0.70, 0.78), opacity=0.6),
    30: dict(name="RV apical coil",          color=(0.64, 0.66, 0.74), opacity=0.6),
    31: dict(name="RV septal coil",          color=(0.60, 0.62, 0.70), opacity=0.6),
    32: dict(name="Subcutaneous ICD lead",   color=(0.56, 0.58, 0.66), opacity=0.6),
    33: dict(name="Left can",                color=(0.80, 0.82, 0.86), opacity=0.8),
    34: dict(name="Right can",               color=(0.78, 0.80, 0.84), opacity=0.8),
    35: dict(name="Subcutaneous ICD can",    color=(0.76, 0.78, 0.82), opacity=0.8),
}

OPACITY_CYCLE = [1.0, 0.2, 0.0]   # solid → ghost → hidden
OPACITY_LABEL = ["Solid", "Ghost", "Hide "]
OPACITY_COLOR = ["#00aa44", "#ffaa00", "#888888"]

# Tag style for CARP meshes (native CARP numbering: 34=RV wall, 35=LV wall,
# 26-33=ICD hardware).  Derived from torso_tags.TAG_INFO so it stays in sync.
CARP_TAG_STYLE: dict[int, dict] = {
    tag: dict(name=info.name, color=info.color, opacity=info.opacity)
    for tag, info in TAG_INFO.items()
}


def _place_ecg_electrodes(
    bone_surf: "pv.PolyData | None",
    skin_surf: "pv.PolyData | None",
    all_pts: np.ndarray,
    lv_surf:  "pv.PolyData | None" = None,
) -> "dict[str, tuple[np.ndarray, np.ndarray]] | None":
    """
    Compute anatomically correct 12-lead ECG electrode skin-contact positions.

    Algorithm
    ---------
    1. Histogram bone density near the sternum along Z to detect rib levels.
    2. Find the midpoint between the 4th/5th ribs  → 4th ICS Z-coordinate.
       Find the midpoint between the 5th/6th ribs  → 5th ICS Z-coordinate.
    3. Find the sternum's right/left border at the 4th ICS level.
    4. For each electrode ray-cast a probe along −Y (front→back) and take the
       most-anterior skin intersection.  The local skin normal at that point
       is retrieved from the pre-computed point-normal array.
    5. The disc is placed with its flat face flush on the skin: the disc axis
       is the outward skin normal, and the disc centre is advanced by half the
       disc depth so the front face is exactly at the skin surface.

    Standard 12-lead placement (IEC 60601-2-25 / AHA)
    --------------------------------------------------
    V1 : 4th ICS, right sternal border
    V2 : 4th ICS, left sternal border
    V3 : midway between V2 and V4
    V4 : 5th ICS, mid-clavicular line (patient left)
    V5 : 5th ICS, anterior axillary line
    V6 : 5th ICS, mid-axillary line
    RA : right upper torso / shoulder region
    LA : left  upper torso / shoulder region
    RL : right lower torso
    LL : left  lower torso

    Coordinate convention (after 180° Z-rotation + shifts)
    -------------------------------------------------------
    +X = patient left,  −X = patient right
     Y = 0 (posterior) → max (anterior)
     Z = 0 (inferior)  → max (superior)
    Units: µm (same as mesh)

    Returns
    -------
    dict { label : (pos_µm, unit_outward_normal) }  or  None
    """
    if bone_surf is None or skin_surf is None:
        return None

    bp = np.asarray(bone_surf.points, dtype=float)
    sp = np.asarray(skin_surf.points, dtype=float)

    x_lo = float(all_pts[:, 0].min())
    x_hi = float(all_pts[:, 0].max())   # +X = patient left
    z_lo = float(all_pts[:, 2].min())   # 0 after shift
    z_hi = float(all_pts[:, 2].max())
    y_hi = float(all_pts[:, 1].max())   # anterior face

    # ── Pre-compute outward skin normals ──────────────────────────────────
    try:
        skin_n = skin_surf.compute_normals(
            cell_normals=False, point_normals=True,
            consistent_normals=True, auto_orient_normals=True,
        )
        skin_normals = np.asarray(skin_n.point_data["Normals"], dtype=float)
    except Exception:
        skin_normals = None

    body_cx = float(all_pts[:, 0].mean())
    body_cy = float(all_pts[:, 1].mean())

    spacing = float((z_hi - z_lo) * 0.050)   # ~5 % body height ≈ 28 mm per ICS

    # ── Step 1: Z-levels for 4th and 5th ICS ─────────────────────────────
    # Primary anchor: LV apex lies at the 5th ICS mid-clavicular line
    # (AHA definition of V4 placement).  Much more reliable than rib counting.
    if lv_surf is not None:
        lv_pts = np.asarray(lv_surf.points, dtype=float)
        # Apex = most inferior portion of LV (low Z, slightly right in AP view)
        z_5ics = float(np.percentile(lv_pts[:, 2], 5)) + spacing * 0.3
        z_4ics = z_5ics + spacing
        print(f"  LV-apex anchor: 5th ICS Z ≈ {int(z_5ics/1000)} mm, "
              f"4th ICS Z ≈ {int(z_4ics/1000)} mm")
    else:
        # Fallback: rib histogram (less reliable but works without cardiac surfaces)
        tz_lo = z_lo + (z_hi - z_lo) * 0.28
        tz_hi = z_lo + (z_hi - z_lo) * 0.82
        ant_bone_thr = float(np.percentile(bp[:, 1], 55))
        ster_mask = (
            (np.abs(bp[:, 0]) < 60_000) &
            (bp[:, 1] > ant_bone_thr) &
            (bp[:, 2] > tz_lo) &
            (bp[:, 2] < tz_hi)
        )
        sbp = bp[ster_mask]
        rib_z: list[float] = []
        if len(sbp) >= 30:
            hist, edges = np.histogram(sbp[:, 2], bins=70, range=(tz_lo, tz_hi))
            ctrs = (edges[:-1] + edges[1:]) / 2.0
            sm   = np.convolve(hist.astype(float), np.ones(5) / 5, mode="same")
            thr_pk = sm.max() * 0.18
            i = 1
            while i < len(sm) - 1:
                if sm[i] > thr_pk and sm[i] >= sm[i-1] and sm[i] >= sm[i+1]:
                    lo, hi = i, i
                    while lo > 0 and sm[lo-1] > thr_pk * 0.4:
                        lo -= 1
                    while hi < len(sm)-1 and sm[hi+1] > thr_pk * 0.4:
                        hi += 1
                    rib_z.append(float(ctrs[(lo+hi)//2]))
                    i = hi + 2
                else:
                    i += 1
        rib_z.sort(reverse=True)
        print(f"  Rib Z-levels (mm): {[int(r/1000) for r in rib_z]}")
        # Use ribs[4]/[5] (5th/6th from top) – ribs[0] is often the clavicle
        if len(rib_z) >= 6:
            z_5ics = (rib_z[4] + rib_z[5]) / 2.0
            z_4ics = (rib_z[3] + rib_z[4]) / 2.0
        elif len(rib_z) >= 5:
            z_5ics = rib_z[4] - spacing / 2.0
            z_4ics = (rib_z[3] + rib_z[4]) / 2.0
        else:
            z_5ics = z_lo + (z_hi - z_lo) * 0.555
            z_4ics = z_5ics + spacing
        print(f"  Fallback: 4th ICS Z ≈ {int(z_4ics/1000)} mm, "
              f"5th ICS Z ≈ {int(z_5ics/1000)} mm")

    # ── Step 2: sternum right / left border X ────────────────────────────
    # Use a narrow ±35 mm window so we capture the sternal BODY only,
    # not the lateral costal-cartilage attachments.
    ster_body = bp[
        (np.abs(bp[:, 2] - z_4ics) < 30_000) &
        (np.abs(bp[:, 0]) < 35_000) &          # sternal body only (±35 mm)
        (bp[:, 1] > np.percentile(bp[:, 1], 60))
    ]
    if len(ster_body) >= 5:
        xs = np.sort(ster_body[:, 0])
        x_ster_R = float(xs[int(len(xs) * 0.05)])   # right sternal border (−X)
        x_ster_L = float(xs[int(len(xs) * 0.95)])   # left  sternal border (+X)
    else:
        x_ster_R = -25_000.0
        x_ster_L =  25_000.0

    print(f"  Sternal borders: R={int(x_ster_R/1000)} mm, "
          f"L={int(x_ster_L/1000)} mm")

    # ── Step 3: left-side anatomical lines (fraction of body half-width) ──
    x_mid_clav = x_hi * 0.48   # mid-clavicular  (~48 % of left half-width)
    x_ant_axil = x_hi * 0.68   # anterior axillary
    x_mid_axil = x_hi * 0.84   # mid-axillary

    # ── Step 4: ray-cast onto skin; return position + outward normal ──────
    def skin_contact(x_t: float, z_t: float):
        """Ray along −Y from outside → first anterior skin hit + local normal."""
        origin = [x_t, y_hi * 1.5, z_t]
        end    = [x_t, -y_hi * 0.5, z_t]

        pos: np.ndarray | None = None
        try:
            hits, _ = skin_surf.ray_trace(origin, end, first_point=False)
            if hits is not None and len(hits) > 0:
                hits = np.asarray(hits, dtype=float)
                pos = hits[np.argmax(hits[:, 1])]   # most anterior
        except Exception:
            pass

        if pos is None:                              # fallback: nearest anterior point
            d = np.hypot(sp[:, 0] - x_t, sp[:, 2] - z_t)
            ant = sp[:, 1] > np.percentile(sp[:, 1], 40)
            d = np.where(ant, d, np.inf)
            pos = sp[int(np.argmin(d))].copy()

        pos = np.asarray(pos, dtype=float)

        # Local outward skin normal
        if skin_normals is not None:
            d_sp  = np.linalg.norm(sp - pos, axis=1)
            n_raw = skin_normals[int(np.argmin(d_sp))].copy()
        else:
            n_raw = np.array([pos[0] - body_cx, pos[1] - body_cy, 0.0], dtype=float)

        mag = float(np.linalg.norm(n_raw))
        n   = n_raw / mag if mag > 1e-9 else np.array([0.0, 1.0, 0.0])

        # Ensure outward (dot with vector from body centre > 0)
        outward = np.array([pos[0] - body_cx, pos[1] - body_cy, 0.0])
        if np.dot(n, outward) < 0:
            n = -n

        return pos, n

    # ── Electrode positions ───────────────────────────────────────────────
    # Note: negate all X targets to follow standard AP radiographic convention
    # (patient RIGHT → viewer LEFT = screen left; patient LEFT → screen right).
    # In our coordinate system after 180° Z-rotation: +X = patient left,
    # screen-left = +X, screen-right = -X.  The negation below corrects for this
    # so that the electrode labels match the standard clinical AP chest view.
    z_v12 = z_4ics + spacing * 0.4   # V1/V2 slightly cranial of pure 4th ICS
    V1 = skin_contact(-x_ster_R,   z_v12)   # right sternal border → screen left
    V2 = skin_contact(-x_ster_L,   z_v12)   # left  sternal border → screen right
    V4 = skin_contact(-x_mid_clav, z_5ics)  # mid-clavicular left  → screen right
    V3 = skin_contact((V2[0][0] + V4[0][0]) / 2, (V2[0][2] + V4[0][2]) / 2)
    V5 = skin_contact(-x_ant_axil, float(V4[0][2]))
    V6 = skin_contact(-x_mid_axil, float(V4[0][2]))

    z_shoulder = z_lo + (z_hi - z_lo) * 0.84
    z_lower    = z_lo + (z_hi - z_lo) * 0.05
    RA = skin_contact( x_hi * 0.78,  z_shoulder)   # right arm — more lateral
    LA = skin_contact(-x_hi * 0.78,  z_shoulder)   # left  arm — more lateral
    RL = skin_contact( x_hi * 0.65,  z_lower)      # right lower torso — more lateral
    LL = skin_contact(-x_hi * 0.65,  z_lower)      # left  lower torso — more lateral

    return {
        "V1": V1, "V2": V2, "V3": V3, "V4": V4,
        "V5": V5, "V6": V6,
        "RA": RA, "LA": LA, "RL": RL, "LL": LL,
    }


def _coord_transform(pts: np.ndarray) -> np.ndarray:
    """
    180° Z-rotation + origin shift (shared by both VTU and CARP paths).
    Input/output: (N, 3) in µm.
    Result: +X=patient left, Y: 0=posterior/max=anterior, Z: 0=inferior/max=superior.
    """
    pts = pts.copy().astype(float)
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]
    pts[:, 1] -= float(pts[:, 1].min())
    pts[:, 2] -= float(pts[:, 2].min())
    return pts


def load_carp_mesh(sim_dir: Path, prefix: str = "torso") -> "pv.UnstructuredGrid":
    """
    Read CARP .pts / .elem files into a PyVista UnstructuredGrid.

    Node coordinates are kept in their original units (µm for KCL meshes,
    mm*1000 for gmsh-generated meshes).  Call _coord_transform() on
    grid.points after loading if you need the standard viewer orientation.

    Parameters
    ----------
    sim_dir : directory containing <prefix>.pts and <prefix>.elem
    prefix  : file name prefix (default "torso")

    Returns
    -------
    pv.UnstructuredGrid with cell_data["RegionTag"]
    """
    pts_file  = Path(sim_dir) / f"{prefix}.pts"
    elem_file = Path(sim_dir) / f"{prefix}.elem"

    if not pts_file.exists():
        raise FileNotFoundError(f"CARP pts file not found: {pts_file}")
    if not elem_file.exists():
        raise FileNotFoundError(f"CARP elem file not found: {elem_file}")

    print(f"Reading {pts_file.name} ...", flush=True)
    nodes = np.loadtxt(pts_file, skiprows=1, dtype=np.float64)

    print(f"Reading {elem_file.name} (may take ~30 s for large files) ...", flush=True)
    try:
        import pandas as pd
        df = pd.read_csv(
            elem_file, sep=r"\s+", header=None, skiprows=1, engine="c",
            dtype={0: str, 1: np.int32, 2: np.int32,
                   3: np.int32, 4: np.int32, 5: np.int32},
        )
        conn = df.iloc[:, 1:5].values.astype(np.int32)
        regs = df.iloc[:, 5].values.astype(np.int32)
    except ImportError:
        # Pure-Python fallback (slow for large files)
        with open(elem_file) as fh:
            n_elems = int(fh.readline())
            rows = []
            for line in fh:
                parts = line.split()
                if len(parts) >= 6:
                    rows.append([int(x) for x in parts[1:6]])
        arr  = np.array(rows, dtype=np.int32)
        conn = arr[:, :4]
        regs = arr[:, 4]

    print(f"  {len(nodes):,} nodes, {len(conn):,} elements, "
          f"tags: {sorted(np.unique(regs).tolist())}")

    # Build PyVista cells array: [4, n0, n1, n2, n3, 4, n0, n1, n2, n3, ...]
    n_cells = len(conn)
    cells = np.empty(n_cells * 5, dtype=np.int64)
    cells[0::5] = 4
    cells[1::5] = conn[:, 0]
    cells[2::5] = conn[:, 1]
    cells[3::5] = conn[:, 2]
    cells[4::5] = conn[:, 3]
    cell_types = np.full(n_cells, 10, dtype=np.uint8)   # VTK_TETRA = 10

    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    grid.cell_data["RegionTag"] = regs
    return grid


def _find_vtu() -> Path | None:
    """Search common output paths for a VTU file."""
    candidates = [
        Path("C:/Users/bulli/cardiac_data/tcia_torso"),
        Path("C:/Users/bulli/cardiac_data/kcl_torso/KCL_torso3"),
        Path("."),
    ]
    for d in candidates:
        if d.is_dir():
            files = list(d.glob("*.vtu"))
            if files:
                # Prefer anatomy (no ICD) if available
                anat = [f for f in files if "ICD" not in f.name]
                return anat[0] if anat else files[0]
    return None


def _run_viewer(
    mesh: "pv.DataSet",
    surfaces: "dict[int, pv.PolyData]",
    tag_style: dict,
    title: str,
    lv_tag: int = 27,
    skin_tag: int = SKIN_TAG,
    bone_tag: int = BONE_TAG,
) -> None:
    """
    Core interactive PyVista viewer — shared by both VTU and CARP paths.

    Parameters
    ----------
    mesh      : full mesh (used for camera bounds and electrode placement)
    surfaces  : per-tag surface meshes (pre-extracted)
    tag_style : dict[int, dict(name, color, opacity)]
    title     : window title string
    lv_tag    : tag number for LV wall (27 in VTU remapping, 35 in CARP native)
    skin_tag  : tag for skin surface (electrode ray-cast target)
    bone_tag  : tag for bone surface (rib detection)
    """
    tags = sorted(surfaces.keys())

    # ── Set up plotter ────────────────────────────────────────────────────────
    pl = pv.Plotter(title=f"Tissue Viewer — {title}", window_size=(1200, 800))
    pl.set_background("white")
    pl.add_axes(line_width=3)
    # Flip the X arrow so it points screen-right (+X = patient left, DICOM convention).
    # Try every known PyVista/VTK path to reach the orientation marker actor.
    try:
        import vtk as _vtk
        _t = _vtk.vtkTransform()
        _t.Scale(-1.0, 1.0, 1.0)   # reflect X only
        _ow = (
            getattr(pl.renderer, 'axes_widget', None)
            or getattr(pl.renderer, '_axes_widget', None)
            or pl.renderer.GetOrientationWidget()
        )
        if _ow is not None:
            _marker = _ow.GetOrientationMarker()
            if _marker is not None:
                _marker.SetUserTransform(_t)
    except Exception as _e:
        print(f"  (axes X-flip skipped: {_e})")

    # ── Add tissue actors ─────────────────────────────────────────────────────
    actors: dict[int, pv.Actor] = {}
    states: dict[int, int] = {}   # index into OPACITY_CYCLE

    for tag in tags:
        if tag not in surfaces:
            continue
        style = tag_style.get(tag, dict(
            name=f"Tag {tag}", color=(0.5, 0.5, 0.5), opacity=0.5
        ))
        init_opacity = style["opacity"]
        # Find closest cycle index
        diffs = [abs(init_opacity - o) for o in OPACITY_CYCLE]
        states[tag] = int(np.argmin(diffs))

        actor = pl.add_mesh(
            surfaces[tag],
            color=style["color"],
            opacity=init_opacity,
            smooth_shading=True,
            show_edges=False,
            name=f"tissue_{tag}",
        )
        actors[tag] = actor

    # ── Load ECG electrodes and create flat disc actors ───────────────────────
    # IEC 60601-2-25 lead colours
    ELECTRODE_INFO = [
        ("V1", (0.80, 0.20, 0.20)),
        ("V2", (0.85, 0.75, 0.10)),
        ("V3", (0.10, 0.65, 0.20)),
        ("V4", (0.50, 0.25, 0.05)),
        ("V5", (0.25, 0.25, 0.25)),
        ("V6", (0.45, 0.10, 0.70)),
        ("RA", (0.90, 0.05, 0.05)),
        ("LA", (0.95, 0.90, 0.05)),
        ("RL", (0.05, 0.65, 0.10)),
        ("LL", (0.25, 0.25, 0.25)),
    ]

    elec_actors: dict[str, pv.Actor] = {}
    elec_states: dict[str, int] = {}
    elec_lbl_actors: list = []   # label actors collected for group visibility toggle

    # ── Auto-place ECG electrodes using anatomical landmarks ─────────────────
    # Electrode positions are derived from the bone surface (rib detection) and
    # the skin surface (ray-cast contact point + local outward normal).
    # The disc flat face is placed flush against the skin; disc axis = skin normal.
    print("Placing ECG electrodes anatomically (standard 12-lead positions)...")
    disc_r     = 10_000.0   # 10 mm radius → 20 mm diameter (µm)
    disc_depth =  2_000.0   # 2 mm depth (µm)

    elec_pos = _place_ecg_electrodes(
        surfaces.get(bone_tag),
        surfaces.get(skin_tag),
        mesh.points,
        lv_surf=surfaces.get(lv_tag),   # LV wall — anchors 5th ICS Z level
    )

    if elec_pos:
        for label, color in ELECTRODE_INFO:
            if label not in elec_pos:
                continue
            pos, normal = elec_pos[label]

            # Disc centre pushed outward by half-depth so face is flush with skin
            centre = pos + normal * (disc_depth * 0.5)
            disc = pv.Cylinder(
                center=centre.tolist(), direction=normal.tolist(),
                radius=disc_r, height=disc_depth, resolution=48, capping=True,
            )
            actor = pl.add_mesh(
                disc, color=color, opacity=1.0,
                smooth_shading=True, show_edges=False, name=f"elec_{label}",
            )
            elec_actors[label] = actor
            elec_states[label] = 0   # solid

            # Floating label anchored slightly beyond the disc face
            lbl_off = normal * disc_r * 1.6 + np.array([0.0, 0.0, disc_r * 0.5])
            lbl_actor = pl.add_point_labels(
                [pos + lbl_off], [label],
                font_size=10, text_color="black", bold=True,
                always_visible=True, show_points=False, name=f"lbl_{label}",
            )
            elec_lbl_actors.append(lbl_actor)

        print(f"  Placed {len(elec_actors)} electrodes (anatomical auto-placement)")
    else:
        print("  Bone/skin surfaces unavailable — skipping electrode placement")

    # ── Camera: DICOM anterior view, +Z up ────────────────────────────────────
    # After DICOM transform: +x=left, y: 0=posterior, max=anterior, +z=superior.
    # Camera placed at high Y (anterior side) facing toward posterior (-Y).
    # Screen-right = -X → patient right (+X original = -X_new) on viewer's LEFT ✓
    b = mesh.bounds
    cx = (b[0] + b[1]) / 2
    cy = (b[2] + b[3]) / 2
    cz = (b[4] + b[5]) / 2
    view_dist = max(b[1]-b[0], b[5]-b[4]) * 2.2
    pl.camera.position    = (cx, b[3] + view_dist, cz)
    pl.camera.focal_point = (cx, cy, cz)
    pl.camera.up          = (0.0, 0.0, 1.0)
    pl.reset_camera_clipping_range()

    # ── Single-column legend: tissues + ECG electrodes ────────────────────────
    BTN_SIZE    = 16
    BTN_GAP     = 3
    ROW_H       = BTN_SIZE + BTN_GAP
    COL_W       = BTN_SIZE + BTN_GAP
    LEFT_X      = 8
    WIN_H       = 950
    pl.window_size = (1300, WIN_H)

    start_y     = WIN_H - 36
    sorted_tags = sorted(actors.keys())
    OPACITIES   = [1.0, 0.2, 0.0]

    btn_widgets: dict = {}   # tag (int) or label (str) -> [w0, w1, w2]

    def make_cb(key, actor_dict, state_dict, idx):
        """Radio-button callback for a single tissue actor."""
        def cb(value):
            if not value:
                btn_widgets[key][idx].GetRepresentation().SetState(1)
                pl.render()
                return
            actor_dict[key].GetProperty().SetOpacity(OPACITIES[idx])
            state_dict[key] = idx
            for j, w in enumerate(btn_widgets[key]):
                if j != idx:
                    w.GetRepresentation().SetState(0)
            pl.render()
        return cb

    def make_group_cb(key, disc_list, lbl_list, state_holder, idx):
        """Radio-button callback that sets opacity on ALL electrode discs at once."""
        def cb(value):
            if not value:
                btn_widgets[key][idx].GetRepresentation().SetState(1)
                pl.render()
                return
            op = OPACITIES[idx]
            for a in disc_list:
                a.GetProperty().SetOpacity(op)
            vis = int(op > 0.0)
            for la in lbl_list:
                try:
                    la.SetVisibility(vis)
                except Exception:
                    pass
            state_holder[0] = idx
            for j, w in enumerate(btn_widgets[key]):
                if j != idx:
                    w.GetRepresentation().SetState(0)
            pl.render()
        return cb

    btn_cfgs = [
        dict(color_on="#00bb55"),   # Solid  green
        dict(color_on="#ff8800"),   # Ghost  orange
        dict(color_on="#cc2222"),   # Hide   red
    ]

    # ── Tissue rows ───────────────────────────────────────────────────────────
    pl.add_text("S  G  H", position=(LEFT_X, start_y + ROW_H + 2),
                font_size=6, color="#444444")

    for i, tag in enumerate(sorted_tags):
        y     = start_y - i * ROW_H
        style = tag_style.get(tag, dict(name=f"Tag {tag}", color=(0.5, 0.5, 0.5)))
        btn_widgets[tag] = []
        for j, cfg in enumerate(btn_cfgs):
            w = pl.add_checkbox_button_widget(
                make_cb(tag, actors, states, j),
                value=(states[tag] == j),
                position=(LEFT_X + j * COL_W, y),
                size=BTN_SIZE, color_off="#cccccc", **cfg,
            )
            btn_widgets[tag].append(w)
        pl.add_text(f" {style['name']}",
                    position=(LEFT_X + 3*COL_W + 3, y + 3),
                    font_size=6, color="black")

    # ── ECG Electrodes: single row controlling all discs together ─────────────
    if elec_actors:
        ECG_KEY      = "__ecg__"
        ecg_state    = [0]   # mutable holder: index into OPACITIES (starts Solid)
        disc_list    = list(elec_actors.values())
        ey           = start_y - len(sorted_tags) * ROW_H
        btn_widgets[ECG_KEY] = []
        for j, cfg in enumerate(btn_cfgs):
            w = pl.add_checkbox_button_widget(
                make_group_cb(ECG_KEY, disc_list, elec_lbl_actors, ecg_state, j),
                value=(j == 0),   # Solid initially
                position=(LEFT_X + j * COL_W, ey),
                size=BTN_SIZE, color_off="#cccccc", **cfg,
            )
            btn_widgets[ECG_KEY].append(w)
        pl.add_text(" ECG Electrodes",
                    position=(LEFT_X + 3*COL_W + 3, ey + 3),
                    font_size=6, color="black")

    pl.add_text(
        f"Tissue Viewer\n{title}",
        position="upper_right",
        font_size=9,
        color="black",
    )

    print("\nControls:")
    print("  Green  button = Solid  (opacity 1.0)")
    print("  Orange button = Ghost  (opacity 0.2)")
    print("  Red    button = Hidden (opacity 0.0)")
    print("  Mouse: left-drag=rotate, right-drag=zoom, middle-drag=pan")
    print("  Press Q to quit.\n")

    pl.show()


def launch(vtu_path: Path) -> None:
    """Launch the interactive viewer for a KCL anatomy VTU file."""
    print(f"Loading: {vtu_path}")
    mesh = pv.read(str(vtu_path))

    if "RegionTag" not in mesh.cell_data:
        raise ValueError("VTU has no 'RegionTag' cell data array.")

    tags = np.unique(mesh.cell_data["RegionTag"]).astype(int).tolist()
    print(f"Found {len(tags)} tissue regions: {tags}")

    # 180° Z-rotation + origin shift (DICOM → viewer coordinates)
    mesh.points = _coord_transform(mesh.points)

    # Per-tissue surfaces (slow first run; cached to disk)
    cache_path = vtu_path.with_suffix(".surfaces_v4.pkl")
    surfaces: dict[int, pv.PolyData] = {}

    if cache_path.exists():
        print(f"Loading cached surfaces: {cache_path.name}")
        with open(cache_path, "rb") as fh:
            surfaces = pickle.load(fh)
    else:
        print(f"Extracting surfaces for {len(tags)} tissues (first run — please wait)...")
        for i, tag in enumerate(tags, 1):
            tname = TAG_STYLE.get(tag, {}).get("name", f"Tag {tag}")
            print(f"  [{i:2d}/{len(tags)}] {tname} ...", end=" ", flush=True)
            sub = mesh.threshold([tag - 0.5, tag + 0.5], scalars="RegionTag")
            if sub.n_cells > 0:
                surfaces[tag] = sub.extract_surface(algorithm="dataset_surface")
                print("done")
            else:
                print("(empty)")
        print(f"Saving surface cache -> {cache_path.name}")
        with open(cache_path, "wb") as fh:
            pickle.dump(surfaces, fh)

    # VTU uses remapped tags: 27 = LV wall (CARP tag 35)
    _run_viewer(mesh, surfaces, TAG_STYLE, vtu_path.stem,
                lv_tag=27, skin_tag=2, bone_tag=3)


def launch_carp(sim_dir: Path, prefix: str = "torso") -> None:
    """
    Launch the interactive viewer for a CARP simulation directory.

    Looks for <sim_dir>/<prefix>.pts and <sim_dir>/<prefix>.elem.
    Works with both KCL-extracted and gmsh-generated CARP meshes.

    Usage
    -----
        python viewer_3d.py /path/to/sim_dir
        python viewer_3d.py /path/to/sim_dir ventricles
    """
    sim_dir = Path(sim_dir)
    mesh = load_carp_mesh(sim_dir, prefix)

    # Apply standard viewer coordinate transform
    mesh.points = _coord_transform(mesh.points)

    tags = np.unique(mesh.cell_data["RegionTag"]).astype(int).tolist()
    print(f"Tags present: {tags}")

    # Per-tissue surfaces (cached alongside the .pts file)
    cache_path = sim_dir / f"{prefix}.surfaces.pkl"
    surfaces: dict[int, pv.PolyData] = {}

    if cache_path.exists():
        print(f"Loading cached surfaces: {cache_path.name}")
        with open(cache_path, "rb") as fh:
            surfaces = pickle.load(fh)
    else:
        print(f"Extracting surfaces for {len(tags)} tissues (first run — please wait)...")
        for i, tag in enumerate(tags, 1):
            tname = CARP_TAG_STYLE.get(tag, {}).get("name", f"Tag {tag}")
            print(f"  [{i:2d}/{len(tags)}] {tname} ...", end=" ", flush=True)
            sub = mesh.threshold([tag - 0.5, tag + 0.5], scalars="RegionTag")
            if sub.n_cells > 0:
                surfaces[tag] = sub.extract_surface(algorithm="dataset_surface")
                print("done")
            else:
                print("(empty)")
        print(f"Saving surface cache -> {cache_path.name}")
        with open(cache_path, "wb") as fh:
            pickle.dump(surfaces, fh)

    title = f"{sim_dir.name}/{prefix}"
    # CARP native: LV_WALL=35, SKIN_TAG=2, BONE_TAG=3
    _run_viewer(mesh, surfaces, CARP_TAG_STYLE, title,
                lv_tag=LV_WALL, skin_tag=SKIN_TAG, bone_tag=BONE_TAG)


def _find_carp_dir() -> Path | None:
    """Search common output paths for a CARP sim directory."""
    candidates = [
        Path("C:/Users/bulli/cardiac_data/kcl_torso/KCL_torso3/sim/torso"),
        Path("C:/Users/bulli/cardiac_data/kcl_torso/KCL_torso3/sim"),
        Path("."),
    ]
    for d in candidates:
        if d.is_dir() and ((d / "torso.pts").exists() or (d / "torso.elem").exists()):
            return d
    return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        prefix = sys.argv[2] if len(sys.argv) > 2 else "torso"
        if arg.is_dir():
            launch_carp(arg, prefix)
        elif arg.suffix == ".vtu":
            launch(arg)
        else:
            print(f"Unknown input: {arg}")
            print("Pass a .vtu file or a CARP sim directory.")
            sys.exit(1)
    else:
        # Auto-discover: prefer VTU, fall back to CARP directory
        vtu_path = _find_vtu()
        if vtu_path is not None:
            launch(vtu_path)
        else:
            carp_dir = _find_carp_dir()
            if carp_dir is not None:
                launch_carp(carp_dir)
            else:
                print("No VTU file or CARP sim directory found.")
                print("Usage:")
                print("  python viewer_3d.py path/to/mesh.vtu")
                print("  python viewer_3d.py path/to/sim_dir [prefix]")
                sys.exit(1)
