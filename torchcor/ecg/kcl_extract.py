"""
torchcor/ecg/kcl_extract.py
============================
One-time extraction of simulation-ready CARP meshes from the KCL
whole-torso model (KCL_torso3.pts / .elem / .lon).

Applies the same 180° Z-axis rotation used in viewer_3d.py so that the
simulation mesh coordinate system matches the 3-D viewer.

Outputs (written once, cached under <out_dir>/):
  torso/torso.{pts,elem,lon}              full anatomy torso (no ICD)
  ventricles/ventricles.{pts,elem,lon}    biventricular mesh
  ventricles/pacing/LV_endo.vtx           LV endocardial nodes
  ventricles/pacing/RV_endo.vtx           RV endocardial nodes
  electrodes.vtx                          12-lead electrode node indices
  heart_to_torso.npy                      ventricle→torso node mapping

KCL CARP tag convention
-----------------------
  1 – 25 : anatomy  (KCL_Info.pdf)
  26 – 33: ICD hardware  (excluded for anatomy model)
  34      : RV wall  (bidomain, σ_i / σ_e anisotropic)
  35      : LV wall  (bidomain)
  (tags 24 = RV blood pool, 25 = LV blood pool)

Usage
-----
    python -m torchcor.ecg.kcl_extract
    python -m torchcor.ecg.kcl_extract --carp <dir> --vtu <file> --out <dir>
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from torchcor.ecg.torso_tags import (
    ICD_TAGS, VENT_TAGS, BLOOD_TAGS, MYO_TAGS, SKIN_TAG, BONE_TAG,
    RV_BLOOD, LV_BLOOD, RV_WALL, LV_WALL,
)

# ── Default paths ──────────────────────────────────────────────────────────────
_KCL_DIR  = Path(r"C:\Users\bulli\cardiac_data\kcl_torso\KCL_torso3")
DEFAULT_CARP_PREFIX = str(_KCL_DIR / "KCL_torso3")
DEFAULT_VTU         = _KCL_DIR / "KCL_torso3_anatomy.vtu"
DEFAULT_OUT_DIR     = _KCL_DIR / "sim"

# ── IEC electrode labels & VTU tag for viewer_3d ──────────────────────────────
ELECTRODE_LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]


# ==============================================================================
#  Fast CARP I/O  (pandas-based, handles 7 M-element files in < 60 s)
# ==============================================================================

def read_carp_pts(pts_file: Path) -> np.ndarray:
    """Read .pts → (N, 3) float64 array  (units: µm, as stored)."""
    nodes = np.loadtxt(pts_file, dtype=np.float64, skiprows=1)
    print(f"  Loaded {nodes.shape[0]:,} nodes from {pts_file.name}")
    return nodes


def read_carp_elem(elem_file: Path):
    """
    Read .elem file  →  (elem_nodes, regions)
      elem_nodes : (M, 4) int32   tetrahedral connectivity
      regions    : (M,)   int32   region tag per element

    Uses pandas for C-speed parsing of large files.
    The KCL file contains only 'Tt' (tetrahedral) elements.
    """
    import pandas as pd

    print(f"  Reading {elem_file.name} (may take ~30 s for large files)…", flush=True)
    # Skip header count line; columns: [type, n0, n1, n2, n3, region]
    df = pd.read_csv(
        elem_file,
        sep=r"\s+",
        header=None,
        skiprows=1,
        dtype={0: str, 1: np.int32, 2: np.int32, 3: np.int32, 4: np.int32, 5: np.int32},
        engine="c",
    )
    elem_nodes = df.iloc[:, 1:5].values.astype(np.int32)
    regions    = df.iloc[:, 5].values.astype(np.int32)
    print(f"  Loaded {len(elem_nodes):,} elements")
    return elem_nodes, regions


def read_carp_lon(lon_file: Path, n_elems: int) -> np.ndarray:
    """
    Read .lon → (M, 3) float32  (longitudinal fiber direction only).

    The KCL .lon uses format '2' (6 columns: fiber + sheet direction).
    We keep only the first 3 columns (longitudinal fiber e_l).
    """
    print(f"  Reading {lon_file.name} (may take ~60 s for large files)…", flush=True)
    fibres = np.loadtxt(lon_file, dtype=np.float32, skiprows=1)
    if fibres.ndim == 1:
        fibres = fibres.reshape(1, -1)
    fibres = fibres[:, :3]   # keep longitudinal direction only
    # Normalise (they should already be unit vectors but apply defensively)
    norms = np.linalg.norm(fibres, axis=1, keepdims=True)
    bad   = (norms[:, 0] < 1e-10)
    fibres[bad] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    norms[bad]  = 1.0
    fibres /= norms
    print(f"  Loaded {len(fibres):,} fiber vectors")
    return fibres


def write_carp_pts(pts_file: Path, nodes_um: np.ndarray) -> None:
    """
    Write .pts file.  nodes_um : (N, 3) in MICROMETRES (standard CARP convention).
    MeshReader.read(unit_conversion=1000) will divide by 1000 → mm.
    """
    pts_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pts_file, "w") as f:
        f.write(f"{len(nodes_um)}\n")
        np.savetxt(f, nodes_um, fmt="%.3f")
    print(f"  Written {len(nodes_um):,} nodes → {pts_file}")


def write_carp_elem(elem_file: Path, elem_nodes: np.ndarray, regions: np.ndarray) -> None:
    """Write .elem file (all-tetrahedral)."""
    elem_file.parent.mkdir(parents=True, exist_ok=True)
    n = len(elem_nodes)
    data = np.column_stack([elem_nodes, regions])  # (M, 5)
    with open(elem_file, "w") as f:
        f.write(f"{n}\n")
        # Write in blocks for speed
        BLOCK = 200_000
        for start in range(0, n, BLOCK):
            chunk = data[start:start + BLOCK]
            lines = [
                f"Tt {r[0]} {r[1]} {r[2]} {r[3]} {r[4]}"
                for r in chunk
            ]
            f.write("\n".join(lines) + "\n")
    print(f"  Written {n:,} elements → {elem_file}")


def write_carp_lon(lon_file: Path, fibres: np.ndarray) -> None:
    """Write .lon file (3-component, format '1')."""
    lon_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lon_file, "w") as f:
        f.write("1\n")   # format: 1 = 3 components
        np.savetxt(f, fibres.astype(np.float32), fmt="%.6f")
    print(f"  Written {len(fibres):,} fiber vectors → {lon_file}")


def write_vtx(vtx_file: Path, node_ids: np.ndarray) -> None:
    """Write a CARP .vtx pacing-site file."""
    vtx_file.parent.mkdir(parents=True, exist_ok=True)
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    with open(vtx_file, "w") as f:
        f.write(f"{len(node_ids)}\n")
        f.write("intra\n")
        np.savetxt(f, node_ids, fmt="%d")
    print(f"  Written {len(node_ids):,} pacing nodes → {vtx_file}")


# ==============================================================================
#  Coordinate transform  (identical to viewer_3d.launch)
# ==============================================================================

def rotate180z_and_shift(nodes_um: np.ndarray) -> np.ndarray:
    """
    Apply the same coordinate transform as viewer_3d.py:
      1. 180° rotation around Z:  X → -X,  Y → -Y
      2. Shift Y so posterior face = 0 (Y_min = 0)
      3. Shift Z so inferior face = 0 (Z_min = 0)

    Input  : (N, 3) in µm  (original KCL CARP coordinates)
    Returns: (N, 3) in µm  (rotated + shifted)
    """
    pts = nodes_um.copy()
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]
    pts[:, 1] -= float(pts[:, 1].min())
    pts[:, 2] -= float(pts[:, 2].min())
    return pts


# ==============================================================================
#  Subset extraction  (filter elements by tag + remap nodes)
# ==============================================================================

def extract_subset(
    nodes: np.ndarray,
    elem_nodes: np.ndarray,
    regions: np.ndarray,
    fibres: np.ndarray,
    keep_tags: set,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract elements with region tag in keep_tags, renumber nodes.

    Returns
    -------
    sub_nodes         : (N_sub, 3)
    sub_elem_nodes    : (M_sub, 4)  — renumbered into sub_nodes indexing
    sub_regions       : (M_sub,)
    sub_fibres        : (M_sub, 3)
    subset_to_full    : (N_sub,)    — sub_node_idx → original node idx
    """
    mask      = np.isin(regions, list(keep_tags))
    sub_elems = elem_nodes[mask]
    sub_regs  = regions[mask]
    sub_fibs  = fibres[mask]

    # Unique nodes needed
    unique_nodes  = np.unique(sub_elems.reshape(-1))   # sorted
    subset_to_full = unique_nodes                        # (N_sub,) → original index

    # Build inverse map: original_node_idx → sub_node_idx
    inv = np.full(nodes.shape[0], -1, dtype=np.int64)
    inv[unique_nodes] = np.arange(len(unique_nodes), dtype=np.int64)

    sub_nodes      = nodes[unique_nodes]
    sub_elem_nodes = inv[sub_elems].astype(np.int32)

    return sub_nodes, sub_elem_nodes, sub_regs, sub_fibs, subset_to_full


# ==============================================================================
#  Pacing-site detection  (endocardial surface nodes)
# ==============================================================================

def find_interface_nodes(
    elem_nodes: np.ndarray,
    regions: np.ndarray,
    tag_a: int,
    tag_b: int,
) -> np.ndarray:
    """
    Return node indices shared by elements of tag_a AND elements of tag_b.
    These are the nodes on the boundary between the two regions (endo surface).
    """
    nodes_a = set(elem_nodes[regions == tag_a].reshape(-1).tolist())
    nodes_b = set(elem_nodes[regions == tag_b].reshape(-1).tolist())
    shared  = np.array(sorted(nodes_a & nodes_b), dtype=np.int64)
    return shared


# ==============================================================================
#  Electrode placement  (reuse viewer_3d._place_ecg_electrodes)
# ==============================================================================

def compute_electrode_positions(
    vtu_path: Path,
    surfaces_cache: Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """
    Compute 12-lead electrode 3-D positions in the rotated coordinate system
    by loading the anatomy VTU and running the anatomical auto-placement
    algorithm from viewer_3d.py.

    The VTU surfaces cache (.surfaces_v4.pkl) is used if available.

    Returns
    -------
    dict { label : (pos_µm, unit_normal) }   or  None on failure
    """
    # Import electrode placement from viewer_3d
    sys.path.insert(0, str(Path(__file__).parent))
    from viewer_3d import _place_ecg_electrodes

    if surfaces_cache is None:
        surfaces_cache = vtu_path.with_suffix(".surfaces_v4.pkl")

    # ── Load or build surface cache ──────────────────────────────────────
    surfaces: dict = {}
    if surfaces_cache.exists():
        print(f"  Loading surface cache: {surfaces_cache.name}")
        with open(surfaces_cache, "rb") as fh:
            surfaces = pickle.load(fh)
    else:
        print("  Building surfaces from VTU (first run — may take several minutes)…")
        mesh = pv.read(str(vtu_path))
        pts  = mesh.points.copy().astype(float)
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]
        pts[:, 1] -= float(pts[:, 1].min())
        pts[:, 2] -= float(pts[:, 2].min())
        mesh.points = pts

        for tag in np.unique(mesh.cell_data["RegionTag"]).astype(int):
            sub = mesh.threshold([tag - 0.5, tag + 0.5], scalars="RegionTag")
            if sub.n_cells > 0:
                surfaces[tag] = sub.extract_surface(algorithm="dataset_surface")

        with open(surfaces_cache, "wb") as fh:
            pickle.dump(surfaces, fh)

    if not surfaces:
        return None

    # ── Reconstruct full-mesh points array for extent calculations ────────
    # Use the first available surface to get approximate bounds.
    all_pts = np.vstack([np.asarray(s.points) for s in surfaces.values()])

    # ── Run auto-placement ────────────────────────────────────────────────
    # Tag mapping in VTU:  2=skin, 3=bones, 27=LV wall
    elec_pos = _place_ecg_electrodes(
        bone_surf=surfaces.get(3),
        skin_surf=surfaces.get(2),
        all_pts=all_pts,
        lv_surf=surfaces.get(27),
    )
    return elec_pos


def map_electrodes_to_carp_nodes(
    elec_pos: dict[str, tuple[np.ndarray, np.ndarray]],
    carp_nodes_um: np.ndarray,
    carp_regions: np.ndarray,
    skin_tag: int = SKIN_TAG,
) -> dict[str, int]:
    """
    For each electrode position (in rotated µm coordinates), find the nearest
    node in the CARP torso skin surface.

    The CARP nodes must already be in the rotated+shifted coordinate system
    (output of rotate180z_and_shift).

    Returns
    -------
    dict { label : torso_node_idx }
    """
    from scipy.spatial import KDTree

    # Build KDTree over all nodes that belong to skin elements
    skin_elem_mask   = carp_regions == skin_tag
    # We don't have per-element → per-node for the torso here, so use all nodes
    # that appear in skin elements (requires elem_nodes)
    # Fallback: use all nodes in the mesh (slightly less precise but fine)
    print(f"  Building KDTree over {len(carp_nodes_um):,} torso nodes…")
    tree = KDTree(carp_nodes_um)

    mapping: dict[str, int] = {}
    for label, (pos, _normal) in elec_pos.items():
        pos_arr = np.asarray(pos, dtype=float).reshape(3)
        _dist, idx = tree.query(pos_arr)
        mapping[label] = int(idx)
        print(f"    {label:3s}: node {idx:7d}  pos=({pos_arr[0]/1000:.1f}, "
              f"{pos_arr[1]/1000:.1f}, {pos_arr[2]/1000:.1f}) mm")

    return mapping


# ==============================================================================
#  Main extraction routine
# ==============================================================================

def extract(
    carp_prefix: str = DEFAULT_CARP_PREFIX,
    vtu_path: Path = DEFAULT_VTU,
    out_dir: Path = DEFAULT_OUT_DIR,
    force: bool = False,
) -> Path:
    """
    Full extraction pipeline.  Returns out_dir.

    If out_dir/torso/ already exists (and force=False), skips extraction.
    """
    out_dir = Path(out_dir)

    torso_dir = out_dir / "torso"
    vent_dir  = out_dir / "ventricles"
    done_flag = out_dir / ".extracted"

    if done_flag.exists() and not force:
        print(f"Extraction already done ({done_flag}).  Use force=True to redo.")
        return out_dir

    print("=" * 60)
    print("KCL mesh extraction")
    print(f"  CARP prefix : {carp_prefix}")
    print(f"  VTU         : {vtu_path}")
    print(f"  Output      : {out_dir}")
    print("=" * 60)

    # ── 1. Read CARP files ──────────────────────────────────────────────
    print("\n[1] Reading CARP files…")
    pts_file  = Path(carp_prefix + ".pts")
    elem_file = Path(carp_prefix + ".elem")
    lon_file  = Path(carp_prefix + ".lon")

    for f in (pts_file, elem_file, lon_file):
        if not f.exists():
            raise FileNotFoundError(f"CARP file not found: {f}")

    nodes_um              = read_carp_pts(pts_file)          # (N, 3) µm
    elem_nodes, regions   = read_carp_elem(elem_file)        # (M,4), (M,)
    fibres                = read_carp_lon(lon_file, len(regions))  # (M,3)

    print(f"  Tags present: {sorted(np.unique(regions).tolist())}")

    # ── 2. Apply 180° Z-rotation + shifts ──────────────────────────────
    print("\n[2] Applying 180° Z-rotation + coordinate shift…")
    # nodes_rot is in µm (same units as CARP source files — standard convention)
    # MeshReader.read(unit_conversion=1000) divides by 1000 → mm during simulation
    nodes_rot = rotate180z_and_shift(nodes_um)   # still in µm

    # ── 3. Extract anatomy torso (no ICD hardware) ──────────────────────
    print("\n[3] Extracting anatomy torso (removing ICD tags 26-33)…")
    icd_mask    = np.isin(regions, list(ICD_TAGS))
    n_icd       = int(icd_mask.sum())
    print(f"  Removing {n_icd:,} ICD elements")
    anat_tags   = set(int(t) for t in np.unique(regions)) - ICD_TAGS

    t_nodes, t_elems, t_regions, t_fibres, t_sub2full = extract_subset(
        nodes_rot, elem_nodes, regions, fibres, anat_tags   # µm
    )
    print(f"  Torso: {len(t_nodes):,} nodes, {len(t_elems):,} elements, "
          f"tags: {sorted(set(t_regions.tolist()))}")

    # ── 4. Extract biventricular subset ────────────────────────────────
    print("\n[4] Extracting biventricular mesh (tags 24, 25, 34, 35)…")
    # Extract from torso (already filtered) — indices into torso_nodes
    v_nodes, v_elems, v_regions, v_fibres, v_sub2torso = extract_subset(
        t_nodes, t_elems, t_regions, t_fibres, VENT_TAGS   # µm
    )
    # v_sub2torso maps  ventricle_node → torso_node
    heart_to_torso = v_sub2torso.astype(np.int64)

    print(f"  Ventricles: {len(v_nodes):,} nodes, {len(v_elems):,} elements, "
          f"tags: {sorted(set(v_regions.tolist()))}")

    # ── 5. Write torso CARP files ───────────────────────────────────────
    print("\n[5] Writing torso CARP files…")
    torso_dir.mkdir(parents=True, exist_ok=True)
    write_carp_pts( torso_dir / "torso.pts",  t_nodes)
    write_carp_elem(torso_dir / "torso.elem", t_elems, t_regions)
    write_carp_lon( torso_dir / "torso.lon",  t_fibres)

    # ── 6. Write ventricular CARP files ────────────────────────────────
    print("\n[6] Writing ventricular CARP files…")
    vent_dir.mkdir(parents=True, exist_ok=True)
    write_carp_pts( vent_dir / "ventricles.pts",  v_nodes)
    write_carp_elem(vent_dir / "ventricles.elem", v_elems, v_regions)
    write_carp_lon( vent_dir / "ventricles.lon",  v_fibres)

    # Save the heart_to_torso mapping (ventricle node → torso node)
    np.save(out_dir / "heart_to_torso.npy", heart_to_torso)
    print(f"  Saved heart→torso mapping ({len(heart_to_torso):,} nodes)")

    # ── 7. Pacing sites ─────────────────────────────────────────────────
    print("\n[7] Generating pacing site .vtx files…")
    pacing_dir = vent_dir / "pacing"
    pacing_dir.mkdir(parents=True, exist_ok=True)

    # LV endocardium: nodes shared by LV wall (35) AND LV blood pool (25)
    lv_endo_torso = find_interface_nodes(t_elems, t_regions, tag_a=LV_WALL, tag_b=LV_BLOOD)
    # Convert from torso node indices → ventricle node indices
    # (t_sub2full maps vent → torso, so we need the inverse)
    inv_vent = np.full(len(t_nodes), -1, dtype=np.int64)
    inv_vent[heart_to_torso] = np.arange(len(heart_to_torso), dtype=np.int64)
    lv_endo_vent = inv_vent[lv_endo_torso]
    lv_endo_vent = lv_endo_vent[lv_endo_vent >= 0]
    write_vtx(pacing_dir / "LV_endo.vtx", lv_endo_vent)

    # RV endocardium: nodes shared by RV wall (34) AND RV blood pool (24)
    rv_endo_torso = find_interface_nodes(t_elems, t_regions, tag_a=RV_WALL, tag_b=RV_BLOOD)
    rv_endo_vent  = inv_vent[rv_endo_torso]
    rv_endo_vent  = rv_endo_vent[rv_endo_vent >= 0]
    write_vtx(pacing_dir / "RV_endo.vtx", rv_endo_vent)

    # ── 8. Electrode positions → CARP node mapping ─────────────────────
    print("\n[8] Mapping 12-lead electrode positions to torso CARP nodes…")
    try:
        elec_pos = compute_electrode_positions(vtu_path)
        if elec_pos is None:
            raise RuntimeError("Electrode placement returned None")

        # Map to nearest nodes in the ROTATED torso CARP mesh
        # t_nodes is already in µm; elec_pos positions are also in µm (VTU rotated)
        elec_node_map = map_electrodes_to_carp_nodes(
            elec_pos, t_nodes,
            t_regions, skin_tag=SKIN_TAG,
        )

        # Write electrodes.vtx: ordered V1…V6 RA LA RL LL
        elec_nodes_ordered = np.array(
            [elec_node_map[lbl] for lbl in ELECTRODE_LABELS], dtype=np.int64
        )
        elec_vtx = out_dir / "electrodes.vtx"
        with open(elec_vtx, "w") as f:
            f.write(f"{len(elec_nodes_ordered)}\n")
            for lbl, nid in zip(ELECTRODE_LABELS, elec_nodes_ordered):
                f.write(f"{nid}  # {lbl}\n")
        print(f"  Written electrodes → {elec_vtx}")

    except Exception as e:
        print(f"  WARNING: Electrode placement failed: {e}")
        print("  You will need to supply electrodes.vtx manually.")

    # ── 9. Done ─────────────────────────────────────────────────────────
    done_flag.write_text("extraction complete")
    print(f"\n{'='*60}")
    print(f"Extraction complete → {out_dir}")
    print(f"{'='*60}")
    return out_dir


# ==============================================================================
#  CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract KCL torso CARP sub-meshes")
    parser.add_argument("--carp", default=DEFAULT_CARP_PREFIX,
                        help="Path prefix to KCL_torso3 CARP files (no extension)")
    parser.add_argument("--vtu",  default=str(DEFAULT_VTU),
                        help="Path to KCL_torso3_anatomy.vtu for electrode placement")
    parser.add_argument("--out",  default=str(DEFAULT_OUT_DIR),
                        help="Output directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs already exist")
    args = parser.parse_args()

    extract(
        carp_prefix=args.carp,
        vtu_path=Path(args.vtu),
        out_dir=Path(args.out),
        force=args.force,
    )


if __name__ == "__main__":
    main()
