"""
Export KCL torso mesh to ParaView VTU files
============================================
Produces two files from the KCL CARP mesh (.pts / .elem):

  KCL_torso3_ICD.vtu     — full mesh, hardware included, tags remapped so
                            true anatomy is sequential (1–27) and ICD device
                            components follow (28–35).

  KCL_torso3.vtu         — anatomy only; ICD hardware elements removed.

Tag remapping (original KCL → new sequential)
----------------------------------------------
  1–25  : unchanged  (inner body through LV blood pool)
  34    → 26          RV wall (myocardium)
  35    → 27          LV wall (myocardium)
  26    → 28          Coronary sinus coil
  27    → 29          SVC coil
  28    → 30          RV apical coil
  29    → 31          RV septal coil
  30    → 32          Subcutaneous ICD lead
  31    → 33          Left can
  32    → 34          Right can
  33    → 35          Subcutaneous ICD can

Usage
-----
    python export_paraview_vtu.py [mesh_dir] [out_dir]

    mesh_dir  — directory containing KCL_torso3.pts / .elem (default: auto-detect)
    out_dir   — where to write the .vtu files (default: same as mesh_dir)

Requires: numpy, meshio  (pip install meshio)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── Tag remapping table ───────────────────────────────────────────────────────
# Maps original KCL tag → new sequential tag.
# Tags 1–25 are identity; ventricular walls move from 34/35 → 26/27;
# old hardware tags 26–33 shift to 28–35.

TAG_REMAP: dict[int, int] = {
    **{t: t for t in range(1, 26)},  # 1–25 unchanged
    34: 26,   # RV wall
    35: 27,   # LV wall
    26: 28,   # coronary sinus coil
    27: 29,   # SVC coil
    28: 30,   # RV apical coil
    29: 31,   # RV septal coil
    30: 32,   # subcutaneous ICD lead
    31: 33,   # left can
    32: 34,   # right can
    33: 35,   # subcutaneous ICD can
}

# New tag numbers that correspond to ICD/device hardware (post-remap)
ICD_TAGS_NEW = set(range(28, 36))

# Human-readable names for the new sequential tags
TAG_NAMES: dict[int, str] = {
    1:  "Inner body (bath)",
    2:  "Skin",
    3:  "Bones",
    4:  "Kidneys",
    5:  "Liver",
    6:  "Stomach",
    7:  "Spleen",
    8:  "Lungs",
    9:  "SVC blood pool",
    10: "Superior vena cava",
    11: "L. pulmonary artery blood",
    12: "L. pulmonary artery",
    13: "Aorta",
    14: "Aorta blood pool",
    15: "SVC plane",
    16: "Mitral valve",
    17: "Pulmonary valve",
    18: "Aortic valve",
    19: "Tricuspid valve",
    20: "L. atrial wall",
    21: "L. atrial blood pool",
    22: "R. atrial blood pool",
    23: "R. atrial wall",
    24: "RV blood pool",
    25: "LV blood pool",
    26: "RV wall (myocardium)",
    27: "LV wall (myocardium)",
    28: "Coronary sinus coil",
    29: "SVC coil",
    30: "RV apical coil",
    31: "RV septal coil",
    32: "Subcutaneous ICD lead",
    33: "Left can",
    34: "Right can",
    35: "Subcutaneous ICD can",
}


# ── CARP reader ───────────────────────────────────────────────────────────────

def _find_mesh(base_dir: Path) -> tuple[Path, Path]:
    """Find .pts and .elem files anywhere under base_dir."""
    pts_files  = list(base_dir.rglob("*.pts"))
    elem_files = list(base_dir.rglob("*.elem"))
    if not pts_files:
        raise FileNotFoundError(f"No .pts file found under {base_dir}")
    if not elem_files:
        raise FileNotFoundError(f"No .elem file found under {base_dir}")
    # Prefer the KCL_torso naming if multiple found
    def _score(p):
        return 0 if "kcl_torso" in p.stem.lower() else 1
    pts_files.sort(key=_score)
    elem_files.sort(key=_score)
    return pts_files[0], elem_files[0]


def _read_pts(path: Path) -> np.ndarray:
    """Read CARP .pts → (N, 3) float32 node coordinates."""
    with open(path) as f:
        n = int(f.readline())
        coords = np.fromstring(f.read(), sep=" ", dtype=np.float32).reshape(n, 3)
    print(f"  Nodes    : {n:,}  ({path.name})")
    return coords


def _read_elem(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read CARP .elem → connectivity (E, 4) int32, region tags (E,) int32.
    Only tetrahedral elements ('Tt') are returned.
    """
    connectivity = []
    tags = []
    with open(path) as f:
        n_total = int(f.readline())
        for line in f:
            parts = line.split()
            if not parts:
                continue
            etype = parts[0]
            if etype == "Tt":
                # Tt  n0 n1 n2 n3  tag
                connectivity.append([int(p) for p in parts[1:5]])
                tags.append(int(parts[5]))
    connectivity = np.array(connectivity, dtype=np.int32)
    tags = np.array(tags, dtype=np.int32)
    print(f"  Tet elems: {len(tags):,}  ({path.name})")
    return connectivity, tags


# ── VTU writer ────────────────────────────────────────────────────────────────

def _write_vtu(
    out_path: Path,
    nodes: np.ndarray,
    connectivity: np.ndarray,
    tags: np.ndarray,
) -> None:
    """Write an unstructured VTU file (VTK XML, ASCII) readable by ParaView."""
    import meshio

    cells = [("tetra", connectivity)]
    cell_data = {"RegionTag": [tags]}

    mesh = meshio.Mesh(
        points=nodes.astype(np.float64),
        cells=cells,
        cell_data=cell_data,
    )
    mesh.write(str(out_path))
    print(f"  Written  : {out_path}  ({len(tags):,} cells)")


# ── Main ──────────────────────────────────────────────────────────────────────

def export(mesh_dir: str | Path, out_dir: str | Path | None = None) -> None:
    mesh_dir = Path(mesh_dir)
    out_dir  = Path(out_dir) if out_dir else mesh_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Locate and read source files
    pts_path, elem_path = _find_mesh(mesh_dir)
    stem = pts_path.stem   # e.g. "KCL_torso3"

    print(f"\nReading mesh: {stem}")
    nodes        = _read_pts(pts_path)
    connectivity, orig_tags = _read_elem(elem_path)

    # ── Apply tag remapping
    new_tags = np.vectorize(TAG_REMAP.get)(orig_tags, orig_tags).astype(np.int32)

    # Summary of tag distribution
    unique, counts = np.unique(new_tags, return_counts=True)
    print("\n  Tag distribution (new numbering):")
    for t, c in zip(unique, counts):
        name = TAG_NAMES.get(int(t), "unknown")
        flag = " ← ICD hardware" if int(t) in ICD_TAGS_NEW else ""
        print(f"    {t:3d}  {name:<35s}  {c:>8,} elements{flag}")

    # ── Export 1: full mesh with ICD hardware (remapped tags)
    out_icd = out_dir / f"{stem}_ICD.vtu"
    print(f"\nExporting ICD model (all tissues) → {out_icd.name}")
    _write_vtu(out_icd, nodes, connectivity, new_tags)

    # ── Export 2: anatomy only — hardware elements reassigned to inner body (tag 1)
    # We keep ALL elements so the mesh has no voids or discontinuities.
    # ICD hardware occupied physical space that anatomically would be subcutaneous
    # tissue / body fluid; tag 1 (inner body, 0.247 S/m) is the correct stand-in.
    tags_anat = new_tags.copy()
    icd_mask = np.isin(tags_anat, list(ICD_TAGS_NEW))
    tags_anat[icd_mask] = 1   # reassign to inner body (bath)

    out_anat = out_dir / f"{stem}.vtu"
    print(f"\nExporting anatomy-only model (ICD elements → inner body) → {out_anat.name}")
    _write_vtu(out_anat, nodes, connectivity, tags_anat)

    print(f"\nDone.  Reassigned {icd_mask.sum():,} ICD element(s) to inner body (tag 1).")
    print(f"\nOpen in ParaView:")
    print(f"  {out_icd}")
    print(f"  {out_anat}")
    print("\nColour by RegionTag to see tissue regions.")
    print("Use Threshold filter (RegionTag 26–27) to isolate ventricular walls.")


if __name__ == "__main__":
    default_mesh_dir = Path("C:/Users/bulli/cardiac_data/kcl_torso/KCL_torso3")

    mesh_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_mesh_dir
    out_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else mesh_dir

    if not mesh_dir.exists():
        print(f"Mesh directory not found: {mesh_dir}")
        print("Usage: python export_paraview_vtu.py [mesh_dir] [out_dir]")
        sys.exit(1)

    export(mesh_dir, out_dir)
