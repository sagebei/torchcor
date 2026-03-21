"""
CT Segmentation → FEM Mesh Pipeline
=====================================
Converts a HEALTHY-TOTAL-BODY-CTS NIfTI segmentation file into a tetrahedral
FEM mesh suitable for 12-lead ECG forward simulation with TorchCor.

Workflow
--------
  1. Load NIfTI label volume (nibabel)
  2. Crop to neck–umbilicus z-range (auto-detected from labels)
  3. Downsample to target voxel spacing (~3 mm isotropic)
  4. Map TCIA labels → ECG conductivity groups (fewer, meaningful regions)
  5. Extract outer body surface via marching cubes (scikit-image)
  6. Clean and smooth surface (pyvista)
  7. Fill interior with quality tetrahedral mesh (tetgen via pyvista)
  8. Assign region tags: for each tet centroid, look up label in voxel data
  9. Export: VTU for ParaView, CARP (.pts/.elem) for TorchCor KCLTorsoECG

TCIA → ECG region mapping
--------------------------
  ECG   Name                    Conductivity   TCIA labels
  ───────────────────────────────────────────────────────────────────────────
   1    Inner body (bulk)         0.247 S/m    0 (background inside body),
                                               adrenal glands, thyroid,
                                               pancreas, stomach-like organs
   2    Subcutaneous fat          0.040 S/m    34 (subcutaneous-fat)
   3    Bones                     0.050 S/m    13–32 (all bone labels)
   4    Skeletal muscle           0.247 S/m    33, 36 (muscle, psoas)
   5    Visceral fat              0.040 S/m    35 (torso-fat)
   6    Lungs                     0.071 S/m    12
   7    Heart                     anisotropic  5
   8    Blood / great vessels     0.667 S/m    2 (aorta), 11 (IVC)
   9    Liver                     0.167 S/m    7
  10    Kidneys                   0.167 S/m    6
  11    Spleen                    0.100 S/m    9
  12    Bladder                   0.100 S/m    3

Usage
-----
    python torchcor/ecg/ct_to_fem.py [nifti_path] [out_dir]

    nifti_path   — path to a *-seg.nii.gz or *.nii.gz file (default: auto-find)
    out_dir      — output directory (default: same as nifti_path parent)

Requires
--------
    pip install nibabel scikit-image pyvista tetgen meshio scipy
    (numpy already present)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── TCIA label → ECG region tag ───────────────────────────────────────────────

# All bone TCIA labels (13–32)
_BONE_LABELS = set(range(13, 33))

# Map every TCIA integer label to an ECG region tag (1-based, sequential)
TCIA_TO_ECG: dict[int, int] = {
    # Background voxels inside the body will be assigned tag 1 (inner body)
    # at the centroid-lookup stage; label 0 = air outside → excluded.
    1:  1,   # adrenal glands  → inner body
    2:  8,   # aorta           → blood/vessels
    3:  12,  # bladder         → bladder
    4:  1,   # brain           → inner body  (outside ROI, won't appear)
    5:  7,   # heart           → heart (anisotropic)
    6:  10,  # kidneys         → kidneys
    7:  9,   # liver           → liver
    8:  1,   # pancreas        → inner body
    9:  11,  # spleen          → spleen
    10: 1,   # thyroid         → inner body
    11: 8,   # IVC             → blood/vessels
    12: 6,   # lung            → lungs
    33: 4,   # skeletal muscle → muscle
    34: 2,   # subcutaneous fat
    35: 5,   # visceral fat
    36: 4,   # psoas           → muscle
}
# All bones → tag 3
for _bl in _BONE_LABELS:
    TCIA_TO_ECG[_bl] = 3

# ECG tag → human-readable name
ECG_TAG_NAMES: dict[int, str] = {
    1:  "Inner body (bulk)",
    2:  "Subcutaneous fat",
    3:  "Bones",
    4:  "Skeletal muscle",
    5:  "Visceral fat",
    6:  "Lungs",
    7:  "Heart",
    8:  "Blood / great vessels",
    9:  "Liver",
    10: "Kidneys",
    11: "Spleen",
    12: "Bladder",
}

# ECG tag → isotropic conductivity in S/m  (tag 7 heart uses bidomain, handled separately)
ECG_CONDUCTIVITY: dict[int, float] = {
    1:  0.247,
    2:  0.040,
    3:  0.050,
    4:  0.247,
    5:  0.040,
    6:  0.0714,
    7:  0.247,   # placeholder — solver replaces with bidomain anisotropic
    8:  0.667,
    9:  0.167,
    10: 0.167,
    11: 0.100,
    12: 0.100,
}


# ── NIfTI loading & preprocessing ────────────────────────────────────────────

def load_nifti(path: Path):
    """Return (label_array uint8, affine 4×4, voxel_size_mm tuple)."""
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asarray(img.dataobj).astype(np.int16)
    affine = img.affine
    voxel_size = tuple(float(v) for v in img.header.get_zooms()[:3])
    print(f"  Loaded : {path.name}")
    print(f"  Shape  : {data.shape}  dtype={data.dtype}")
    print(f"  Voxels : {voxel_size[0]:.2f} × {voxel_size[1]:.2f} × {voxel_size[2]:.2f} mm")
    return data, affine, voxel_size


def detect_neck_umbilicus(data: np.ndarray, voxel_size_z: float) -> tuple[int, int]:
    """
    Auto-detect z-slice range covering neck to umbilicus.

    Uses label presence heuristics:
      - Superior (neck): first slice with thyroid (10) or clavicle (14), minus 20 mm margin
      - Inferior (umbilicus): last slice with liver (7) plus 30 mm margin
        (liver ends roughly at umbilicus level)

    Falls back to middle 60% of volume if labels are missing.
    """
    nz = data.shape[2]
    margin_sup = int(20 / voxel_size_z)   # 20 mm above first thyroid/clavicle
    margin_inf = int(30 / voxel_size_z)   # 30 mm below liver

    # Find z slices where specific labels exist
    def _first_last(label: int):
        mask = (data == label).any(axis=(0, 1))
        idxs = np.where(mask)[0]
        return (int(idxs[0]), int(idxs[-1])) if len(idxs) else (None, None)

    z_thyroid  = _first_last(10)   # thyroid
    z_clavicle = _first_last(14)   # clavicle
    z_liver    = _first_last(7)    # liver

    sup_candidates = [z for z in [z_thyroid[0], z_clavicle[0]] if z is not None]
    z_sup = max(0, min(sup_candidates) - margin_sup) if sup_candidates else int(0.30 * nz)

    if z_liver[1] is not None:
        z_inf = min(nz - 1, z_liver[1] + margin_inf)
    else:
        z_inf = int(0.70 * nz)

    print(f"  Crop z : slices {z_sup}–{z_inf} of {nz}  "
          f"({(z_inf - z_sup) * voxel_size_z / 10:.1f} cm coverage)")
    return z_sup, z_inf


def downsample(data: np.ndarray, voxel_size: tuple,
               target_mm: float = 3.0) -> tuple[np.ndarray, tuple]:
    """
    Downsample label volume to ~target_mm isotropic using nearest-neighbour.
    Returns (downsampled_array, new_voxel_size).
    """
    from scipy.ndimage import zoom
    factors = tuple(v / target_mm for v in voxel_size)
    # Nearest-neighbour to preserve integer labels
    data_ds = zoom(data, factors, order=0, prefilter=False)
    new_vox = tuple(target_mm for _ in voxel_size)
    print(f"  Downsampled: {data.shape} → {data_ds.shape}  "
          f"(voxel ≈ {target_mm:.1f} mm isotropic)")
    return data_ds.astype(np.int16), new_vox


# ── Surface extraction & meshing ─────────────────────────────────────────────

def extract_body_surface(data: np.ndarray, voxel_size: tuple,
                         smooth_iterations: int = 20) -> "pv.PolyData":
    """
    Extract the outer body surface (all non-zero labels) via marching cubes,
    then smooth and clean with pyvista.
    """
    import pyvista as pv
    from skimage.measure import marching_cubes
    from skimage.filters import gaussian

    print("  Extracting outer body surface …")
    # Binary volume: 1 = body, 0 = air
    body = (data > 0).astype(np.float32)
    # Slight Gaussian blur to reduce staircasing artefacts at boundary
    body_smooth = gaussian(body, sigma=1.0)

    verts, faces, _, _ = marching_cubes(
        body_smooth, level=0.5,
        spacing=voxel_size,
        allow_degenerate=False,
    )
    print(f"    Marching cubes: {len(verts):,} vertices, {len(faces):,} faces")

    # Convert to pyvista PolyData
    n_faces = faces.shape[0]
    face_arr = np.hstack([np.full((n_faces, 1), 3), faces]).ravel()
    mesh = pv.PolyData(verts.astype(np.float32), face_arr)

    # Clean and smooth
    mesh = mesh.clean()
    mesh = mesh.smooth(n_iter=smooth_iterations, relaxation_factor=0.1)
    mesh = mesh.triangulate()
    print(f"    After clean/smooth: {mesh.n_points:,} pts, {mesh.n_faces:,} faces")
    return mesh


def generate_tet_mesh(surface: "pv.PolyData",
                      max_volume_mm3: float = 500.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill the interior of surface with quality tetrahedra using TetGen.

    Parameters
    ----------
    surface       : watertight PolyData outer shell
    max_volume_mm3: max tet volume in mm³ (controls mesh density)

    Returns
    -------
    nodes (N,3), tets (E,4) — both numpy arrays
    """
    import tetgen
    print(f"  Generating tet mesh (max vol={max_volume_mm3:.0f} mm³) …")
    tet = tetgen.TetGen(surface)
    # q = quality (radius/edge ratio), a = max volume
    tet.tetrahedralize(order=1, mindihedral=10.0, quality=True,
                       minratio=1.5, verbose=0,
                       switches=f"pq1.5a{max_volume_mm3:.1f}YO")
    nodes = tet.node.copy()
    tets  = tet.elem.copy()
    print(f"    Tet mesh: {len(nodes):,} nodes, {len(tets):,} tetrahedra")
    return nodes, tets


# ── Label assignment ──────────────────────────────────────────────────────────

def assign_ecg_tags(
    nodes: np.ndarray,
    tets: np.ndarray,
    label_vol: np.ndarray,
    voxel_size: tuple,
    origin: np.ndarray,
) -> np.ndarray:
    """
    For each tet centroid, look up the TCIA label at that voxel position,
    then map to ECG region tag.

    Parameters
    ----------
    nodes     : (N,3) float mm coordinates
    tets      : (E,4) int node indices
    label_vol : cropped, downsampled label array
    voxel_size: (dx, dy, dz) in mm
    origin    : (3,) mm coordinate of voxel [0,0,0] corner

    Returns
    -------
    ecg_tags : (E,) int32 array of ECG region tags
    """
    print("  Assigning ECG region tags …")
    centroids = nodes[tets].mean(axis=1)   # (E,3)

    # Convert mm coordinates to voxel indices
    dx, dy, dz = voxel_size
    ox, oy, oz = origin
    ix = np.clip(((centroids[:, 0] - ox) / dx).astype(int), 0, label_vol.shape[0] - 1)
    iy = np.clip(((centroids[:, 1] - oy) / dy).astype(int), 0, label_vol.shape[1] - 1)
    iz = np.clip(((centroids[:, 2] - oz) / dz).astype(int), 0, label_vol.shape[2] - 1)

    tcia_labels = label_vol[ix, iy, iz]   # (E,) TCIA label per tet

    # Vectorised map: TCIA label → ECG tag (default to inner body = 1)
    ecg_tags = np.ones(len(tets), dtype=np.int32)
    for tcia_val, ecg_val in TCIA_TO_ECG.items():
        ecg_tags[tcia_labels == tcia_val] = ecg_val

    # Print summary
    unique, counts = np.unique(ecg_tags, return_counts=True)
    print("  Tag distribution:")
    for t, c in zip(unique, counts):
        name = ECG_TAG_NAMES.get(int(t), "?")
        sig  = ECG_CONDUCTIVITY.get(int(t), 0.0)
        print(f"    {t:3d}  {name:<30s}  {sig:.4f} S/m  {c:>8,} tets")

    return ecg_tags


# ── Export ────────────────────────────────────────────────────────────────────

def export_vtu(nodes, tets, tags, out_path: Path) -> None:
    import meshio
    mesh = meshio.Mesh(
        points=nodes.astype(np.float64),
        cells=[("tetra", tets)],
        cell_data={"RegionTag": [tags]},
    )
    mesh.write(str(out_path))
    print(f"  VTU → {out_path}  ({len(tets):,} cells)")


def export_carp(nodes, tets, tags, stem: Path) -> None:
    """Write CARP .pts and .elem files for use with KCLTorsoECG."""
    pts_path  = stem.with_suffix(".pts")
    elem_path = stem.with_suffix(".elem")

    # .pts
    with open(pts_path, "w") as f:
        f.write(f"{len(nodes)}\n")
        for x, y, z in nodes:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    # .elem  (TetGen uses 0-based; CARP also 0-based)
    with open(elem_path, "w") as f:
        f.write(f"{len(tets)}\n")
        for (n0, n1, n2, n3), tag in zip(tets, tags):
            f.write(f"Tt {n0} {n1} {n2} {n3} {tag}\n")

    print(f"  CARP → {pts_path.name} + {elem_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(nifti_path: Path, out_dir: Path, target_mm: float = 3.0,
        max_vol_mm3: float = 500.0) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = nifti_path.name.replace(".nii.gz", "").replace(".nii", "")

    # 1 — Load
    print("\n[1/6] Loading NIfTI …")
    data, affine, vox = load_nifti(nifti_path)

    # 2 — Crop to neck–umbilicus
    print("\n[2/6] Detecting neck–umbilicus crop …")
    z_sup, z_inf = detect_neck_umbilicus(data, vox[2])
    data_crop = data[:, :, z_sup:z_inf + 1]
    # Adjust origin for the crop (z only)
    origin = affine[:3, 3].copy()
    origin[2] += z_sup * vox[2]
    print(f"  Cropped shape: {data_crop.shape}")

    # 3 — Downsample
    print("\n[3/6] Downsampling to target resolution …")
    data_ds, vox_ds = downsample(data_crop, vox, target_mm=target_mm)
    # Recompute origin in downsampled space (origin stays the same; voxel size changes)

    # 4 — Extract outer surface
    print("\n[4/6] Extracting body surface …")
    surface = extract_body_surface(data_ds, vox_ds)

    # 5 — Tetrahedral mesh
    print("\n[5/6] Generating tetrahedral mesh …")
    nodes, tets = generate_tet_mesh(surface, max_volume_mm3=max_vol_mm3)

    # 6 — Assign ECG region tags
    print("\n[6/6] Assigning region tags …")
    # Origin of the downsampled voxel grid (corner of voxel [0,0,0])
    ds_origin = np.array(origin)
    ecg_tags = assign_ecg_tags(nodes, tets, data_ds, vox_ds, ds_origin)

    # 7 — Export
    print("\nExporting …")
    vtu_path  = out_dir / f"{stem_name}_ecg.vtu"
    carp_stem = out_dir / stem_name
    export_vtu(nodes, tets, ecg_tags, vtu_path)
    export_carp(nodes, tets, ecg_tags, carp_stem)

    print(f"\nDone!")
    print(f"  ParaView : {vtu_path}")
    print(f"  TorchCor : {carp_stem}.pts / .elem")
    print("\nNext step: open the VTU in ParaView, colour by RegionTag.")
    print("Then run KCLTorsoECG with this mesh directory.")


if __name__ == "__main__":
    import zipfile

    downloads = Path("C:/Users/bulli/Downloads")

    # Auto-find a NIfTI file (extract from zip if needed)
    nifti_path = None
    if len(sys.argv) > 1:
        nifti_path = Path(sys.argv[1])
    else:
        # Look for already-extracted NIfTI
        extracted = list(downloads.rglob("*Healthy-Total-Body-CTs-0??.nii.gz"))
        if extracted:
            nifti_path = extracted[0]
        else:
            # Extract first NIfTI from zip
            zips = sorted(downloads.glob("*Healthy*Body*.zip"))
            if not zips:
                print("No TCIA zip found in Downloads. Pass path explicitly.")
                sys.exit(1)
            extract_dir = downloads / "_tcia_nifti"
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zips[0]) as z:
                niftis = [f for f in z.namelist()
                          if f.endswith(".nii.gz") and "__MACOSX" not in f]
                if not niftis:
                    print("No .nii.gz found in zip.")
                    sys.exit(1)
                print(f"Extracting: {niftis[0]}")
                z.extract(niftis[0], extract_dir)
                nifti_path = extract_dir / niftis[0]

    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else \
              Path("C:/Users/bulli/cardiac_data/tcia_torso")

    print(f"Input  : {nifti_path}")
    print(f"Output : {out_dir}")

    run(nifti_path, out_dir, target_mm=3.0, max_vol_mm3=500.0)
