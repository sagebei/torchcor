"""
Inspect HEALTHY-TOTAL-BODY-CTS NIfTI download
==============================================
Run this first to see what files and label values are inside the zip.

Usage:
    python torchcor/ecg/inspect_tcia.py [path_to_zip]

Default zip path: C:/Users/bulli/Downloads/Healthy-Total-Body-CTs-...zip
"""

from __future__ import annotations
import sys
import os
import zipfile
from pathlib import Path

import numpy as np

# ── Locate zip ────────────────────────────────────────────────────────────────
def _find_zip(hint: Path) -> Path:
    if hint.is_file():
        return hint
    # Search Downloads for any matching zip
    downloads = Path("C:/Users/bulli/Downloads")
    candidates = sorted(downloads.glob("*Healthy*Body*CT*.zip")) + \
                 sorted(downloads.glob("*healthy*body*.zip")) + \
                 sorted(downloads.glob("*TCIA*.zip")) + \
                 sorted(downloads.glob("*NIfTI*.zip"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not find TCIA zip under {downloads}\n"
        "Pass the path explicitly:  python inspect_tcia.py  path/to/file.zip"
    )


def main():
    hint = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    zip_path = _find_zip(hint)
    print(f"Zip file : {zip_path}")
    print(f"Size     : {zip_path.stat().st_size / 1e6:.1f} MB\n")

    # ── List zip contents ─────────────────────────────────────────────────────
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

    print(f"Total entries: {len(names)}")

    # Group by extension
    from collections import defaultdict
    by_ext: dict[str, list[str]] = defaultdict(list)
    for n in names:
        ext = Path(n).suffix.lower()
        if n.endswith(".nii.gz"):
            ext = ".nii.gz"
        by_ext[ext].append(n)

    print("\nFile types:")
    for ext, files in sorted(by_ext.items(), key=lambda x: -len(x[1])):
        print(f"  {ext or '(no ext)':15s}  {len(files):5d} files")

    # Show directory tree (first 60 entries)
    print("\nTop-level structure (first 60 entries):")
    for n in names[:60]:
        print(f"  {n}")
    if len(names) > 60:
        print(f"  ... ({len(names) - 60} more)")

    # ── If NIfTI files found, inspect one label map ───────────────────────────
    nifti_files = by_ext.get(".nii.gz", []) + by_ext.get(".nii", [])
    seg_files = [f for f in nifti_files if any(
        kw in f.lower() for kw in ("seg", "label", "mask", "annot")
    )]
    if not seg_files:
        seg_files = nifti_files  # fall back to all NIfTI

    if seg_files:
        print(f"\nFound {len(seg_files)} segmentation NIfTI file(s).")
        sample = seg_files[0]
        print(f"\nInspecting: {sample}")

        # Extract the sample to a temp location
        extract_dir = zip_path.parent / "_tcia_inspect_tmp"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path) as z:
            z.extract(sample, extract_dir)

        sample_path = extract_dir / sample

        try:
            import nibabel as nib
            img = nib.load(str(sample_path))
            data = np.asarray(img.dataobj)
            print(f"  Shape  : {data.shape}")
            print(f"  Dtype  : {data.dtype}")
            print(f"  Voxel size: {img.header.get_zooms()}")

            labels, counts = np.unique(data, return_counts=True)
            print(f"\n  Unique label values ({len(labels)} total):")
            for lbl, cnt in zip(labels, counts):
                pct = 100 * cnt / data.size
                print(f"    {int(lbl):4d}   {cnt:>10,} voxels  ({pct:.2f}%)")

        except ImportError:
            print("  nibabel not installed — run:  pip install nibabel")
            print("  Then re-run this script to see label values.")

    # ── Check for spreadsheet ─────────────────────────────────────────────────
    sheets = [f for f in names if Path(f).suffix.lower() in (".xlsx", ".csv", ".xls")]
    if sheets:
        print(f"\nSpreadsheet(s) found:")
        for s in sheets:
            print(f"  {s}")
        # Extract and peek at it
        with zipfile.ZipFile(zip_path) as z:
            z.extract(sheets[0], extract_dir)
        sheet_path = extract_dir / sheets[0]
        try:
            import pandas as pd
            if sheet_path.suffix == ".csv":
                df = pd.read_csv(sheet_path)
            else:
                df = pd.read_excel(sheet_path)
            print(f"\n  Spreadsheet preview ({sheet_path.name}):")
            print(df.to_string(max_rows=50))
        except ImportError:
            print("  pandas/openpyxl not installed:  pip install pandas openpyxl")

    print("\nDone — use this label information to build the FEM mesh pipeline.")


if __name__ == "__main__":
    main()
