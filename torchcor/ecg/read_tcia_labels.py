"""Read all sheets from the TCIA label spreadsheet."""
import zipfile
from pathlib import Path
import pandas as pd

downloads = Path("C:/Users/bulli/Downloads")
zips = sorted(downloads.glob("*Healthy*Body*.zip")) + sorted(downloads.glob("*NIfTI*.zip"))
zip_path = zips[0]

extract_dir = downloads / "_tcia_inspect_tmp"
extract_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path) as z:
    sheets = [f for f in z.namelist() if f.endswith(".xlsx") and "__MACOSX" not in f]
    z.extract(sheets[0], extract_dir)

sheet_path = extract_dir / sheets[0]
print(f"File: {sheet_path}\n")

xl = pd.ExcelFile(sheet_path)
print(f"Sheet names: {xl.sheet_names}\n")

for sheet in xl.sheet_names:
    print(f"{'='*60}")
    print(f"Sheet: '{sheet}'")
    print(f"{'='*60}")
    # Try with no header first to see raw content
    df_raw = pd.read_excel(sheet_path, sheet_name=sheet, header=None)
    print(df_raw.to_string())
    print()
