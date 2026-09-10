"""Collect the P1-P3 benchmark results into one CSV for plotting.

Reads the JSON each runner writes (``p1/``, ``p2/``, ``p3/``) and emits
``torchcor_land.csv``: one row per plotted point, long format, tagged with the
Land et al. (2015) figure it belongs to.  Filter on ``problem`` and ``figure``.

    python export_csv.py            # after running p1.py, p2.py and p3.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "torchcor_land.csv"
COLUMNS = ["problem", "figure", "quantity", "mesh", "dofs", "finest", "layer",
           "component", "station", "parameter", "x_mm", "y_mm", "z_mm", "value",
           "unit", "reference_value", "notes"]

# What each problem's strain stations mean, for the ``notes`` column.
BEAM_STRAIN = ("pair 1 mm apart along x, labelled by the first point x=parameter",
               "pair 0.4 mm apart in y at (x=parameter, 0.5, 0.5)",
               "pair 0.4 mm apart in z at (x=parameter, 0.5, 0.5)")
VENTRICLE_STRAIN = {
    "long": "pair of neighbouring stations along the layer, labelled by the first (s=parameter)",
    "circ": "station paired with its image rotated by pi/10 about the axis",
    "trans": "across the wall: endo-mid at endo, mid-epi at epi, endo-epi at midwall"}


def rows_for(problem, data):
    results = [r for r in data["results"] if r.get("valid")]
    skipped = [r for r in data["results"] if not r.get("valid")]
    finest = max((r["displacement_dofs"] for r in results), default=None)
    ref = data.get("reference", {})
    guide = ref.get("strain_guide_percent") or {}
    rows = []

    def row(**kw):
        base = dict.fromkeys(COLUMNS, "")
        base.update(problem=problem, mesh="x".join(map(str, kw.pop("divisions"))),
                    dofs=kw.pop("dofs"), finest=kw.pop("dofs_finest"))
        base.update(kw)
        rows.append(base)

    for r in results:
        d, n = r["divisions"], r["displacement_dofs"]
        common = dict(divisions=d, dofs=n, dofs_finest=(n == finest))
        mid = np.asarray(r["midline_mm"])
        if problem == 1:
            row(figure="3", quantity="tip_z", value=r["tip_z"], unit="mm",
                reference_value=ref.get("tip_z_mm", ""),
                notes="deformed z of the point (10, 0.5, 1); reference = consensus", **common)
            for i, (x, y, z) in enumerate(mid):
                row(figure="4", quantity="midline", station=i, parameter=10.0*i/(len(mid)-1),
                    x_mm=x, y_mm=y, z_mm=z, unit="mm",
                    notes="deformed image of the line (x=parameter, 0.5, 0.5)", **common)
            for comp, note in zip(("x", "y", "z"), BEAM_STRAIN):
                vals, g = r["strains_percent"][comp], guide.get(comp, [])
                for i, v in enumerate(vals):
                    row(figure="5", quantity="strain", component=comp, station=i+1,
                        parameter=float(i), value=v, unit="percent",
                        reference_value=(g[i] if i < len(g) else ""), notes=note, **common)
        else:
            fig_apex, fig_mid, fig_strain = ("6", "7", "8") if problem == 2 else ("9", "10-11", "12")
            for layer in ("endo", "epi"):
                row(figure=fig_apex, quantity="apex_z", layer=layer, value=r[f"{layer}_apex_z"],
                    unit="mm", reference_value=(ref.get("apex_z_mm") or {}).get(layer, ""),
                    notes="deformed z of the apex on this surface", **common)
            for i, (x, y, z) in enumerate(mid):
                row(figure=fig_mid, quantity="midline", layer="mid", station=i,
                    parameter=i/(len(mid)-1), x_mm=x, y_mm=y, z_mm=z, unit="mm",
                    notes="deformed midwall line t=0.5, v=0, apex (s=0) to base (s=1)", **common)
            if problem == 3:
                row(figure="10-11", quantity="twist_peak_y", layer="mid", value=r["peak_y_mm"],
                    unit="mm", reference_value=", ".join(map(str, ref.get("approximate_peak_y_mm", []))),
                    notes="largest signed y of the deformed midwall line", **common)
            s = np.asarray(r["strain_stations_s"])
            for key, vals in r["strains_percent"].items():
                layer, comp = key.split("/")
                g = guide.get(key, [])
                for i, v in enumerate(vals):
                    row(figure=fig_strain, quantity="strain", layer=layer, component=comp,
                        station=i+1, parameter=float(s[i]), value=v, unit="percent",
                        reference_value=(g[i] if i < len(g) else ""),
                        notes=VENTRICLE_STRAIN[comp], **common)
    return rows, skipped


def main():
    sources = {1: HERE/"p1"/"beam_results.json", 2: HERE/"p2"/"ventricle_results.json",
               3: HERE/"p3"/"p3_results.json"}
    rows = []
    for problem, path in sources.items():
        if not path.exists():
            print(f"problem {problem}: {path.name} not found, run p{problem}.py first")
            continue
        data = json.load(open(path))
        got, skipped = rows_for(problem, data)
        rows += got
        meshes = sorted({r["mesh"] for r in got}, key=lambda m: tuple(map(int, m.split("x"))))
        print(f"problem {problem}: {len(got):5d} rows from meshes {meshes}"
              + (f"; {len(skipped)} invalid solve(s) skipped" if skipped else ""))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
