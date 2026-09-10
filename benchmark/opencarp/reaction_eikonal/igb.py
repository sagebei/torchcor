"""Read openCARP IGB output, and save this run's results for comparison.

openCARP writes an IGB file as a 1024-byte ASCII header -- space-separated
``key:value`` pairs -- followed by the raw samples, ``x`` values per frame and
``t`` frames.  Only the fields this benchmark needs are interpreted.

Run as a script it converts everything ``run_case_AF.slurm`` produced in
``biv/`` into ``results/``: the activation times and the transmembrane voltage
as ``.npy``, plus a small JSON summary of the run.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TYPES = {"float": np.float32, "double": np.float64, "short": np.int16}


def read_igb(path):
    """Return ``(values, header)`` -- values shaped ``(t, x)``, header a dict."""
    with open(path, "rb") as fh:
        header = fh.read(1024).decode("ascii", "replace")
        # Later keys win: openCARP repeats ``dim_x`` with a different meaning.
        meta = dict(tok.split(":", 1) for tok in header.split() if ":" in tok)
        dtype = TYPES.get(meta.get("type", "float"), np.float32)
        values = np.frombuffer(fh.read(), dtype=dtype)
    # The frame count comes from the data, not from the header: openCARP
    # stamps the simulation's frame count on the activation-time file even
    # though it holds a single frame.
    x = int(meta["x"])
    return values.reshape(-1, x), meta


def summarise(name, values, meta):
    """The few numbers worth eyeballing before a full comparison."""
    return {"file": name, "nodes": values.shape[1], "frames": values.shape[0],
            "unit_t": meta.get("unites_t", ""), "unit": meta.get("unites", ""),
            "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean())}


def main():
    # RE+ writes to biv_rep/ and is summarised into results_rep/.
    re_plus = "--re-plus" in sys.argv
    src = HERE/("biv_rep" if re_plus else "biv")
    out = HERE/("results_rep" if re_plus else "results")
    out.mkdir(exist_ok=True)
    summary = {}
    for igb in sorted(src.glob("*.igb")):
        values, meta = read_igb(igb)
        np.save(out/f"{igb.stem}.npy", values)
        summary[igb.stem] = summarise(igb.name, values, meta)
        print(f"{igb.name}: {values.shape} -> {out.name}/{igb.stem}.npy")
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out.name}/summary.json")


if __name__ == "__main__":
    main()
