"""Published participant results for benchmark 2 of Arostica et al.

Source
------
Zenodo record 14260459, ``results_time_curves/data``, downloaded into
``reference/``.  Nine participating codes at each mesh resolution, sampled at
10 ms over the one-second beat, giving the ``T = 101`` datapoints of
equation (21).

The discrepancy measure is the paper's RED, the same one benchmark 1 is scored
by and taken from the same ``figures.py``: the time average of the
displacement error relative to the mean over participants,

.. math::

    \\mathrm{RED}(p) = \\frac{1}{T}\\sum_{t_n}
        \\frac{\\|u(t_n, p) - \\bar{u}(t_n, p)\\|}{\\|\\bar{u}(t_n, p)\\|}

Benchmark 2 tracks three points rather than two, and its population is smaller
than benchmark 1's, so the two are not directly comparable to one another --
only each against its own participants.
"""

import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent/"reference"
#: Mesh resolution -> the filename stem the archive stores it under.
CASES = {"coarse": "biventricular_coarse", "fine": "biventricular_fine"}
#: Section 4: the three tracked material points [m].  ``p0`` and ``p1`` are
#: benchmark 1's; ``p2`` is on the right ventricle and is new here.
PROBES = ("p0", "p1", "p2")
#: The nine datasets the paper's own comparison uses, in its order.  The
#: archive also ships ``_p1`` variants, which are a separate fibre-space
#: comparison and would change the mean every RED is measured against.
TEAMS = ("carpentry", "ambit", "4c", "simula", "chimera", "cheart", "lifex",
         "simvascular", "comsol")


def participants(case):
    """Every published history for a case, as ``{name: {probe: (T, 3) array}}``."""
    out = {}
    for team in TEAMS:
        path = HERE/f"{CASES[case]}_group_{team}.pkl"
        if not path.exists():
            continue
        raw = pickle.load(open(path, "rb"))
        out[team] = {"time": np.asarray(raw["time"], dtype=float),
                     **{p: np.stack([raw["displacement"][p][c]
                                     for c in ("ux", "uy", "uz")], axis=-1)
                        for p in PROBES}}
    return out


def red(ours, mean):
    """Relative discrepancy of one history against the participant mean.

    ``ours`` and ``mean`` are ``(T, 3)``.  Samples where the mean vanishes --
    the start of the beat, where every participant is still at rest -- carry
    no information about relative error and are left out.
    """
    scale = np.linalg.norm(mean, axis=-1)
    keep = scale > 0.0
    return float(np.mean(np.linalg.norm(ours - mean, axis=-1)[keep]/scale[keep]))


def compare(case, time, history, complete=True):
    """Score our history for ``case`` against the published participants.

    ``history`` maps each probe to an ``(n, 3)`` array on our own time grid; it
    is resampled onto the published 10 ms grid before scoring, as the paper
    does for groups that ran a different step.

    Returns ``None`` rather than a score when the comparison would not mean
    what it says: a run that stopped early, or one that produced a non-finite
    value.  Interpolating a short history onto the full grid silently holds its
    last displacement to the end of the beat and scores that as agreement.
    """
    teams = participants(case)
    if len(teams) != len(TEAMS):
        missing = sorted(set(TEAMS) - set(teams))
        raise FileNotFoundError(
            f"reference case {case!r} is missing {missing}; the comparison "
            f"population must match the published one")
    grid = teams[TEAMS[0]]["time"]
    time = np.asarray(time, dtype=float)
    if not complete:
        return None
    # Finiteness before monotonicity: [-inf, ..., +inf] has positive
    # differences and would otherwise be resampled onto the published grid and
    # scored, spanning any interval asked of it.
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
        raise ValueError("our time history must be a finite one-dimensional array")
    if not np.all(np.diff(time) > 0):
        raise ValueError("our time history must be increasing")
    if not all(np.all(np.isfinite(history[p])) for p in PROBES):
        return None
    if time[-1] < grid[-1] - 1e-9 or time[0] > grid[0] + 1e-9:
        return None
    result = {"case": case, "n_participants": len(teams), "time": grid,
              "participants": sorted(teams)}
    for probe in PROBES:
        stack = np.stack([t[probe] for t in teams.values()])       # (teams, T, 3)
        mean = stack.mean(axis=0)
        ours = np.stack([np.interp(grid, time, history[probe][:, i])
                         for i in range(3)], axis=-1)
        result[probe] = {
            "mean": mean, "std": stack.std(axis=0), "ours": ours,
            "red": red(ours, mean),
            "participant_red": {n: red(t[probe], mean) for n, t in teams.items()},
        }
    return result


def summary(result) -> str:
    """One line per probe: our RED against the range the participants span."""
    lines = []
    for probe in PROBES:
        r = result[probe]
        theirs = np.array(list(r["participant_red"].values()))
        lines.append(f"  {probe}: RED {r['red']:.3f}   participants "
                     f"{theirs.min():.3f}-{theirs.max():.3f} "
                     f"(median {np.median(theirs):.3f}, n={len(theirs)})")
    return "\n".join(lines)
