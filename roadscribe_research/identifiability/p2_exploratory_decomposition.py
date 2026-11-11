"""EXPLORATORY (not pre-registered): where does R3_cal's Spearman advantage over a_dec come from?

Spearman across locations is invariant to a per-vehicle gain, so the gain calibration cannot
change it. The advantage splits into (i) the index definition (spatial-band log-spectral level,
R3_uncal vs a_dec) and (ii) frequency-dependent channel removal (R3_cal vs R3_uncal).
Reads results/p2_heldout_indices.csv; block bootstrap as in P1/P2.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import p1_cross_vehicle as p1  # noqa: E402
from roadscribe.paired.agreement import block_indices  # noqa: E402


def sp(d, col):
    P = d.pivot(index="loc", columns="vehicle", values=col)
    return float(np.nanmean([stats.spearmanr(P[a], P[b], nan_policy="omit").statistic
                             for a, b in combinations(sorted(d.vehicle.unique()), 2)]))


def main():
    d = pd.read_csv(HERE / "results" / "p2_heldout_indices.csv")
    pt = {c: sp(d, c) for c in ("log_a_dec", "R3_uncal", "R3_cal")}
    rng = np.random.default_rng(p1.SEED + 4)
    by_route = {r: sorted(g["loc"].unique(), key=lambda s: int(s.split(":")[1])) for r, g in d.groupby("route")}
    groups = {k: g for k, g in d.groupby("loc")}
    defn, chan = [], []
    for _ in range(2000):
        parts = []
        for r, locs in by_route.items():
            for j, i in enumerate(block_indices(len(locs), p1.BLOCK, rng)):
                g = groups[locs[i]].copy()
                g["loc"] = f"{locs[i]}#{j}"
                parts.append(g)
        x = pd.concat(parts, ignore_index=True)
        s = {c: sp(x, c) for c in ("log_a_dec", "R3_uncal", "R3_cal")}
        defn.append(s["R3_uncal"] - s["log_a_dec"])
        chan.append(s["R3_cal"] - s["R3_uncal"])
    res = {"label": "EXPLORATORY, not pre-registered", "spearman": pt,
           "index_definition_effect_R3uncal_minus_adec": [pt["R3_uncal"] - pt["log_a_dec"], p1.ci(defn)],
           "channel_calibration_effect_R3cal_minus_R3uncal": [pt["R3_cal"] - pt["R3_uncal"], p1.ci(chan)]}
    (HERE / "results" / "p2_exploratory_decomposition.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
