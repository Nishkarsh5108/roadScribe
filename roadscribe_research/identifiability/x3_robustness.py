"""EXPLORATORY robustness checks on P1's two main findings (not pre-registered).

X3a  Is the speed-elasticity sign flip only surface composition (rough surfaces driven slowly)?
     Recompute pooled vs within-place slopes on stretches whose ground-truth surface is asphalt
     for all three vehicles, and within each surface type.
X3b  Placebo for the vehicle share: shuffle vehicle labels within each place (keeping speeds and
     values) 1,000 times; the vehicle share of log a_dec should collapse to ~0 if P1's 18% is real.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import p1_cross_vehicle as p1  # noqa: E402
from roadscribe.paired.agreement import variance_shares  # noqa: E402


def slopes(d: pd.DataFrame) -> dict:
    lv, ly = np.log(d["v_mean"].to_numpy()), np.log(d["a_wz"].to_numpy())
    pooled = float(np.polyfit(lv, ly, 1)[0])
    within = variance_shares(ly, d["loc"].to_numpy(), d["vehicle"].to_numpy(), lv)["slope_log_v_within"]
    return dict(n_locations=int(d["loc"].nunique()), gamma_pooled=pooled, gamma_within=within)


def main():
    d = p1.triplets(pd.read_csv(HERE / "results" / "p1_stretches_100m.csv"))
    out = {"label": "EXPLORATORY, not pre-registered"}
    surf = d.groupby("loc").gt_surface.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
    out["surface_counts"] = surf.value_counts().to_dict()
    out["X3a_by_surface"] = {}
    for s in sorted(surf.unique()):
        locs = surf[surf == s].index
        if len(locs) >= 15:
            out["X3a_by_surface"][s] = slopes(d[d["loc"].isin(locs)])
    out["X3a_all"] = slopes(d)
    rng = np.random.default_rng(p1.SEED + 5)
    y = np.log(d["a_dec"].to_numpy())
    lv = np.log(d["v_mean"].to_numpy())
    loc = d["loc"].to_numpy()
    veh = d["vehicle"].to_numpy()
    observed = variance_shares(y, loc, veh, lv)["vehicle"]
    order = np.argsort(loc, kind="stable")
    null = []
    for _ in range(1000):
        v2 = veh.copy()
        for g in np.unique(loc):
            idx = np.flatnonzero(loc == g)
            v2[idx] = rng.permutation(veh[idx])
        null.append(variance_shares(y, loc, v2, lv)["vehicle"])
    null = np.asarray(null)
    out["X3b_placebo"] = dict(observed_vehicle_share=observed, null_mean=float(null.mean()),
                              null_p99=float(np.percentile(null, 99)), p_value=float((np.sum(null >= observed) + 1) / (len(null) + 1)))
    (HERE / "results" / "x3_robustness.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
