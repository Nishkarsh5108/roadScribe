"""EXPLORATORY (not pre-registered): does the vehicle share of road claims depend on sensor placement?

Same trips, same alignment, same analysis as P1 (no bootstrap), for the dashboard, above-suspension
and below-suspension IMUs of the left logger. Physics predicts the vehicle (channel) share is lowest
below the suspension, where the sensor sees the wheel rather than the body.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import p1_cross_vehicle as p1  # noqa: E402
from roadscribe.io.pvs import ROUTES  # noqa: E402


def main():
    out = {"label": "EXPLORATORY, not pre-registered"}
    for placement in ("dashboard", "above_suspension", "below_suspension"):
        tabs = []
        for route, trips in ROUTES.items():
            t, _ = p1.load_route(route, trips, placement=placement, bins=(100.0,))
            tabs.append(t[100.0])
        d = p1.triplets(pd.concat(tabs, ignore_index=True))
        r = p1.analyse(d, placement, boot=False)
        out[placement] = {k: r[k] for k in ("n_locations", "kappa_felt_class", "kappa_road_class", "spearman_a_dec",
                                             "gamma_pooled", "gamma_within", "lovo_acc_a_dec", "lovo_acc_a_v")}
        out[placement]["vehicle_share_log_a_dec"] = r["variance_shares"]["log_a_dec"]["vehicle"]
        out[placement]["residual_share_log_a_dec"] = r["variance_shares"]["log_a_dec"]["residual"]
        print(placement, json.dumps(out[placement], default=float), flush=True)
    (HERE / "results" / "x1_placements.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
