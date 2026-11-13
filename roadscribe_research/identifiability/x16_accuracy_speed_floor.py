"""EXPLORATORY (added 2026-11-13, second review): road-quality accuracy without RoadScribe's 5 m/s floor.

Repeats P7-H5 (calibration leave-one-route-out; cut-points fitted without the test route and the test vehicle;
the test vehicle's gain comes from the other routes) on all places all three vehicles drove at mean speed of
at least 0, 3 and 5 m/s. Reports accuracy, balanced accuracy, block-bootstrap intervals and confusion matrices.
Run from the repo root after p1_cross_vehicle.py.
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
import p7_review_checks as p7  # noqa: E402
from roadscribe.paired.agreement import apply_cutpoints, fit_cutpoints  # noqa: E402


def confusion(y, p):
    m = np.zeros((3, 3), int)
    np.add.at(m, (y, p), 1)
    return m.tolist()


def main():
    left = pd.read_csv(HERE / "results" / "p1_stretches_100m.csv")
    out = {"label": "EXPLORATORY, not pre-registered", "scheme": "route held out for calibration and cut-points; cut-points also without the test vehicle"}
    v0 = p1.V_MIN
    for vmin in (0.0, 3.0, 5.0):
        p1.V_MIN = vmin
        prim = p1.triplets(left)
        res = p7.correctness(left, prim)
        # confusion matrices for the route-and-vehicle-held-out predictions
        cal = p7.loro(left)
        d = prim.assign(a_cal=p7.apply_cal(prim, cal))
        d = d[np.isfinite(d.gt_quality)].reset_index(drop=True)
        y = d.gt_quality.astype(int).to_numpy()
        conf = {}
        for name, x in {"a_dec": np.log(d.a_dec.to_numpy()), "a_cal": np.log(d.a_cal.to_numpy())}.items():
            pred = np.full(len(d), -1)
            for r in p7.ROUTES:
                for c in (1, 2, 3):
                    tr = ((d.route != r) & (d.vehicle != c)).to_numpy()
                    te = ((d.route == r) & (d.vehicle == c)).to_numpy()
                    pred[te] = apply_cutpoints(x[te], fit_cutpoints(x[tr], y[tr]))
            conf[name] = confusion(y, pred)
        rv = res["route_and_vehicle_out"]
        out[str(vmin)] = dict(n_places=int(prim["loc"].nunique()), n_rows=res["n_rows"], class_counts=res["class_counts"],
                              majority=res["majority_baseline"],
                              accuracy={k: rv[k]["accuracy"] for k in ("a_v", "a_dec", "a_cal")},
                              balanced={k: rv[k]["balanced_accuracy"] for k in ("a_v", "a_dec", "a_cal")},
                              ci95=rv["ci95"], cal_minus_dec=rv["cal_minus_dec"],
                              confusion_rows_true_cols_pred=conf)
        print(vmin, json.dumps({k: out[str(vmin)][k] for k in ("n_places", "class_counts", "accuracy", "balanced")}, default=float), flush=True)
    p1.V_MIN = v0
    (HERE / "results" / "x16_accuracy_speed_floor.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
