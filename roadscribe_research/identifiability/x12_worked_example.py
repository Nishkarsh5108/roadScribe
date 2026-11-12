"""Worked example for the paper (Table 3): one place, three vehicles, road wording before and after calibration.

Selects places where the three vehicles' uncalibrated road classes all differ and the calibrated classes all
agree (leave-one-route-out calibration from p4_results.json), and writes the first such place on route A plus
the count. Run from the repo root after p1_cross_vehicle.py and p4_calibrated_claims.py.
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
from roadscribe.layer1.iso2631 import comfort_class  # noqa: E402
from roadscribe.layer3.lexicon import ROAD_WORDS  # noqa: E402


def main():
    d = p1.triplets(pd.read_csv(HERE / "results" / "p1_stretches_100m.csv"))
    cal = json.loads((HERE / "results" / "p4_results.json").read_text(encoding="utf-8"))["calibration"]
    gam = d.route.map(lambda r: cal[r]["gamma"])
    gain = np.array([cal[r]["gains"][str(v)] for r, v in zip(d.route, d.vehicle)])
    d = d.assign(a_cal=d.a_wz * (10.0 / d.v_mean) ** gam * np.exp(-gain))
    d["cal_class"] = [comfort_class(a) for a in d.a_cal]
    sel = [loc for loc, g in d.groupby("loc") if g.road_class.nunique() == 3 and g.cal_class.nunique() == 1]
    any_differ = int(sum(1 for _, g in d.groupby("loc") if g.road_class.nunique() > 1))
    ex = d[d["loc"] == sel[0]].sort_values("vehicle")
    rows = [dict(vehicle=int(r.vehicle), v_mean_mps=round(r.v_mean, 1), a_wz=round(r.a_wz, 3), a_dec=round(r.a_dec, 3),
                 road_class=int(r.road_class), wording=f"the surface is {ROAD_WORDS[int(r.road_class)][0]}",
                 gain_log=round(cal[r.route]["gains"][str(int(r.vehicle))], 3), a_cal=round(r.a_cal, 3),
                 cal_class=int(r.cal_class), cal_wording=f"the surface is {ROAD_WORDS[int(r.cal_class)][0]}")
            for r in ex.itertuples()]
    out = dict(place=sel[0], gt_quality=float(ex.gt_quality.iloc[0]), gt_surface=str(ex.gt_surface.iloc[0]),
               gamma_hat=cal[ex.route.iloc[0]]["gamma"], rows=rows, n_places=int(d["loc"].nunique()),
               n_three_classes_then_agree=len(sel), n_any_disagreement_uncal=any_differ)
    (HERE / "results" / "x12_worked_example.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
