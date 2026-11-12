"""EXPLORATORY (added 2026-11-13): are RoadScribe's event claims properties of the place?

The world-level test applied to events instead of severity classes, without labels: an event detected by one
vehicle is *reproduced* if another vehicle's detector fires within 50 m of the same place on the common route
coordinate (P1's GNSS alignment). Real road features should be reproduced across vehicles; events produced by
the vehicle, driver or detector noise should not. Chance level: the same statistic after circularly shifting
the other vehicles' events along the route (200 random shifts).

Also reported: reproduction of events near a labelled speed bump vs elsewhere, by detector type and confidence.
Run from the repo root:  python roadscribe_research/identifiability/x11_event_reproducibility.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.io.pvs import ROUTES, bump_mask, load_pvs_trip, pvs_root, vehicle_of  # noqa: E402
from roadscribe.layer1.events import detect_events  # noqa: E402
from roadscribe.layer1.iso2631 import weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.layer3.hedging import MENTION_THRESHOLD  # noqa: E402
from roadscribe.paired.route_align import densify, enu, match_forward, sample_theta  # noqa: E402

TOL = 50.0
N_SHIFT = 200
SEED = 20261113


def runs(mask):
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def route_events(trips):
    cfg = RoadScribeConfig()
    raws = {t: load_pvs_trip(pvs_root(), t, side="left", with_labels=True) for t in trips}
    g = {t: raws[t].gnss.dropna(subset=["lat", "lon"]).sort_values("t").reset_index(drop=True) for t in trips}
    lat0 = float(np.mean([g[t].lat.mean() for t in trips]))
    lon0 = float(np.mean([g[t].lon.mean() for t in trips]))
    ref_pts, ref_s = densify(enu(g[trips[0]].lat, g[trips[0]].lon, lat0, lon0), 1.0)
    out = {}
    for t in trips:
        raw = raws[t]
        vft = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
        aw = weight_vehicle_axes(vft.a_v, raw.fs, cfg.iso)
        det = detect_events(vft, cfg.events, cfg.iso, gnss=raw.gnss, aw_z=aw[:, 2])
        th_fix, d_fix = match_forward(enu(g[t].lat, g[t].lon, lat0, lon0), ref_pts, ref_s)
        theta, valid = sample_theta(vft.t, g[t].t.to_numpy(float), th_fix, d_fix, max_dist_m=15.0)
        evs = []
        for e in det.events:
            i = int(np.clip(np.searchsorted(vft.t, 0.5 * (e.t0 + e.t1)), 0, len(vft.t) - 1))
            if valid[i] and e.p_defect >= MENTION_THRESHOLD and e.label != "artifact":
                evs.append(dict(theta=float(theta[i]), kind=e.label, conf=float(e.p_defect)))
        bumps = [float(np.mean(theta[a:b][valid[a:b]])) for a, b in runs(bump_mask(raw.labels)) if valid[a:b].any()]
        out[t] = dict(events=evs, bumps=bumps, vehicle=vehicle_of(t), length=float(ref_s[-1]))
        print(f"trip {t}: {len(evs)} mentionable events on the route coordinate, {len(bumps)} bumps", flush=True)
    return out


def reproduced(theta, others):
    """Number of other vehicles with an event within TOL of theta."""
    return sum(1 for o in others if len(o) and np.min(np.abs(o - theta)) <= TOL)


def main():
    rng = np.random.default_rng(SEED)
    rows, chance_any, chance_all, local_any, local_all = [], [], [], [], []
    for route, trips in ROUTES.items():
        R = route_events(trips)
        L = max(r["length"] for r in R.values())
        for t in trips:
            others = [np.array([e["theta"] for e in R[u]["events"]]) for u in trips if u != t]
            bumps = np.array(sorted({b for u in trips for b in R[u]["bumps"]}))
            for e in R[t]["events"]:
                k = reproduced(e["theta"], others)
                near = bool(len(bumps) and np.min(np.abs(bumps - e["theta"])) <= TOL)
                rows.append(dict(route=route, trip=t, kind=e["kind"], conf=e["conf"], n_rep=k, near_bump=near))
            for _ in range(N_SHIFT // 10):
                sh = [np.mod(o + rng.uniform(0, L), L) for o in others]
                ks = [reproduced(e["theta"], sh) for e in R[t]["events"]]
                chance_any += [k >= 1 for k in ks]
                chance_all += [k == 2 for k in ks]
                # local null: shift by 100-500 m, keeping the local event density of rough stretches
                sh = [o + rng.choice([-1, 1]) * rng.uniform(100, 500) for o in others]
                ks = [reproduced(e["theta"], sh) for e in R[t]["events"]]
                local_any += [k >= 1 for k in ks]
                local_all += [k == 2 for k in ks]
    rows_np = lambda f: [r for r in rows if f(r)]  # noqa: E731

    def summ(rs):
        return dict(n=len(rs), reproduced_by_any_other=float(np.mean([r["n_rep"] >= 1 for r in rs])) if rs else None,
                    reproduced_by_both_others=float(np.mean([r["n_rep"] == 2 for r in rs])) if rs else None)
    out = {"label": "EXPLORATORY, not pre-registered", "tolerance_m": TOL,
           "all": summ(rows), "near_labelled_bump": summ(rows_np(lambda r: r["near_bump"])),
           "not_near_bump": summ(rows_np(lambda r: not r["near_bump"])),
           "by_kind": {k: summ(rows_np(lambda r, k=k: r["kind"] == k)) for k in sorted({r["kind"] for r in rows})},
           "chance_any": float(np.mean(chance_any)), "chance_both": float(np.mean(chance_all)),
           "local_shift_100_500m_any": float(np.mean(local_any)), "local_shift_100_500m_both": float(np.mean(local_all))}
    (HERE / "results" / "x11_event_reproducibility.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
