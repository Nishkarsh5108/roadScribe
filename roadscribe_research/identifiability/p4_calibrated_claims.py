"""P4: road claims after calibrating each vehicle on shared roads (PREREG_P4.md).

Leave one route out: gamma and vehicle gains are fitted on the other two routes (P1 stretch
tables, log a_wz ~ location FE + vehicle FE + log v), then applied on the held-out route both
at stretch level and inside the full RoadScribe pipeline (template narratives, capsules).
Run from the repo root after p1_cross_vehicle.py and x2_narrative_contradictions.py.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

import p1_cross_vehicle as p1  # noqa: E402
import x2_narrative_contradictions as x2  # noqa: E402
from roadscribe.config import RoadScribeConfig, SpeedConfig  # noqa: E402
from roadscribe.eval.claims import verify_text  # noqa: E402
from roadscribe.io.pvs import ROUTES, load_pvs_trip, pvs_root, route_of, vehicle_of  # noqa: E402
from roadscribe.layer1.events import detect_events  # noqa: E402
from roadscribe.layer1.iso2631 import comfort_class, weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.layer2.arbitrate import arbitrate, build_hrnus  # noqa: E402
from roadscribe.layer2.distance import build_grid  # noqa: E402
from roadscribe.layer2.mipelt import segment  # noqa: E402
from roadscribe.layer3.kst import RegexTokenizer, build_capsule  # noqa: E402
from roadscribe.layer3.ledger import LedgerView, TripMeta, compile_ledger  # noqa: E402
from roadscribe.layer3.templates import ground, realize, rule_plan  # noqa: E402
from roadscribe.paired.agreement import (_demean, apply_cutpoints, block_indices, fit_cutpoints,  # noqa: E402
                                         variance_shares, weighted_kappa)
from roadscribe.paired.route_align import densify, enu, match_forward, sample_theta  # noqa: E402

OUT = HERE / "results"
CAPS = REPO / "data" / "processed" / "pvs_capsules_cal"


def calibrate(tab: pd.DataFrame, routes) -> tuple[float, dict]:
    d = p1.triplets(tab[tab.route.isin(routes)])
    y, lv = np.log(d.a_wz.to_numpy()), np.log(d.v_mean.to_numpy())
    loc, veh = d["loc"].to_numpy(), d.vehicle.to_numpy()
    D = (veh[:, None] == np.array([2, 3])[None, :]).astype(float)
    beta, *_ = np.linalg.lstsq(np.c_[_demean(D, loc), _demean(lv, loc)], _demean(y, loc), rcond=None)
    g = np.array([0.0, beta[0], beta[1]])
    g -= g.mean()
    return float(beta[2]), {v: float(g[v - 1]) for v in (1, 2, 3)}


def stretch_level(tab: pd.DataFrame, cal: dict) -> pd.DataFrame:
    d = p1.triplets(tab)
    gam = d.route.map(lambda r: cal[r][0])
    gain = [cal[r][1][v] for r, v in zip(d.route, d.vehicle)]
    d = d.assign(a_cal=d.a_wz * (10.0 / d.v_mean) ** gam * np.exp(-np.asarray(gain)))
    d["cal_class"] = [comfort_class(a) for a in d.a_cal]
    return d


def agreement(d: pd.DataFrame, col_class: str, col_val: str) -> dict:
    pairs = list(combinations([1, 2, 3], 2))
    P = d.pivot(index="loc", columns="vehicle", values=col_class)
    V = d.pivot(index="loc", columns="vehicle", values=col_val)
    lv = np.log(d.v_mean.to_numpy())
    sh = variance_shares(np.log(d[col_val].to_numpy()), d["loc"].to_numpy(), d.vehicle.to_numpy(), lv)
    return dict(kappa=float(np.mean([weighted_kappa(P[a], P[b], 6) for a, b in pairs])),
                all_three_equal=float(np.mean(P.nunique(axis=1) == 1)),
                pairs_differ=float(np.mean(np.concatenate([(P[a] != P[b]).to_numpy() for a, b in pairs]))),
                spearman=float(np.mean([stats.spearmanr(np.log(V[a]), np.log(V[b])).statistic for a, b in pairs])),
                vehicle_share=sh["vehicle"], within_slope=sh["slope_log_v_within"])


def lovo(d: pd.DataFrame, col: str) -> float:
    dd = d[np.isfinite(d.gt_quality)]
    x, y = np.log(dd[col].to_numpy()), dd.gt_quality.astype(int).to_numpy()
    hit = 0
    for h in (1, 2, 3):
        tr, te = (dd.vehicle != h).to_numpy(), (dd.vehicle == h).to_numpy()
        hit += int(np.sum(apply_cutpoints(x[te], fit_cutpoints(x[tr], y[tr])) == y[te]))
    return hit / len(y)


def narratives(cal: dict) -> dict:
    base = RoadScribeConfig()
    tok = RegexTokenizer()
    CAPS.mkdir(parents=True, exist_ok=True)
    totals = dict(pairs=0, differ=0, distinct_road_claims=0, narratives_with_road_claim=0)
    attrib = dict(pairs=0, disagree=0)
    for route, trips in ROUTES.items():
        gam, gains = cal[route]
        raws = {t: load_pvs_trip(pvs_root(), t, side="left", with_labels=False) for t in trips}
        g = {t: raws[t].gnss.dropna(subset=["lat", "lon"]).sort_values("t").reset_index(drop=True) for t in trips}
        lat0 = float(np.mean([g[t].lat.mean() for t in trips]))
        lon0 = float(np.mean([g[t].lon.mean() for t in trips]))
        ref_pts, ref_s = densify(enu(g[trips[0]].lat, g[trips[0]].lon, lat0, lon0), 1.0)
        n_bins = int(ref_s[-1] // x2.BIN) + 1
        claims = {}
        for t in trips:
            cfg = dataclasses.replace(base, speed=SpeedConfig(gamma=gam, vehicle_gain_log=gains[vehicle_of(t)]))
            raw = raws[t]
            th_fix, d_fix = match_forward(enu(g[t].lat, g[t].lon, lat0, lon0), ref_pts, ref_s)
            theta, _ = sample_theta(raw.t, g[t].t.to_numpy(float), th_fix, d_fix)
            vft = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
            aw = weight_vehicle_axes(vft.a_v, raw.fs, cfg.iso)
            det = detect_events(vft, cfg.events, cfg.iso, gnss=raw.gnss, aw_z=aw[:, 2])
            grid = build_grid(vft, aw, det.events, cfg.distance, cfg.speed, cfg.spectral)
            res, _ = segment(grid.Y, grid.s, [], [], cfg.mipelt)
            end = float(vft.s[-1])
            bounds = arbitrate(res.changepoints, grid.s, grid.Y, [], cfg.arbitration, s_start=float(vft.s[0]), s_end=end)
            units = build_hrnus(bounds, vft, aw, det.events, [], grid_s=grid.s, grid_Y=grid.Y,
                                iso_cfg=cfg.iso, speed_cfg=cfg.speed, spectral_cfg=cfg.spectral)
            meta = TripMeta(raw.trip_id, duration_s=float(raw.t[-1] - raw.t[0]), length_m=end, map_coverage=0.0)
            ledger = compile_ledger(units, det.events, meta, cfg.ledger)
            kst = build_capsule(units, det.events, meta, 128, tok, cfg.kst, ledger=ledger)
            (CAPS / f"pvs_{t}.json").write_text(json.dumps(dict(
                trip_id=raw.trip_id, route_id=route_of(t), vehicle_id=vehicle_of(t), calibration=dict(gamma=gam, vehicle_gain_log=gains[vehicle_of(t)]),
                capsule_text=kst.capsule.text, capsule_tokens=kst.capsule.n_tokens), indent=1), encoding="utf-8")
            L = LedgerView(ledger)
            must = [f.key for f in L.facts if f.kind == "EVT" and f.conf >= 0.85]
            s_sorted = np.maximum.accumulate(vft.s)
            claims[t] = {}
            for persona in x2.PERSONAS:
                plan = rule_plan(ledger, kst.capsule.selection, persona, rng=None, must_mention_events=must)
                rep = verify_text(realize(ground(plan, ledger), persona, vary=False), ledger, must_mention_events=must)
                cl = []
                for c in rep.claims:
                    if c.unit_id is None or (c.kind, c.key) not in (("SEV", "road_class"), ("CMP", "speed_cause")):
                        continue
                    th0, th1 = np.interp(list(L.extent(c.unit_id)), s_sorted, theta)
                    cl.append(dict(kind=c.key, value=int(c.value) if c.key != "speed_cause" else 1, th0=float(th0), th1=float(th1)))
                claims[t][persona] = cl
                n_road = sum(1 for c in cl if c["kind"] == "road_class")
                totals["distinct_road_claims"] += n_road
                totals["narratives_with_road_claim"] += int(n_road > 0)
        for persona in x2.PERSONAS:
            grids = {t: {k: x2.per_stretch(claims[t][persona], k, n_bins) for k in ("road_class", "speed_cause")} for t in trips}
            for a, b in combinations(trips, 2):
                for xa, xb, sa, sb in zip(grids[a]["road_class"], grids[b]["road_class"], grids[a]["speed_cause"], grids[b]["speed_cause"]):
                    if xa is not None and xb is not None:
                        totals["pairs"] += 1
                        totals["differ"] += int(xa != xb)
                        attrib["pairs"] += 1
                        attrib["disagree"] += int((sa is not None) != (sb is not None))
        print(f"route {route}: gamma={gam:.3f} gains={gains}", flush=True)
    totals["differ_rate"] = totals["differ"] / max(1, totals["pairs"])
    attrib["disagree_rate"] = attrib["disagree"] / max(1, attrib["pairs"])
    return dict(road_claims=totals, speed_attribution=attrib)


def main():
    tab = pd.read_csv(OUT / "p1_stretches_100m.csv")
    cal = {r: calibrate(tab, [q for q in ROUTES if q != r]) for r in ROUTES}
    d = stretch_level(tab, cal)
    res = {"calibration": {r: dict(gamma=c[0], gains=c[1]) for r, c in cal.items()}}
    res["uncalibrated"] = {**agreement(d, "road_class", "a_dec"), "lovo_acc": lovo(d, "a_dec")}
    res["calibrated"] = {**agreement(d, "cal_class", "a_cal"), "lovo_acc": lovo(d, "a_cal")}
    rng = np.random.default_rng(p1.SEED + 6)
    by_route = {r: sorted(g["loc"].unique(), key=lambda s: int(s.split(":")[1])) for r, g in d.groupby("route")}
    groups = {k: g for k, g in d.groupby("loc")}
    dk, vs = [], []
    for _ in range(2000):
        parts = []
        for r, locs in by_route.items():
            for j, i in enumerate(block_indices(len(locs), p1.BLOCK, rng)):
                gg = groups[locs[i]].copy()
                gg["loc"] = f"{locs[i]}#{j}"
                parts.append(gg)
        x = pd.concat(parts, ignore_index=True)
        a_u, a_c = agreement(x, "road_class", "a_dec"), agreement(x, "cal_class", "a_cal")
        dk.append(a_c["kappa"] - a_u["kappa"])
        vs.append(a_c["vehicle_share"])
    res["ci95"] = {"kappa_cal_minus_uncal": p1.ci(dk), "vehicle_share_cal": p1.ci(vs)}
    print(json.dumps(res, indent=1, default=float), flush=True)
    res["narratives"] = narratives(cal)
    unc = json.loads((OUT / "x2_narrative_contradictions.json").read_text(encoding="utf-8"))
    res["narratives_uncalibrated_x2"] = dict(road_claims=unc["totals"]["road_class"], speed_attribution=unc["speed_attribution"])
    c, u = res["calibrated"], res["uncalibrated"]
    res["verdicts"] = {
        "P4_H1": bool(c["kappa"] > 0.718 and c["all_three_equal"] > 0.045),
        "P4_H2": bool(c["vehicle_share"] < 0.05),
        "P4_H3": bool(abs(c["within_slope"]) < 0.2),
        "P4_H4": bool(res["narratives"]["road_claims"]["differ_rate"] < 0.797),
    }
    (OUT / "p4_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("narratives", "verdicts")}, indent=1, default=float))


if __name__ == "__main__":
    main()
