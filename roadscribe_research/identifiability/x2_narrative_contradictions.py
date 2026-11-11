"""EXPLORATORY (not pre-registered): do RoadScribe's own narratives contradict each other about the same road?

For each route, the template narrator T writes one narrative per trip (3 vehicles) and persona.
Claims are extracted with the repository's verifier, mapped from each trip's own distance s to
the shared route coordinate theta, and compared per 100 m stretch across vehicles:
  * road claims ("the surface itself is ...", SEV/road_class): should agree; a difference is a contradiction;
  * felt claims (SEV/felt_class): may legitimately differ (reported for contrast);
  * speed attributions (CMP/speed_cause): "felt X mostly because of speed".
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.eval.claims import verify_text  # noqa: E402
from roadscribe.io.pvs import ROUTES, load_pvs_trip, pvs_root, vehicle_of  # noqa: E402
from roadscribe.layer1.events import detect_events  # noqa: E402
from roadscribe.layer1.iso2631 import weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.layer2.arbitrate import arbitrate, build_hrnus  # noqa: E402
from roadscribe.layer2.distance import build_grid  # noqa: E402
from roadscribe.layer2.mipelt import segment  # noqa: E402
from roadscribe.layer3.kst import RegexTokenizer, build_capsule  # noqa: E402
from roadscribe.layer3.ledger import LedgerView, TripMeta, compile_ledger  # noqa: E402
from roadscribe.layer3.templates import ground, realize, rule_plan  # noqa: E402
from roadscribe.paired.route_align import densify, enu, match_forward, sample_theta  # noqa: E402

BIN = 100.0
PERSONAS = ("commuter", "sports_car")


def trip_claims(raw, cfg, tok, theta):
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
    L = LedgerView(ledger)
    must = [f.key for f in L.facts if f.kind == "EVT" and f.conf >= 0.85]
    s_sorted = np.maximum.accumulate(vft.s)
    out = {}
    for persona in PERSONAS:
        plan = rule_plan(ledger, kst.capsule.selection, persona, rng=None, must_mention_events=must)
        rep = verify_text(realize(ground(plan, ledger), persona, vary=False), ledger, must_mention_events=must)
        claims = []
        for c in rep.claims:
            if c.unit_id is None or (c.kind, c.key) not in (("SEV", "road_class"), ("SEV", "felt_class"), ("CMP", "speed_cause")):
                continue
            s0, s1 = L.extent(c.unit_id)
            th0, th1 = np.interp([s0, s1], s_sorted, theta)
            claims.append(dict(kind=c.key, value=int(c.value) if c.key != "speed_cause" else 1,
                               th0=float(th0), th1=float(th1), supported=bool(c.supported)))
        out[persona] = claims
    return out


def per_stretch(claims, kind, n_bins):
    """Claimed value per 100 m stretch (claim must cover >= half the stretch); None if no claim."""
    vals = [None] * n_bins
    for c in claims:
        if c["kind"] != kind:
            continue
        for b in range(int(c["th0"] // BIN), int(c["th1"] // BIN) + 1):
            if 0 <= b < n_bins and min(c["th1"], (b + 1) * BIN) - max(c["th0"], b * BIN) >= 0.5 * BIN:
                vals[b] = c["value"]
    return vals


def main():
    cfg, tok = RoadScribeConfig(), RegexTokenizer()
    summary = {"label": "EXPLORATORY, not pre-registered", "routes": {}}
    totals = {k: dict(pairs=0, differ=0, differ2=0) for k in ("road_class", "felt_class")}
    attrib = dict(pairs=0, disagree=0)
    for route, trips in ROUTES.items():
        raws = {t: load_pvs_trip(pvs_root(), t, side="left", with_labels=False) for t in trips}
        g = {t: raws[t].gnss.dropna(subset=["lat", "lon"]).sort_values("t").reset_index(drop=True) for t in trips}
        lat0 = float(np.mean([g[t].lat.mean() for t in trips]))
        lon0 = float(np.mean([g[t].lon.mean() for t in trips]))
        ref_pts, ref_s = densify(enu(g[trips[0]].lat, g[trips[0]].lon, lat0, lon0), 1.0)
        n_bins = int(ref_s[-1] // BIN) + 1
        claims = {}
        for t in trips:
            th_fix, d_fix = match_forward(enu(g[t].lat, g[t].lon, lat0, lon0), ref_pts, ref_s)
            theta, _ = sample_theta(raws[t].t, g[t].t.to_numpy(float), th_fix, d_fix)
            claims[t] = trip_claims(raws[t], cfg, tok, theta)
        rs = {}
        for persona in PERSONAS:
            grids = {t: {k: per_stretch(claims[t][persona], k, n_bins) for k in ("road_class", "felt_class", "speed_cause")}
                     for t in trips}
            for k in ("road_class", "felt_class"):
                pairs = differ = differ2 = 0
                for a, b in combinations(trips, 2):
                    for x, y in zip(grids[a][k], grids[b][k]):
                        if x is not None and y is not None:
                            pairs += 1
                            differ += int(x != y)
                            differ2 += int(abs(x - y) >= 2)
                rs[f"{persona}:{k}"] = dict(pairs=pairs, differ=differ, differ2=differ2)
                totals[k]["pairs"] += pairs
                totals[k]["differ"] += differ
                totals[k]["differ2"] += differ2
            # attribution: one narrative blames speed where another describes the road but does not
            for a, b in combinations(trips, 2):
                for sa, sb, ra, rb in zip(grids[a]["speed_cause"], grids[b]["speed_cause"],
                                          grids[a]["road_class"], grids[b]["road_class"]):
                    if ra is not None and rb is not None:
                        attrib["pairs"] += 1
                        attrib["disagree"] += int((sa is not None) != (sb is not None))
        rs["distinct_claims"] = {f"pvs{t}:{persona}:{k}": sum(1 for c in claims[t][persona] if c["kind"] == k)
                                 for t in trips for persona in PERSONAS for k in ("road_class", "felt_class", "speed_cause")}
        summary["routes"][route] = rs
        print(route, json.dumps(rs), flush=True)
    for k, v in totals.items():
        v["differ_rate"] = v["differ"] / max(1, v["pairs"])
        v["differ2_rate"] = v["differ2"] / max(1, v["pairs"])
    attrib["disagree_rate"] = attrib["disagree"] / max(1, attrib["pairs"])
    summary["totals"] = totals
    summary["speed_attribution"] = attrib
    (HERE / "results" / "x2_narrative_contradictions.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"totals": totals, "speed_attribution": attrib}, indent=1))


if __name__ == "__main__":
    main()
