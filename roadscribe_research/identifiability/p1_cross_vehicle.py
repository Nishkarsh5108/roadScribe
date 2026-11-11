"""Pilot P1: cross-vehicle agreement of RoadScribe's felt and road claims on PVS.

Implements PREREG_P1.md exactly; anything beyond it is labelled exploratory in the output.
Run from the repo root:  python roadscribe_research/identifiability/p1_cross_vehicle.py
"""
from __future__ import annotations

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
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.io.pvs import ROUTES, load_pvs_trip, pvs_root, vehicle_of  # noqa: E402
from roadscribe.layer1.iso2631 import weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.paired.agreement import (apply_cutpoints, block_indices, fit_cutpoints,  # noqa: E402
                                         variance_shares, weighted_kappa)
from roadscribe.paired.route_align import densify, enu, match_forward, sample_theta  # noqa: E402
from roadscribe.paired.stretches import stretch_table  # noqa: E402

OUT = HERE / "results"
V_MIN = 5.0
N_BOOT = 2000
N_BOOT_H5 = 500
BLOCK = 10
SEED = 20261112


def load_route(route: str, trips, placement: str = "dashboard", bins=(100.0, 50.0)):
    cfg = RoadScribeConfig()
    raws, vfts = {}, {}
    for t in trips:
        raw = load_pvs_trip(pvs_root(), t, side="left", placement=placement, with_labels=True)
        vfts[t] = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
        raws[t] = raw
    g = {t: raws[t].gnss.dropna(subset=["lat", "lon"]).sort_values("t").reset_index(drop=True) for t in trips}
    lat0 = float(np.mean([g[t].lat.mean() for t in trips]))
    lon0 = float(np.mean([g[t].lon.mean() for t in trips]))
    ref_pts, ref_s = densify(enu(g[trips[0]].lat, g[trips[0]].lon, lat0, lon0), 1.0)
    tables = {b: [] for b in bins}
    align = {}
    for t in trips:
        raw, vft = raws[t], vfts[t]
        th_fix, d_fix = match_forward(enu(g[t].lat, g[t].lon, lat0, lon0), ref_pts, ref_s)
        theta, valid = sample_theta(vft.t, g[t].t.to_numpy(float), th_fix, d_fix, max_dist_m=15.0)
        align[t] = dict(median_dist_m=float(np.median(d_fix)), p95_dist_m=float(np.percentile(d_fix, 95)),
                        valid_frac=float(valid.mean()), theta_end_m=float(th_fix[-1]), ref_len_m=float(ref_s[-1]))
        aw = weight_vehicle_axes(vft.a_v, raw.fs, cfg.iso)
        for b in bins:
            tab = stretch_table(theta, valid, vft.v, aw, vft.a_v[:, 2], raw.fs, raw.labels, bin_m=b,
                                gamma=cfg.speed.gamma, v_ref=cfg.speed.v_ref)
            tab.insert(0, "vehicle", vehicle_of(t))
            tab.insert(0, "trip", t)
            tab.insert(0, "route", route)
            tables[b].append(tab)
    return {b: pd.concat(v, ignore_index=True) for b, v in tables.items()}, align


def triplets(df: pd.DataFrame, exclude_bumps: bool = False) -> pd.DataFrame:
    """Rows for stretches valid in all three vehicles with v_mean >= V_MIN in all three."""
    d = df.copy()
    d["loc"] = d["route"] + ":" + d["stretch"].astype(str)
    ok = d.groupby("loc").agg(nv=("vehicle", "nunique"), vmin=("v_mean", "min"), anybump=("gt_bump", "any"))
    keep = ok[(ok.nv == 3) & (ok.vmin >= V_MIN)]
    if exclude_bumps:
        keep = keep[~keep.anybump]
    d = d[d["loc"].isin(keep.index)].copy()
    return d.sort_values(["route", "stretch", "vehicle"]).reset_index(drop=True)


def pair_stats(d: pd.DataFrame) -> dict:
    """Mean pairwise statistics across the 3 vehicles (each location contributes one value per pair)."""
    piv = {c: d.pivot(index="loc", columns="vehicle", values=c) for c in
           ("felt_class", "road_class", "gt_quality", "a_v", "a_dec", "r2", "a_wz")}
    out = {}
    pairs = list(combinations(sorted(d.vehicle.unique()), 2))
    for name, ncls in (("felt_class", 6), ("road_class", 6)):
        P = piv[name]
        out[f"kappa_{name}"] = float(np.nanmean([weighted_kappa(P[a], P[b], ncls) for a, b in pairs]))
        diffs = np.concatenate([np.abs(P[a] - P[b]).to_numpy() for a, b in pairs])
        out[f"{name}_pairs_differ_ge1"] = float(np.mean(diffs >= 1))
        out[f"{name}_pairs_differ_ge2"] = float(np.mean(diffs >= 2))
        out[f"{name}_all_three_equal"] = float(np.mean(P.nunique(axis=1) == 1))
    Q = piv["gt_quality"].dropna()
    out["kappa_gt_quality"] = float(np.nanmean([weighted_kappa(Q[a].astype(int), Q[b].astype(int), 3) for a, b in pairs])) \
        if len(Q) else float("nan")
    out["n_locations_gt_quality"] = int(len(Q))
    for name in ("a_v", "a_dec", "r2"):
        P = piv[name]
        out[f"spearman_{name}"] = float(np.nanmean([stats.spearmanr(np.log(P[a]) if name != "r2" else P[a],
                                                                       np.log(P[b]) if name != "r2" else P[b],
                                                                       nan_policy="omit").statistic for a, b in pairs]))
    return out


def model_stats(d: pd.DataFrame) -> dict:
    out = {}
    lv = np.log(d["v_mean"].to_numpy())
    for name, y in (("log_a_dec", np.log(d["a_dec"])), ("log_a_v", np.log(d["a_v"])),
                    ("log_a_wz", np.log(d["a_wz"])), ("r2", d["r2"])):
        y = np.asarray(y, float)
        m = np.isfinite(y)
        sh = variance_shares(y[m], d["loc"].to_numpy()[m], d["vehicle"].to_numpy()[m], lv[m])
        out[name] = sh
    y = np.log(d["a_wz"].to_numpy())
    out["gamma_pooled"] = float(np.polyfit(lv, y, 1)[0])
    out["gamma_within"] = out["log_a_wz"]["slope_log_v_within"]
    out["gamma_within_minus_pooled"] = out["gamma_within"] - out["gamma_pooled"]
    loc = d.groupby("loc").agg(rough=("a_wz", lambda s: float(np.mean(np.log(s)))), vbar=("v_mean", "mean"))
    out["spearman_loc_roughness_vs_speed"] = float(stats.spearmanr(loc.rough, loc.vbar).statistic)
    return out


def lovo_accuracy(d: pd.DataFrame, n_grid: int = 60) -> dict:
    """Leave-one-vehicle-out accuracy predicting GT quality from a 1-D index (cut-points from the other two)."""
    dd = d[np.isfinite(d["gt_quality"])].copy()
    y = dd["gt_quality"].astype(int).to_numpy()
    res = {}
    idx = {"a_v": np.log(dd["a_v"].to_numpy()), "a_dec": np.log(dd["a_dec"].to_numpy()), "r2": dd["r2"].to_numpy()}
    for name, x in idx.items():
        correct = 0
        bal = []
        for h in sorted(dd.vehicle.unique()):
            tr = (dd.vehicle != h).to_numpy() & np.isfinite(x)
            te = (dd.vehicle == h).to_numpy() & np.isfinite(x)
            cuts = fit_cutpoints(x[tr], y[tr], n_grid=n_grid)
            pred = apply_cutpoints(x[te], cuts)
            correct += int(np.sum(pred == y[te]))
            bal.append(np.mean([np.mean(pred[y[te] == k] == k) for k in np.unique(y[te])]))
        res[f"lovo_acc_{name}"] = correct / int(np.isfinite(x).sum())
        res[f"lovo_balanced_acc_{name}"] = float(np.mean(bal))
    res["majority_baseline"] = float(np.max(np.bincount(y)) / len(y))
    res["class_counts"] = np.bincount(y, minlength=3).tolist()
    return res


def bootstrap(d: pd.DataFrame, fn, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_route = {r: sorted(g["loc"].unique(), key=lambda s: int(s.split(":")[1])) for r, g in d.groupby("route")}
    groups = {k: g for k, g in d.groupby("loc")}
    out = []
    for _ in range(n):
        parts = []
        for r, locs in by_route.items():
            for j, i in enumerate(block_indices(len(locs), BLOCK, rng)):
                g = groups[locs[i]].copy()
                g["loc"] = f"{locs[i]}#{j}"  # duplicates are distinct locations
                parts.append(g)
        out.append(fn(pd.concat(parts, ignore_index=True)))
    return out


def ci(vals) -> list:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def analyse(d: pd.DataFrame, label: str, boot: bool = True) -> dict:
    res = {"label": label, "n_locations": int(d["loc"].nunique()), "n_rows": int(len(d))}
    res.update(pair_stats(d))
    ms = model_stats(d)
    res["variance_shares"] = {k: ms[k] for k in ("log_a_dec", "log_a_v", "log_a_wz", "r2")}
    for k in ("gamma_pooled", "gamma_within", "gamma_within_minus_pooled", "spearman_loc_roughness_vs_speed"):
        res[k] = ms[k]
    res.update(lovo_accuracy(d))
    if boot:
        def core(x):
            p = pair_stats(x)
            m = model_stats(x)
            return dict(kappa_felt=p["kappa_felt_class"], kappa_road=p["kappa_road_class"], kappa_gt=p["kappa_gt_quality"],
                        veh_share_dec=m["log_a_dec"]["vehicle"], veh_share_v=m["log_a_v"]["vehicle"],
                        g_diff=m["gamma_within_minus_pooled"], rho_rough_speed=m["spearman_loc_roughness_vs_speed"])
        bs = bootstrap(d, core, N_BOOT, SEED)
        res["ci95"] = {k: ci([b[k] for b in bs]) for k in bs[0]}
        bs5 = bootstrap(d, lambda x: lovo_accuracy(x, n_grid=30), N_BOOT_H5, SEED + 1)
        res["ci95"]["h5_diff_road_minus_felt"] = ci([b["lovo_acc_a_dec"] - b["lovo_acc_a_v"] for b in bs5])
    return res


def verdicts(r: dict) -> dict:
    return {
        "H1_felt_kappa_lt_0.6": r["kappa_felt_class"] < 0.6,
        "H2a_road_kappa_lt_0.8": r["kappa_road_class"] < 0.8,
        "H2b_vehicle_share_ge_5pct": r["variance_shares"]["log_a_dec"]["vehicle"] >= 0.05,
        "H2_main": (r["kappa_road_class"] < 0.8) and (r["variance_shares"]["log_a_dec"]["vehicle"] >= 0.05),
        "H3a_rho_lt_minus_0.2": r["spearman_loc_roughness_vs_speed"] < -0.2,
        "H3b_gamma_diff_gt_0.1": r["gamma_within_minus_pooled"] > 0.1,
        "H4_gt_kappa_ge_0.6": r["kappa_gt_quality"] >= 0.6,
        "H5_road_minus_felt_ge_5pts": (r["lovo_acc_a_dec"] - r["lovo_acc_a_v"]) >= 0.05,
        "KILL_roads_show_no_problem": (r["kappa_road_class"] >= 0.8) and (r["variance_shares"]["log_a_dec"]["vehicle"] < 0.05),
    }


def main():
    OUT.mkdir(exist_ok=True)
    all_tabs = {100.0: [], 50.0: []}
    alignment = {}
    for route, trips in ROUTES.items():
        tabs, align = load_route(route, trips)
        alignment.update({f"pvs{t}": a for t, a in align.items()})
        for b in all_tabs:
            all_tabs[b].append(tabs[b])
        print(f"route {route}: loaded {trips}; alignment {align}", flush=True)
    tab100 = pd.concat(all_tabs[100.0], ignore_index=True)
    tab50 = pd.concat(all_tabs[50.0], ignore_index=True)
    tab100.to_csv(OUT / "p1_stretches_100m.csv", index=False)
    tab50.to_csv(OUT / "p1_stretches_50m.csv", index=False)

    prim = triplets(tab100)
    results = {"alignment": alignment, "primary_100m": analyse(prim, "primary: 100 m, v>=5 m/s in all three vehicles")}
    results["primary_100m"]["verdicts"] = verdicts(results["primary_100m"])
    results["per_route_100m"] = {r: analyse(g, f"route {r}", boot=False) for r, g in prim.groupby("route")}
    results["secondary_100m_no_bumps"] = analyse(triplets(tab100, exclude_bumps=True), "100 m, no bump stretches", boot=False)
    results["secondary_50m"] = analyse(triplets(tab50), "50 m", boot=False)
    (OUT / "p1_results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    r = results["primary_100m"]
    print(json.dumps({k: r[k] for k in r if k not in ("variance_shares",)}, indent=1, default=float))
    print(json.dumps(r["variance_shares"], indent=1))
    for k in ("per_route_100m",):
        for rt, rr in results[k].items():
            print(rt, {x: round(rr[x], 3) for x in ("kappa_felt_class", "kappa_road_class", "kappa_gt_quality", "spearman_a_dec")},
                  "veh share dec", round(rr["variance_shares"]["log_a_dec"]["vehicle"], 3))
    for k in ("secondary_100m_no_bumps", "secondary_50m"):
        rr = results[k]
        print(k, {x: round(rr[x], 3) for x in ("n_locations", "kappa_felt_class", "kappa_road_class", "kappa_gt_quality")},
              "veh share dec", round(rr["variance_shares"]["log_a_dec"]["vehicle"], 3))


if __name__ == "__main__":
    main()
