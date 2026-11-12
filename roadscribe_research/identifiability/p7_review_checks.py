"""P7 (PREREG_P7.md): noise floor, replication on the right logger, data efficiency, held-out correctness,
speed-floor sensitivity, and the confidence intervals the paper lacked.

Run from the repo root after p1_cross_vehicle.py:
    python roadscribe_research/identifiability/p7_review_checks.py
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

import p1_cross_vehicle as p1  # noqa: E402
from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.io.pvs import GNSS_COLS, ROUTES, load_pvs_trip, pvs_root, read_logger, vehicle_of  # noqa: E402
from roadscribe.layer1.iso2631 import comfort_class, weight_vehicle_axes  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.paired.agreement import (_demean, apply_cutpoints, block_indices, fit_cutpoints,  # noqa: E402
                                         variance_shares, weighted_kappa)
from roadscribe.paired.route_align import densify, enu, match_forward, sample_theta  # noqa: E402
from roadscribe.paired.stretches import stretch_table  # noqa: E402

OUT = HERE / "results"
SEED = 20261113
N_BOOT = 2000
N_DRAWS = 200
KM = (0.5, 1, 2, 4, 8, "all")
PAIRS = list(combinations([1, 2, 3], 2))
V_REF = 10.0


# ----------------------------------------------------------------------------- right logger
def right_tables() -> pd.DataFrame:
    """Right-logger dashboard stretch table on the *left* logger's reference track (same grid as P1)."""
    cfg = RoadScribeConfig()
    tabs = []
    for route, trips in ROUTES.items():
        gl = {}
        for t in trips:
            df = read_logger(pvs_root(), t, "left", usecols=GNSS_COLS)
            g = df[GNSS_COLS].rename(columns={"timestamp_gps": "t", "latitude": "lat", "longitude": "lon"})
            gl[t] = g.drop_duplicates("t").dropna(subset=["lat", "lon"]).sort_values("t").reset_index(drop=True)
        lat0 = float(np.mean([gl[t].lat.mean() for t in trips]))
        lon0 = float(np.mean([gl[t].lon.mean() for t in trips]))
        ref_pts, ref_s = densify(enu(gl[trips[0]].lat, gl[trips[0]].lon, lat0, lon0), 1.0)
        for t in trips:
            raw = load_pvs_trip(pvs_root(), t, side="right", placement="dashboard", with_labels=False)
            vft = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
            g = raw.gnss.dropna(subset=["lat", "lon"]).sort_values("t").reset_index(drop=True)
            th_fix, d_fix = match_forward(enu(g.lat, g.lon, lat0, lon0), ref_pts, ref_s)
            theta, valid = sample_theta(vft.t, g.t.to_numpy(float), th_fix, d_fix, max_dist_m=15.0)
            aw = weight_vehicle_axes(vft.a_v, raw.fs, cfg.iso)
            tab = stretch_table(theta, valid, vft.v, aw, vft.a_v[:, 2], raw.fs, None, bin_m=100.0,
                                gamma=cfg.speed.gamma, v_ref=cfg.speed.v_ref)
            tab.insert(0, "vehicle", vehicle_of(t))
            tab.insert(0, "trip", t)
            tab.insert(0, "route", route)
            tab["align_median_m"] = float(np.median(d_fix))
            tabs.append(tab)
            print(f"right logger trip {t}: {len(tab)} stretches, median match {np.median(d_fix):.1f} m", flush=True)
    return pd.concat(tabs, ignore_index=True)


# ----------------------------------------------------------------------------- calibration helpers
def fit_fe(d: pd.DataFrame, gamma_fixed: float | None = None) -> tuple[float, dict]:
    """log a_wz = place FE + vehicle gain + gamma log v (gamma fitted unless fixed); gains centred."""
    y, lv = np.log(d.a_wz.to_numpy()), np.log(d.v_mean.to_numpy())
    loc, veh = d["loc"].to_numpy(), d.vehicle.to_numpy()
    D = (veh[:, None] == np.array([2, 3])[None, :]).astype(float)
    if gamma_fixed is None:
        beta, *_ = np.linalg.lstsq(np.c_[_demean(D, loc), _demean(lv, loc)], _demean(y, loc), rcond=None)
        gam = float(beta[2])
    else:
        beta, *_ = np.linalg.lstsq(_demean(D, loc), _demean(y - gamma_fixed * lv, loc), rcond=None)
        gam = float(gamma_fixed)
    g = np.array([0.0, beta[0], beta[1]])
    g -= g.mean()
    return gam, {v: float(g[v - 1]) for v in (1, 2, 3)}


def apply_cal(d: pd.DataFrame, cal: dict, use_gain=True) -> np.ndarray:
    gam = d.route.map(lambda r: cal[r][0]).to_numpy(float)
    gain = np.array([cal[r][1][v] if use_gain else 0.0 for r, v in zip(d.route, d.vehicle)])
    return d.a_wz.to_numpy() * (V_REF / d.v_mean.to_numpy()) ** gam * np.exp(-gain)


def kappa_cross(d: pd.DataFrame, col: str) -> float:
    P = d.pivot(index="loc", columns="vehicle", values=col)
    return float(np.mean([weighted_kappa(P[a], P[b], 6) for a, b in PAIRS]))


def classes(x) -> np.ndarray:
    return np.array([comfort_class(float(a)) for a in x])


def loro(tab: pd.DataFrame, **kw) -> dict:
    return {r: fit_fe(p1.triplets(tab[tab.route != r]), **kw) for r in ROUTES}


def ci(v) -> list:
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def block_boot_rows(d: pd.DataFrame, n: int, seed: int):
    """Yield index arrays of d resampled by contiguous 10-place blocks within each route (as P1)."""
    rng = np.random.default_rng(seed)
    by_route = {r: sorted(g["loc"].unique(), key=lambda s: int(s.split(":")[1])) for r, g in d.groupby("route")}
    rows = {k: g.index.to_numpy() for k, g in d.groupby("loc")}
    for _ in range(n):
        yield np.concatenate([rows[locs[i]] for r, locs in by_route.items() for i in block_indices(len(locs), p1.BLOCK, rng)])


def n_blocks(d: pd.DataFrame) -> dict:
    return {r: int(np.ceil(g["loc"].nunique() / p1.BLOCK)) for r, g in d.groupby("route")}


# ----------------------------------------------------------------------------- analyses
def noise_floor(left: pd.DataFrame, right: pd.DataFrame, prim: pd.DataFrame) -> dict:
    """P7-H1: same vehicle, same pass, left vs right dashboard module."""
    keys = ["route", "trip", "stretch"]
    m = left.merge(right, on=keys, suffixes=("_L", "_R"))
    m["loc"] = m.route + ":" + m.stretch.astype(str)
    m = m[m["loc"].isin(set(prim["loc"])) & (m.v_mean_R >= p1.V_MIN) & (m.v_mean_L >= p1.V_MIN)].reset_index(drop=True)
    out = dict(n_rows=int(len(m)), n_places=int(m["loc"].nunique()))
    for name in ("road_class", "felt_class"):
        a, b = m[f"{name}_L"].to_numpy(int), m[f"{name}_R"].to_numpy(int)
        out[f"kappa_{name}"] = weighted_kappa(a, b, 6)
        out[f"{name}_differ"] = float(np.mean(a != b))
        out[f"{name}_differ_ge2"] = float(np.mean(np.abs(a - b) >= 2))
        out[f"kappa_{name}_by_vehicle"] = {int(v): weighted_kappa(g[f"{name}_L"].astype(int), g[f"{name}_R"].astype(int), 6)
                                           for v, g in m.groupby("vehicle_L")}
    # side share: log a_dec ~ (trip, place) FE + side, the analogue of P1's vehicle share
    y = np.log(np.r_[m.a_dec_L, m.a_dec_R])
    grp = np.r_[m.trip.astype(str) + "|" + m["loc"], m.trip.astype(str) + "|" + m["loc"]]
    side = np.r_[np.zeros(len(m)), np.ones(len(m))]
    lv = np.log(np.r_[m.v_mean_L, m.v_mean_R])
    out["side_share_log_a_dec"] = variance_shares(y, grp, side, lv)["vehicle"]
    out["side_log_gain_R_minus_L"] = float(np.mean(np.log(m.a_dec_R) - np.log(m.a_dec_L)))
    bs = []
    for idx in block_boot_rows(m.assign(route=m.route), N_BOOT, SEED):
        mm = m.loc[idx]
        bs.append(weighted_kappa(mm.road_class_L.astype(int), mm.road_class_R.astype(int), 6))
    out["ci95_kappa_road_class"] = ci(bs)
    return out, m


def replication(right: pd.DataFrame, left: pd.DataFrame) -> dict:
    """P7-H3: P1 vehicle share and P4 LORO calibration on the right logger (places valid there)."""
    gt = left[["route", "trip", "stretch", "gt_quality", "gt_surface", "gt_bump"]]
    r = right.merge(gt, on=["route", "trip", "stretch"], how="left")
    r["gt_bump"] = r.gt_bump.fillna(False).astype(bool)
    d = p1.triplets(r)
    ms = p1.model_stats(d)
    ps = p1.pair_stats(d)
    cal = loro(r)
    d = d.assign(a_cal=apply_cal(d, cal))
    d["cal_class"] = classes(d.a_cal)
    sh = variance_shares(np.log(d.a_cal.to_numpy()), d["loc"].to_numpy(), d.vehicle.to_numpy(), np.log(d.v_mean.to_numpy()))
    return dict(n_places=int(d["loc"].nunique()), vehicle_share_log_a_dec=ms["log_a_dec"]["vehicle"],
                vehicle_share_log_a_v=ms["log_a_v"]["vehicle"], kappa_road_uncal=ps["kappa_road_class"],
                kappa_felt=ps["kappa_felt_class"], kappa_road_cal=kappa_cross(d, "cal_class"),
                vehicle_share_cal=sh["vehicle"], gamma_hat={k: v[0] for k, v in cal.items()},
                gains={k: v[1] for k, v in cal.items()}, gamma_within=ms["gamma_within"], gamma_pooled=ms["gamma_pooled"])


def data_efficiency(left: pd.DataFrame, prim: pd.DataFrame) -> dict:
    """P7-H4: kappa_cal over all places when calibration uses K km of shared road from the other routes."""
    rng = np.random.default_rng(SEED + 4)
    blocks = {}
    for r in ROUTES:
        cal_d = p1.triplets(left[left.route != r])
        bl = []
        for rr, g in cal_d.groupby("route"):
            locs = sorted(g["loc"].unique(), key=lambda s: int(s.split(":")[1]))
            bl += [locs[i:i + p1.BLOCK] for i in range(0, len(locs), p1.BLOCK)]
        blocks[r] = (cal_d, bl)
    out = {}
    for K in KM:
        ks, gs = [], []
        for _ in range(N_DRAWS if K != "all" else 1):
            cal = {}
            for r in ROUTES:
                cal_d, bl = blocks[r]
                if K == "all":
                    sel = cal_d
                elif K < 1:
                    b = bl[rng.integers(len(bl))]
                    sel = cal_d[cal_d["loc"].isin(b[: int(round(K * p1.BLOCK))])]
                else:
                    pick = rng.choice(len(bl), size=min(int(K), len(bl)), replace=False)
                    sel = cal_d[cal_d["loc"].isin([l for i in pick for l in bl[i]])]
                cal[r] = fit_fe(sel)
            d = prim.assign(a_cal=apply_cal(prim, cal))
            d["cal_class"] = classes(d.a_cal)
            ks.append(kappa_cross(d, "cal_class"))
            gs.append(np.mean([c[0] for c in cal.values()]))
        out[str(K)] = dict(median_kappa=float(np.median(ks)), p2_5=float(np.percentile(ks, 2.5)), p97_5=float(np.percentile(ks, 97.5)),
                           share_ge_0_90=float(np.mean(np.asarray(ks) >= 0.90)), median_gamma=float(np.median(gs)), draws=len(ks))
    return out


def ablation(left: pd.DataFrame, prim: pd.DataFrame) -> dict:
    out = {}
    variants = {
        "none (Eq. dec, gamma 0.35)": (dict(gamma_fixed=0.35), False),
        "gain only, gamma 0.35": (dict(gamma_fixed=0.35), True),
        "gain only, gamma 0.5 (theory, w = 2)": (dict(gamma_fixed=0.5), True),
        "elasticity only (gamma hat, no gain)": ({}, False),
        "gain + gamma hat (P4)": ({}, True),
    }
    for name, (kw, use_gain) in variants.items():
        cal = loro(left, **kw)
        d = prim.assign(a_cal=apply_cal(prim, cal, use_gain=use_gain))
        d["cal_class"] = classes(d.a_cal)
        sh = variance_shares(np.log(d.a_cal.to_numpy()), d["loc"].to_numpy(), d.vehicle.to_numpy(), np.log(d.v_mean.to_numpy()))
        out[name] = dict(kappa=kappa_cross(d, "cal_class"), vehicle_share=sh["vehicle"], within_slope=sh["slope_log_v_within"])
    return out


def accuracy(y, pred) -> tuple[float, float]:
    acc = float(np.mean(pred == y))
    bal = float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))
    return acc, bal


def correctness(left: pd.DataFrame, prim: pd.DataFrame) -> dict:
    """P7-H5 (route and vehicle held out) and CIs for P4's leave-one-vehicle-out accuracies."""
    cal = loro(left)
    d = prim.assign(a_cal=apply_cal(prim, cal))
    d = d[np.isfinite(d.gt_quality)].reset_index(drop=True)
    y = d.gt_quality.astype(int).to_numpy()
    idx = {"a_v": np.log(d.a_v.to_numpy()), "a_dec": np.log(d.a_dec.to_numpy()), "a_cal": np.log(d.a_cal.to_numpy())}
    preds = {"route_and_vehicle_out": {}, "vehicle_out": {}}
    for name, x in idx.items():
        p_rv, p_v = np.full(len(d), -1), np.full(len(d), -1)
        for r in ROUTES:
            for c in (1, 2, 3):
                tr = ((d.route != r) & (d.vehicle != c)).to_numpy()
                te = ((d.route == r) & (d.vehicle == c)).to_numpy()
                p_rv[te] = apply_cutpoints(x[te], fit_cutpoints(x[tr], y[tr]))
        for c in (1, 2, 3):
            tr, te = (d.vehicle != c).to_numpy(), (d.vehicle == c).to_numpy()
            p_v[te] = apply_cutpoints(x[te], fit_cutpoints(x[tr], y[tr]))
        preds["route_and_vehicle_out"][name], preds["vehicle_out"][name] = p_rv, p_v
    out = {"n_rows": int(len(d)), "class_counts": np.bincount(y, minlength=3).tolist(),
           "majority_baseline": float(np.max(np.bincount(y)) / len(y))}
    for scheme, P in preds.items():
        res = {}
        for name, p in P.items():
            a, b = accuracy(y, p)
            res[name] = dict(accuracy=a, balanced_accuracy=b)
        bs = {k: [] for k in ("a_v", "a_dec", "a_cal", "cal_minus_dec", "bal_cal", "bal_dec")}
        for ii in block_boot_rows(d, N_BOOT, SEED + 5):
            yy = y[ii]
            acc = {k: float(np.mean(P[k][ii] == yy)) for k in ("a_v", "a_dec", "a_cal")}
            for k in ("a_v", "a_dec", "a_cal"):
                bs[k].append(acc[k])
            bs["cal_minus_dec"].append(acc["a_cal"] - acc["a_dec"])
            bs["bal_cal"].append(accuracy(yy, P["a_cal"][ii])[1])
            bs["bal_dec"].append(accuracy(yy, P["a_dec"][ii])[1])
        res["ci95"] = {k: ci(v) for k, v in bs.items()}
        res["cal_minus_dec"] = res["a_cal"]["accuracy"] - res["a_dec"]["accuracy"]
        out[scheme] = res
    return out


def speed_floor(left: pd.DataFrame) -> dict:
    """P7-H6 and the counts behind the 5 m/s filter."""
    out = {}
    v0 = p1.V_MIN
    for vmin in (0.0, 3.0, 5.0):
        p1.V_MIN = vmin
        d = p1.triplets(left)
        cal = loro(left)
        dc = d.assign(a_cal=apply_cal(d, cal))
        dc["cal_class"] = classes(dc.a_cal)
        q = d.groupby("loc").gt_quality.agg(lambda s: s.max())
        out[str(vmin)] = dict(n_places=int(d["loc"].nunique()),
                              vehicle_share_log_a_dec=p1.model_stats(d)["log_a_dec"]["vehicle"],
                              kappa_road_uncal=p1.pair_stats(d)["kappa_road_class"],
                              kappa_road_cal=kappa_cross(dc, "cal_class"),
                              gamma_hat={k: v[0] for k, v in cal.items()},
                              places_by_worst_quality=q.value_counts().sort_index().to_dict())
    p1.V_MIN = v0
    allp = left.assign(loc=left.route + ":" + left.stretch.astype(str)).groupby("loc").vehicle.nunique()
    out["places_any_coverage_all_three"] = int((allp == 3).sum())
    out["places_total"] = int(len(allp))
    return out


def missing_cis(left: pd.DataFrame, prim: pd.DataFrame) -> dict:
    """Block-bootstrap CIs for kappa_cal and kappa_uncal (P4 bootstrapped only the difference),
    the calibrated vehicle share, gamma hat on all routes, and the felt vehicle share (from P1)."""
    cal = loro(left)
    d = prim.assign(a_cal=apply_cal(prim, cal)).reset_index(drop=True)
    d["cal_class"] = classes(d.a_cal)
    gam_all, gains_all = fit_fe(prim)
    bs = {k: [] for k in ("kappa_cal", "kappa_uncal", "kappa_felt", "gamma_all_routes", "all_three_equal_cal", "all_three_equal_uncal")}
    for ii in block_boot_rows(d, N_BOOT, SEED + 6):
        dd = d.loc[ii].copy()
        # every place has exactly one row per vehicle and block_boot_rows keeps a place's rows together,
        # so consecutive triplets are the resampled places; duplicates become distinct places
        assert len(dd) % 3 == 0
        dd["loc"] = (np.arange(len(dd)) // 3).astype(str)
        bs["kappa_cal"].append(kappa_cross(dd, "cal_class"))
        bs["kappa_uncal"].append(kappa_cross(dd, "road_class"))
        bs["kappa_felt"].append(kappa_cross(dd, "felt_class"))
        P = dd.pivot(index="loc", columns="vehicle", values="cal_class")
        bs["all_three_equal_cal"].append(float(np.mean(P.nunique(axis=1) == 1)))
        P = dd.pivot(index="loc", columns="vehicle", values="road_class")
        bs["all_three_equal_uncal"].append(float(np.mean(P.nunique(axis=1) == 1)))
        bs["gamma_all_routes"].append(fit_fe(dd)[0])
    p1r = json.loads((OUT / "p1_results.json").read_text(encoding="utf-8"))["primary_100m"]
    return dict(point=dict(kappa_cal=kappa_cross(d, "cal_class"), kappa_uncal=kappa_cross(d, "road_class"),
                           kappa_felt=kappa_cross(d, "felt_class"), gamma_all_routes=gam_all, gains_all_routes=gains_all),
                ci95={k: ci(v) for k, v in bs.items()},
                felt_vehicle_share_ci95_from_p1=p1r["ci95"]["veh_share_v"],
                bootstrap=dict(resamples=N_BOOT, block_places=p1.BLOCK, blocks_per_route=n_blocks(d)))


def main():
    left = pd.read_csv(OUT / "p1_stretches_100m.csv")
    prim = p1.triplets(left)
    res = {"prereg": "PREREG_P7.md (commit 5891276)"}
    res["H4_data_efficiency"] = data_efficiency(left, prim)
    print("H4", json.dumps(res["H4_data_efficiency"], default=float), flush=True)
    res["ablation"] = ablation(left, prim)
    print("ablation", json.dumps(res["ablation"], default=float), flush=True)
    res["H6_speed_floor"] = speed_floor(left)
    print("H6", json.dumps(res["H6_speed_floor"], default=float), flush=True)
    res["H5_correctness"] = correctness(left, prim)
    print("H5", json.dumps({k: v for k, v in res["H5_correctness"].items()}, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)), flush=True)
    res["missing_cis"] = missing_cis(left, prim)
    print("CIs", json.dumps(res["missing_cis"], default=float), flush=True)
    right = right_tables()
    right.to_csv(OUT / "p7_right_stretches_100m.csv", index=False)
    res["H1_noise_floor"], m = noise_floor(left, right, prim)
    print("H1", json.dumps(res["H1_noise_floor"], default=float), flush=True)
    res["H3_replication_right"] = replication(right, left)
    print("H3", json.dumps(res["H3_replication_right"], default=float), flush=True)
    k_cal = res["missing_cis"]["point"]["kappa_cal"]
    nf = res["H1_noise_floor"]
    rep = res["H3_replication_right"]
    sf = res["H6_speed_floor"]["3.0"]
    res["verdicts"] = {
        "P7_H1_kappa_LR_ge_0.90": nf["kappa_road_class"] >= 0.90,
        "P7_H2_abs_kcal_minus_kLR_le_0.05": abs(k_cal - nf["kappa_road_class"]) <= 0.05,
        "P7_H3_right_share_ge_0.10_and_kcal_ge_0.90": rep["vehicle_share_log_a_dec"] >= 0.10 and rep["kappa_road_cal"] >= 0.90,
        "P7_H4_median_kcal_ge_0.90_at_2km": res["H4_data_efficiency"]["2"]["median_kappa"] >= 0.90,
        "P7_H5_cal_beats_dec_ci_above_0": res["H5_correctness"]["route_and_vehicle_out"]["ci95"]["cal_minus_dec"][0] > 0,
        "P7_H6_at_3mps_share_ge_0.10_and_kcal_ge_0.90": sf["vehicle_share_log_a_dec"] >= 0.10 and sf["kappa_road_cal"] >= 0.90,
    }
    (OUT / "p7_results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)), encoding="utf-8")
    print(json.dumps(res["verdicts"], indent=1))


if __name__ == "__main__":
    main()
