"""Pilot P2: channel-calibrated road index versus RoadScribe's a_dec and a gain-only calibration.

Implements PREREG_P2.md. Needs results/p1_stretches_100m.csv from p1_cross_vehicle.py.
Run from the repo root:  python roadscribe_research/identifiability/p2_channel_calibration.py
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
sys.path.insert(0, str(HERE))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

import p1_cross_vehicle as p1  # noqa: E402
from roadscribe.config import RoadScribeConfig  # noqa: E402
from roadscribe.io.pvs import ROUTES, load_pvs_trip, pvs_root, vehicle_of  # noqa: E402
from roadscribe.layer1.orientation import estimate_orientation, to_vehicle_frame  # noqa: E402
from roadscribe.paired.agreement import apply_cutpoints, block_indices, fit_cutpoints, variance_shares  # noqa: E402
from roadscribe.paired.route_align import densify, enu, match_forward, sample_theta  # noqa: E402
from roadscribe.scr.whittle import SCRData, fit_scr, multitaper_frames  # noqa: E402

OUT = HERE / "results"
V_MIN = 5.0
BIN = 100.0
NU_BAND = (0.05, 2.0)
NU_REF = 0.3
INDICES = ("log_a_dec", "log_a_dec_gain", "R3_uncal", "R3_cal")


def route_frames(route, trips):
    """Per-trip frames: periodograms, speed, theta, stretch, vehicle (dashboard, unweighted vertical)."""
    cfg = RoadScribeConfig()
    raws, vfts = {}, {}
    for t in trips:
        raw = load_pvs_trip(pvs_root(), t, side="left", placement="dashboard", with_labels=False)
        raws[t] = raw
        vfts[t] = to_vehicle_frame(raw, cfg.orientation, estimate_orientation(raw, cfg.orientation))
    g = {t: raws[t].gnss.dropna(subset=["lat", "lon"]).sort_values("t").reset_index(drop=True) for t in trips}
    lat0 = float(np.mean([g[t].lat.mean() for t in trips]))
    lon0 = float(np.mean([g[t].lon.mean() for t in trips]))
    ref_pts, ref_s = densify(enu(g[trips[0]].lat, g[trips[0]].lon, lat0, lon0), 1.0)
    out = {}
    for t in trips:
        vft = vfts[t]
        th_fix, d_fix = match_forward(enu(g[t].lat, g[t].lon, lat0, lon0), ref_pts, ref_s)
        theta, valid = sample_theta(vft.t, g[t].t.to_numpy(float), th_fix, d_fix, max_dist_m=15.0)
        f, I, centres, k = multitaper_frames(vft.a_v[:, 2], raws[t].fs)
        half = 128
        rho = np.array([vft.v[max(0, c - half):c + half].mean() for c in centres])
        ok = valid[centres] & (rho >= V_MIN)
        out[t] = dict(f=f, I=I[ok], rho=rho[ok], theta=theta[centres][ok],
                      stretch=np.floor(theta[centres][ok] / BIN).astype(int), vehicle=vehicle_of(t), k=k)
    return out


def calibrate(frames_by_route, cal_routes, lam=(10.0, 10.0, 1.0)):
    """SCR fit on the calibration routes; returns A_hat(f) per vehicle (index vehicle-1) and the fit."""
    I, rho, ch, seg_keys = [], [], [], []
    f = None
    for r in cal_routes:
        for t, fr in frames_by_route[r].items():
            f = fr["f"]
            I.append(fr["I"])
            rho.append(fr["rho"])
            ch.append(np.full(len(fr["rho"]), fr["vehicle"] - 1))
            seg_keys += [(r, int(s)) for s in fr["stretch"]]
    keys = sorted(set(seg_keys))
    kid = {key: i for i, key in enumerate(keys)}
    seg = np.array([kid[key] for key in seg_keys])
    edges = [(kid[(r, s)], kid[(r, s + 1)]) for (r, s) in keys if (r, s + 1) in kid]
    data = SCRData(f=f, I=np.vstack(I), rho=np.concatenate(rho), channel=np.concatenate(ch), segment=seg,
                   k_tapers=3, segment_edges=edges)
    fit = fit_scr(data, lam_a=lam[0], lam_b=lam[1], lam_theta=lam[2])
    A = np.array([fit.A(f, c) for c in range(3)])
    return A, fit


def gain_calibration(p1tab, cal_routes):
    d = p1.triplets(p1tab[p1tab.route.isin(cal_routes)])
    y = np.log(d["a_dec"].to_numpy())
    lv = np.log(d["v_mean"].to_numpy())
    veh = d["vehicle"].to_numpy()
    loc = d["loc"].to_numpy()
    from roadscribe.paired.agreement import _demean
    D = (veh[:, None] == np.array([2, 3])[None, :]).astype(float)
    X = np.c_[_demean(D, loc), _demean(lv, loc)]
    beta, *_ = np.linalg.lstsq(X, _demean(y, loc), rcond=None)
    g = np.array([0.0, beta[0], beta[1]])
    return g - g.mean()


def stretch_level(fr, A_hat_v, s):
    m = fr["stretch"] == s
    if m.sum() < 2:
        return float("nan"), float("nan")
    I, rho, f = fr["I"][m], fr["rho"][m], fr["f"]
    nu = f[None, :] / rho[:, None]
    band = (nu >= NU_BAND[0]) & (nu <= NU_BAND[1])
    x = (np.log(nu) - np.log(NU_REF))[band]
    if band.sum() < 5:
        return float("nan"), float("nan")
    r_unc = (np.log(I) + np.log(rho)[:, None])[band]
    r_cal = (np.log(I) - A_hat_v[None, :] + np.log(rho)[:, None])[band]
    X = np.c_[np.ones_like(x), x]
    c_unc = np.linalg.lstsq(X, r_unc, rcond=None)[0][0]
    c_cal = np.linalg.lstsq(X, r_cal, rcond=None)[0][0]
    return float(c_unc), float(c_cal)


def metrics(d: pd.DataFrame, name: str) -> dict:
    y = d[name].to_numpy(float)
    m = np.isfinite(y)
    sh = variance_shares(y[m], d["loc"].to_numpy()[m], d["vehicle"].to_numpy()[m], np.log(d["v_mean"].to_numpy()[m]))
    P = d.pivot(index="loc", columns="vehicle", values=name)
    pairs = list(combinations(sorted(d.vehicle.unique()), 2))
    rho = float(np.nanmean([stats.spearmanr(P[a], P[b], nan_policy="omit").statistic for a, b in pairs]))
    return dict(vehicle_share=sh["vehicle"], location_share=sh["location"], within_slope_log_v=sh["slope_log_v_within"],
                spearman=rho)


def lovo(d: pd.DataFrame, name: str, n_grid: int = 60) -> dict:
    dd = d[np.isfinite(d["gt_quality"]) & np.isfinite(d[name])]
    y = dd["gt_quality"].astype(int).to_numpy()
    x = dd[name].to_numpy(float)
    correct, bal = 0, []
    for h in sorted(dd.vehicle.unique()):
        tr = (dd.vehicle != h).to_numpy()
        te = ~tr
        pred = apply_cutpoints(x[te], fit_cutpoints(x[tr], y[tr], n_grid=n_grid))
        correct += int(np.sum(pred == y[te]))
        bal.append(np.mean([np.mean(pred[y[te] == k] == k) for k in np.unique(y[te])]))
    return dict(lovo_acc=correct / len(y), lovo_balanced_acc=float(np.mean(bal)))


def boot(d: pd.DataFrame, n: int, seed: int):
    rng = np.random.default_rng(seed)
    by_route = {r: sorted(g["loc"].unique(), key=lambda s: int(s.split(":")[1])) for r, g in d.groupby("route")}
    groups = {k: g for k, g in d.groupby("loc")}
    out = []
    for _ in range(n):
        parts = []
        for r, locs in by_route.items():
            for j, i in enumerate(block_indices(len(locs), p1.BLOCK, rng)):
                g = groups[locs[i]].copy()
                g["loc"] = f"{locs[i]}#{j}"
                parts.append(g)
        x = pd.concat(parts, ignore_index=True)
        out.append({k: metrics(x, k) for k in INDICES})
    return out


def evaluate(p1tab, frames, lam=(10.0, 10.0, 1.0)):
    rows = []
    fits = {}
    for r in ROUTES:
        cal = [q for q in ROUTES if q != r]
        A_hat, fit = calibrate(frames, cal, lam)
        gains = gain_calibration(p1tab, cal)
        fits[r] = dict(converged=fit.converged, message=fit.message, gains=gains.tolist(),
                       A_hat_at_1_2_5_10Hz={f"veh{v + 1}": np.interp([1, 2, 5, 10], frames[r][ROUTES[r][0]]["f"], A_hat[v]).round(3).tolist()
                                            for v in range(3)})
        d = p1.triplets(p1tab[p1tab.route == r])
        for _, row in d.iterrows():
            t = int(row["trip"])
            v = int(row["vehicle"])
            c_unc, c_cal = stretch_level(frames[r][t], A_hat[v - 1], int(row["stretch"]))
            rows.append(dict(route=r, stretch=int(row["stretch"]), loc=row["loc"], vehicle=v, trip=t,
                             v_mean=row["v_mean"], gt_quality=row["gt_quality"], gt_bump=row["gt_bump"],
                             log_a_dec=float(np.log(row["a_dec"])), log_a_dec_gain=float(np.log(row["a_dec"]) - gains[v - 1]),
                             R3_uncal=c_unc, R3_cal=c_cal))
    return pd.DataFrame(rows), fits


def main():
    p1tab = pd.read_csv(OUT / "p1_stretches_100m.csv")
    frames = {r: route_frames(r, trips) for r, trips in ROUTES.items()}
    print("frames loaded:", {r: {t: len(fr["rho"]) for t, fr in fs.items()} for r, fs in frames.items()}, flush=True)
    d, fits = evaluate(p1tab, frames)
    d.to_csv(OUT / "p2_heldout_indices.csv", index=False)
    res = {"fits": fits, "n_locations": int(d["loc"].nunique()), "n_rows": int(len(d))}
    res["point"] = {k: {**metrics(d, k), **lovo(d, k)} for k in INDICES}
    bs = boot(d, 2000, p1.SEED + 2)
    res["ci95"] = {k: {m: p1.ci([b[k][m] for b in bs]) for m in ("vehicle_share", "spearman", "within_slope_log_v")} for k in INDICES}
    res["ci95_diff"] = {
        "vehicle_share_R3cal_minus_adec": p1.ci([b["R3_cal"]["vehicle_share"] - b["log_a_dec"]["vehicle_share"] for b in bs]),
        "vehicle_share_R3cal_minus_R3uncal": p1.ci([b["R3_cal"]["vehicle_share"] - b["R3_uncal"]["vehicle_share"] for b in bs]),
        "vehicle_share_R3cal_minus_adecgain": p1.ci([b["R3_cal"]["vehicle_share"] - b["log_a_dec_gain"]["vehicle_share"] for b in bs]),
        "spearman_R3cal_minus_adec": p1.ci([b["R3_cal"]["spearman"] - b["log_a_dec"]["spearman"] for b in bs]),
        "spearman_R3cal_minus_adecgain": p1.ci([b["R3_cal"]["spearman"] - b["log_a_dec_gain"]["spearman"] for b in bs]),
    }
    pt = res["point"]
    res["verdicts"] = {
        "P2_H1": bool(pt["R3_cal"]["vehicle_share"] < pt["log_a_dec"]["vehicle_share"]
                      and pt["R3_cal"]["vehicle_share"] < pt["R3_uncal"]["vehicle_share"]),
        "P2_H2_gate": bool(pt["R3_cal"]["spearman"] > pt["log_a_dec"]["spearman"]),
        "P2_H3": bool((pt["log_a_dec_gain"]["vehicle_share"] - pt["R3_cal"]["vehicle_share"] >= 0.02)
                      or (pt["R3_cal"]["spearman"] - pt["log_a_dec_gain"]["spearman"] >= 0.05)),
        "P2_H4": bool(abs(pt["R3_cal"]["within_slope_log_v"]) < abs(pt["log_a_dec"]["within_slope_log_v"])),
    }
    # exploratory: penalty sensitivity
    res["exploratory_lambda"] = {}
    for tag, lam in (("x0.1", (1.0, 1.0, 0.1)), ("x10", (100.0, 100.0, 10.0))):
        d2, _ = evaluate(p1tab, frames, lam)
        res["exploratory_lambda"][tag] = {k: metrics(d2, k) for k in ("R3_cal",)}
    (OUT / "p2_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "fits"}, indent=1, default=float))
    print(json.dumps(res["fits"], indent=1, default=float))


if __name__ == "__main__":
    main()
