"""Pilot P3: predict a held-out vehicle's felt vertical vibration at each place (PREREG_P3.md).

SCR prediction: the place's road term (a power law in nu) from the other two vehicles, plus the
target vehicle's channel calibrated on other routes. Baselines: calibrated gamma + gains, and naive.
Run from the repo root:  python roadscribe_research/identifiability/p3_counterfactual.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
os.environ.setdefault("ROADSCRIBE_PVS_DIR", r"d:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle")

import p1_cross_vehicle as p1  # noqa: E402
import p2_channel_calibration as p2  # noqa: E402
from roadscribe.io.pvs import ROUTES  # noqa: E402
from roadscribe.layer1.iso2631 import analog_response, comfort_class  # noqa: E402
from roadscribe.paired.agreement import _demean, block_indices, weighted_kappa  # noqa: E402

OUT = HERE / "results"
NU_REF = 0.3
K = 3
BIAS = float(digamma(K) - np.log(K))  # E[log(chi2_2K / 2K)]


def band_a2(I, W2, df):
    return float(np.mean(np.sum(W2[None, :] * I, axis=1) * df))


def main():
    p1tab = pd.read_csv(OUT / "p1_stretches_100m.csv")
    frames = {r: p2.route_frames(r, trips) for r, trips in ROUTES.items()}
    f = next(iter(next(iter(frames.values())).values()))["f"]
    df = float(f[1] - f[0])
    W2 = np.abs(analog_response("Wk", f)) ** 2

    # band-limited measured values per (route, trip, stretch) on the P1 primary set
    meas = []
    for r in ROUTES:
        d = p1.triplets(p1tab[p1tab.route == r])
        for _, row in d.iterrows():
            fr = frames[r][int(row["trip"])]
            m = fr["stretch"] == int(row["stretch"])
            if m.sum() < 2:
                continue
            meas.append(dict(route=r, loc=row["loc"], stretch=int(row["stretch"]), trip=int(row["trip"]),
                             vehicle=int(row["vehicle"]), vbar=float(fr["rho"][m].mean()),
                             log_a=0.5 * np.log(band_a2(fr["I"][m], W2, df))))
    M = pd.DataFrame(meas)
    full = M.groupby("loc").vehicle.nunique()
    M = M[M["loc"].isin(full[full == 3].index)].reset_index(drop=True)

    rows = []
    for r in ROUTES:
        cal = [q for q in ROUTES if q != r]
        A_hat, fit = p2.calibrate(frames, cal)
        # gamma + gains on calibration routes (within-location OLS on the same band-limited quantity)
        C = M[M.route.isin(cal)]
        D = (C.vehicle.to_numpy()[:, None] == np.array([2, 3])[None, :]).astype(float)
        X = np.c_[_demean(D, C["loc"].to_numpy()), _demean(np.log(C.vbar.to_numpy()), C["loc"].to_numpy())]
        beta, *_ = np.linalg.lstsq(X, _demean(C.log_a.to_numpy(), C["loc"].to_numpy()), rcond=None)
        gains = np.array([0.0, beta[0], beta[1]])
        gains -= gains.mean()
        gamma = float(beta[2])
        H = M[M.route == r]
        for loc, g in H.groupby("loc"):
            s = int(g.stretch.iloc[0])
            for _, tgt in g.iterrows():
                others = g[g.vehicle != tgt.vehicle]
                # SCR: road power law from the other two vehicles' frames at this place
                xs, ys = [], []
                for _, o in others.iterrows():
                    fr = frames[r][int(o.trip)]
                    m = fr["stretch"] == s
                    nu = f[None, :] / fr["rho"][m][:, None]
                    xs.append((np.log(nu) - np.log(NU_REF)).ravel())
                    ys.append((np.log(fr["I"][m]) - A_hat[int(o.vehicle) - 1][None, :] + np.log(fr["rho"][m])[:, None]).ravel())
                xs, ys = np.concatenate(xs), np.concatenate(ys)
                c0, c1 = np.linalg.lstsq(np.c_[np.ones_like(xs), xs], ys, rcond=None)[0]
                frt = frames[r][int(tgt.trip)]
                mt = frt["stretch"] == s
                rho_t = frt["rho"][mt][:, None]
                logI = A_hat[int(tgt.vehicle) - 1][None, :] + c0 + c1 * (np.log(f[None, :] / rho_t) - np.log(NU_REF)) - np.log(rho_t)
                pred_scr = 0.5 * np.log(band_a2(np.exp(logI - BIAS), W2, df))
                pred_gg = float(np.mean([o.log_a + gamma * (np.log(tgt.vbar) - np.log(o.vbar)) + gains[int(tgt.vehicle) - 1]
                                         - gains[int(o.vehicle) - 1] for _, o in others.iterrows()]))
                pred_naive = float(others.log_a.mean())
                rows.append(dict(route=r, loc=loc, stretch=s, vehicle=int(tgt.vehicle), vbar=tgt.vbar, measured=tgt.log_a,
                                 scr=pred_scr, gamma_gain=pred_gg, naive=pred_naive, gamma_cal=gamma))
        print(f"held-out route {r}: SCR fit converged={fit.converged}; calibrated gamma={gamma:.3f}; gains={np.round(gains, 3).tolist()}",
              flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "p3_predictions.csv", index=False)

    def summarise(P):
        out = {}
        mc = np.array([comfort_class(np.exp(v)) for v in P.measured])
        for k in ("scr", "gamma_gain", "naive"):
            e = P[k] - P.measured
            pc = np.array([comfort_class(np.exp(v)) for v in P[k]])
            out[k] = dict(rmse=float(np.sqrt(np.mean(e**2))), bias=float(np.mean(e)),
                          class_exact=float(np.mean(pc == mc)), class_kappa=weighted_kappa(pc, mc, 6))
        return out

    res = {"n": int(len(P)), "n_locations": int(P["loc"].nunique()), "point": summarise(P),
           "calibrated_gamma_per_heldout_route": P.groupby("route").gamma_cal.first().to_dict()}
    rng = np.random.default_rng(p1.SEED + 3)
    by_route = {r: sorted(g["loc"].unique(), key=lambda x: int(x.split(":")[1])) for r, g in P.groupby("route")}
    groups = {k: g for k, g in P.groupby("loc")}
    diffs, diffs_exact = [], []
    for _ in range(2000):
        parts = [groups[locs[i]] for rr, locs in by_route.items() for i in block_indices(len(locs), p1.BLOCK, rng)]
        Q = pd.concat(parts, ignore_index=True)
        s_ = summarise(Q)
        diffs.append(s_["scr"]["rmse"] - s_["gamma_gain"]["rmse"])
        diffs_exact.append(s_["scr"]["class_exact"] - s_["gamma_gain"]["class_exact"])
    res["ci95_rmse_scr_minus_gamma_gain"] = p1.ci(diffs)
    res["ci95_class_exact_scr_minus_gamma_gain"] = p1.ci(diffs_exact)
    pt = res["point"]
    res["verdicts"] = {"P3_H1": pt["scr"]["rmse"] < pt["gamma_gain"]["rmse"],
                       "P3_H2": pt["scr"]["class_exact"] > pt["gamma_gain"]["class_exact"],
                       "sanity_both_beat_naive": (pt["scr"]["rmse"] < pt["naive"]["rmse"]) and (pt["gamma_gain"]["rmse"] < pt["naive"]["rmse"])}
    (OUT / "p3_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
