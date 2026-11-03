"""(1) Left/right dashboard clock alignment via cross-correlation.
(2) Wheelbase double-hit test at labelled speed bumps: does the lag between
    the two vertical impacts scale as L/v (axle spacing) or stay constant (body mode)?"""
import os, json
import numpy as np
import pandas as pd
from scipy import signal, stats

RAW = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle"
FS = 100.0
COLS = ["timestamp", "acc_z_dashboard", "gyro_z_dashboard", "acc_z_above_suspension", "acc_z_below_suspension", "speed"]


def xcorr_lag(a, b, maxlag):
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    c = signal.correlate(a, b, mode="full", method="fft") / n
    lags = np.arange(-n + 1, n)
    k = np.abs(lags) <= maxlag
    i = np.argmax(c[k])
    return int(lags[k][i]), float(c[k][i]), float(c[lags == 0][0])


def runs(mask):
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def tkeo_env(x, smooth=5):
    psi = np.zeros_like(x)
    psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    psi = np.maximum(psi, 0)
    return np.convolve(psi, np.ones(smooth) / smooth, mode="same")


sos_bp = signal.butter(4, [4 / (FS / 2), 25 / (FS / 2)], "band", output="sos")
sos_yaw = signal.butter(2, [0.05 / (FS / 2), 2 / (FS / 2)], "band", output="sos")
sos_z = signal.butter(2, [0.5 / (FS / 2), 5 / (FS / 2)], "band", output="sos")

align = []
bumps = []
for i in range(1, 10):
    d = os.path.join(RAW, f"PVS {i}")
    L = pd.read_csv(os.path.join(d, "dataset_gps_mpu_left.csv"), usecols=COLS)
    R = pd.read_csv(os.path.join(d, "dataset_gps_mpu_right.csv"), usecols=COLS)
    Y = pd.read_csv(os.path.join(d, "dataset_labels.csv"), usecols=["speed_bump_asphalt", "speed_bump_cobblestone"])
    n = min(len(L), len(R))
    gzl = signal.sosfiltfilt(sos_yaw, L.gyro_z_dashboard.values[:n])
    gzr = signal.sosfiltfilt(sos_yaw, R.gyro_z_dashboard.values[:n])
    azl = signal.sosfiltfilt(sos_z, L.acc_z_dashboard.values[:n])
    azr = signal.sosfiltfilt(sos_z, R.acc_z_dashboard.values[:n])
    lag_g, pk_g, z0_g = xcorr_lag(gzl, gzr, int(30 * FS))
    lag_z, pk_z, z0_z = xcorr_lag(azl, azr, int(30 * FS))
    # drift: lag in first and last thirds (yaw channel)
    th = n // 3
    lag_a, pk_a, _ = xcorr_lag(gzl[:th], gzr[:th], int(30 * FS))
    lag_b, pk_b, _ = xcorr_lag(gzl[-th:], gzr[-th:], int(30 * FS))
    # same-side, same-logger check: dashboard vs above-suspension (left file)
    al = signal.sosfiltfilt(sos_z, L.acc_z_above_suspension.values[:n])
    lag_s, pk_s, z0_s = xcorr_lag(azl, al, int(5 * FS))
    align.append(dict(trip=i, yaw_lag_s=lag_g / FS, yaw_peak=round(pk_g, 3), yaw_r_at0=round(z0_g, 3),
                      az_lag_s=lag_z / FS, az_peak=round(pk_z, 3), az_r_at0=round(z0_z, 3),
                      yaw_lag_first_third_s=lag_a / FS, yaw_lag_last_third_s=lag_b / FS,
                      sameside_dash_vs_above_lag_s=lag_s / FS, sameside_r_at0=round(z0_s, 3)))
    print(json.dumps(align[-1]), flush=True)

    # ---- double-hit at speed bumps (left dashboard and left above-suspension, no L/R averaging)
    sb = (Y.speed_bump_asphalt.values[:n] == 1) | (Y.speed_bump_cobblestone.values[:n] == 1)
    v = L.speed.values[:n]
    for ch in ["acc_z_dashboard", "acc_z_above_suspension"]:
        env = tkeo_env(signal.sosfiltfilt(sos_bp, L[ch].values[:n]))
        for (s, e) in runs(sb):
            a0, a1 = max(0, s - 100), min(n, e + 100)
            w = env[a0:a1]
            vb = float(np.mean(v[s:e]))
            if vb < 1.5 or len(w) < 200:
                continue
            w0 = w - w.mean()
            ac = np.correlate(w0, w0, mode="full")[len(w0) - 1:]
            ac = ac / (ac[0] + 1e-12)
            lo, hi = int(0.12 * FS), min(int(1.6 * FS), len(ac) - 1)
            k = lo + int(np.argmax(ac[lo:hi]))
            # two-largest-peaks method
            pk, _ = signal.find_peaks(w, distance=int(0.1 * FS))
            if len(pk) >= 2:
                top = pk[np.argsort(w[pk])[-2:]]
                tau_pk = abs(top[1] - top[0]) / FS
            else:
                tau_pk = np.nan
            bumps.append(dict(trip=i, veh=(i - 1) // 3 + 1, route="ABC"[(i - 1) % 3], ch=ch, v=vb,
                              tau_ac=k / FS, ac_peak=float(ac[k]), tau_pk=tau_pk))

A = pd.DataFrame(align)
B = pd.DataFrame(bumps)
B.to_csv(os.path.join(os.path.dirname(__file__), "pvs_doublehit.csv"), index=False)
print()
for ch, g in B.groupby("ch"):
    for meth in ["tau_ac", "tau_pk"]:
        gg = g.dropna(subset=[meth])
        rho, p = stats.spearmanr(gg[meth], 1.0 / gg.v)
        Lhat = gg.v * gg[meth]
        # through-origin LS slope tau = L * (1/v)
        x = 1.0 / gg.v.values
        slope = float(np.sum(x * gg[meth].values) / np.sum(x * x))
        per_veh = {int(k): round(float(np.median(gv.v * gv[meth])), 2) for k, gv in gg.groupby("veh")}
        iqr_veh = {int(k): [round(float(np.percentile(gv.v * gv[meth], 25)), 2), round(float(np.percentile(gv.v * gv[meth], 75)), 2)] for k, gv in gg.groupby("veh")}
        print(f"{ch:24s} {meth}: n={len(gg)} spearman(tau,1/v)={rho:.3f} p={p:.2g} "
              f"L_slope={slope:.2f} m  median L per vehicle={per_veh} IQR={iqr_veh}")
