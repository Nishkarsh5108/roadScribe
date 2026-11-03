"""Measure PVS facts needed for the RoadScribe schema/method (read-only on raw data)."""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import signal

RAW = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle"
COLS = ["timestamp", "acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard",
        "gyro_x_dashboard", "gyro_y_dashboard", "gyro_z_dashboard",
        "acc_z_above_suspension", "acc_z_below_suspension",
        "timestamp_gps", "latitude", "longitude", "speed"]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(x, y))


def runs(mask):
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


rows = []
for i in range(1, 10):
    d = os.path.join(RAW, f"PVS {i}")
    L = pd.read_csv(os.path.join(d, "dataset_gps_mpu_left.csv"), usecols=COLS)
    R = pd.read_csv(os.path.join(d, "dataset_gps_mpu_right.csv"), usecols=COLS)
    Y = pd.read_csv(os.path.join(d, "dataset_labels.csv"))
    m = min(len(L), len(R), len(Y))
    t = L.timestamp.values
    dt = np.diff(t)
    same_t = float(np.mean(np.isclose(L.timestamp.values[:m], R.timestamp.values[:m], atol=5e-3)))

    # GPS fixes (unique timestamp_gps)
    g = L[["timestamp_gps", "latitude", "longitude", "speed"]].drop_duplicates("timestamp_gps")
    tg = g.timestamp_gps.values
    gps_dt = np.diff(tg)
    seg = haversine(g.latitude.values[:-1], g.longitude.values[:-1], g.latitude.values[1:], g.longitude.values[1:])
    path_len = float(np.nansum(seg))
    v = L.speed.values
    int_v = float(np.nansum(v[:-1] * dt))  # if speed in m/s this ~ path_len

    # forward axis: LP IMU averaged per GPS fix vs GPS dv/dt
    fs = 1.0 / np.median(dt)
    b, a = signal.butter(2, 0.5 / (fs / 2), "low")
    ax = signal.filtfilt(b, a, L.acc_x_dashboard.values)
    ay = signal.filtfilt(b, a, L.acc_y_dashboard.values)
    gz = signal.filtfilt(b, a, L.gyro_z_dashboard.values)
    tg_mid = 0.5 * (tg[:-1] + tg[1:])
    dvdt = np.diff(g.speed.values) / gps_dt
    ax_g = np.interp(tg_mid, t, ax)
    ay_g = np.interp(tg_mid, t, ay)
    gz_g = np.interp(tg_mid, t, gz)
    brg = bearing(g.latitude.values[:-1], g.longitude.values[:-1], g.latitude.values[1:], g.longitude.values[1:])
    # course rate from consecutive bearings, only when moving
    dbrg = np.diff(brg)
    dbrg = (dbrg + 180) % 360 - 180
    crate = dbrg / gps_dt[1:]
    gz_c = gz_g[1:]
    mov = (g.speed.values[1:-1] > 3.0) & np.isfinite(crate) & (np.abs(crate) < 60)
    ok = np.isfinite(dvdt) & (gps_dt > 0.5) & (gps_dt < 1.5)

    def c(x, y, mk):
        x, y = x[mk], y[mk]
        return float(np.corrcoef(x - x.mean(), y - y.mean())[0, 1]) if len(x) > 10 else float("nan")

    # left vs right dashboard (band 0.2-5 Hz)
    bb, ab = signal.butter(2, [0.2 / (fs / 2), 5 / (fs / 2)], "band")
    lr = {}
    for col in ["acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard", "gyro_x_dashboard", "gyro_y_dashboard", "gyro_z_dashboard"]:
        xl = signal.filtfilt(bb, ab, L[col].values[:m])
        xr = signal.filtfilt(bb, ab, R[col].values[:m])
        lr[col.replace("_dashboard", "")] = round(float(np.corrcoef(xl, xr)[0, 1]), 3)

    # labels
    surf = np.full(m, "none", dtype=object)
    surf[Y.asphalt_road.values[:m] == 1] = "asphalt"
    surf[Y.cobblestone_road.values[:m] == 1] = "cobble"
    surf[Y.dirt_road.values[:m] == 1] = "dirt"
    multi = int(np.sum((Y.asphalt_road.values[:m] + Y.cobblestone_road.values[:m] + Y.dirt_road.values[:m]) != 1))
    surf_changes = int(np.sum(surf[1:] != surf[:-1]))
    sb = ((Y.speed_bump_asphalt.values[:m] == 1) | (Y.speed_bump_cobblestone.values[:m] == 1))
    sb_runs = runs(sb)
    sb_dur = [(e - s) / fs for s, e in sb_runs]
    sb_v = [float(np.nanmean(v[s:e])) for s, e in sb_runs]
    qL = np.argmax(Y[["good_road_left", "regular_road_left", "bad_road_left"]].values[:m], axis=1)
    qR = np.argmax(Y[["good_road_right", "regular_road_right", "bad_road_right"]].values[:m], axis=1)
    q_changes = int(np.sum(np.maximum(qL, qR)[1:] != np.maximum(qL, qR)[:-1]))

    rows.append(dict(
        trip=i, n_left=len(L), n_right=len(R), n_lab=len(Y), same_t_frac=round(same_t, 4),
        fs_median=round(fs, 2), dt_p99=round(float(np.percentile(dt, 99)), 4), n_gaps_gt_50ms=int(np.sum(dt > 0.05)),
        gps_n=len(g), gps_dt_median=round(float(np.median(gps_dt)), 3),
        dur_min=round((t[-1] - t[0]) / 60, 1), path_km=round(path_len / 1000, 2), int_v_km=round(int_v / 1000, 2),
        v_max=round(float(np.nanmax(v)), 2), v_mean_moving=round(float(np.nanmean(v[v > 1])), 2),
        frac_stationary=round(float(np.mean(v < 1)), 3),
        corr_ax_dvdt=round(c(ax_g, dvdt, ok), 3), corr_ay_dvdt=round(c(ay_g, dvdt, ok), 3),
        corr_gz_courserate=round(c(gz_c, crate, mov), 3),
        lr_corr=lr,
        surf_multi_or_none=multi, surf_changes=surf_changes,
        surf_frac={k: round(float(np.mean(surf == k)), 3) for k in ["asphalt", "cobble", "dirt", "none"]},
        n_speed_bumps=len(sb_runs), sb_dur_med=round(float(np.median(sb_dur)), 2) if sb_dur else None,
        sb_v_med=round(float(np.median(sb_v)), 2) if sb_v else None, quality_changes=q_changes,
    ))
    print(json.dumps(rows[-1]), flush=True)

df = pd.DataFrame(rows)
print("\nTOTAL dur_min=%.1f path_km=%.2f speed_bumps=%d surf_changes=%d" % (
    df.dur_min.sum(), df.path_km.sum(), df.n_speed_bumps.sum(), df.surf_changes.sum()))
