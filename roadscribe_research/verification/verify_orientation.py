"""Validate the orientation estimator on PVS (left logger, dashboard IMU):
(1) gated quasi-static gravity -> Rodrigues tilt; (2) closed-form 2-D Procrustes yaw against
GPS longitudinal accel and v*yaw-rate. Expected on unrotated PVS: yaw ~ +90 deg (forward = +y_s).
(3) Equivariance under 20 random 3-D rotations per trip."""
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial.transform import Rotation as Rot

RAW = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle"
FS, G = 100.0, 9.80665
COLS = ["timestamp", "acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard",
        "gyro_x_dashboard", "gyro_y_dashboard", "gyro_z_dashboard", "timestamp_gps", "speed"]
sos_lp = signal.butter(2, 0.5 / (FS / 2), "low", output="sos")
rng = np.random.default_rng(11)


def rodrigues_to_z(u):
    u = u / np.linalg.norm(u)
    ez = np.array([0.0, 0.0, 1.0])
    k = np.cross(u, ez)
    s, c = np.linalg.norm(k), float(u @ ez)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = k / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def estimate(f_b, w_b, t, tg, vg):
    """Return R_vb (vehicle <- body), yaw (deg) and number of quasi-static samples."""
    fl = signal.sosfiltfilt(sos_lp, f_b, axis=0)
    wl = signal.sosfiltfilt(sos_lp, w_b, axis=0)
    v = np.interp(t, tg, vg)
    alon = np.interp(t, tg, np.gradient(vg, tg))
    qs = (np.abs(np.linalg.norm(fl, axis=1) - G) < 0.3) & (np.linalg.norm(wl, axis=1) < np.radians(3)) \
        & (np.abs(alon) < 0.3) & (v > 3.0)
    g_hat = np.median(fl[qs], axis=0)
    R1 = rodrigues_to_z(g_hat)                         # levels body: R1 @ up_b = e_z
    hl = (fl @ R1.T)[:, :2]                            # leveled horizontal specific force (LP)
    wz = (wl @ R1.T)[:, 2]                             # yaw rate, invariant to heading misalignment
    tm = tg[1:-1]
    H = np.column_stack([np.interp(tm, t, hl[:, 0]), np.interp(tm, t, hl[:, 1])])
    r = np.column_stack([np.interp(tm, t, alon), np.interp(tm, t, v * wz)])
    ok = np.interp(tm, t, v) > 3.0
    M = H[ok].T @ r[ok]                                 # M_ij = sum_k H_ki r_kj
    psi = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    c, s = np.cos(psi), np.sin(psi)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])  # (h_x,h_y) -> (a_lon, a_lat[left])
    return Rz @ R1, np.degrees(psi), int(qs.sum())


rows = []
for i in range(1, 10):
    L = pd.read_csv(os.path.join(RAW, f"PVS {i}", "dataset_gps_mpu_left.csv"), usecols=COLS)
    t = L.timestamp.values
    f_s = L[["acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard"]].values
    w_s = np.radians(L[["gyro_x_dashboard", "gyro_y_dashboard", "gyro_z_dashboard"]].values)
    g = L[["timestamp_gps", "speed"]].drop_duplicates("timestamp_gps")
    tg, vg = g.timestamp_gps.values, g.speed.values
    R0, yaw0, nqs = estimate(f_s, w_s, t, tg, vg)
    tilt0 = np.degrees(np.arccos(np.clip(R0[2] @ np.array([0, 0, 1.0]), -1, 1)))
    errs = []
    for _ in range(20):
        Rr = Rot.random(random_state=rng).as_matrix()  # body = Rr @ sensor
        Rb, _, _ = estimate(f_s @ Rr.T, w_s @ Rr.T, t, tg, vg)
        Rexp = R0 @ Rr.T                                # expected vehicle <- body
        errs.append(np.degrees(Rot.from_matrix(Rb @ Rexp.T).magnitude()))
    rows.append(dict(trip=i, veh=(i - 1) // 3 + 1, yaw_deg=round(yaw0, 2), sensor_tilt_deg=round(tilt0, 2), n_quasistatic=nqs,
                     equivariance_err_med_deg=round(float(np.median(errs)), 4), equivariance_err_max_deg=round(float(np.max(errs)), 4)))
    print(rows[-1], flush=True)
