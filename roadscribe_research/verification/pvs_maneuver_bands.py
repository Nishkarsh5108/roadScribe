"""Pre-specified test: do driver maneuvers (braking/acceleration/turning) raise the
dashboard vertical vibration energy, and in which band? Asphalt only, no speed bumps
within +-2 s, 5-20 m/s, 2-s non-overlapping windows, left logger only (no L/R averaging).
Model: log RMS ~ log v + maneuver class + trip fixed effects, cluster-robust SE by trip."""
import os
import numpy as np
import pandas as pd
from scipy import signal
import statsmodels.formula.api as smf

RAW = r"D:\Coding\Hackathon\GFG\ARM\ARM\ml_model_work\data\raw\multi_class_kaggle"
FS = 100.0
W = 200
COLS = ["timestamp", "acc_z_dashboard", "acc_z_below_suspension", "gyro_z_dashboard", "timestamp_gps", "speed"]
sos_lo = signal.butter(4, [0.5 / (FS / 2), 4 / (FS / 2)], "band", output="sos")
sos_hi = signal.butter(4, [4 / (FS / 2), 25 / (FS / 2)], "band", output="sos")

recs = []
for i in range(1, 10):
    d = os.path.join(RAW, f"PVS {i}")
    L = pd.read_csv(os.path.join(d, "dataset_gps_mpu_left.csv"), usecols=COLS)
    Y = pd.read_csv(os.path.join(d, "dataset_labels.csv"), usecols=["asphalt_road", "speed_bump_asphalt", "speed_bump_cobblestone"])
    n = min(len(L), len(Y))
    t = L.timestamp.values[:n]
    v = L.speed.values[:n]
    g = L[["timestamp_gps", "speed"]].iloc[:n].drop_duplicates("timestamp_gps")
    tg, vg = g.timestamp_gps.values, g.speed.values
    alon_g = np.gradient(vg, tg)
    alon = np.interp(t, tg, alon_g)
    wz = np.radians(L.gyro_z_dashboard.values[:n])
    wz = signal.sosfiltfilt(signal.butter(2, 1 / (FS / 2), "low", output="sos"), wz)
    alat = v * wz
    zlo = signal.sosfiltfilt(sos_lo, L.acc_z_dashboard.values[:n])
    zhi = signal.sosfiltfilt(sos_hi, L.acc_z_dashboard.values[:n])
    whi = signal.sosfiltfilt(sos_hi, L.acc_z_below_suspension.values[:n])
    bump = ((Y.speed_bump_asphalt.values[:n] == 1) | (Y.speed_bump_cobblestone.values[:n] == 1)).astype(float)
    bump_near = np.convolve(bump, np.ones(401), mode="same") > 0
    asph = Y.asphalt_road.values[:n] == 1
    for s in range(0, n - W, W):
        sl = slice(s, s + W)
        if not asph[sl].all() or bump_near[sl].any():
            continue
        vb = v[sl].mean()
        if not (5 <= vb <= 20):
            continue
        al, at = alon[sl].mean(), np.abs(alat[sl]).max()
        if al < -1.0:
            cls = "brake"
        elif al > 1.0:
            cls = "accel"
        elif at > 1.5 and abs(al) < 0.5:
            cls = "turn"
        elif abs(al) < 0.3 and at < 0.5:
            cls = "cruise"
        else:
            continue
        recs.append(dict(trip=i, cls=cls, logv=np.log(vb),
                         lo=np.log(np.sqrt(np.mean(zlo[sl] ** 2))),
                         hi=np.log(np.sqrt(np.mean(zhi[sl] ** 2))),
                         wheel_hi=np.log(np.sqrt(np.mean(whi[sl] ** 2)))))

D = pd.DataFrame(recs)
print(D.groupby("cls").size().to_dict())
for y in ["lo", "hi", "wheel_hi"]:
    m = smf.ols(f"{y} ~ logv + C(cls, Treatment('cruise')) + C(trip)", data=D).fit(
        cov_type="cluster", cov_kwds={"groups": D.trip})
    out = {k.split("[T.")[1].rstrip("]"): (round(m.params[k], 3), round(m.bse[k], 3), round(m.pvalues[k], 4))
           for k in m.params.index if k.startswith("C(cls")}
    print(f"{y:9s} logv={m.params['logv']:.3f}  class effects vs cruise (coef, SE, p): {out}")
# dashboard-high minus wheel-high: is any dashboard high-band change road-explained?
D["dash_minus_wheel_hi"] = D.hi - D.wheel_hi
m = smf.ols("dash_minus_wheel_hi ~ logv + C(cls, Treatment('cruise')) + C(trip)", data=D).fit(cov_type="cluster", cov_kwds={"groups": D.trip})
print("dash-wheel ", {k.split("[T.")[1].rstrip("]"): (round(m.params[k], 3), round(m.bse[k], 3), round(m.pvalues[k], 4)) for k in m.params.index if k.startswith("C(cls")})
