"""PVS dataset facts: ports pvs_facts.py, pvs_align_doublehit.py and pvs_maneuver_bands.py.

All tests here need the raw PVS data (``ROADSCRIBE_PVS_DIR``) and are skipped
without it. Expected values are the RESULTS.md / Appendix A numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import signal, stats

from roadscribe.io.pvs import (
    ROUTES,
    bump_mask,
    clock_offset_s,
    read_labels,
    read_logger,
    route_of,
    surface_labels,
    vehicle_of,
    xcorr_lag,
)
from roadscribe.layer1.events import bandpass, find_runs, tkeo_envelope
from roadscribe.layer1.gnss import bearing_deg, haversine_m

pytestmark = [pytest.mark.pvs, pytest.mark.slow]

FS = 100.0
TRIPS = range(1, 10)
FACT_COLS = ["timestamp", "acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard", "gyro_z_dashboard",
             "timestamp_gps", "latitude", "longitude", "speed"]

# RESULTS.md, pvs_facts.py
DUR_MIN = {1: 24.0, 2: 20.8, 3: 17.6, 4: 22.1, 5: 22.3, 6: 16.0, 7: 21.4, 8: 20.6, 9: 15.3}
PATH_KM = {1: 13.83, 2: 11.63, 3: 10.74, 4: 13.83, 5: 11.64, 6: 10.74, 7: 13.79, 8: 11.64, 9: 10.76}
SURF_CHANGES = {1: 9, 2: 4, 3: 5, 4: 9, 5: 4, 6: 5, 7: 9, 8: 4, 9: 5}
N_BUMPS = {"A": 11, "B": 7, "C": 2}
SB_V_MED = {1: 3.62, 2: 4.60, 3: 4.52, 4: 3.81, 5: 5.09, 6: 4.70, 7: 4.04, 8: 5.13, 9: 4.13}
Q_CHANGES = {1: 230, 2: 194, 3: 115, 4: 272, 5: 138, 6: 66, 7: 202, 8: 90, 9: 68}
CORR_AY = {1: 0.43, 2: 0.412, 3: 0.405, 4: 0.692, 5: 0.561, 6: 0.531, 7: 0.663, 8: 0.572, 9: 0.583}
LR_AZ_R0 = {1: 0.491, 2: 0.87, 3: 0.851, 4: 0.641, 5: -0.09, 6: -0.192, 7: 0.699, 8: 0.689, 9: 0.5}
# RESULTS.md, pvs_align_doublehit.py
YAW_LAG_S = {1: -0.09, 2: -0.01, 3: 0.03, 4: 0.03, 5: 0.54, 6: 0.22, 7: -0.01, 8: 0.02, 9: 0.04}


def _gps(L: pd.DataFrame) -> pd.DataFrame:
    return L[["timestamp_gps", "latitude", "longitude", "speed"]].drop_duplicates("timestamp_gps")


@pytest.fixture(scope="module")
def facts(pvs_root):
    rows = {}
    for i in TRIPS:
        L = read_logger(pvs_root, i, "left", FACT_COLS)
        Rr = read_logger(pvs_root, i, "right", ["acc_z_dashboard"])
        Y = read_labels(pvs_root, i)
        m = min(len(L), len(Rr), len(Y))
        t = L.timestamp.to_numpy()
        dt = np.diff(t)
        g = _gps(L)
        tg = g.timestamp_gps.to_numpy()
        seg = haversine_m(g.latitude.values[:-1], g.longitude.values[:-1], g.latitude.values[1:], g.longitude.values[1:])
        v = L.speed.to_numpy()
        fs = 1.0 / np.median(dt)
        b, a = signal.butter(2, 0.5 / (fs / 2), "low")
        ax = signal.filtfilt(b, a, L.acc_x_dashboard.values)
        ay = signal.filtfilt(b, a, L.acc_y_dashboard.values)
        gz = signal.filtfilt(b, a, L.gyro_z_dashboard.values)
        gps_dt = np.diff(tg)
        tg_mid = 0.5 * (tg[:-1] + tg[1:])
        dvdt = np.diff(g.speed.values) / gps_dt
        ok = np.isfinite(dvdt) & (gps_dt > 0.5) & (gps_dt < 1.5)
        brg = bearing_deg(g.latitude.values[:-1], g.longitude.values[:-1], g.latitude.values[1:], g.longitude.values[1:])
        dbrg = (np.diff(brg) + 180) % 360 - 180
        crate = dbrg / gps_dt[1:]
        mov = (g.speed.values[1:-1] > 3.0) & np.isfinite(crate) & (np.abs(crate) < 60)

        def corr(x, y, mk):
            x, y = x[mk], y[mk]
            return float(np.corrcoef(x - x.mean(), y - y.mean())[0, 1])

        bb, ab = signal.butter(2, [0.2 / (fs / 2), 5 / (fs / 2)], "band")
        lr = float(np.corrcoef(signal.filtfilt(bb, ab, L.acc_z_dashboard.values[:m]),
                               signal.filtfilt(bb, ab, Rr.acc_z_dashboard.values[:m]))[0, 1])
        Ym = Y.iloc[:m]
        surf = surface_labels(Ym)
        sb_runs = find_runs(bump_mask(Ym))
        q = np.maximum(np.argmax(Ym[["good_road_left", "regular_road_left", "bad_road_left"]].values, axis=1),
                       np.argmax(Ym[["good_road_right", "regular_road_right", "bad_road_right"]].values, axis=1))
        rows[i] = dict(
            fs=fs, gaps=int(np.sum(dt > 0.05)), gps_dt=float(np.median(gps_dt)), dur_min=(t[-1] - t[0]) / 60,
            path_km=float(np.nansum(seg)) / 1000, int_v_km=float(np.nansum(v[:-1] * dt)) / 1000,
            corr_ax=corr(np.interp(tg_mid, t, ax), dvdt, ok), corr_ay=corr(np.interp(tg_mid, t, ay), dvdt, ok),
            corr_gz=corr(np.interp(tg_mid, t, gz)[1:], crate, mov), lr_az=lr,
            surf_changes=int(np.sum(surf[1:] != surf[:-1])), n_bumps=len(sb_runs),
            sb_v_med=float(np.median([np.nanmean(v[s:e]) for s, e in sb_runs])), q_changes=int(np.sum(q[1:] != q[:-1])),
        )
    return rows


def test_trip_count_duration_distance(facts):
    """Appendix A: 9 trips; 180.1 min; 108.6 km."""
    assert sorted(facts) == list(TRIPS)
    assert round(sum(r["dur_min"] for r in facts.values()), 1) == 180.1
    assert round(sum(r["path_km"] for r in facts.values()), 1) == 108.6
    for i, r in facts.items():
        assert round(r["dur_min"], 1) == DUR_MIN[i]
        assert round(r["path_km"], 2) == PATH_KM[i]


def test_imu_rate_and_gaps(facts):
    """Appendix A: 100.0 Hz; no gap > 50 ms."""
    for r in facts.values():
        assert round(r["fs"], 1) == 100.0 and r["gaps"] == 0


def test_gnss_rate_and_speed_units(facts):
    """Appendix A: 1 Hz GNSS; speed in m/s (integral of v dt within 1 % of the haversine path)."""
    for r in facts.values():
        assert round(r["gps_dt"], 3) == 1.0
        assert abs(r["int_v_km"] / r["path_km"] - 1) < 0.01


def test_routes():
    """Appendix A: routes {1,4,7}, {2,5,8}, {3,6,9}, 13.8 / 11.6 / 10.7 km."""
    assert ROUTES == {"A": (1, 4, 7), "B": (2, 5, 8), "C": (3, 6, 9)}
    for r, trips in ROUTES.items():
        assert all(route_of(i) == r for i in trips)
    assert [vehicle_of(i) for i in TRIPS] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    for r, km in {"A": 13.8, "B": 11.6, "C": 10.7}.items():
        assert round(np.mean([PATH_KM[i] for i in ROUTES[r]]), 1) == km


def test_route_lengths_measured(facts):
    for r, km in {"A": 13.8, "B": 11.6, "C": 10.7}.items():
        assert round(np.mean([facts[i]["path_km"] for i in ROUTES[r]]), 1) == km


def test_labels(facts):
    """Appendix A: 54 surface transitions; 60 bump traversals (11/7/2 per route);
    bump speed median 3.6-5.1 m/s; quality labels flip 66-272 times per trip."""
    assert sum(r["surf_changes"] for r in facts.values()) == 54
    assert sum(r["n_bumps"] for r in facts.values()) == 60
    for i, r in facts.items():
        assert r["surf_changes"] == SURF_CHANGES[i]
        assert r["n_bumps"] == N_BUMPS[route_of(i)]
        assert round(r["sb_v_med"], 2) == SB_V_MED[i]
        assert 3.6 <= round(r["sb_v_med"], 1) <= 5.1
        assert r["q_changes"] == Q_CHANGES[i]
        assert 66 <= r["q_changes"] <= 272


def test_axis_convention(facts):
    """Appendix A: forward = +y_s (corr with GNSS dv/dt +0.41 to +0.69); z up."""
    for i, r in facts.items():
        assert r["corr_ay"] == pytest.approx(CORR_AY[i], abs=1e-3)
        assert 0.40 <= r["corr_ay"] <= 0.70
        assert abs(r["corr_ax"]) < 0.15
        assert r["corr_gz"] < 0  # gyro z up vs clockwise course rate


def test_left_right_zero_lag_correlation(facts):
    for i, r in facts.items():
        assert r["lr_az"] == pytest.approx(LR_AZ_R0[i], abs=1e-3)


# --------------------------------------------------------------------------- pvs_align_doublehit.py
@pytest.fixture(scope="module")
def loggers(pvs_root):
    cols = ["timestamp", "acc_z_dashboard", "gyro_z_dashboard", "acc_z_above_suspension", "acc_z_below_suspension", "speed"]
    out = {}
    for i in TRIPS:
        L = read_logger(pvs_root, i, "left", cols)
        R = read_logger(pvs_root, i, "right", cols)
        Y = read_labels(pvs_root, i, ["speed_bump_asphalt", "speed_bump_cobblestone"])
        out[i] = (L, R, Y)
    return out


def test_left_right_clock_offsets(loggers):
    """Appendix A: trip 1 -0.09 s; trip 4 +0.10 -> -0.05 s; trip 5 +0.54 s; trip 6 +0.22 s;
    others <= 0.05 s; same-side <= 0.01 s."""
    sos_yaw = signal.butter(2, [0.05 / (FS / 2), 2 / (FS / 2)], "band", output="sos")
    sos_z = signal.butter(2, [0.5 / (FS / 2), 5 / (FS / 2)], "band", output="sos")
    for i, (L, R, _) in loggers.items():
        n = min(len(L), len(R))
        off, pk = clock_offset_s(L.gyro_z_dashboard.values[:n], R.gyro_z_dashboard.values[:n], FS)
        assert round(off, 2) == YAW_LAG_S[i] and pk > 0.98
        if i not in (1, 4, 5, 6):
            assert abs(off) <= 0.05
        gzl = signal.sosfiltfilt(sos_yaw, L.gyro_z_dashboard.values[:n])
        gzr = signal.sosfiltfilt(sos_yaw, R.gyro_z_dashboard.values[:n])
        th = n // 3
        if i == 4:
            assert xcorr_lag(gzl[:th], gzr[:th], int(30 * FS))[0] / FS == pytest.approx(0.10)
            assert xcorr_lag(gzl[-th:], gzr[-th:], int(30 * FS))[0] / FS == pytest.approx(-0.05)
        azl = signal.sosfiltfilt(sos_z, L.acc_z_dashboard.values[:n])
        al = signal.sosfiltfilt(sos_z, L.acc_z_above_suspension.values[:n])
        assert abs(xcorr_lag(azl, al, int(5 * FS))[0] / FS) <= 0.01


def test_wheelbase_double_hit_cue_fails(loggers):
    """Appendix A: Spearman rho(lag, 1/v) between -0.12 and 0.00, n = 60, all p >= 0.36."""
    rows = []
    for i, (L, R, Y) in loggers.items():
        n = min(len(L), len(R))
        sb = bump_mask(Y.iloc[:n])
        v = L.speed.values[:n]
        for ch in ["acc_z_dashboard", "acc_z_above_suspension"]:
            env = tkeo_envelope(bandpass(L[ch].values[:n], FS, 4, 25, 4), FS, 0.05)
            for s, e in find_runs(sb):
                a0, a1 = max(0, s - 100), min(n, e + 100)
                w = env[a0:a1]
                vb = float(np.mean(v[s:e]))
                if vb < 1.5 or len(w) < 200:
                    continue
                w0 = w - w.mean()
                ac = np.correlate(w0, w0, mode="full")[len(w0) - 1 :]
                ac = ac / (ac[0] + 1e-12)
                lo, hi = int(0.12 * FS), min(int(1.6 * FS), len(ac) - 1)
                k = lo + int(np.argmax(ac[lo:hi]))
                pk, _ = signal.find_peaks(w, distance=int(0.1 * FS))
                tau_pk = abs(np.diff(pk[np.argsort(w[pk])[-2:]])[0]) / FS if len(pk) >= 2 else np.nan
                rows.append(dict(ch=ch, v=vb, tau_ac=k / FS, tau_pk=tau_pk))
    B = pd.DataFrame(rows)
    for ch, g in B.groupby("ch"):
        for meth in ["tau_ac", "tau_pk"]:
            gg = g.dropna(subset=[meth])
            rho, p = stats.spearmanr(gg[meth], 1.0 / gg.v)
            assert len(gg) == 60, (ch, meth)
            assert -0.125 <= rho <= 0.005, (ch, meth, rho)
            assert p >= 0.355, (ch, meth, p)


# --------------------------------------------------------------------------- pvs_maneuver_bands.py
def test_maneuver_effects_and_speed_elasticity(pvs_root):
    """§3.5 table and Appendix A speed elasticities 0.32 / 0.36 / 0.62."""
    smf = pytest.importorskip("statsmodels.formula.api")
    W = 200
    cols = ["timestamp", "acc_z_dashboard", "acc_z_below_suspension", "gyro_z_dashboard", "timestamp_gps", "speed"]
    sos_lo = signal.butter(4, [0.5 / (FS / 2), 4 / (FS / 2)], "band", output="sos")
    sos_hi = signal.butter(4, [4 / (FS / 2), 25 / (FS / 2)], "band", output="sos")
    recs = []
    for i in TRIPS:
        L = read_logger(pvs_root, i, "left", cols)
        Y = read_labels(pvs_root, i, ["asphalt_road", "speed_bump_asphalt", "speed_bump_cobblestone"])
        n = min(len(L), len(Y))
        t = L.timestamp.values[:n]
        v = L.speed.values[:n]
        g = L[["timestamp_gps", "speed"]].iloc[:n].drop_duplicates("timestamp_gps")
        alon = np.interp(t, g.timestamp_gps.values, np.gradient(g.speed.values, g.timestamp_gps.values))
        wz = signal.sosfiltfilt(signal.butter(2, 1 / (FS / 2), "low", output="sos"), np.radians(L.gyro_z_dashboard.values[:n]))
        alat = v * wz
        zlo = signal.sosfiltfilt(sos_lo, L.acc_z_dashboard.values[:n])
        zhi = signal.sosfiltfilt(sos_hi, L.acc_z_dashboard.values[:n])
        whi = signal.sosfiltfilt(sos_hi, L.acc_z_below_suspension.values[:n])
        bump = bump_mask(Y.iloc[:n]).astype(float)
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
            rms = lambda x: np.log(np.sqrt(np.mean(x[sl] ** 2)))  # noqa: E731
            recs.append(dict(trip=i, cls=cls, logv=np.log(vb), lo=rms(zlo), hi=rms(zhi), wheel_hi=rms(whi)))
    D = pd.DataFrame(recs)
    assert D.groupby("cls").size().to_dict() == {"accel": 23, "brake": 75, "cruise": 225, "turn": 92}

    def fit(y):
        m = smf.ols(f"{y} ~ logv + C(cls, Treatment('cruise')) + C(trip)", data=D).fit(
            cov_type="cluster", cov_kwds={"groups": D.trip})
        eff = {k.split("[T.")[1].rstrip("]"): (m.params[k], m.pvalues[k]) for k in m.params.index if k.startswith("C(cls")}
        return m.params["logv"], eff

    expected = {  # coefficient, p-value (RESULTS.md)
        "lo": (0.323, {"accel": (-0.175, 0.070), "brake": (-0.179, 0.0006), "turn": (0.066, 0.41)}),
        "hi": (0.363, {"accel": (-0.057, 0.47), "brake": (-0.128, 0.025), "turn": (-0.044, 0.15)}),
        "wheel_hi": (0.615, {"accel": (-0.367, 0.0028), "brake": (-0.160, 0.058), "turn": (0.055, 0.25)}),
    }
    for y, (gamma, eff) in expected.items():
        g_hat, got = fit(y)
        assert g_hat == pytest.approx(gamma, abs=1e-3), y
        for cls, (coef, p) in eff.items():
            assert got[cls][0] == pytest.approx(coef, abs=1e-3), (y, cls)
            assert got[cls][1] == pytest.approx(p, rel=0.05, abs=5e-4), (y, cls)
    D["dash_minus_wheel_hi"] = D.hi - D.wheel_hi
    _, got = fit("dash_minus_wheel_hi")
    for cls, (coef, p) in {"accel": (0.310, 0.0004), "brake": (0.032, 0.34), "turn": (-0.099, 0.0001)}.items():
        assert got[cls][0] == pytest.approx(coef, abs=1e-3), cls
        assert got[cls][1] == pytest.approx(p, rel=0.05, abs=5e-4), cls
