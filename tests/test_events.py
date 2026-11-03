"""TKEO impulse detection, maneuver states and the decision rule (§3.5)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from roadscribe.config import EventConfig
from roadscribe.layer1 import events as ev
from synth import bump, vehicle_frame

FS = 100.0


def _background(n: int, sd: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sos = signal.butter(2, [1 / (FS / 2), 30 / (FS / 2)], "band", output="sos")
    x = signal.sosfiltfilt(sos, rng.normal(0, 1, n))
    return x * sd / x.std()


@pytest.fixture(scope="module")
def bumpy():
    """300 s at 8 m/s: six bumps on smooth road and a 60 s rough stretch with no bump."""
    n = int(300 * FS)
    t = np.arange(n) / FS
    az = _background(n, 0.2, 0)
    rough = (t >= 150) & (t < 210)
    az[rough] *= 5
    bumps_t = [20.0, 55.0, 90.0, 120.0, 240.0, 270.0]
    b = bump(FS)
    for tb in bumps_t:
        i = int(tb * FS)
        az[i : i + len(b)] += b
    rng = np.random.default_rng(1)
    a_v = np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), az])
    w_v = np.column_stack([np.zeros(n), np.zeros(n), rng.normal(0, 0.002, n)])
    vft = vehicle_frame(a_v, np.full(n, 8.0), FS, w_v=w_v, a_lon=np.zeros(n))
    return vft, bumps_t


# --------------------------------------------------------------------------- operators
@pytest.mark.parametrize("A,Omega,phi", [(1.0, 0.3, 0.0), (2.5, 1.1, 0.7), (0.4, 2.0, -1.2)])
def test_tkeo_of_a_cosine(A, Omega, phi):
    """Psi = A^2 sin^2(Omega) for x = A cos(Omega n + phi) (Kaiser 1990)."""
    n = np.arange(500)
    psi = ev.tkeo(A * np.cos(Omega * n + phi))
    np.testing.assert_allclose(psi[1:-1], A**2 * np.sin(Omega) ** 2, rtol=1e-9, atol=1e-12)
    assert psi[0] == psi[-1] == 0.0


def test_envelope_is_nonnegative_and_smoothed():
    x = np.zeros(1000)
    x[500] = 1.0
    E = ev.tkeo_envelope(x, FS, 0.05)
    assert E.min() >= 0
    assert (E > 0).sum() == 5  # 50 ms boxcar at 100 Hz


def test_adaptive_threshold_tracks_background():
    n = int(60 * FS)
    E = np.abs(_background(n, 1.0, 2))
    E[n // 2 :] *= 10
    theta = ev.adaptive_threshold(E, FS, 10.0, 15.0)
    assert theta[n // 2 + int(15 * FS)] > 5 * theta[int(15 * FS)]


def test_adaptive_threshold_exact_at_hop_centres():
    rng = np.random.default_rng(3)
    E = rng.exponential(size=5000)
    theta = ev.adaptive_threshold(E, FS, window_s=10.0, kappa=3.0, hop_s=0.1)
    W = 1001
    c = 2000
    win = E[c - W // 2 : c + W // 2 + 1]
    med = np.median(win)
    assert theta[c] == pytest.approx(med + 3.0 * 1.4826 * np.median(np.abs(win - med)))


def test_runs_and_merging():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1], bool)
    runs = ev.find_runs(m)
    assert runs == [(1, 3), (5, 6), (10, 12)]
    assert ev.merge_runs(runs, max_gap=3) == [(1, 6), (10, 12)]
    assert ev.merge_runs(runs, max_gap=5) == [(1, 12)]


def test_max_event_duration():
    cfg = EventConfig()
    assert ev.max_event_duration(10.0, cfg) == pytest.approx(0.8)
    assert ev.max_event_duration(3.0, cfg) == pytest.approx(1.5)
    assert ev.max_event_duration(0.0, cfg) == pytest.approx(3.5)  # floored speed


# --------------------------------------------------------------------------- detection
def test_detects_every_bump_and_ignores_rough_background(bumpy):
    vft, bumps_t = bumpy
    res = ev.detect_events(vft)
    imp = [e for e in res.events if e.kind == "impulse"]
    assert len(imp) == len(bumps_t)
    for e, tb in zip(imp, bumps_t):
        assert e.t0 - 0.1 <= tb <= e.t1 + 0.1
        assert e.label == "defect" and e.maneuver == "cruise"
        assert e.s == pytest.approx(8.0 * tb, abs=3.0)
        assert e.mtvv_z > 0.3
    assert not any(150 <= e.t0 < 210 for e in imp)
    assert [e.event_id for e in res.events] == [f"E{k + 1}" for k in range(len(res.events))]


def test_long_runs_go_to_layer2(bumpy):
    vft, _ = bumpy
    a_v = vft.a_v.copy()
    i = int(30 * FS)
    a_v[i : i + int(3 * FS), 2] += 4.0 * np.sin(2 * np.pi * 12 * np.arange(int(3 * FS)) / FS)
    vft2 = vehicle_frame(a_v, vft.v, FS, w_v=vft.w_v, a_lon=vft.a_lon_gnss)
    res = ev.detect_events(vft2)
    assert any(t0 <= 31.5 <= t1 for t0, t1 in res.rough_runs)
    assert not any(30.5 <= e.t0 <= 32.5 for e in res.events)


def test_event_coordinates_from_gnss(bumpy):
    import pandas as pd

    vft, _ = bumpy
    g = pd.DataFrame({"t": [0.0, 300.0], "lat": [52.0, 52.01], "lon": [4.0, 4.02]})
    e = ev.detect_events(vft, gnss=g).events[0]
    assert e.lat == pytest.approx(52.0 + 0.01 * e.t0 / 300, abs=1e-5)
    assert ev.detect_events(vft).events[0].lat != ev.detect_events(vft).events[0].lat  # NaN without GNSS


# --------------------------------------------------------------------------- maneuvers and rule
def test_maneuver_states():
    n = int(60 * FS)
    t = np.arange(n) / FS
    v = np.full(n, 10.0)
    a_lon = np.where((t > 10) & (t < 15), -2.0, np.where((t > 20) & (t < 25), 1.5, 0.0))
    wz = np.where((t > 30) & (t < 36), 0.3, 0.0)
    # lane change: +0.25 rad/s then -0.25 rad/s, 1 s each -> net heading ~0
    wz = wz + np.where((t > 45) & (t < 46), 0.25, 0.0) - np.where((t > 46) & (t < 47), 0.25, 0.0)
    st = ev.maneuver_states(v, a_lon, np.zeros(n), wz, FS)
    assert st.brake[int(12 * FS)] and not st.brake[int(5 * FS)]
    assert st.accel[int(22 * FS)]
    assert st.turn[int(33 * FS)] and not st.turn[int(46 * FS)]
    assert st.swerve[int(46 * FS)] and not st.swerve[int(33 * FS)]
    assert st.at(int(12 * FS), int(13 * FS)) == "brake"
    assert st.at(0, int(2 * FS)) == "cruise"


def test_stop_go_needs_low_speed_and_jerk():
    n = int(20 * FS)
    t = np.arange(n) / FS
    v = np.where(t < 10, 1.0, 8.0)
    a_x = np.where((t > 4) & (t < 4.3), 3.0, 0.0) + np.where((t > 14) & (t < 14.3), 3.0, 0.0)
    st = ev.maneuver_states(v, np.zeros(n), a_x, None, FS)
    assert st.stop_go[int(4 * FS) : int(4.5 * FS)].any()
    assert not st.stop_go[int(14 * FS) : int(14.5 * FS)].any()
    assert not st.turn.any() and not st.swerve.any()  # no gyroscope


@pytest.mark.parametrize(
    "m,jx,rho,d,dmax,label",
    [
        ("accel", 5.0, 0.5, 0.2, 0.8, "artifact"),
        ("stop_go", 5.0, 0.5, 0.2, 0.8, "artifact"),
        ("accel", 1.0, 0.5, 0.2, 0.8, "ambiguous"),  # no jerk -> not an artifact; rho < rho0
        ("accel", 5.0, 2.0, 0.2, 0.8, "defect"),  # high band ratio wins
        ("brake", 9.0, 2.0, 0.2, 0.8, "defect"),  # braking never vetoes
        ("turn", 9.0, 2.0, 0.2, 0.8, "defect"),  # turning never vetoes
        ("cruise", 0.0, 2.0, 1.0, 0.8, "ambiguous"),  # too long
        ("cruise", 0.0, 0.5, 0.2, 0.8, "ambiguous"),
    ],
)
def test_decision_rule(m, jx, rho, d, dmax, label):
    assert ev.label_event(m, jx, rho, d, dmax, EventConfig()) == label


def test_band_ratio_separates_impact_from_body_motion():
    n = int(10 * FS)
    t = np.arange(n) / FS
    body = np.sin(2 * np.pi * 1.5 * t)
    hf = body.copy()
    i = int(5 * FS)
    b = bump(FS)
    hf[i : i + len(b)] += b
    assert ev.band_ratio(hf, FS, i, i + len(b)) > 10 * ev.band_ratio(body, FS, i, i + len(b))


def test_avoidance_event_from_isolated_swerve():
    n = int(60 * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(4)
    wz = np.where((t > 30) & (t < 31), 0.3, 0.0) - np.where((t > 31) & (t < 32), 0.3, 0.0)
    a_v = np.column_stack([np.zeros(n), np.zeros(n), _background(n, 0.2, 5)])
    w_v = np.column_stack([np.zeros(n), np.zeros(n), wz + rng.normal(0, 0.001, n)])
    vft = vehicle_frame(a_v, np.full(n, 10.0), FS, w_v=w_v, a_lon=np.zeros(n))
    res = ev.detect_events(vft)
    av = [e for e in res.events if e.kind == "avoidance"]
    assert len(av) == 1
    assert av[0].maneuver == "swerve" and av[0].label == "ambiguous"
    assert 0.4 <= av[0].p_defect < 0.7  # hedged as "possibly"
    # a bump within +-2 s suppresses the avoidance event
    b = bump(FS)
    a_v[int(31.5 * FS) : int(31.5 * FS) + len(b), 2] += b
    vft2 = vehicle_frame(a_v, np.full(n, 10.0), FS, w_v=w_v, a_lon=np.zeros(n))
    assert not [e for e in ev.detect_events(vft2).events if e.kind == "avoidance"]


def test_defect_calibrator():
    rng = np.random.default_rng(6)
    pos = [ev.event_feature_vector(rng.uniform(5, 50), 0.2, rng.uniform(2, 20), 0.5, "cruise") for _ in range(80)]
    neg = [ev.event_feature_vector(rng.uniform(1, 10), 0.4, rng.uniform(0.05, 0.8), 6.0, "accel") for _ in range(80)]
    X = np.vstack(pos + neg)
    y = np.r_[np.ones(80), np.zeros(80)]
    cal = ev.DefectCalibrator().fit(X, y)
    p = cal.predict_proba(X)
    assert p[:80].mean() > 0.8 and p[80:].mean() < 0.2
    assert np.all((p >= 0) & (p <= 1))
    with pytest.raises(RuntimeError):
        ev.DefectCalibrator().predict_proba(X)


def test_calibrator_is_used_for_p_defect(bumpy):
    class Const:
        def predict_proba(self, X):
            return np.full(len(X), 0.93)

    vft, _ = bumpy
    res = ev.detect_events(vft, calibrator=Const())
    assert all(e.p_defect == pytest.approx(0.93) for e in res.events if e.kind == "impulse")
