"""Orientation estimator (§3.1): synthetic ground truth plus the PVS port of verify_orientation.py."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from roadscribe.config import OrientationConfig
from roadscribe.layer1.orientation import (
    estimate_orientation,
    geometric_median,
    procrustes_yaw,
    rodrigues_to_z,
    rotation_angle_deg,
    to_vehicle_frame,
    yaw_matrix,
)
from synth import make_trip

E_Z = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- primitives
def test_geometric_median_is_rotation_equivariant_componentwise_median_is_not():
    rng = np.random.default_rng(0)
    X = rng.normal([0.3, -0.2, 9.8], [0.4, 0.4, 0.2], size=(2000, 3))
    X[:100] += rng.normal(0, 5, (100, 3))  # outliers
    gm = geometric_median(X)
    worst_gm = worst_cw = 0.0
    for _ in range(20):
        R = Rot.random(random_state=rng).as_matrix()
        Xr = X @ R.T
        worst_gm = max(worst_gm, np.linalg.norm(geometric_median(Xr) - R @ gm))
        worst_cw = max(worst_cw, np.linalg.norm(np.median(Xr, axis=0) - R @ np.median(X, axis=0)))
    assert worst_gm < 1e-8
    assert worst_cw > 1e-3


def test_geometric_median_robust_and_exact_on_points():
    X = np.vstack([np.zeros((60, 2)), np.full((40, 2), 100.0)])
    assert np.allclose(geometric_median(X), 0.0, atol=1e-6)  # majority point, not the mean
    assert np.allclose(geometric_median(np.array([[1.0, 2.0, 3.0]])), [1.0, 2.0, 3.0])
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    assert np.allclose(geometric_median(tri), tri.mean(0), atol=1e-9)  # Fermat point = centroid


@pytest.mark.parametrize("u", [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0.1, -0.3, 0.9], [0, 1e-12, -1]])
def test_rodrigues_levels_and_is_a_rotation(u):
    u = np.asarray(u, float)
    R = rodrigues_to_z(u)
    assert np.allclose(R @ (u / np.linalg.norm(u)), E_Z, atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rodrigues_degenerate_cases():
    assert np.array_equal(rodrigues_to_z(np.array([0, 0, 2.0])), np.eye(3))
    assert np.array_equal(rodrigues_to_z(np.array([0, 0, -1.0])), np.diag([1.0, -1.0, -1.0]))


def test_procrustes_closed_form_is_the_argmin():
    rng = np.random.default_rng(1)
    psi_true = np.radians(73.0)
    h = rng.normal(size=(300, 2))
    r = h @ yaw_matrix(psi_true)[:2, :2].T + rng.normal(0, 0.05, (300, 2))
    psi = procrustes_yaw(h, r)
    grid = np.radians(np.arange(-180, 180, 0.01))
    loss = [np.sum((h @ yaw_matrix(p)[:2, :2].T - r) ** 2) for p in grid]
    assert np.degrees(psi) == pytest.approx(np.degrees(grid[int(np.argmin(loss))]), abs=0.01)
    assert np.degrees(psi) == pytest.approx(73.0, abs=0.5)


def test_pvs_convention_forward_plus_y_gives_plus_90():
    """A sensor whose forward axis is +y must give psi = +90 deg (§3.0 PVS facts)."""
    a = np.linspace(-2, 2, 50)
    h = np.column_stack([np.zeros(50), a])  # all excitation on sensor y
    r = np.column_stack([a, np.zeros(50)])  # ... which is the vehicle's forward axis
    assert np.degrees(procrustes_yaw(h, r)) == pytest.approx(90.0)


def test_leveled_yaw_rate_does_not_depend_on_heading_offset():
    rng = np.random.default_rng(2)
    R1 = rodrigues_to_z(np.array([0.1, 0.2, 0.97]))
    w = rng.normal(size=(100, 3))
    for psi in np.radians([0, 37, 145, -90]):
        assert np.allclose((w @ (yaw_matrix(psi) @ R1).T)[:, 2], (w @ R1.T)[:, 2])


# --------------------------------------------------------------------------- synthetic trips
@pytest.fixture(scope="module")
def trip():
    return make_trip(seed=1)


def test_recovers_known_pvs_like_mount(trip):
    res = estimate_orientation(trip.raw)
    assert len(res.episodes) == 1 and res.splits_t == []
    e = res.episodes[0]
    assert rotation_angle_deg(e.R_vb, trip.R_vb) < 0.5
    assert e.yaw_deg == pytest.approx(90.0, abs=1.0)
    lo, hi = e.yaw_ci_deg
    assert lo < e.yaw_deg < hi and hi - lo < 2.0
    assert e.n_quasistatic > 10_000
    assert e.g_norm == pytest.approx(9.80665, abs=0.05)
    assert e.excitation > 0


def test_equivariant_to_arbitrary_phone_orientation(trip):
    """Appendix A: equivariance <= 0.63 deg; the geometric median makes it exact."""
    rng = np.random.default_rng(11)
    R0 = estimate_orientation(trip.raw).episodes[0].R_vb
    errs = []
    for _ in range(8):
        Rr = Rot.random(random_state=rng).as_matrix()  # body' = Rr @ body
        raw = dataclasses.replace(trip.raw, f_b=trip.raw.f_b @ Rr.T, w_b=trip.raw.w_b @ Rr.T)
        Rb = estimate_orientation(raw).episodes[0].R_vb
        errs.append(rotation_angle_deg(Rb, R0 @ Rr.T))
    assert max(errs) <= 0.63
    assert max(errs) < 1e-3


@pytest.mark.parametrize("euler", [(0, 0, 0), (-135, 20, 10), (30, -60, 170), (200, 5, -95)])
def test_recovers_arbitrary_mounts(euler):
    R_vb = Rot.from_euler("zyx", euler, degrees=True).as_matrix()
    st = make_trip(R_vb=R_vb, seed=3, duration=450)
    e = estimate_orientation(st.raw).episodes[0]
    assert rotation_angle_deg(e.R_vb, R_vb) < 0.5


def test_gating_matters():
    """Without the acceleration and gyro gates the tilt absorbs maneuvers (arctan(a/g))."""
    st = make_trip(seed=4, duration=450)
    gated = estimate_orientation(st.raw).episodes[0]
    loose = OrientationConfig(eps_g=5.0, eps_omega_deg=1e3, eps_a=1e3, v_min=-1.0)
    ungated = estimate_orientation(st.raw, loose).episodes[0]
    assert gated.n_quasistatic < ungated.n_quasistatic
    assert rotation_angle_deg(gated.R_vb, st.R_vb) <= rotation_angle_deg(ungated.R_vb, st.R_vb) + 1e-6


def test_handling_event_splits_episodes():
    st = make_trip(seed=2, handling_at=300.0)
    res = estimate_orientation(st.raw)
    assert len(res.episodes) == 2
    assert res.splits_t[0] == pytest.approx(300.5, abs=1.0)
    t = st.raw.t
    assert np.all(res.episode_index[np.abs(t - 300.5) < 4.5] == -1)  # +-5 s discarded
    assert rotation_angle_deg(res.episodes[0].R_vb, st.R_vb) < 0.5
    assert rotation_angle_deg(res.episodes[1].R_vb, st.R_vb_after) < 0.5
    vft = to_vehicle_frame(st.raw, result=res)
    assert len(vft.R_vb) == 2 and set(np.unique(vft.episode)) == {-1, 0, 1}


def test_no_gyro_falls_back_to_gnss_course():
    st = make_trip(seed=5, gyro=False)
    e = estimate_orientation(st.raw).episodes[0]
    assert e.yaw_reference == "gnss_course"
    assert rotation_angle_deg(e.R_vb, st.R_vb) < 1.5


def test_vehicle_frame_signals(trip):
    vft = to_vehicle_frame(trip.raw)
    truth = trip.f_v - np.array([0, 0, 9.80665])
    assert np.sqrt(np.mean((vft.a_v - truth) ** 2)) < 0.05
    assert np.corrcoef(vft.w_v[:, 2], trip.w_z)[0, 1] > 0.99
    assert vft.s[-1] == pytest.approx(np.trapezoid(trip.v, trip.raw.t), rel=0.01)
    assert np.all(np.diff(vft.s) >= 0)


def test_insufficient_data_raises():
    from roadscribe.layer1.orientation import OrientationError

    st = make_trip(seed=6, duration=120)
    with pytest.raises(OrientationError):
        estimate_orientation(st.raw, OrientationConfig(min_quasistatic=10**7))


# --------------------------------------------------------------------------- PVS (verify_orientation.py)
@pytest.mark.pvs
@pytest.mark.slow
def test_pvs_yaw_tilt_and_equivariance(pvs_root):
    """Appendix A: yaw 87.9-92.5 deg (vehicles 1, 3), 103.1-105.5 deg (vehicle 2);
    30k-51k quasi-static samples per trip; equivariance <= 0.63 deg.

    The package uses PCHIP speed and a geometric median, where the script
    used linear interpolation and a componentwise median, so the yaw bounds
    carry 0.5 deg of slack.
    """
    from roadscribe.io.pvs import load_pvs_trip, vehicle_of

    rng = np.random.default_rng(11)
    for trip in range(1, 10):
        raw = load_pvs_trip(pvs_root, trip, with_labels=False)
        e = estimate_orientation(raw).episodes
        assert len(e) == 1, f"trip {trip}: a fixed mount must give one episode"
        yaw = e[0].yaw_deg
        if vehicle_of(trip) == 2:
            assert 103.1 - 0.5 <= yaw <= 105.5 + 0.5, (trip, yaw)
        else:
            assert 87.9 - 0.5 <= yaw <= 92.5 + 0.5, (trip, yaw)
        assert 25_000 <= e[0].n_quasistatic <= 55_000, (trip, e[0].n_quasistatic)
        for _ in range(3):
            Rr = Rot.random(random_state=rng).as_matrix()
            rot = dataclasses.replace(raw, f_b=raw.f_b @ Rr.T, w_b=raw.w_b @ Rr.T)
            err = rotation_angle_deg(estimate_orientation(rot).episodes[0].R_vb, e[0].R_vb @ Rr.T)
            assert err <= 0.63, (trip, err)
