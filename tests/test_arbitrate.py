"""Distance grid (§4.2) and arbitration into HRNUs (§4.4)."""
from __future__ import annotations

import numpy as np
import pytest

from roadscribe.config import ArbitrationConfig
from roadscribe.layer1.iso2631 import comfort_class, weight_vehicle_axes
from roadscribe.layer2.arbitrate import (
    Boundary,
    MapTransition,
    WaySpan,
    arbitrate,
    build_hrnus,
    hotelling_t2,
    map_only_boundaries,
    merge_short_units,
    snap_changepoints,
    transition_weights,
)
from roadscribe.layer2.distance import FEATURE_NAMES, build_grid, robust_standardize
from synth import vehicle_frame

FS = 100.0


# --------------------------------------------------------------------------- A1-A3
def test_transition_weights():
    tr = [MapTransition(0, "surface"), MapTransition(1, "highway"), MapTransition(2, "junction"),
          MapTransition(3, "traffic_calming"), MapTransition(4, "surface", weight=0.5)]
    assert transition_weights(tr).tolist() == [1.0, 0.6, 0.3, 0.3, 0.5]


def test_snap_within_25m_and_typing():
    tr = [MapTransition(500.0, "surface"), MapTransition(1000.0, "highway")]
    bs = snap_changepoints([480.0, 760.0, 1030.0], tr, 25.0)
    assert [(b.s, b.kind) for b in bs] == [(500.0, "map+signal"), (760.0, "signal"), (1030.0, "signal")]
    assert bs[0].s_signal == 480.0 and bs[0].transition is tr[0]


def test_two_changepoints_do_not_snap_to_one_transition():
    tr = [MapTransition(500.0, "surface")]
    bs = snap_changepoints([490.0, 505.0], tr, 25.0)
    assert sorted((b.s, b.kind) for b in bs) == [(490.0, "signal"), (500.0, "map+signal")]


def test_hotelling():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (30, 3))
    b = rng.normal(0, 1, (30, 3))
    _, p_same = hotelling_t2(a, b)
    _, p_diff = hotelling_t2(a, b + [1.5, 0, 0])
    assert p_same > 0.01 and p_diff < 1e-3
    assert np.isnan(hotelling_t2(a[:1], b)[1])
    # with one variable T^2 is the squared pooled two-sample t statistic
    from scipy import stats

    t2, p1 = hotelling_t2(a[:, :1], b[:, :1] + 0.4, ridge=0.0)
    t = stats.ttest_ind(a[:, 0], b[:, 0] + 0.4)
    assert t2 == pytest.approx(t.statistic**2) and p1 == pytest.approx(t.pvalue)


def test_map_only_boundary_needs_a_signal_change():
    rng = np.random.default_rng(1)
    s = np.arange(0, 1000, 5.0)
    Y = rng.normal(0, 1, (len(s), 2))
    Y[s >= 500, 0] += 3.0
    tr = [MapTransition(500.0, "surface"), MapTransition(250.0, "surface"), MapTransition(750.0, "highway")]
    out = map_only_boundaries([], tr, s, Y, ArbitrationConfig())
    assert [(b.s, b.kind) for b in out] == [(500.0, "map")]  # 250 m has no change; highway is not tested
    assert out[0].p_value < 0.01
    # a nearby signal boundary pre-empts it
    assert map_only_boundaries([Boundary(510.0, "signal")], tr[:1], s, Y) == []


def test_merge_short_units_into_closer_neighbour():
    s = np.arange(0, 600, 5.0)
    Y = np.zeros((len(s), 1))
    Y[(s >= 200) & (s < 230), 0] = 1.0  # 30 m unit, similar to the right neighbour
    Y[s >= 230, 0] = 1.2
    bs = [Boundary(200.0, "signal"), Boundary(230.0, "map+signal")]
    out = merge_short_units(bs, 0.0, 600.0, s, Y, 50.0)
    assert [b.s for b in out] == [200.0]


def test_arbitrate_end_to_end_on_grid():
    rng = np.random.default_rng(2)
    s = np.arange(0, 1500, 5.0)
    Y = rng.normal(0, 0.3, (len(s), 2))
    Y[s >= 600, 0] += 2.0
    Y[s >= 1100, 1] += 2.0
    cps = [int(np.searchsorted(s, 590.0)), int(np.searchsorted(s, 1100.0)), int(np.searchsorted(s, 1120.0))]
    tr = [MapTransition(600.0, "surface")]
    out = arbitrate(cps, s, Y, tr)
    kinds = [(round(b.s), b.kind) for b in out]
    assert kinds[0] == (600, "map+signal")
    assert len(out) == 2 and out[1].kind == "signal"  # 1097.5 and 1117.5 merged (20 m unit)


# --------------------------------------------------------------------------- grid and HRNUs
@pytest.fixture(scope="module")
def two_surface_trip():
    """1 km smooth, then 1 km rough, at 10 m/s, with one bump at 1500 m."""
    n = int(200 * FS)
    rng = np.random.default_rng(3)
    v = np.full(n, 10.0)
    az = rng.normal(0, 0.15, n)
    az[n // 2 :] = rng.normal(0, 0.9, n - n // 2)
    a_v = np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), az])
    vft = vehicle_frame(a_v, v, FS, w_v=np.zeros((n, 3)), a_lon=np.zeros(n))
    aw = weight_vehicle_axes(vft.a_v, FS)
    return vft, aw


def test_grid_features(two_surface_trip):
    vft, aw = two_surface_trip
    g = build_grid(vft, aw)
    assert g.names == FEATURE_NAMES
    assert g.Y.shape == (len(g.s), 6) and np.all(np.isfinite(g.Y))
    assert g.s[1] - g.s[0] == pytest.approx(5.0)
    a_wz = np.exp(g.raw[:, 1])
    assert np.median(a_wz[g.s > 1100]) > 4 * np.median(a_wz[g.s < 900])
    assert np.median(g.Y[:, 1]) == pytest.approx(0.0, abs=1e-9)
    # speed is not a feature, and at constant speed the decoupled column is a shift of the raw one
    np.testing.assert_allclose(g.raw[:, 0] - g.raw[:, 1], g.raw[0, 0] - g.raw[0, 1], atol=1e-6)


def test_robust_standardize_sparse_column():
    X = np.zeros((100, 2))
    X[:, 0] = np.arange(100)
    X[50, 1] = 4.0
    Z = robust_standardize(X, scale_floor=np.array([0.0, 4.0]))
    assert Z[50, 1] == pytest.approx(1.0)


def test_build_hrnus(two_surface_trip):
    vft, aw = two_surface_trip
    ways = [WaySpan(0, 1000, 11, {"name": "Road A", "surface": "asphalt", "highway": "residential"}),
            WaySpan(1000, 2100, 12, {"name": "Road B", "surface": "sett"})]
    units = build_hrnus([Boundary(1000.0, "map+signal")], vft, aw, ways=ways,
                        map_match_conf=np.full(vft.n, 0.95))
    assert [u.unit_id for u in units] == ["U1", "U2"]
    u1, u2 = units
    assert u1.boundary_left == "trip_start" and u2.boundary_left == "map+signal"
    assert u1.names == ["Road A"] and u2.osm_tags["surface"] == "sett" and u2.osm_way_ids == [12]
    assert u2.a_v > 3 * u1.a_v
    assert u1.iso_felt == comfort_class(u1.a_v)
    assert u1.a_wz_dec == pytest.approx(u1.a_wz)  # v = v_ref = 10 m/s
    assert u1.iso_road == comfort_class(u1.a_wz_dec)
    assert u1.conf["map_match"] == pytest.approx(0.95)
    assert u1.conf["channels"] == ["acc", "gyro", "gps", "map"]
    assert u1.map_signal_disagreement is None  # no classifier
    assert u2.s1 == pytest.approx(vft.s[-1])


def test_hrnu_speed_low_and_disagreement(two_surface_trip):
    vft, aw = two_surface_trip
    slow = vehicle_frame(vft.a_v, np.full(vft.n, 3.0), FS, w_v=vft.w_v, a_lon=np.zeros(vft.n))
    ways = [WaySpan(0, 1000, 1, {"surface": "asphalt"})]
    g = build_grid(slow, aw)
    units = build_hrnus([], slow, aw, ways=ways, grid_s=g.s, grid_Y=g.Y,
                        surface_proba=lambda Y: {"asphalt": 0.2, "cobble": 0.8})
    assert units[0].a_wz_dec is None and units[0].iso_road is None
    assert units[0].map_signal_disagreement == pytest.approx(0.8)
