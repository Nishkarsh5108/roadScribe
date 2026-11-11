"""Tests for roadscribe.paired (route alignment, stretch features, agreement statistics)."""
from __future__ import annotations

import numpy as np
import pytest

from roadscribe.paired.agreement import apply_cutpoints, fit_cutpoints, variance_shares, weighted_kappa
from roadscribe.paired.route_align import densify, match_forward, sample_theta


def _loop(n=400, r=200.0):
    a = np.linspace(0, 2 * np.pi, n)
    return np.c_[r * np.cos(a), r * np.sin(a)]


def test_forward_matching_handles_a_loop_whose_start_is_its_end():
    ref_pts, ref_s = densify(_loop())
    L = ref_s[-1]
    rng = np.random.default_rng(0)
    track = _loop(300) + rng.normal(0, 3.0, (300, 2))  # same loop, noisy GNSS, different sampling
    theta, dist = match_forward(track, ref_pts, ref_s)
    assert theta[0] < 30 and theta[-1] > L - 30  # start near 0, end near L (not confused)
    assert np.all(np.diff(theta) > -31)  # never jumps back further than the window
    assert np.median(dist) < 6


def test_sample_theta_marks_off_route_samples_invalid():
    t_fix = np.arange(5.0)
    th, ok = sample_theta(np.array([0.5, 1.5, 3.5]), t_fix, np.arange(5.0) * 10, np.array([1, 1, 50, 1, 1.0]))
    assert np.allclose(th, [5, 15, 35]) and list(ok) == [True, False, True]


def test_weighted_kappa_limits():
    a = np.array([0, 1, 2, 3, 4, 5] * 10)
    assert weighted_kappa(a, a, 6) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    b = rng.integers(0, 6, 6000)
    c = rng.integers(0, 6, 6000)
    assert abs(weighted_kappa(b, c, 6)) < 0.05
    assert np.isnan(weighted_kappa(np.zeros(5, int), np.zeros(5, int), 3))


def test_variance_shares_recover_a_vehicle_offset():
    rng = np.random.default_rng(2)
    loc = np.repeat(np.arange(200), 3)
    veh = np.tile([1, 2, 3], 200)
    lv = rng.normal(2.3, 0.3, 600)
    y = rng.normal(0, 1.0, 200)[loc] + np.array([0, 0.5, -0.5])[veh - 1] + 0.35 * lv + rng.normal(0, 0.1, 600)
    sh = variance_shares(y, loc, veh, lv)
    assert sh["slope_log_v_within"] == pytest.approx(0.35, abs=0.05)
    assert sh["vehicle"] > 0.05 and sh["location"] > 0.5 and sh["residual"] < 0.05


def test_cutpoints_separate_three_classes():
    x = np.r_[np.full(30, 0.1), np.full(30, 1.0), np.full(30, 2.0)]
    y = np.r_[np.zeros(30), np.ones(30), np.full(30, 2)].astype(int)
    cuts = fit_cutpoints(x, y)
    assert np.mean(apply_cutpoints(x, cuts) == y) == 1.0
