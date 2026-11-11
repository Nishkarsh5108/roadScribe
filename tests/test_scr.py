"""Synthetic recovery tests for the SCR Whittle fit, including the identifiability cases (WORKLOG C6)."""
from __future__ import annotations

import numpy as np
import pytest

from roadscribe.scr.whittle import SCRData, fit_scr

F = np.linspace(0.5, 20.0, 50)


def _truth():
    # three channels with different resonances, power-law roads with segment-specific level and slope
    def A(c, f):
        f0 = (1.2, 1.8, 2.6)[c]
        return np.log(1.0 / ((1 - (f / f0) ** 2) ** 2 + (0.6 * f / f0) ** 2)) + 0.3 * c
    rng = np.random.default_rng(3)
    level = rng.normal(0.0, 0.7, 30)
    slope = rng.uniform(1.6, 2.4, 30)

    def B(s, nu):
        return level[s] - slope[s] * np.log(nu / 0.3)
    return A, B


def _simulate(speed_mode: str, n_frames_per_pass: int = 6, seed: int = 0):
    A, B = _truth()
    rng = np.random.default_rng(seed)
    rows_I, rho, ch, seg = [], [], [], []
    seg_speed = rng.uniform(6, 24, 30)  # used when every channel drives a segment at the same speed
    for s in range(30):
        for c in range(3):
            v = seg_speed[s] if speed_mode == "shared" else rng.uniform(6, 24)
            for _ in range(n_frames_per_pass):
                logS = A(c, F) + B(s, F / v) - np.log(v)
                I = np.exp(logS) * rng.gamma(3, 1 / 3, size=F.size)  # K = 3 tapers
                rows_I.append(I)
                rho.append(v)
                ch.append(c)
                seg.append(s)
    return SCRData(f=F, I=np.array(rows_I), rho=np.array(rho), channel=np.array(ch), segment=np.array(seg), k_tapers=3,
                   segment_edges=[]), A, B


def _channel_errors(fit, A):
    est = np.array([fit.A(F, c) for c in range(3)])
    tru = np.array([A(c, F) for c in range(3)])
    common_err = (est.mean(0) - est.mean()) - (tru.mean(0) - tru.mean())  # shape of the mean channel
    diff_err = (est - est.mean(0)) - (tru - tru.mean(0))  # channel differences
    return float(np.sqrt(np.mean(common_err**2))), float(np.sqrt(np.mean(diff_err**2)))


@pytest.mark.slow
def test_speed_diversity_identifies_the_common_channel_and_shared_speed_does_not():
    data_div, A, _ = _simulate("diverse")
    data_shr, _, _ = _simulate("shared")
    fit_div = fit_scr(data_div, lam_a=1.0, lam_b=1.0, lam_theta=0.0)
    fit_shr = fit_scr(data_shr, lam_a=1.0, lam_b=1.0, lam_theta=0.0)
    common_div, diff_div = _channel_errors(fit_div, A)
    common_shr, diff_shr = _channel_errors(fit_shr, A)
    # differences between channels are identified either way (they share the road)
    assert diff_div < 0.25 and diff_shr < 0.25
    # the common channel shape is identified only with speed diversity
    assert common_div < 0.25
    assert common_shr > 2 * common_div


@pytest.mark.slow
def test_recovered_road_level_ranks_segments():
    data, A, B = _simulate("diverse", seed=1)
    fit = fit_scr(data, lam_a=1.0, lam_b=1.0, lam_theta=0.0)
    nu = np.array([0.3])
    est = np.array([fit.B(nu, s)[0] for s in range(30)])
    tru = np.array([B(s, nu)[0] for s in range(30)])
    assert np.corrcoef(est, tru)[0, 1] > 0.95
