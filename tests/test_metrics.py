"""Narrative metrics: rho_KSF, BC, OI, anomaly grounding, ECE_ling (§6.3-6.5)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from roadscribe.eval.metrics import (
    Mention,
    RefEvent,
    anomaly_grounding,
    band_consistency,
    ece_ling,
    overstatement_index,
    rho_ksf,
    severity_fidelity,
)
from roadscribe.layer1.iso2631 import comfort_class


def test_rho_ksf_is_spearman_with_midranks():
    rng = np.random.default_rng(0)
    x = rng.lognormal(-0.5, 0.6, 200)
    k = np.array([comfort_class(v) for v in x]) + rng.integers(-1, 2, 200)
    k = np.clip(k, 0, 5)
    assert rho_ksf(x, k) == pytest.approx(stats.spearmanr(x, k).statistic, abs=1e-12)


def test_three_numbers_separate_failure_modes():
    x = np.array([0.2, 0.4, 0.7, 1.1, 1.8, 2.6])
    exact = [comfort_class(v) for v in x]
    plus_one = [min(5, c + 1) for c in exact]
    f0, f1 = severity_fidelity(x, exact), severity_fidelity(x, plus_one)
    assert f0.rho_ksf == pytest.approx(1.0) and f1.rho_ksf > 0.98  # rho ignores the shift
    assert f0.bc == 1.0 and f1.bc < f0.bc  # BC catches miscalibration ...
    assert f0.oi == 0.0 and f1.oi > 0  # ... and OI gives its direction
    assert overstatement_index([0.2], [0]) == 0 and band_consistency([0.9], [2]) == 1.0  # overlap 0.8-1.0


def test_anomaly_grounding_matching():
    mentions = [Mention("bump", 2050, 2150), Mention("bump", 2380, 2420), Mention("bump", 5000, 5000)]
    refs = [RefEvent("bump", 2100, 1.8), RefEvent("bump", 2410, 1.3), RefEvent("bump", 3000, 2.0),
            RefEvent("jolt", 2100, 0.5)]
    ag = anomaly_grounding(mentions, refs, delta_s=25)
    assert ag.agp == pytest.approx(2 / 3) and ag.agr == pytest.approx(2 / 4)
    assert ag.wagr == pytest.approx((1.8 + 1.3) / (1.8 + 1.3 + 2.0 + 0.5))
    any_type = anomaly_grounding(mentions, refs, compatible=lambda a, b: True)
    assert any_type.agr == pytest.approx(2 / 4)  # still one mention per event


def test_wagr_prefers_heavier_events_among_maximum_matchings():
    # one mention covers two events; the heavier one must be counted
    m = [Mention("bump", 0, 100)]
    r = [RefEvent("bump", 10, 0.5), RefEvent("bump", 90, 3.0)]
    ag = anomaly_grounding(m, r)
    assert ag.wagr == pytest.approx(3.0 / 3.5) and ag.matches == ((0, 1),)
    # and it never sacrifices cardinality for weight
    m2 = [Mention("bump", 0, 100), Mention("bump", 80, 120)]
    r2 = [RefEvent("bump", 10, 0.5), RefEvent("bump", 90, 3.0)]
    assert anomaly_grounding(m2, r2).agr == 1.0


def test_ece_ling():
    hedges = ["assertive"] * 10 + ["likely"] * 10 + ["possibly"] * 10
    correct = [1] * 9 + [0] + [1] * 7 + [0] * 3 + [1] * 5 + [0] * 5
    nominal = {"assertive": 0.95, "likely": 0.75, "possibly": 0.5}
    assert ece_ling(hedges, correct, nominal) == pytest.approx((0.05 + 0.05 + 0.0) / 3)
