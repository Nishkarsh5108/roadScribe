"""MI-PELT: ports verify_math.py (b) and checks the corrected pruning rule."""
from __future__ import annotations

import numpy as np
import pytest

from roadscribe.config import MIPELTConfig
from roadscribe.layer2.mipelt import (
    MeanShiftCost,
    base_penalty,
    changepoint_positions,
    location_penalty,
    map_prior,
    mipelt,
    optimal_partitioning,
    segment,
)


def _verify_math_series(trials: int = 300, seed: int = 7):
    """The exact random series of verify_math.py (b), in the same RNG order."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        n = 120
        cps = np.sort(rng.choice(np.arange(10, n - 10), size=4, replace=False))
        mu = np.repeat(rng.normal(0, 1.2, (5, 2)), np.diff(np.concatenate([[0], cps, [n]])), axis=0)
        Y = mu + rng.normal(0, 1, (n, 2))
        beta = np.full(n + 1, 12.0)
        mapnodes = rng.choice(np.arange(5, n - 5), size=10, replace=False)
        beta_const = beta.copy()
        beta[mapnodes] = 2.0  # cheaper change points at map transitions
        yield Y, beta, beta_const


def _script_classical_rule(Y: np.ndarray, beta: np.ndarray) -> float:
    """The 'classical' rule exactly as written in verify_math.py: keep t iff F[t] + C < F[s]."""
    C = MeanShiftCost(Y)
    n = len(Y)
    F = np.full(n + 1, np.inf)
    F[0] = 0.0
    G = lambda t: F[t] + (beta[t] if t > 0 else 0.0)  # noqa: E731
    R = [0]
    for s in range(1, n + 1):
        F[s] = min(G(t) + C(t, s) for t in R)
        R = [t for t in R if F[t] + C(t, s) < F[s]] + [s]
    return F[n]


@pytest.fixture(scope="module")
def series():
    return list(_verify_math_series())


@pytest.mark.slow
def test_corrected_rule_matches_exhaustive_300_of_300(series):
    """Appendix A: corrected rule, 0/300 mismatches against exhaustive optimal partitioning."""
    bad = 0
    for Y, beta, _ in series:
        C = MeanShiftCost(Y)
        bad += not np.isclose(mipelt(C, beta).cost, optimal_partitioning(C, beta).cost)
    assert bad == 0


@pytest.mark.slow
def test_script_classical_rule_is_suboptimal_300_of_300(series):
    """Appendix A: the textbook rule as implemented in verify_math.py is sub-optimal in 300/300."""
    bad = sum(
        not np.isclose(_script_classical_rule(Y, beta), optimal_partitioning(MeanShiftCost(Y), beta).cost)
        for Y, beta, _ in series
    )
    assert bad == 300


@pytest.mark.slow
def test_script_classical_rule_fails_even_with_constant_penalty(series):
    """The 300/300 figure is not caused by the location-dependent penalty alone.

    verify_math.py's rule compares F[0] + C(0, s) with F[s], but G(0) = F(0)
    carries no penalty, so whenever t = 0 is optimal (always at s = 1) it is
    pruned. That rule therefore fails with a *constant* penalty too, where
    the textbook PELT condition is exact.
    """
    bad = sum(
        not np.isclose(_script_classical_rule(Y, bc), optimal_partitioning(MeanShiftCost(Y), bc).cost)
        for Y, _, bc in series
    )
    assert bad == 300


@pytest.mark.slow
def test_textbook_rule_exact_for_constant_penalty_but_not_for_map_penalty(series):
    """The fair textbook rule (t = 0 handled correctly) is exact for constant beta and
    still sub-optimal under the map penalty: 136/300 series with these seeds."""
    bad_map = bad_const = 0
    for Y, beta, bc in series:
        C = MeanShiftCost(Y)
        bad_map += not np.isclose(mipelt(C, beta, pruning="textbook").cost, optimal_partitioning(C, beta).cost)
        bad_const += not np.isclose(mipelt(C, bc, pruning="textbook").cost, optimal_partitioning(C, bc).cost)
    assert bad_const == 0
    assert 0 < bad_map < 300


@pytest.mark.slow
@pytest.mark.parametrize("min_size", [2, 5, 9])
def test_corrected_rule_exact_with_min_size(series, min_size):
    for Y, beta, _ in series[:100]:
        C = MeanShiftCost(Y)
        ref = optimal_partitioning(C, beta, min_size)
        got = mipelt(C, beta, min_size)
        assert np.isclose(got.cost, ref.cost)
        segs = np.diff([0, *got.changepoints, len(Y)])
        assert segs.min() >= min_size


def test_cost_cumsum_matches_direct():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(50, 3))
    C = MeanShiftCost(Y)
    for t, s in [(0, 50), (3, 7), (10, 11), (20, 45)]:
        seg = Y[t:s]
        assert C(t, s) == pytest.approx(np.sum((seg - seg.mean(0)) ** 2))
    np.testing.assert_allclose(C.many(np.array([0, 3, 10]), 20), [C(0, 20), C(3, 20), C(10, 20)])


def test_cost_with_diagonal_covariance():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(40, 2)) * [1.0, 10.0]
    D = np.array([1.0, 100.0])
    C = MeanShiftCost(Y, D)
    seg = Y / np.sqrt(D)
    assert C(0, 40) == pytest.approx(np.sum((seg - seg.mean(0)) ** 2))
    auto = MeanShiftCost(Y, "auto").D
    assert auto[1] / auto[0] == pytest.approx(100, rel=0.6)


def test_cost_is_superadditive():
    """The lemma's assumption: C(t+1:s) + C(s+1:T) <= C(t+1:T)."""
    rng = np.random.default_rng(2)
    C = MeanShiftCost(rng.normal(size=(60, 2)))
    for _ in range(200):
        t, s, T = np.sort(rng.choice(np.arange(61), 3, replace=False))
        assert C(t, s) + C(s, T) <= C(t, T) + 1e-9


def test_lambda_zero_recovers_pelt():
    """lambda = 0 gives a constant penalty, and PELT then equals optimal partitioning."""
    rng = np.random.default_rng(3)
    Y = np.concatenate([rng.normal(0, 1, (60, 2)), rng.normal(3, 1, (60, 2)), rng.normal(-1, 1, (60, 2))])
    s_grid = np.arange(len(Y)) * 5.0
    beta = location_penalty(s_grid, [100.0, 400.0], [1.0, 1.0], beta0=12.0, lam=0.0)
    assert np.allclose(beta, 12.0)
    res = mipelt(Y, beta)
    assert res.changepoints == optimal_partitioning(Y, 12.0).changepoints
    assert res.changepoints == (60, 120)


def test_pruning_reduces_work():
    rng = np.random.default_rng(4)
    Y = np.concatenate([rng.normal(m, 1, (100, 1)) for m in (0, 4, 0, 4, 0, 4, 0, 4)])
    pruned = mipelt(Y, 15.0)
    full = mipelt(Y, 15.0, pruning="none")
    assert pruned.cost == pytest.approx(full.cost)
    assert pruned.n_evaluations < full.n_evaluations / 5
    assert pruned.max_candidates < 150


def test_map_prior_and_penalty():
    s_grid = np.arange(0, 500, 5.0)
    pos = changepoint_positions(s_grid)
    assert len(pos) == len(s_grid) + 1
    assert pos[1] == pytest.approx(2.5)
    pi = map_prior(pos, [200.0, 400.0], [1.0, 0.3], sigma_b=15.0)
    assert pi.max() <= 1.0
    assert pi[np.argmin(np.abs(pos - 200.0))] == pytest.approx(np.exp(-(2.5**2) / (2 * 15**2)))
    assert map_prior(pos, [], [], 15.0).sum() == 0
    beta = location_penalty(s_grid, [200.0], [1.0], beta0=10.0, lam=0.5)
    assert beta.min() == pytest.approx(10.0 * (1 - 0.5 * np.exp(-(2.5**2) / 450)))
    assert beta.max() == pytest.approx(10.0)
    with pytest.raises(ValueError):
        location_penalty(s_grid, [200.0], [1.0], beta0=10.0, lam=1.0)
    assert base_penalty(2.0, 6, 100) == pytest.approx(2.0 * 6 * np.log(100))


def test_map_penalty_moves_a_boundary_onto_the_transition():
    """A weak change is found only where the map makes it cheap."""
    rng = np.random.default_rng(0)
    Y = np.concatenate([rng.normal(0, 1, (80, 1)), rng.normal(0.5, 1, (80, 1))])
    s_grid = np.arange(len(Y)) * 5.0
    cfg_plain = MIPELTConfig(c=4.0, lam=0.0, min_size=4)
    cfg_map = MIPELTConfig(c=4.0, lam=0.9, min_size=4)
    plain, _ = segment(Y, s_grid, [400.0], [1.0], cfg_plain)
    mapped, beta = segment(Y, s_grid, [400.0], [1.0], cfg_map)
    assert plain.changepoints == ()
    assert len(mapped.changepoints) == 1
    assert abs(changepoint_positions(s_grid)[mapped.changepoints[0]] - 400.0) <= 25.0
    assert beta.min() < base_penalty(4.0, 1, len(Y))


def test_input_validation():
    with pytest.raises(ValueError):
        MeanShiftCost(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError):
        mipelt(np.zeros((3, 1)), 1.0, min_size=5)
    with pytest.raises(ValueError):
        mipelt(np.zeros((10, 1)), np.ones(5))
    with pytest.raises(ValueError):
        mipelt(np.zeros((10, 1)), 1.0, pruning="bogus")
