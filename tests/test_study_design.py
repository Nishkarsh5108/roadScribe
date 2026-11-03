"""User-study design numbers: ports verify_math.py (c), (d) and the inline non-inferiority check.

These are properties of the study design rather than of package code; they
are kept so that every Appendix A number has a test.
"""
from __future__ import annotations

import numpy as np
import pytest


def williams(k: int) -> np.ndarray:
    """Williams design for an even number of conditions: first row 0, 1, k-1, 2, k-2, ..."""
    first = [0]
    lo, hi = 1, k - 1
    while len(first) < k:
        first.append(lo)
        lo += 1
        if len(first) < k:
            first.append(hi)
            hi -= 1
    return np.array([[(x + r) % k for x in first] for r in range(k)])


def test_williams_4x4():
    """Appendix A: rows [[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]]; balanced."""
    W = williams(4)
    assert W.tolist() == [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
    pairs: dict[tuple[int, int], int] = {}
    for row in W:
        for a, b in zip(row[:-1], row[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    assert len(pairs) == 12 and set(pairs.values()) == {1}  # each ordered pair once
    assert all(sorted(W[:, j]) == list(range(4)) for j in range(4))  # column-balanced
    # §7.1 table: A = T, T+V, S-PGR, T->P with conditions (T, T+V, T->P, S-PGR) = 0..3
    names = ["T", "T+V", "T->P", "S-PGR"]
    seqs = [[names[c] for c in [0, 1, 3, 2]], [names[c] for c in [1, 2, 0, 3]]]
    assert seqs[0] == ["T", "T+V", "S-PGR", "T->P"] and seqs[1] == ["T+V", "T->P", "T", "S-PGR"]
    assert 28 % 4 == 0 and 28 % 6 != 0  # 4 sequences divide 28; a 3-condition design's 6 do not


def test_power_n28():
    """Appendix A: dz >= 0.549 (alpha .05), 0.646 (.05/3), 0.703 (.05/6); N = 33.4 for dz = 0.5."""
    power = pytest.importorskip("statsmodels.stats.power")
    tp = power.TTestPower()
    got = [round(tp.solve_power(nobs=28, alpha=0.05 / k, power=0.8, alternative="two-sided"), 3) for k in (1, 3, 6)]
    assert got == [0.549, 0.646, 0.703]
    assert round(tp.solve_power(effect_size=0.5, alpha=0.05, power=0.8), 1) == 33.4


def test_noninferiority_power():
    """Appendix A: 82.5 % power at margin = 0.5 SD with N = 28; N = 26.1 at d = 0.5; N = 40 at 0.4 SD."""
    power = pytest.importorskip("statsmodels.stats.power")
    tp = power.TTestPower()
    assert round(tp.power(effect_size=0.5, nobs=28, alpha=0.05, alternative="larger"), 3) == 0.825
    assert round(tp.solve_power(effect_size=0.5, alpha=0.05, power=0.8, alternative="larger"), 1) == 26.1
    assert round(tp.solve_power(effect_size=0.4, alpha=0.05, power=0.8, alternative="larger"), 1) == 40.0
