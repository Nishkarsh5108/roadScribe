r"""Narrative metrics (METHODOLOGY §6.3-6.5).

* Kinetic-semantic fidelity: :math:`\rho_{KSF}` (Pearson correlation of
  mid-ranks, i.e. Spearman), Kendall's :math:`\tau_b` as a check, band
  consistency BC and overstatement index OI.
* Anomaly grounding: AGP, AGR and wAGR from a maximum bipartite matching of
  narrated mentions to reference events.
* Linguistic calibration: :math:`\mathrm{ECE}_{ling}`.

Claims are pooled over a corpus before calling these functions; a single
narrative with two severity claims has :math:`\rho=\pm1` trivially.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from ..layer1.iso2631 import COMFORT_BANDS, COMFORT_CUTPOINTS

__all__ = [
    "midranks",
    "rho_ksf",
    "kendall_tau_b",
    "band_consistency",
    "overstatement_index",
    "SeverityFidelity",
    "severity_fidelity",
    "Mention",
    "RefEvent",
    "AnomalyGrounding",
    "anomaly_grounding",
    "RecordingGrounding",
    "anomaly_grounding_by_recording",
    "ece_ling",
]


# --------------------------------------------------------------------------- severity
def midranks(x: Sequence[float]) -> np.ndarray:
    """Ranks with ties given the mean of the ranks they span."""
    return stats.rankdata(np.asarray(x, dtype=float), method="average")


def rho_ksf(x: Sequence[float], k: Sequence[int]) -> float:
    """Pearson correlation of the mid-ranks of measured intensity x_c and expressed class k_c."""
    R, Q = midranks(x), midranks(k)
    R, Q = R - R.mean(), Q - Q.mean()
    den = np.sqrt(np.sum(R**2) * np.sum(Q**2))
    return float(np.sum(R * Q) / den) if den > 0 else float("nan")


def kendall_tau_b(x: Sequence[float], k: Sequence[int]) -> float:
    return float(stats.kendalltau(x, k, variant="b").statistic)


def band_consistency(x: Sequence[float], k: Sequence[int], bands=COMFORT_BANDS) -> float:
    """BC = mean 1[x_c in B(k_c)] with the overlapping ISO 2631-1 Annex C bands."""
    x, k = np.asarray(x, float), np.asarray(k, int)
    lo = np.array([bands[j][0] for j in k])
    hi = np.array([bands[j][1] for j in k])
    return float(np.mean((x >= lo) & (x <= hi)))


def overstatement_index(x: Sequence[float], k: Sequence[int], cutpoints=COMFORT_CUTPOINTS) -> float:
    """OI = mean (k_c - k*(x_c)), k*(x) = sum_j 1[x >= theta_j]; positive = overstated."""
    x, k = np.asarray(x, float), np.asarray(k, int)
    kstar = (x[:, None] >= np.asarray(cutpoints)[None, :]).sum(axis=1)
    return float(np.mean(k - kstar))


@dataclass(frozen=True)
class SeverityFidelity:
    n: int
    rho_ksf: float
    tau_b: float
    bc: float
    oi: float


def severity_fidelity(x: Sequence[float], k: Sequence[int]) -> SeverityFidelity:
    """All four severity numbers; report them separately, never as a composite."""
    if len(x) != len(k) or len(x) < 2:
        raise ValueError("need at least two paired severity claims")
    return SeverityFidelity(len(x), rho_ksf(x, k), kendall_tau_b(x, k), band_consistency(x, k), overstatement_index(x, k))


# --------------------------------------------------------------------------- anomaly grounding
@dataclass(frozen=True)
class Mention:
    """A narrated event mention with the along-track interval it resolves to (m)."""

    type: str
    s_lo: float
    s_hi: float


@dataclass(frozen=True)
class RefEvent:
    """A reference event (ledger or ground truth) at position s with weight w (MTVV)."""

    type: str
    s: float
    w: float = 1.0
    id: str = ""


@dataclass(frozen=True)
class AnomalyGrounding:
    agp: float
    agr: float
    wagr: float
    matches: tuple[tuple[int, int], ...]  # (mention index, reference index)


def _dist(s: float, lo: float, hi: float) -> float:
    return max(lo - s, 0.0, s - hi)


def anomaly_grounding(
    mentions: Sequence[Mention],
    refs: Sequence[RefEvent],
    delta_s: float = 25.0,
    compatible: Optional[Callable[[str, str], bool]] = None,
) -> AnomalyGrounding:
    """AGP = |M*|/|mentions|, AGR = |M*|/|refs|, wAGR = matched weight / total weight.

    An edge joins a mention and a reference event of compatible type whose
    position lies within ``delta_s`` of the mention's interval. M* is a
    maximum matching; among maximum matchings it maximises the matched
    weight, because augmenting paths are grown from reference events in
    decreasing weight order (the matchable reference sets form a transversal
    matroid, so this greedy order is optimal) and an augmenting path never
    unmatches a reference that is already matched.
    """
    comp = compatible or (lambda a, b: a == b)
    order = sorted(range(len(refs)), key=lambda j: -refs[j].w)
    adj = {j: [i for i, m in enumerate(mentions) if comp(m.type, refs[j].type) and _dist(refs[j].s, m.s_lo, m.s_hi) <= delta_s]
           for j in range(len(refs))}
    mention_of: dict[int, int] = {}  # mention index -> reference index

    def augment(j: int, seen: set[int]) -> bool:
        for i in adj[j]:
            if i in seen:
                continue
            seen.add(i)
            if i not in mention_of or augment(mention_of[i], seen):
                mention_of[i] = j
                return True
        return False

    for j in order:
        augment(j, set())
    m = len(mention_of)
    w_tot = sum(r.w for r in refs)
    w_hit = sum(refs[j].w for j in mention_of.values())
    return AnomalyGrounding(
        agp=m / len(mentions) if mentions else float("nan"),
        agr=m / len(refs) if refs else float("nan"),
        wagr=w_hit / w_tot if w_tot > 0 else float("nan"),
        matches=tuple(sorted(mention_of.items())),
    )


# --------------------------------------------------------------------------- calibration
def ece_ling(hedges: Sequence[str], correct: Sequence[bool], nominal: Mapping[str, float]) -> float:
    """ECE_ling = sum_h (n_h / n) |P_G(h) - p_h| over hedge classes h."""
    hedges = np.asarray(hedges)
    correct = np.asarray(correct, dtype=float)
    n = len(hedges)
    if n == 0:
        return float("nan")
    total = 0.0
    for h in np.unique(hedges):
        sel = hedges == h
        total += sel.sum() / n * abs(correct[sel].mean() - nominal[str(h)])
    return float(total)


# --------------------------------------------------------------------------- grounding per recording
@dataclass(frozen=True)
class RecordingGrounding:
    """Anomaly grounding scored inside each recording, with counts summed over recordings."""

    n_mentions: int
    n_refs: int
    n_matched: int
    agp: float
    agr: float
    chance_agr: float  # NaN unless chance draws were requested
    per_recording: tuple  # (recording_id, n_mentions, n_refs, n_matched, mean chance matches)


def anomaly_grounding_by_recording(
    recordings: Sequence[tuple],
    delta_s: float = 25.0,
    chance_draws: int = 0,
    seed: int = 0,
    compatible: Optional[Callable[[str, str], bool]] = None,
) -> RecordingGrounding:
    """AGP/AGR with matching restricted to each recording, plus a chance level.

    ``recordings`` holds ``(recording_id, mentions, refs, length)`` tuples, with
    positions in that recording's own coordinate. Positions from different
    recordings are not comparable, so a mention may only match a reference
    event of the same recording. Pooling all mentions and references into one
    matching lets a mention in one trip match a bump in another, and inflated
    a reported AGR to 100%. Each recording's references are counted once.

    With ``chance_draws > 0``, each recording's mention intervals are also
    dropped uniformly at random within ``[0, length]`` (same widths and types),
    and the mean number of matches gives ``chance_agr``. If the observed AGR is
    close to it, the metric is measuring how wide the mentions are, not
    whether they point at the events.
    """
    rng = np.random.default_rng(seed)
    tot_m = tot_r = tot_hit = 0
    tot_chance = 0.0
    rows = []
    for rec_id, mentions, refs, length in recordings:
        mentions, refs = list(mentions), list(refs)
        hit = len(anomaly_grounding(mentions, refs, delta_s, compatible).matches) if mentions and refs else 0
        chance = float("nan")
        if chance_draws > 0:
            if mentions and refs:
                draws = []
                for _ in range(chance_draws):
                    rnd = []
                    for m in mentions:
                        w = max(0.0, m.s_hi - m.s_lo)
                        lo = float(rng.uniform(0.0, max(1e-9, float(length) - w)))
                        rnd.append(Mention(m.type, lo, lo + w))
                    draws.append(len(anomaly_grounding(rnd, refs, delta_s, compatible).matches))
                chance = float(np.mean(draws))
            else:
                chance = 0.0
            tot_chance += chance
        tot_m += len(mentions)
        tot_r += len(refs)
        tot_hit += hit
        rows.append((rec_id, len(mentions), len(refs), hit, chance))
    return RecordingGrounding(
        n_mentions=tot_m,
        n_refs=tot_r,
        n_matched=tot_hit,
        agp=tot_hit / tot_m if tot_m else float("nan"),
        agr=tot_hit / tot_r if tot_r else float("nan"),
        chance_agr=(tot_chance / tot_r if tot_r else float("nan")) if chance_draws > 0 else float("nan"),
        per_recording=tuple(rows),
    )
