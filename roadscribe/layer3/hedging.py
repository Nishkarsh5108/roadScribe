"""Confidence -> hedge policy (METHODOLOGY §5.4).

| p          | wording                          |
|------------|----------------------------------|
| >= 0.9     | assertive (no hedge)             |
| 0.7 - 0.9  | "likely", "probably"             |
| 0.4 - 0.7  | "possibly", "may have"           |
| < 0.4      | not mentioned unless asked       |

A claim is an *overclaim* when its hedge is stronger than its confidence
allows ("definitely" on a 0.6 event); that is a verification failure.
"""
from __future__ import annotations

from typing import Literal, Optional

__all__ = ["Hedge", "HEDGE_THRESHOLDS", "MENTION_THRESHOLD", "hedge_for", "allows", "strength"]

Hedge = Literal["assertive", "likely", "possibly"]

#: Minimum confidence each hedge may be used for.
HEDGE_THRESHOLDS: dict[str, float] = {"assertive": 0.9, "likely": 0.7, "possibly": 0.4}
MENTION_THRESHOLD = 0.4
_ORDER = {"possibly": 0, "likely": 1, "assertive": 2}


def hedge_for(p: float) -> Optional[str]:
    """Required hedge for confidence ``p``; ``None`` means do not mention unprompted."""
    if p >= HEDGE_THRESHOLDS["assertive"]:
        return "assertive"
    if p >= HEDGE_THRESHOLDS["likely"]:
        return "likely"
    if p >= HEDGE_THRESHOLDS["possibly"]:
        return "possibly"
    return None


def allows(hedge: str, p: float) -> bool:
    """True if ``hedge`` is not stronger than confidence ``p`` permits."""
    return p >= HEDGE_THRESHOLDS[hedge]


def strength(hedge: str) -> int:
    return _ORDER[hedge]
