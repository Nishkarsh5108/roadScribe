"""Fixed lexicon shared by the template realizer and the claim extractor.

The severity lexicon is fixed *a priori* (§6.3): felt-comfort words follow
the ISO 2631-1 Annex C labels plus synonyms; road-condition words describe the
speed-decoupled surface class. The two lists are disjoint so that a sentence
can be parsed into felt and road claims without ambiguity, and no phrase is a
word-boundary substring of another phrase in a different class.
"""
from __future__ import annotations

import re

__all__ = [
    "FELT_WORDS",
    "ROAD_WORDS",
    "HEDGE_WORDS",
    "EVENT_NOUNS",
    "COUNT_WORDS",
    "MANEUVER_PHRASES",
    "SURFACE_WORDS",
    "number_word",
    "phrase_pattern",
    "contains_lexicon_word",
]

FELT_WORDS: dict[int, tuple[str, ...]] = {
    0: ("comfortable", "perfectly comfortable", "calm"),
    1: ("a little uncomfortable", "slightly uncomfortable", "mildly bumpy"),
    2: ("fairly uncomfortable", "fairly bumpy", "noticeably bumpy"),
    3: ("uncomfortable", "bumpy", "rough"),
    4: ("very uncomfortable", "very bumpy", "very rough"),
    5: ("extremely uncomfortable", "extremely rough", "punishing"),
}

ROAD_WORDS: dict[int, tuple[str, ...]] = {
    0: ("in excellent condition",),
    1: ("in good condition",),
    2: ("in fair condition",),
    3: ("in poor condition",),
    4: ("in very poor condition",),
    5: ("badly broken up",),
}

HEDGE_WORDS: dict[str, tuple[str, ...]] = {
    "assertive": ("definitely", "certainly", "clearly"),
    "likely": ("most likely", "probably", "likely"),
    "possibly": ("possibly", "may have", "may also have", "might have", "could have", "perhaps"),
}

#: noun -> event type it may refer to (bump/pothole: defect; jolt: defect or ambiguous)
EVENT_NOUNS: dict[str, str] = {"bump": "bump", "pothole": "bump", "jolt": "jolt", "swerve": "swerve"}

COUNT_WORDS: dict[str, int] = {
    "a few": 2,
    "several": 3,
    "another": 1,
    "an": 1,
    "a": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}

MANEUVER_PHRASES: dict[str, tuple[str, ...]] = {
    "brake": ("as you braked", "your braking", "under braking"),
    "accel": ("pulling away", "accelerating hard"),
    "stop_go": ("stop-and-go driving",),
    "turn": ("in a turn", "while turning"),
}

#: signal-classifier surface class -> words in text
SURFACE_WORDS: dict[str, str] = {"asphalt": "asphalt", "cobble": "cobblestones", "unpaved": "an unpaved road"}

_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def number_word(n: int) -> str:
    return _NUMBER_WORDS.get(n, str(n))


def phrase_pattern(phrases) -> re.Pattern:
    """Word-bounded alternation, longest phrase first."""
    alts = sorted(set(phrases), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(p) for p in alts) + r")\b", re.IGNORECASE)


_ALL_LEXICON = phrase_pattern(
    [w for ws in FELT_WORDS.values() for w in ws] + [w for ws in ROAD_WORDS.values() for w in ws]
)


def contains_lexicon_word(text: str) -> bool:
    return bool(_ALL_LEXICON.search(text))
