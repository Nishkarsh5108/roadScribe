"""Persona cards for planning, realization and prompts.

METHODOLOGY §2.2 (E2) holds out *motion-sensitive passenger*, *fleet
manager* and *municipal road engineer* as unseen test personas, and §8.6
forbids test personas in any few-shot or training material. Cards carry
``e2_test_persona`` so that training data can be filtered accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Persona", "PERSONAS", "get_persona", "E2_TEST_PERSONAS"]


@dataclass(frozen=True)
class Persona:
    key: str
    label: str
    card: str  # what the model is told about the reader
    numbers: bool  # state numeric values (speeds, m/s^2, lengths)
    road_focus: bool  # describe the surface itself (road class, surface type)
    maneuver_focus: bool  # attribute jolts to driving (braking, pulling away)
    always_range: bool  # always give the km range of a stretch
    report_style: bool  # one line per stretch instead of prose
    e2_test_persona: bool
    max_units: int = 6  # stretches described individually; the rest go into the REST summary


PERSONAS: dict[str, Persona] = {
    "commuter": Persona(
        "commuter",
        "daily commuter",
        "Drives the same routes to work. Wants to know which stretches were rough, where the jolts "
        "were, and whether to drive any part differently next time. Short, plain, friendly.",
        numbers=False, road_focus=False, maneuver_focus=True, always_range=False, report_style=False,
        e2_test_persona=False, max_units=4,
    ),
    "sports_car": Persona(
        "sports_car",
        "sports car driver",
        "Drives a low, stiffly sprung car for enjoyment. Cares about the surface itself and the pace "
        "it allows: where the road is in good shape, where speed made it harsh, what the surface is "
        "made of. Comfortable with speeds and numbers.",
        numbers=True, road_focus=True, maneuver_focus=False, always_range=False, report_style=False,
        e2_test_persona=False, max_units=5,
    ),
    "car_sick_passenger": Persona(
        "car_sick_passenger",
        "car-sick passenger",
        "Rides as a passenger and gets motion sick. Cares about how the ride felt, about jolts and "
        "about driving that makes it worse (hard braking, pulling away). Gentle wording, no numbers.",
        numbers=False, road_focus=False, maneuver_focus=True, always_range=False, report_style=False,
        e2_test_persona=True, max_units=3,
    ),
    "fleet_manager": Persona(
        "fleet_manager",
        "fleet manager",
        "Manages delivery vans. Wants a terse per-stretch report: locations as km ranges, severity, "
        "weighted vibration values, shock-dominated stretches, defect positions, map disagreements "
        "and data gaps. One line per stretch.",
        numbers=True, road_focus=True, maneuver_focus=False, always_range=True, report_style=True,
        e2_test_persona=True, max_units=6,
    ),
}

E2_TEST_PERSONAS = frozenset(k for k, p in PERSONAS.items() if p.e2_test_persona)


def get_persona(key: str) -> Persona:
    try:
        return PERSONAS[key]
    except KeyError:
        raise KeyError(f"unknown persona {key!r}; known: {sorted(PERSONAS)}") from None
