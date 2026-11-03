"""Teacher generation with a verification filter (METHODOLOGY §8.6, ``synth/teacher.py``).

The teacher here is *programmatic*: the rule planner and the T+V template
realizer of :mod:`roadscribe.layer3.templates`. Every output passes through
the same filter the methodology prescribes for a frontier-model teacher:

1. the plan validates against the capsule's ledger (the §5.3 grammar);
2. every claim the extractor finds in the narrative is supported by the
   ledger, nothing MISSING is stated, and every must-mention event is
   mentioned.

Outputs that fail are dropped, not repaired.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..eval.claims import VerificationReport, verify_text
from ..layer3.ledger import format_ledger
from ..layer3.personas import get_persona
from ..layer3.plan_grammar import canonical_json, validate_plan
from ..layer3.templates import ground, realize, rule_plan
from .sampler import SyntheticTrip

__all__ = ["TEACHER_ID", "TeacherOutput", "teach", "visible_units", "sft_record"]

TEACHER_ID = "roadscribe.synth.teacher/rule-planner+template-realizer v1"


@dataclass
class TeacherOutput:
    plan: dict
    plan_errors: list[str]
    frames: list[dict]
    text: str
    report: VerificationReport
    seed: int

    @property
    def ok(self) -> bool:
        return not self.plan_errors and self.report.passed


def visible_units(trip: SyntheticTrip) -> list[str]:
    return [u for u, lvl in trip.capsule.selection.items() if lvl != "OMIT"]


def teach(trip: SyntheticTrip, seed: int) -> TeacherOutput:
    rng = random.Random(f"plan:{seed}")
    plan = rule_plan(trip.ledger, trip.capsule.selection, trip.persona, rng, trip.must_mention_events)
    errs = validate_plan(plan, trip.ledger, visible_units(trip))
    frames = ground(plan, trip.ledger)
    text = realize(frames, trip.persona, seed=seed, vary=True)
    report = verify_text(text, trip.ledger, omitted_units=plan["omit"], must_mention_events=trip.must_mention_events)
    return TeacherOutput(plan, errs, frames, text, report, seed)


def prompt_ledger(trip: SyntheticTrip) -> str:
    """Fact index shown to the planner: trip-level facts and the visible units' facts."""
    return format_ledger(trip.ledger, [None, *visible_units(trip)], compact=True)


def sft_record(trip: SyntheticTrip, out: TeacherOutput, record_id: str, split: str, config_hash: str) -> dict:
    P = get_persona(trip.persona)
    rep = out.report
    return {
        "id": record_id,
        "split": split,
        "persona": trip.persona,
        "e2_test_persona": P.e2_test_persona,
        "intent": out.plan["intent"],
        "task_mix": trip.task_mix,
        "degradations": trip.patterns,
        "regime_max_felt_class": trip.regime,
        "budget": trip.capsule.budget,
        "budget_effective": trip.kst.budget_effective,
        "tokenizer": trip.capsule.tokenizer,
        "capsule_id": trip.capsule.capsule_id,
        "capsule": trip.capsule.text,
        "ledger": prompt_ledger(trip),
        "must_mention_events": trip.must_mention_events,
        "plan": canonical_json(out.plan),
        "frames": out.frames,
        "narrative": out.text,
        "verification": {
            "passed": rep.passed,
            "plan_valid": not out.plan_errors,
            "n_claims": rep.n_claims,
            "p_ledger": rep.p_ledger,
            "hallucinated_missing": rep.hallucinated_missing,
            "event_mentions": rep.event_mentions,
            "must_mention_missed": rep.must_mention_missed,
        },
        "generator": {"teacher": TEACHER_ID, "seed": out.seed, "config_hash": config_hash},
    }
