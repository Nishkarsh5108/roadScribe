"""Prompt formats for the dataset records (PLAN, REALIZE, QA, DPO).

Records store the raw parts (capsule, fact index, plan, frames, text) once;
:func:`to_chat` turns a record into chat messages for a given task, so the
static system text (the KV-cached prefix of §5.3) is defined in one place.
"""
from __future__ import annotations

import json
from typing import Literal

from ..layer3.personas import get_persona

__all__ = ["SYSTEM_PLAN", "SYSTEM_REALIZE", "SYSTEM_QA", "persona_card", "to_chat"]

SYSTEM_PLAN = (
    "You are RoadScribe's planner. Read the trip capsule and the fact index, then choose which stretches "
    "and facts to mention for this reader, in what order, with a rhetorical relation (elaboration, contrast, "
    "cause, comparison_history) and a hedge (none, likely, possibly). Reply with one JSON plan. Use only ids "
    "that appear in the fact index; omitted stretches go in \"omit\" and are summarised by a REST item. "
    "Never mention events with confidence below 0.4."
)
SYSTEM_REALIZE = (
    "You are RoadScribe's narrator. Write the trip narrative from the fact frames for the reader described. "
    "Use only the values, names and events in the frames; do not add any number, place or event. Hedge "
    "events by confidence: no hedge at 0.9 or above, 'probably' or 'likely' from 0.7, 'possibly' or "
    "'may have' from 0.4. Say plainly when a value is missing instead of guessing it. Describe how the ride "
    "felt separately from the condition of the surface."
)
SYSTEM_QA = (
    "You are RoadScribe. Answer the driver's question about this trip using only the capsule and the fact "
    "index. If the trip data cannot answer it, say so and why, starting with \"I can't tell that from this drive\"."
)


def persona_card(persona: str) -> str:
    P = get_persona(persona)
    return f"Reader: {P.label}. {P.card}"


def to_chat(record: dict, task: Literal["plan", "realize", "qa", "dpo"]) -> dict:
    """Chat messages for one record.

    * ``plan``:    capsule + fact index -> plan JSON (SFT record)
    * ``realize``: fact frames -> narrative (SFT record)
    * ``qa``:      capsule + fact index + question -> answer (QA record)
    * ``dpo``:     realization prompt with ``chosen`` and ``rejected`` (DPO record)
    """
    card = persona_card(record["persona"])
    if task == "plan":
        user = f"{card}\n\nCAPSULE\n{record['capsule']}\n\nFACT INDEX\n{record['ledger']}"
        return {"messages": [{"role": "system", "content": SYSTEM_PLAN}, {"role": "user", "content": user},
                             {"role": "assistant", "content": record["plan"]}]}
    if task in ("realize", "dpo"):
        frames = json.dumps(record["frames"], separators=(",", ":"), sort_keys=True)
        prompt = [{"role": "system", "content": SYSTEM_REALIZE}, {"role": "user", "content": f"{card}\n\nFRAMES\n{frames}"}]
        if task == "realize":
            return {"messages": prompt + [{"role": "assistant", "content": record["narrative"]}]}
        return {"prompt": prompt, "chosen": [{"role": "assistant", "content": record["chosen"]}],
                "rejected": [{"role": "assistant", "content": record["rejected"]}]}
    if task == "qa":
        user = f"{card}\n\nCAPSULE\n{record['capsule']}\n\nFACT INDEX\n{record['ledger']}\n\nQUESTION\n{record['question']}"
        return {"messages": [{"role": "system", "content": SYSTEM_QA}, {"role": "user", "content": user},
                             {"role": "assistant", "content": record["gold_answer"]}]}
    raise ValueError(f"unknown task {task!r}")
