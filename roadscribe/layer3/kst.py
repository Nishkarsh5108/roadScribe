r"""Kinetic-Spatial Tokenizer (METHODOLOGY §5.2, Algorithm 1 in sec3_system.tex).

For each unit the salience is

.. math:: \sigma_m=w_1k_m+w_2\sum_{e\in\mathcal E_m}p_e\,\mathrm{sev}(e)+w_3\Delta_{\mathcal H}(u_m)+w_4\delta^{map}_m,

with every component z-scored across the trip's units and all weights 1.
Each unit has three serialisations :math:`\ell\in\{\text{OMIT},\text{BRIEF},\text{FULL}\}`
with token cost :math:`c_{m,\ell}` and gain :math:`g_{m,\ell}=\sigma^+_m\eta_\ell`
(:math:`\eta_{OMIT}=0`). Must-mention units (an event with
:math:`p_e\,\mathrm{sev}(e)\ge\eta^\star`) cannot be omitted. Selection is
the exact multiple-choice knapsack

.. math:: V_m(b)=\max_{\ell\in\Lambda_m}\{V_{m-1}(b-c_{m,\ell})+g_{m,\ell}\},\qquad V_0(b)=0,

solved in :math:`O(3MB)`.

Two implementation choices that the pseudo-code leaves open:

* z-scored salience can be negative, and a negative gain would make the
  knapsack omit a unit even when budget is left over. Gains therefore use
  :math:`\sigma^+_m=\sigma_m-\min_m\sigma_m+\text{salience\_floor}`, which keeps
  the ranking and spends the whole budget.
* If even BRIEF for every must-mention unit does not fit, Algorithm 1 merges
  adjacent units outside :math:`\mathcal M` and retries. OMIT costs nothing,
  so merging non-must-mention units cannot restore feasibility here; the
  budget is raised to the smallest feasible value instead and the increase is
  logged and reported (``KSTResult.budget_increased``).
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol, Sequence

import numpy as np

from ..config import KSTConfig, LedgerConfig
from ..layer1.iso2631 import comfort_class
from ..schemas import HRNU, Capsule, Event, Fact
from .ledger import Anchor, TripMeta, compile_ledger, event_word, facts_by_unit, nearest_anchor

__all__ = [
    "Tokenizer",
    "RegexTokenizer",
    "CallableTokenizer",
    "hf_tokenizer",
    "LEVELS",
    "event_severity",
    "salience_components",
    "salience",
    "must_mention",
    "serialize_unit",
    "header_lines",
    "rest_line",
    "mckp",
    "KSTResult",
    "build_capsule",
]

log = logging.getLogger(__name__)
LEVELS: tuple[str, str, str] = ("OMIT", "BRIEF", "FULL")


# --------------------------------------------------------------------------- tokenizers
class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class RegexTokenizer:
    """Deterministic stand-in for a BPE tokenizer (letters in chunks of up to 6,
    digits in groups of up to 3, one token per punctuation mark).

    Use the target model's tokenizer (:func:`hf_tokenizer`) for reported
    numbers; this one exists so that the pipeline and its tests have no model
    dependency.
    """

    name = "approx-regex-v1"
    _pat = re.compile(r"[A-Za-z]{1,6}|\d{1,3}|[^\sA-Za-z\d]")

    def count(self, text: str) -> int:
        return len(self._pat.findall(text))


@dataclass
class CallableTokenizer:
    """Adapter for any ``text -> token ids`` (or ``text -> int``) function."""

    fn: Callable[[str], object]
    name: str

    def count(self, text: str) -> int:
        out = self.fn(text)
        return int(out) if isinstance(out, (int, np.integer)) else len(out)  # type: ignore[arg-type]


def hf_tokenizer(model_name: str) -> CallableTokenizer:
    """Tokenizer of a Hugging Face model (requires ``transformers``)."""
    from transformers import AutoTokenizer  # optional dependency

    tok = AutoTokenizer.from_pretrained(model_name)
    return CallableTokenizer(lambda s: tok.encode(s, add_special_tokens=False), model_name)


# --------------------------------------------------------------------------- salience
def event_severity(e: Event) -> float:
    """sev(e) = 1 + ISO comfort class of the event's MTVV (1..6)."""
    m = e.mtvv_z if np.isfinite(e.mtvv_z) else 0.0
    return 1.0 + comfort_class(m)


def _events_by_unit(units: Sequence[HRNU], events: Sequence[Event]) -> list[list[Event]]:
    by_id = {e.event_id: e for e in events}
    return [[by_id[i] for i in u.event_ids if i in by_id] for u in units]


def salience_components(
    units: Sequence[HRNU],
    events: Sequence[Event],
    history_delta: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Raw (M, 4) components [k_m, sum p_e sev(e), Delta_H, delta_map] (missing -> 0)."""
    ev = _events_by_unit(units, events)
    rows = []
    for u, eu in zip(units, ev):
        rows.append(
            [
                float(u.iso_felt),
                float(sum(e.p_defect * event_severity(e) for e in eu)),
                float(history_delta.get(u.unit_id, 0.0)) if history_delta else 0.0,
                float(u.map_signal_disagreement) if u.map_signal_disagreement is not None else 0.0,
            ]
        )
    return np.array(rows, dtype=float).reshape(len(units), 4)


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return np.where(sd > 1e-12, (X - mu) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def salience(
    units: Sequence[HRNU],
    events: Sequence[Event],
    cfg: KSTConfig | None = None,
    history_delta: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """sigma_m with z-scored components and weights (w1..w4)."""
    cfg = cfg or KSTConfig()
    if not units:
        return np.zeros(0)
    Z = _zscore(salience_components(units, events, history_delta))
    return Z @ np.array([cfg.w1, cfg.w2, cfg.w3, cfg.w4])


def must_mention(units: Sequence[HRNU], events: Sequence[Event], cfg: KSTConfig | None = None) -> set[int]:
    """M = {m : exists e in E_m with p_e sev(e) >= eta*}."""
    cfg = cfg or KSTConfig()
    ev = _events_by_unit(units, events)
    return {m for m, eu in enumerate(ev) if any(e.p_defect * event_severity(e) >= cfg.eta_star for e in eu)}


# --------------------------------------------------------------------------- serialisation
def _p2(p: float) -> str:
    return f"{p:.2f}".lstrip("0") if p < 1 else "1.0"


def _events_text(eu: Sequence[Event], brief: bool) -> str:
    if not eu:
        return "" if brief else "ev -"
    groups: dict[str, list[Event]] = {}
    for e in eu:
        groups.setdefault(event_word(e), []).append(e)
    parts = []
    for word, es in groups.items():
        if brief:
            parts.append(f"{word} x{len(es)}")
            continue
        ctx = sorted({e.maneuver.replace("_", "-") for e in es if e.maneuver != "cruise"})
        txt = f"{word} x{len(es)} p" + ",".join(_p2(e.p_defect) for e in es)
        if ctx:
            txt += " " + ",".join(ctx)
        parts.append(txt)
    return "; ".join(parts)


def serialize_unit(
    u: HRNU,
    level: str,
    events: Sequence[Event] = (),
    anchor: Optional[Anchor] = None,
    missing: Sequence[Fact] = (),
) -> str:
    """Compact line format of §5.2. OMIT serialises to the empty string."""
    if level == "OMIT":
        return ""
    name = u.names[0] if u.names else "-"
    head = f"{u.unit_id} {u.s0 / 1000:.1f}-{u.s1 / 1000:.1f}km {name}"
    felt = f"felt {u.iso_felt}" + (" SHOCK" if u.shock else "")
    if level == "BRIEF":
        ev = _events_text(events, brief=True)
        return " | ".join(p for p in (head, felt, ev) if p)
    if level != "FULL":
        raise ValueError(f"unknown level {level!r}")
    tag = u.osm_tags.get("surface") if u.osm_tags else None
    pred = u.conf.get("surface_pred") if isinstance(u.conf, Mapping) else None
    surface = f"{pred} (osm {tag or '-'})" if pred else f"osm {tag or '-'}"
    road = f"road {u.iso_road}" if u.iso_road is not None else "road -"
    parts = [head, surface, felt, road, _events_text(events, brief=False)]
    if anchor is not None:
        parts.append(f"near {anchor.name} @{anchor.s / 1000:.1f}km")
    mm = u.conf.get("map_match") if isinstance(u.conf, Mapping) else None
    if mm is not None:
        parts.append(f"mm {_p2(float(mm))}")
    miss = [f"{f.key}:{f.note}" for f in missing if f.kind == "MISSING"]
    if miss:
        parts.append("missing " + ",".join(miss))
    return " | ".join(parts)


def header_lines(trip: TripMeta, ledger: Sequence[Fact]) -> list[str]:
    dur = f"{trip.duration_s / 60:.1f} min" if np.isfinite(trip.duration_s) else "- min"
    km = f"{trip.length_m / 1000:.1f} km" if np.isfinite(trip.length_m) else "- km"
    lines = [f"TRIP {trip.trip_id} | {dur} | {km} | persona={trip.persona} | ch={','.join(trip.channels)} | hist={trip.history}"]
    for f in ledger:
        if f.kind == "MISSING" and f.unit_id is None:
            lines.append(f"MISSING {f.key} ({f.note})")
    return lines


def rest_line(omitted: Sequence[HRNU], events: Sequence[Event]) -> str:
    """One line summarising the omitted units; empty if nothing is omitted."""
    if not omitted:
        return ""
    km = sum(u.length_m for u in omitted) / 1000
    surfaces: list[str] = []
    for u in sorted(omitted, key=lambda u: -u.length_m):
        tag = u.osm_tags.get("surface") if u.osm_tags else None
        if tag and tag not in surfaces:
            surfaces.append(tag)
    felt = [u.iso_felt for u in omitted]
    n_ev = sum(len(eu) for eu in _events_by_unit(omitted, events))
    ev = "no events" if n_ev == 0 else f"{n_ev} event" + ("s" if n_ev > 1 else "")
    return (
        f"REST {len(omitted)} unit{'s' if len(omitted) > 1 else ''}, {km:.1f} km, "
        f"{','.join(surfaces) if surfaces else 'surface -'}, felt {min(felt)}-{max(felt)}, {ev}"
    )


# --------------------------------------------------------------------------- knapsack
def mckp(
    costs: np.ndarray, gains: np.ndarray, allowed: np.ndarray, budget: int
) -> tuple[Optional[np.ndarray], float]:
    """Exact multiple-choice knapsack by dynamic programming, O(L M B).

    ``costs`` (M, L) non-negative integers, ``gains`` (M, L), ``allowed`` (M, L)
    bool. Returns the chosen option per item (or ``None`` if infeasible) and
    the optimal total gain. Ties prefer the earlier (cheaper) option.
    """
    costs = np.asarray(costs, dtype=np.int64)
    gains = np.asarray(gains, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    M, L = costs.shape
    if np.any(costs < 0):
        raise ValueError("costs must be non-negative")
    if budget < 0:
        return None, float("-inf")
    B = int(budget)
    V = np.zeros(B + 1)  # V_0(b) = 0: capacity "at most b"
    choice = np.zeros((M, B + 1), dtype=np.int8)
    for m in range(M):
        best = np.full(B + 1, -np.inf)
        arg = np.zeros(B + 1, dtype=np.int8)
        for ell in range(L):
            if not allowed[m, ell]:
                continue
            c = int(costs[m, ell])
            if c > B:
                continue
            cand = np.full(B + 1, -np.inf)
            cand[c:] = V[: B + 1 - c] + gains[m, ell]
            better = cand > best
            best[better] = cand[better]
            arg[better] = ell
        V = best
        choice[m] = arg
    if not np.isfinite(V[B]):
        return None, float("-inf")
    out = np.zeros(M, dtype=int)
    b = B
    for m in range(M - 1, -1, -1):
        ell = int(choice[m, b])
        out[m] = ell
        b -= int(costs[m, ell])
    return out, float(V[B])


# --------------------------------------------------------------------------- capsule
@dataclass
class KSTResult:
    capsule: Capsule
    salience: np.ndarray
    costs: np.ndarray  # (M, 3) token costs for OMIT/BRIEF/FULL
    must_mention: set[str]
    budget_effective: int
    budget_increased: bool
    log: list[str] = field(default_factory=list)


def build_capsule(
    units: Sequence[HRNU],
    events: Sequence[Event],
    trip: TripMeta,
    budget: Optional[int] = None,
    tokenizer: Optional[Tokenizer] = None,
    cfg: KSTConfig | None = None,
    ledger: Optional[Sequence[Fact]] = None,
    ledger_cfg: LedgerConfig | None = None,
    anchors: Sequence[Anchor] = (),
    history_delta: Optional[Mapping[str, float]] = None,
) -> KSTResult:
    """Algorithm 1: ledger, salience, token costs, knapsack, capsule text.

    The capsule is guaranteed to have at most ``budget`` tokens under
    ``tokenizer`` unless the must-mention units alone do not fit, in which
    case the effective budget is raised and ``budget_increased`` is set.
    """
    cfg = cfg or KSTConfig()
    tok = tokenizer or RegexTokenizer()
    B = int(cfg.budget if budget is None else budget)
    notes: list[str] = []
    if ledger is None:
        ledger = compile_ledger(units, events, trip, ledger_cfg, anchors=anchors, anchor_rmax_m=cfg.anchor_rmax_m)
    ledger = list(ledger)
    fb = facts_by_unit(ledger)
    ev_units = _events_by_unit(units, events)
    M = len(units)

    sig = salience(units, events, cfg, history_delta)
    sig_pos = sig - (sig.min() if M else 0.0) + cfg.salience_floor
    eta = np.array([0.0, cfg.eta_brief, cfg.eta_full])
    gains = sig_pos[:, None] * eta[None, :]
    mm_idx = must_mention(units, events, cfg)
    allowed = np.ones((M, 3), dtype=bool)
    for m in mm_idx:
        allowed[m, 0] = False

    lines = {}
    costs = np.zeros((M, 3), dtype=np.int64)
    for m, u in enumerate(units):
        a = nearest_anchor(u, anchors, cfg.anchor_rmax_m)
        miss = [f for f in fb.get(u.unit_id, []) if f.kind == "MISSING"]
        for ell, level in enumerate(LEVELS):
            text = serialize_unit(u, level, ev_units[m], a, miss)
            lines[(m, ell)] = text
            costs[m, ell] = tok.count(text + "\n") if text else 0

    header = "\n".join(header_lines(trip, ledger))
    fixed = tok.count(header + "\n")
    c_rest = tok.count(rest_line(list(units), events) + "\n") if M else 0

    def assemble(choice: np.ndarray) -> str:
        body = [lines[(m, int(ell))] for m, ell in enumerate(choice) if int(ell) > 0]
        rest = rest_line([u for m, u in enumerate(units) if int(choice[m]) == 0], events)
        return "\n".join([header] + body + ([rest] if rest else []))

    B_eff = B
    increased = False
    cap = B - fixed - c_rest
    choice: Optional[np.ndarray] = None
    text = header
    for _ in range(cfg.max_refits + 1):
        choice, _ = mckp(costs, gains, allowed, cap) if M else (np.zeros(0, dtype=int), 0.0)
        if choice is None:
            need = int(sum(min(costs[m, 1], costs[m, 2]) for m in mm_idx))
            B_eff = fixed + c_rest + need
            increased = True
            msg = f"budget {B} infeasible for {len(mm_idx)} must-mention units; raised to {B_eff}"
            log.warning(msg)
            notes.append(msg)
            cap = need
            choice, _ = mckp(costs, gains, allowed, cap)
            assert choice is not None
        text = assemble(choice)
        n_tok = tok.count(text)
        if n_tok <= B_eff:
            break
        cap -= n_tok - B_eff  # the joined text cost more than the parts; tighten and retry
        notes.append(f"refit: capsule had {n_tok} > {B_eff} tokens")
    else:  # pragma: no cover - only reachable with a pathological tokenizer
        raise RuntimeError("could not fit the capsule into the budget")

    selection = {u.unit_id: LEVELS[int(choice[m])] for m, u in enumerate(units)} if M else {}
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    capsule = Capsule(
        capsule_id=f"{trip.trip_id}:{trip.persona}:{B}:{digest}",
        trip_id=trip.trip_id,
        persona=trip.persona,
        budget=B,  # requested budget; see KSTResult.budget_effective
        text=text,
        n_tokens=tok.count(text),
        tokenizer=tok.name,
        selection=selection,
        ledger=ledger,
    )
    return KSTResult(
        capsule=capsule,
        salience=sig,
        costs=costs,
        must_mention={units[m].unit_id for m in mm_idx},
        budget_effective=B_eff,
        budget_increased=increased,
        log=notes,
    )
