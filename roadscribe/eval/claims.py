r"""Closed-vocabulary claim extraction and ledger verification (METHODOLOGY §6.1-6.4).

This extractor parses text written in the fixed lexicon of
:mod:`roadscribe.layer3.lexicon` (the template realizer and its controlled
perturbations). It is the "rules" extractor of §8.2 ``eval/claims.py``, not
the validated LLM extractor that §6.1 requires for free text: it only
recognises claims phrased the way the template system phrases them.

Support rules (§6.2):

* NUM: within ``max(tol, half the displayed precision)`` of the ledger value.
* SEV: band consistency, i.e. the measured value lies inside the ISO band of
  the expressed class (bands overlap).
* EVT: event mentions are matched to ledger events by maximum bipartite
  matching; an edge needs a compatible type, a confidence that allows the
  hedge (§5.4), and, when stated, a position within the tolerance and the
  same maneuver.
* ANCHOR: the name or place is attached to that unit.
* CMP: the direction holds with the ledger's margin.
* Claims about a field the ledger marks MISSING count as hallucinated-missing.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence

from ..layer1.iso2631 import COMFORT_BANDS
from ..layer3.hedging import HEDGE_THRESHOLDS, MENTION_THRESHOLD
from ..layer3.ledger import LedgerView
from ..layer3.lexicon import (
    COUNT_WORDS,
    FELT_WORDS,
    HEDGE_WORDS,
    MANEUVER_PHRASES,
    ROAD_WORDS,
    SURFACE_WORDS,
    phrase_pattern,
)
from ..schemas import Fact

__all__ = ["ExtractedClaim", "VerificationReport", "verify_text"]

_FELT = {w.lower(): k for k, ws in FELT_WORDS.items() for w in ws}
_ROAD = {w.lower(): k for k, ws in ROAD_WORDS.items() for w in ws}
_FELT_RE = phrase_pattern(_FELT)
_ROAD_RE = phrase_pattern(_ROAD)
_HEDGE = {h: phrase_pattern(ws) for h, ws in HEDGE_WORDS.items()}
_MAN = {m: phrase_pattern(ws) for m, ws in MANEUVER_PHRASES.items()}
_SURF = {v: k for k, v in SURFACE_WORDS.items()}

_COUNT_ALT = "|".join(re.escape(w) for w in sorted(COUNT_WORDS, key=len, reverse=True))
_EVENT_RE = re.compile(
    rf"\b({_COUNT_ALT}|\d+)\s+(?:(?:sharp|hard|small|single|further|more)\s+)*(bumps?|potholes?|jolts?|swerves?)\b",
    re.IGNORECASE,
)
_RANGE_PAREN = re.compile(r"\((\d+\.\d)[–-](\d+\.\d) km\)")
_RANGE_FROM = re.compile(r"\bfrom (\d+\.\d) to (\d+\.\d) km\b")
_ROADNAME = re.compile(
    r"\b((?:[A-Z][a-z']+ )+(?:Road|Street|Lane|Avenue|Way|Drive|Track|Ring|Boulevard|Row|Close|Parade))\b"
)
_POS = re.compile(r"\baround the ((?:\d+\.\d)(?:(?:, | and )\d+\.\d)*) km marks?\b")
_SPEED = re.compile(r"\b(\d+) km/h")
_ACC = re.compile(r"\b(\d+(?:\.\d+)?) m/s²")
_LEN = re.compile(r"\b(\d+(?:\.\d+)?) (km|m)\b")
_TRIP_MIN = re.compile(r"\b(\d+)(?:-minute| minutes)\b")
_TRIP_KM = re.compile(r"\b(\d+(?:\.\d+)?) km (?:drive|run|trip)\b")
_GPS_GAP = re.compile(r"\b(\d+)-second GPS gap\b")
_ONE_S = re.compile(r"\b1-s\b")
_DIGIT = re.compile(r"\d+(?:\.\d+)?")
_REST = re.compile(r"\b(?:other|remaining) (?:(\w+) )?stretch(?:es)?\b", re.IGNORECASE)
_REST_NOEV = re.compile(r"\bno bumps or jolts\b")
_REST_EV = re.compile(rf"\b({_COUNT_ALT}|\d+) lower-priority events?\b", re.IGNORECASE)
_SPEEDCAUSE = re.compile(r"\bbecause of (?:your |the )?speed\b")
_SHOCK = re.compile(r"\b(shock-dominated|jarring)\b")
_SUPER = re.compile(r"\b(roughest|smoothest) stretch\b")
_CMP = re.compile(r"\b(rougher|smoother) than (?!usual)")
_HIST = re.compile(r"\b(rougher|smoother) than usual\b|\bthe same as usual\b")
_OSM = re.compile(r"\bthe map lists it as ([a-z][a-z ]*?)(?=[,.;]|$)")
_SIGNAL = re.compile(r"\brides like (" + "|".join(re.escape(v) for v in SURFACE_WORDS.values()) + r")\b")
_NEAR = re.compile(r"\bnear ([A-Z][\w']*(?: [A-Z][\w']*)*)")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\-])")


def _band_ok(x: Optional[float], k: int) -> bool:
    if x is None:
        return False
    lo, hi = COMFORT_BANDS[k]
    return lo - 1e-9 <= x <= hi + 1e-9


def _half_prec(num: str) -> float:
    d = len(num.split(".")[1]) if "." in num else 0
    return 0.5 * 10 ** (-d)


def _count(w: str) -> int:
    w = w.lower()
    return int(w) if w.isdigit() else COUNT_WORDS[w]


@dataclass
class ExtractedClaim:
    kind: str
    unit_id: Optional[str]
    key: str
    value: Any
    span: tuple[int, int]
    hedge: str = "assertive"
    supported: Optional[bool] = None
    reason: str = ""


@dataclass
class VerificationReport:
    passed: bool
    claims: list[ExtractedClaim]
    n_claims: int
    n_supported: int
    p_ledger: float
    hallucinated_missing: int
    must_mention_missed: list[str]
    reasons: list[str]
    event_mentions: int = 0
    matched_events: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("claims")
        d["unsupported"] = [
            {"kind": c.kind, "unit": c.unit_id, "key": c.key, "value": c.value, "reason": c.reason,
             "span": list(c.span)}
            for c in self.claims if not c.supported
        ]
        return d


class _Ctx:
    def __init__(self, ledger: Sequence[Fact], cmp_margin: float, v_ref: float):
        self.L = LedgerView(ledger)
        self.margin = cmp_margin
        self.v_ref = v_ref
        self.name_re = phrase_pattern(self.L.name_units) if self.L.name_units else None
        self.claims: list[ExtractedClaim] = []
        self.slots: list[dict] = []
        self.a_v = {u: self.L.value(u, "a_v") for u in self.L.units}

    def add(self, kind, unit, key, value, span, ok: bool, reason: str = "", hedge="assertive"):
        self.claims.append(ExtractedClaim(kind, unit, key, value, span, hedge, bool(ok), "" if ok else reason))

    def unit_by_range(self, a: str, b: str) -> Optional[str]:
        for u in self.L.units:
            s0, s1 = self.L.extent(u)
            if abs(s0 / 1000 - float(a)) <= 0.0501 and abs(s1 / 1000 - float(b)) <= 0.0501:
                return u
        return None

    def num(self, unit, key, shown: str, x_si: float, scale: float, span, kind="NUM"):
        f = self.L.get(unit, key)
        if f is None or f.kind == "MISSING":
            reason = "hallucinated_missing" if f is not None else "no_value"
            self.add(kind, unit, key, x_si, span, False, reason)
            return
        tol = max(f.tol or 0.0, _half_prec(shown) * scale)
        self.add(kind, unit, key, x_si, span, abs(float(f.value) - x_si) <= tol + 1e-9, "number_mismatch")


def _sentences(text: str) -> list[tuple[int, int]]:
    out = []
    pos = 0
    for line in text.split("\n"):
        start = pos
        last = 0
        for m in _SENT.finditer(line):
            out.append((start + last, start + m.start()))
            last = m.end()
        out.append((start + last, start + len(line)))
        pos += len(line) + 1
    return [(a, b) for a, b in out if text[a:b].strip()]


def _clauses(a: int, b: int, text: str) -> list[tuple[int, int]]:
    out, s = [], a
    for i in range(a, b):
        if text[i] == ";":
            out.append((s, i))
            s = i + 1
    out.append((s, b))
    return out


def _clause_hedge(c: str) -> str:
    if _HEDGE["assertive"].search(c):
        return "assertive"
    if _HEDGE["possibly"].search(c):
        return "possibly"
    if _HEDGE["likely"].search(c):
        return "likely"
    return "assertive"


def _overlaps(span, used) -> bool:
    return any(span[0] < b and a < span[1] for a, b in used)


def verify_text(
    text: str,
    ledger: Sequence[Fact],
    *,
    omitted_units: Optional[Iterable[str]] = None,
    must_mention_events: Iterable[str] = (),
    cmp_margin: float = 0.2,
    v_ref: float = 10.0,
) -> VerificationReport:
    """Extract every claim from ``text`` and check it against ``ledger``.

    ``omitted_units`` are the units summarised by a REST sentence (the plan's
    ``omit`` list). ``must_mention_events`` are event ids that must be matched
    by some mention (events below the mention threshold are exempt).
    """
    C = _Ctx(ledger, cmp_margin, v_ref)
    L = C.L
    omitted = list(omitted_units) if omitted_units is not None else None
    prev_unit: Optional[str] = None
    for a, b in _sentences(text):
        s = text[a:b]
        used: list[tuple[int, int]] = []

        # ---------------- references (ranges, names) -> sentence unit
        refs: list[tuple[int, int, Optional[str], bool]] = []  # start, end, unit, is_comparison_target
        for m in list(_RANGE_PAREN.finditer(s)) + list(_RANGE_FROM.finditer(s)):
            u = C.unit_by_range(m.group(1), m.group(2))
            C.add("RANGE", u, "extent", (m.group(1), m.group(2)), (a + m.start(), a + m.end()), u is not None, "unknown_range")
            used.append((m.start(), m.end()))
            refs.append((m.start(), m.end(), u, False))
        name_spans = []
        if C.name_re is not None:
            name_spans = [(m.start(), m.end(), m.group(1)) for m in C.name_re.finditer(s)]
        for m in _ROADNAME.finditer(s):
            if not any(m.start() < e and st < m.end() for st, e, _ in name_spans):
                name_spans.append((m.start(), m.end(), m.group(1)))
        for st, e, name in sorted(name_spans):
            before = s[:st].rstrip()
            if before.endswith(" near") or before == "near":
                continue
            rng = _RANGE_PAREN.match(s, e + 1) if e + 1 <= len(s) else None
            known = [k for k in L.name_units if k.lower() == name.lower()]
            cands = L.name_units[known[0]] if known else []
            if rng is not None:
                u = C.unit_by_range(rng.group(1), rng.group(2))
            elif len(cands) == 1:
                u = cands[0]
            elif prev_unit in cands:
                u = prev_unit
            else:
                u = None
            if u is None:
                C.add("NAME", None, "name", name, (a + st, a + e), False, "ambiguous_reference" if cands else "unknown_name")
            elif name in L.names(u):
                C.add("NAME", u, "name", name, (a + st, a + e), True)
            else:
                C.add("NAME", u, "name", name, (a + st, a + e), False,
                      "hallucinated_missing" if "name" in L.missing(u) else "wrong_name")
            used.append((st, e))
            refs.append((st, e, u, False))
        refs.sort()
        # the reference right after "rougher/smoother than" (plus a range glued to it) is a comparison target
        for m in _CMP.finditer(s):
            nxt = next((i for i, r_ in enumerate(refs) if r_[0] >= m.end() - 1), None)
            if nxt is None:
                continue
            refs[nxt] = (*refs[nxt][:3], True)
            j = nxt + 1
            while j < len(refs) and refs[j][0] - refs[j - 1][1] <= 2 and refs[j][2] == refs[nxt][2]:
                refs[j] = (*refs[j][:3], True)
                j += 1
        primary = [r for r in refs if not r[3]]
        rest_m = _REST.search(s)
        if rest_m:
            unit = "REST"
        elif primary:
            unit = primary[0][2]
        elif _TRIP_MIN.search(s) or _TRIP_KM.search(s) or s.startswith("Trip report"):
            unit = "TRIP"
        else:
            unit = prev_unit
        if unit not in ("TRIP", "REST"):
            prev_unit = unit

        # ---------------- trip-level and special numbers
        for m in _TRIP_MIN.finditer(s):
            C.num(None, "trip_duration_s", m.group(1), float(m.group(1)) * 60, 60, (a + m.start(), a + m.end()))
            used.append((m.start(), m.end()))
        for m in _TRIP_KM.finditer(s):
            C.num(None, "trip_length_m", m.group(1), float(m.group(1)) * 1000, 1000, (a + m.start(), a + m.end()))
            used.append((m.start(), m.end()))
        for m in _GPS_GAP.finditer(s):
            notes = set(L.missing(unit if unit not in ("TRIP", "REST") else None).values()) | set(L.missing(None).values())
            C.add("MISSING_ACK", unit, "anchor", m.group(1), (a + m.start(), a + m.end()), f"gps_gap_{m.group(1)}s" in notes, "number_mismatch")
            used.append((m.start(), m.end()))
        for m in _ONE_S.finditer(s):
            used.append((m.start(), m.end()))

        # ---------------- REST sentence
        if unit == "REST":
            used.append(rest_m.span())
            n = _count(rest_m.group(1)) if rest_m.group(1) else 1
            om = omitted if omitted is not None else []
            C.add("REST", "REST", "n_units", n, (a + rest_m.start(), a + rest_m.end()), n == len(om), "rest_count")
            classes = sorted(_FELT[m.group(1).lower()] for m in _FELT_RE.finditer(s))
            if classes:
                lo, hi = classes[0], classes[-1]
                ok = all(any(_band_ok(C.a_v.get(u), k) for k in range(lo, hi + 1)) for u in om)
                C.add("REST", "REST", "felt_range", (lo, hi), (a, b), ok, "severity_mismatch")
            mentionable = sum(1 for u in om for f in L.events(u) if f.conf >= MENTION_THRESHOLD and f.value.get("label") != "artifact")
            if _REST_NOEV.search(s):
                C.add("REST", "REST", "n_events", 0, (a, b), mentionable == 0, "event_count")
            for m in _REST_EV.finditer(s):
                C.add("REST", "REST", "n_events", _count(m.group(1)), (a + m.start(), a + m.end()), _count(m.group(1)) == mentionable, "event_count")
                used.append((m.start(), m.end()))
            for m in _LEN.finditer(s):
                if _overlaps(m.span(), used):
                    continue
                x = float(m.group(1)) * (1000 if m.group(2) == "km" else 1)
                tot = sum(L.value(u, "length_m", 0.0) for u in om)
                tol = max(10.0, 0.1 * tot, _half_prec(m.group(1)) * (1000 if m.group(2) == "km" else 1))
                C.add("REST", "REST", "length_m", x, (a + m.start(), a + m.end()), abs(x - tot) <= tol, "number_mismatch")
                used.append(m.span())
            for m in _DIGIT.finditer(s):
                if not _overlaps(m.span(), used):
                    C.add("UNPARSED", "REST", "number", m.group(0), (a + m.start(), a + m.end()), False, "unparsed_number")
            continue

        u = unit if unit not in ("TRIP",) else None
        resolved = u is not None

        def need_unit(kind, key, value, span) -> bool:
            if not resolved:
                C.add(kind, None, key, value, span, False, "unresolved")
            return resolved

        # ---------------- unit-level claims (sentence level)
        for m in _FELT_RE.finditer(s):
            k = _FELT[m.group(1).lower()]
            if need_unit("SEV", "felt_class", k, (a + m.start(), a + m.end())):
                C.add("SEV", u, "felt_class", k, (a + m.start(), a + m.end()), _band_ok(C.a_v.get(u), k), "severity_mismatch")
        for m in _ROAD_RE.finditer(s):
            k = _ROAD[m.group(1).lower()]
            if need_unit("SEV", "road_class", k, (a + m.start(), a + m.end())):
                if "road_class" in L.missing(u):
                    C.add("SEV", u, "road_class", k, (a + m.start(), a + m.end()), False, "hallucinated_missing")
                else:
                    C.add("SEV", u, "road_class", k, (a + m.start(), a + m.end()), _band_ok(L.value(u, "a_wz_dec"), k), "severity_mismatch")
        for m in _SPEEDCAUSE.finditer(s):
            if need_unit("CMP", "speed_cause", True, (a + m.start(), a + m.end())):
                if "road_class" in L.missing(u):
                    C.add("CMP", u, "speed_cause", True, (a + m.start(), a + m.end()), False, "hallucinated_missing")
                else:
                    fc, rc, v = L.value(u, "felt_class"), L.value(u, "road_class"), L.value(u, "v_mean_mps")
                    ok = None not in (fc, rc, v) and fc > rc and v > C.v_ref
                    C.add("CMP", u, "speed_cause", True, (a + m.start(), a + m.end()), ok, "unsupported_cause")
        for m in _REST_NOEV.finditer(s):
            if need_unit("EVT", "no_events", True, (a + m.start(), a + m.end())):
                n_ev = sum(1 for f in L.events(u) if f.conf >= MENTION_THRESHOLD and f.value.get("label") != "artifact")
                C.add("EVT", u, "no_events", True, (a + m.start(), a + m.end()), n_ev == 0, "fabricated_absence")
        for m in _SHOCK.finditer(s):
            if need_unit("SEV", "shock", True, (a + m.start(), a + m.end())):
                C.add("SEV", u, "shock", True, (a + m.start(), a + m.end()), L.get(u, "shock", "SEV") is not None, "severity_mismatch")
        for m in _SUPER.finditer(s):
            if need_unit("CMP", "superlative", m.group(1), (a + m.start(), a + m.end())):
                vals = [x for x in C.a_v.values() if x is not None]
                x = C.a_v.get(u)
                tol = (L.get(u, "a_v").tol or 0.0) if L.get(u, "a_v") else 0.0
                ok = x is not None and (x >= max(vals) - tol if m.group(1) == "roughest" else x <= min(vals) + tol)
                C.add("CMP", u, "superlative", m.group(1), (a + m.start(), a + m.end()), ok, "comparison_mismatch")
        for m in _CMP.finditer(s):
            tgt = next((r for r in refs if r[0] >= m.end() - 1 and r[3]), None)
            other = tgt[2] if tgt else None
            if need_unit("CMP", "a_v_vs_unit", m.group(1), (a + m.start(), a + m.end())):
                x, y = C.a_v.get(u), C.a_v.get(other) if other else None
                ok = x is not None and y is not None and (
                    x > (1 + C.margin) * y if m.group(1) == "rougher" else x < y / (1 + C.margin))
                C.add("CMP", u, "a_v_vs_unit", {"relation": m.group(1), "other": other}, (a + m.start(), a + m.end()), ok, "comparison_mismatch")
        for m in _HIST.finditer(s):
            rel = m.group(1) or "similar"
            if need_unit("CMP", "a_v_vs_history", rel, (a + m.start(), a + m.end())):
                f = L.get(u, "a_v_vs_history")
                if f is None:
                    C.add("CMP", u, "a_v_vs_history", rel, (a + m.start(), a + m.end()), False, "no_history")
                elif f.kind == "MISSING":
                    C.add("CMP", u, "a_v_vs_history", rel, (a + m.start(), a + m.end()), False, "hallucinated_missing")
                else:
                    C.add("CMP", u, "a_v_vs_history", rel, (a + m.start(), a + m.end()), f.value["relation"] == rel, "comparison_mismatch")
        for m in _OSM.finditer(s):
            if need_unit("ANCHOR", "osm_surface", m.group(1), (a + m.start(), a + m.end())):
                tag = L.value(u, "osm_surface")
                C.add("ANCHOR", u, "osm_surface", m.group(1), (a + m.start(), a + m.end()),
                      tag is not None and tag.replace("_", " ") == m.group(1).strip(), "wrong_map_tag")
        for m in _SIGNAL.finditer(s):
            if need_unit("SEV", "surface_signal", m.group(1), (a + m.start(), a + m.end())):
                C.add("SEV", u, "surface_signal", m.group(1), (a + m.start(), a + m.end()),
                      L.value(u, "surface_signal") == _SURF[m.group(1)], "wrong_surface")
        for m in _NEAR.finditer(s):
            if need_unit("ANCHOR", "near", m.group(1), (a + m.start(), a + m.end())):
                pois = [f.value["name"] for f in L.all(u, "near", "ANCHOR")]
                C.add("ANCHOR", u, "near", m.group(1), (a + m.start(), a + m.end()), m.group(1) in pois, "wrong_anchor")
            used.append(m.span())

        # ---------------- clause-level: positions, events, numbers
        for ca, cb in _clauses(0, len(s), s):
            c = s[ca:cb]
            hedge = _clause_hedge(c)
            positions: list[float] = []
            for m in _POS.finditer(c):
                positions += [float(x) * 1000 for x in re.findall(r"\d+\.\d", m.group(1))]
                used.append((ca + m.start(), ca + m.end()))
            man = next((k for k, rx in _MAN.items() if rx.search(c)), None)
            mentions = list(_EVENT_RE.finditer(c))
            for m in mentions:
                used.append((ca + m.start(), ca + m.end()))
                n = _count(m.group(1))
                noun = m.group(2).lower().rstrip("s")
                typ = {"bump": "bump", "pothole": "bump", "jolt": "jolt", "swerve": "swerve"}[noun]
                for i in range(n):
                    pos = positions[i] if len(positions) == n and len(mentions) == 1 else None
                    C.slots.append({"unit": u, "type": typ, "hedge": hedge, "pos": pos, "man": man,
                                    "span": (a + ca + m.start(), a + ca + m.end())})
            for m in _SPEED.finditer(c):
                if not _overlaps((ca + m.start(), ca + m.end()), used) and need_unit("NUM", "v_mean_mps", m.group(1), (a + ca + m.start(), a + ca + m.end())):
                    C.num(u, "v_mean_mps", m.group(1), float(m.group(1)) / 3.6, 1 / 3.6, (a + ca + m.start(), a + ca + m.end()))
                used.append((ca + m.start(), ca + m.end()))
            for m in _ACC.finditer(c):
                key = "mtvv_z" if "peak" in c[: m.start()] else "a_v"
                if need_unit("NUM", key, m.group(1), (a + ca + m.start(), a + ca + m.end())):
                    C.num(u, key, m.group(1), float(m.group(1)), 1.0, (a + ca + m.start(), a + ca + m.end()))
                used.append((ca + m.start(), ca + m.end()))
            for m in _LEN.finditer(c):
                sp = (ca + m.start(), ca + m.end())
                if _overlaps(sp, used):
                    continue
                scale = 1000.0 if m.group(2) == "km" else 1.0
                if need_unit("NUM", "length_m", m.group(1), (a + sp[0], a + sp[1])):
                    C.num(u, "length_m", m.group(1), float(m.group(1)) * scale, scale, (a + sp[0], a + sp[1]))
                used.append(sp)
        for m in _DIGIT.finditer(s):
            if not _overlaps(m.span(), used):
                C.add("UNPARSED", u, "number", m.group(0), (a + m.start(), a + m.end()), False, "unparsed_number")

    # ---------------- event matching per unit (Kuhn's algorithm, must-mention events first)
    must = set(must_mention_events)
    matched_events: list[str] = []
    for unit in {sl["unit"] for sl in C.slots} | {L.unit_of_event(e) for e in must}:
        slots = [sl for sl in C.slots if sl["unit"] == unit]
        evs = sorted(
            [{"id": f.key, "p": f.conf, **f.value} for f in L.events(unit)] if unit else [],
            key=lambda e: (e["id"] not in must, -e["p"]),
        )

        def compat(e, sl, relax=()):
            t = sl["type"]
            ok_type = (
                (t == "bump" and e["kind"] == "impulse" and e["label"] == "defect")
                or (t == "jolt" and e["kind"] == "impulse" and e["label"] in ("defect", "ambiguous"))
                or (t == "swerve" and e["kind"] == "avoidance")
            )
            if not ok_type:
                return False
            if "hedge" not in relax and e["p"] < HEDGE_THRESHOLDS[sl["hedge"]] - 1e-12:
                return False
            if "pos" not in relax and sl["pos"] is not None and abs(e["s"] - sl["pos"]) > 50.0 + 1e-6:
                return False
            if "man" not in relax and sl["man"] is not None and e["maneuver"] != sl["man"]:
                return False
            return True

        match_slot: dict[int, int] = {}

        def try_ev(i, seen):
            for j, sl in enumerate(slots):
                if j in seen or not compat(evs[i], sl):
                    continue
                seen.add(j)
                if j not in match_slot or try_ev(match_slot[j], seen):
                    match_slot[j] = i
                    return True
            return False

        for i in range(len(evs)):
            try_ev(i, set())
        got = {i for i in match_slot.values()}
        matched_events += [evs[i]["id"] for i in got]
        for j, sl in enumerate(slots):
            ok = j in match_slot
            reason = ""
            if not ok:
                free = [e for i, e in enumerate(evs) if i not in got]
                if sl["unit"] is None:
                    reason = "unresolved"
                elif any(compat(e, sl, ("hedge",)) for e in free):
                    reason = "overclaimed_hedge"
                elif any(compat(e, sl, ("man",)) for e in free):
                    reason = "misattributed_maneuver"
                elif any(compat(e, sl, ("pos",)) for e in free):
                    reason = "wrong_position"
                elif any(compat(e, sl, ("hedge", "pos", "man")) for e in evs):
                    reason = "event_count"
                else:
                    reason = "fabricated_event"
            C.claims.append(ExtractedClaim("EVT", sl["unit"], sl["type"], {"pos": sl["pos"], "maneuver": sl["man"]},
                                           sl["span"], sl["hedge"], ok, reason))

    missed = sorted(
        e for e in must
        if e not in matched_events
        and (f := next((x for x in L.facts if x.kind == "EVT" and x.key == e), None)) is not None
        and f.conf >= MENTION_THRESHOLD
    )
    claims = sorted(C.claims, key=lambda c: c.span)
    n_sup = sum(1 for c in claims if c.supported)
    reasons = sorted({c.reason for c in claims if not c.supported} | ({"missed_must_mention"} if missed else set()))
    return VerificationReport(
        passed=n_sup == len(claims) and not missed,
        claims=claims,
        n_claims=len(claims),
        n_supported=n_sup,
        p_ledger=n_sup / len(claims) if claims else 1.0,
        hallucinated_missing=sum(1 for c in claims if c.reason == "hallucinated_missing"),
        must_mention_missed=missed,
        reasons=reasons,
        event_mentions=len(C.slots),
        matched_events=sorted(matched_events),
    )
