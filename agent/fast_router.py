"""Fast Router -- deterministic local text classifier
(HERMES-FAST-ROUTER-001).

No LLM classification anywhere in this module: every pattern is a
plain regex matched against the ENTIRE stripped input (anchored
``^...$``), never a substring match inside a longer sentence, keeping
false positives low -- this task's own instruction prioritizes
avoiding misfires ("誤爆防止を優先する") over maximal phrase coverage.

This module never hardcodes a Persona name, alias, or canonical ID --
a name/alias candidate extracted by regex is only ever validated via
agent.pp_context_bridge.resolve_persona_id() (the same resolver the
explicit "/pp ..." commands use). When a structurally-matching phrase
turns out not to name a real Persona (e.g. "警察について教えて" --
"警察" is the generic Japanese word for "police", not a registered
alias), this module falls through to LIGHT_CHAT rather than guessing
or raising -- ordinary conversation must never misroute into the
Bridge.
"""
from __future__ import annotations

import re
from typing import Any, Optional

ROUTE_LIGHT_CHAT = "LIGHT_CHAT"
ROUTE_PERSONA_LIST = "PERSONA_LIST"
ROUTE_PERSONA_LOOKUP = "PERSONA_LOOKUP"
ROUTE_CHATTER = "CHATTER"

_CHATTER_PATTERNS = (
    re.compile(r"^(?P<a>.+?)と(?P<b>.+?)で(話して|会話して)[。.!！]*$"),
    re.compile(r"^(?P<a>.+?)と(?P<b>.+?)を(話させて|会話させて)[。.!！]*$"),
)

_LOOKUP_PATTERNS = (
    re.compile(r"^(?P<name>.+?)は誰(ですか)?[?？]?$"),
    re.compile(r"^(?P<name>.+?)って誰(ですか)?[?？]?$"),
    re.compile(r"^(?P<name>.+?)について(教えて)?[。.]?$"),
    re.compile(r"^(?P<name>.+?)の役割は?[?？]?$"),
)

_PERSONA_LIST_INTENT_PATTERNS = (
    re.compile(r"^今いる人格[はの]?[?？]?$"),
    re.compile(r"^人格一覧[を見せてくださいる]*[。.]?$"),
    re.compile(r"^人格を見せて[。.]?$"),
    re.compile(r"^[Pp]ersona一覧[。.]?$"),
    re.compile(r"^ペルソナ一覧[。.]?$"),
)

# Deliberately an EXACT-match-only safe set, never a substring/regex --
# "誰がいる？" alone is genuinely ambiguous outside a Persona context
# (could be about people in a room), so it is only routed when it is
# the WHOLE message, not embedded in a longer sentence. See module
# docstring.
_PERSONA_LIST_EXACT_SAFE_SET = frozenset({"誰がいる？", "誰がいるの？", "誰がいる", "誰がいるの"})


def _try_chatter(text: str) -> Optional[tuple[str, str]]:
    for pattern in _CHATTER_PATTERNS:
        m = pattern.match(text)
        if m:
            return m.group("a").strip(), m.group("b").strip()
    return None


def _try_lookup(text: str) -> Optional[str]:
    for pattern in _LOOKUP_PATTERNS:
        m = pattern.match(text)
        if m:
            return m.group("name").strip()
    return None


def _is_persona_list(text: str) -> bool:
    if text in _PERSONA_LIST_EXACT_SAFE_SET:
        return True
    return any(p.match(text) for p in _PERSONA_LIST_INTENT_PATTERNS)


def route(text: str, pp_repo_path: Optional[str] = None) -> dict[str, Any]:
    """Deterministically classifies `text`. Returns
    {"route": ROUTE_*, ...}. Never calls an LLM. Any Persona name
    candidate is resolved via agent.pp_context_bridge.resolve_persona_id()
    -- this module keeps no alias table of its own, so a Bell/Verba/
    Police-class alias added to PP data later works here automatically,
    with zero code changes."""
    stripped = (text or "").strip()
    if not stripped:
        return {"route": ROUTE_LIGHT_CHAT}

    from agent.pp_context_bridge import resolve_persona_id

    chatter_names = _try_chatter(stripped)
    if chatter_names:
        a_raw, b_raw = chatter_names
        cid_a = resolve_persona_id(a_raw, pp_repo_path)
        cid_b = resolve_persona_id(b_raw, pp_repo_path)
        if cid_a and cid_b:
            return {"route": ROUTE_CHATTER, "persona_a": a_raw, "persona_b": b_raw}
        # Structurally "AとBで話して" but A/B aren't both real
        # Personas -- not actually a CHATTER request.

    lookup_name = _try_lookup(stripped)
    if lookup_name:
        if resolve_persona_id(lookup_name, pp_repo_path):
            return {"route": ROUTE_PERSONA_LOOKUP, "query": lookup_name}
        # Structurally "...について教えて" etc. but the subject isn't a
        # known Persona -- ordinary conversation.

    if _is_persona_list(stripped):
        return {"route": ROUTE_PERSONA_LIST}

    return {"route": ROUTE_LIGHT_CHAT}
