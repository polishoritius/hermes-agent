"""Deterministic human-friendly formatters for PP Context Bridge
results reached via the Fast Router's natural-language routes
(HERMES-FAST-ROUTER-001).

Plain string templating only -- no LLM involved. The explicit
"/pp ..." commands are unaffected by this module and keep printing
the existing raw structured JSON (see cli.py); only the natural-
language routes (PERSONA_LIST / PERSONA_LOOKUP / CHATTER reached via
agent/fast_router.py) use these formatters. The Bridge's own return
values stay structured data -- these functions only shape a display
string on top of that data, never mutate it.
"""
from __future__ import annotations

from typing import Any


def format_persona_list(data: dict[str, Any]) -> str:
    personas = data.get("personas", [])
    lines = [f"現在のPersonaは{len(personas)}名です。"]
    for p in personas:
        lines.append(f"- {p['name']}")
    return "\n".join(lines)


def format_persona(data: dict[str, Any], query: str = "") -> str:
    if "error" in data:
        return f"「{query}」に該当するPersonaは見つかりませんでした。"
    name = data["name"]
    header = name
    if query and query.strip() and query.strip() != name:
        header = f"{name}（{query.strip()}）"
    lines = [header, f"ID: {data['id']}", f"役割: {data['role']}"]
    if data.get("description"):
        lines.append(data["description"])
    return "\n".join(lines)


def format_chatter(data: dict[str, Any]) -> str:
    if data.get("status") == "PERSONA_NOT_FOUND":
        return (
            f"Persona解決に失敗しました: persona_a={data.get('persona_a')}, "
            f"persona_b={data.get('persona_b')}"
        )
    lines = []
    for turn in data.get("turns", []):
        lines.append(f"{turn['speaker_display_name']}:")
        lines.append(turn["text"])
        lines.append("")
    lines.append(f"[status: {data.get('status')}]")
    return "\n".join(lines).strip()
