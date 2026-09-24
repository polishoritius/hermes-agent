"""Light Chat -- minimal local-Ollama fast path (HERMES-LIGHT-CHAT-001).

Bypasses the normal agent bootstrap entirely: no HermesCLI(), no
_init_agent(), no MCP discovery, no model_tools.get_tool_definitions(),
no tools/tool_search.py bridge, no skills, no kanban, no
agent/conversation_loop.py. Talks to a local OpenAI-compatible endpoint
directly for short, tool-free conversational turns.

Never falls back to another provider. A connection failure raises
LightChatUnavailableError -- callers must surface this as an explicit
error, not retry against openrouter/anthropic/etc.

History is kept in a small per-session JSON file under
HERMES_HOME/light_chat_sessions/, capped at MAX_HISTORY_MESSAGES. This
is deliberately separate from hermes_state.SessionDB (state.db): that
store's schema/alternation-repair logic belongs to the normal Agent
route, and a light-only writer risks breaking the normal route's later
resume of the same session (see HERMES-LIGHT-CHAT-001 audit, landmine
#7) -- so Light Chat never touches state.db.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_PROVIDER = "custom"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_SESSION_ID = "default"

# 4 user/assistant turns == 8 messages.
MAX_HISTORY_MESSAGES = 8

LIGHT_CHAT_BOUNDARY_PROMPT = (
    "あなたはHermesのLight Chatモードです。ツールは一切使用できません。"
    "簡潔で自然な日本語(または相手の言語)で、雑談や短い質問に会話のみで応答してください。"
    "知らないことや確認できないことは断定せず、正直に分からないと伝えてください。"
    "特にProject Poiesis (PP) の正式な情報については、このモードでは確認できないため、"
    "断定的な回答をしないでください。"
)


class LightChatUnavailableError(RuntimeError):
    """The local endpoint could not be reached or returned an error.

    Raising this is the only failure behavior Light Chat has -- there
    is no fallback_model / openrouter / anthropic wiring in this
    module. Callers should print this and exit non-zero, never retry
    against a different provider.
    """


def _resolve_hermes_home(hermes_home: Optional[Path]) -> Path:
    if hermes_home is not None:
        return Path(hermes_home)
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".hermes"


def _history_path(hermes_home: Path, session_id: str) -> Path:
    safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")) or "default"
    directory = hermes_home / "light_chat_sessions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{safe_id}.json"


def _load_history(hermes_home: Path, session_id: str) -> list[dict[str, str]]:
    path = _history_path(hermes_home, session_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return data[-MAX_HISTORY_MESSAGES:]


def _save_history(hermes_home: Path, session_id: str, history: list[dict[str, str]]) -> None:
    path = _history_path(hermes_home, session_id)
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    path.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_system_prompt(hermes_home: Path) -> str:
    """Minimal identity: SOUL.md verbatim (if present and non-empty) plus
    the fixed Light Chat boundary text. No AGENTS.md, no skills
    catalog, no tool metadata, no execution policy -- see
    HERMES-LIGHT-CHAT-001 STEP 4."""
    soul_path = hermes_home / "SOUL.md"
    try:
        soul_text = soul_path.read_text(encoding="utf-8").strip()
    except OSError:
        soul_text = ""
    if soul_text:
        return f"{soul_text}\n\n{LIGHT_CHAT_BOUNDARY_PROMPT}"
    return LIGHT_CHAT_BOUNDARY_PROMPT


def run_light_chat(
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    hermes_home: Optional[Path] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> dict[str, Any]:
    """Runs one Light Chat turn. Never builds a ``tools=`` array, never
    imports tools/tool_search.py or model_tools.py, never touches
    agent/conversation_loop.py or agent/agent_init.py's init_agent().

    Returns a dict with the response text and call metadata
    (tool_search_calls / tool_calls / external_api_calls are always 0
    by construction -- this path cannot invoke them).
    """
    if not query or not query.strip():
        raise ValueError("Light Chat: query must be non-empty")

    home = _resolve_hermes_home(hermes_home)
    history = _load_history(home, session_id)
    system_prompt = _build_system_prompt(home)
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": query}]
    )

    if client_factory is None:
        import openai

        def client_factory():
            return openai.OpenAI(base_url=base_url, api_key=api_key or "no-key", max_retries=0)

    try:
        client = client_factory()
        t0 = time.time()
        response = client.chat.completions.create(model=model, messages=messages)
        latency_s = time.time() - t0
    except Exception as exc:  # noqa: BLE001 -- converted to an explicit
        # error type; this except block never attempts another
        # provider/base_url.
        raise LightChatUnavailableError(
            f"Light Chat: local endpoint unavailable (provider={provider} base_url={base_url} model={model}): {exc}"
        ) from exc

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None

    new_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": text},
    ]
    _save_history(home, session_id, new_history)

    return {
        "text": text,
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "latency_s": latency_s,
        "tool_search_calls": 0,
        "tool_calls": 0,
        "external_api_calls": 0,
    }


# ── PP Context Bridge command routing (PP-CONTEXT-BRIDGE-001) ──────────
#
# Explicit-command dispatch ONLY -- a hard "/pp ..." prefix check, never
# natural-language intent classification (explicit non-goal). Anything
# that doesn't start with "/pp " goes to run_light_chat() exactly as
# before -- ordinary greetings/questions never reach the Bridge.

_PP_PERSONAS_PREFIX = "/pp personas"
_PP_PERSONA_PREFIX = "/pp persona "
_PP_CHATTER_PREFIX = "/pp chatter "


def handle_light_input(text: str, **run_light_chat_kwargs: Any) -> dict[str, Any]:
    """Single entry point for the Light Chat CLI path: routes an
    explicit "/pp ..." command to agent/pp_context_bridge.py (read +
    explicit CHATTER dispatch only), otherwise forwards unchanged to
    run_light_chat(). The returned dict always carries "kind" so the
    caller knows how to render it ("pp_bridge" vs "light_chat")."""
    stripped = (text or "").strip()

    if stripped == _PP_PERSONAS_PREFIX or stripped.startswith(_PP_PERSONAS_PREFIX + " "):
        from agent.pp_context_bridge import list_personas

        return {"kind": "pp_bridge", "command": "personas", "data": list_personas(), "external_api_calls": 0}

    if stripped.startswith(_PP_PERSONA_PREFIX):
        query = stripped[len(_PP_PERSONA_PREFIX):].strip()
        from agent.pp_context_bridge import get_persona

        return {"kind": "pp_bridge", "command": "persona", "data": get_persona(query), "external_api_calls": 0}

    if stripped.startswith(_PP_CHATTER_PREFIX):
        args = stripped[len(_PP_CHATTER_PREFIX):].strip().split()
        if len(args) != 2:
            return {
                "kind": "pp_bridge",
                "command": "chatter",
                "data": {"error": "USAGE: /pp chatter <persona_a> <persona_b>"},
                "external_api_calls": 0,
            }
        from agent.pp_context_bridge import start_chatter

        result = start_chatter(args[0], args[1])
        return {"kind": "pp_bridge", "command": "chatter", "data": result, "external_api_calls": 0}

    # Not a /pp command -- ordinary Light Chat, Bridge is never touched.
    result = run_light_chat(text, **run_light_chat_kwargs)
    result["kind"] = "light_chat"
    return result
