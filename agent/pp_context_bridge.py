"""PP Context Bridge -- minimal read + explicit CHATTER dispatch
(PP-CONTEXT-BRIDGE-001).

Scope, enforced by this module's own structure:

  READ:
    - list_personas() / get_persona() read Project Poiesis's existing
      canonical/runtime sources directly (persona-ontology-registry.ts
      as the machine-readable mirror of
      docs/03_pps/Persona_Ontology_vNext.md, and
      config/virtual-office/personas.yaml for the runtime
      system_prompt/aliases fields) -- via the SAME reader functions
      scripts/persona_chatter_prompt.py already uses for CHATTER, so
      no second Persona registry is created here.

  ACTION:
    - start_chatter() only ever calls scripts/persona_chatter.py's
      existing run_chatter() entry point. It does not reimplement any
      chat/turn logic.

Never imports scripts/operaos_growth_loop.py, scripts/hermes_write_boundary.py,
scripts/council_record_validator.py, scripts/verify_execution_result.py,
or any Discord/Git-write module -- no Growth mutation, no VERIFY, no
Council, no Git write, no Discord post, no external API, no arbitrary
shell execution. Persona *definitions* are never edited by this
module; it only reads them and forwards to the existing CHATTER
runtime.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

DEFAULT_PP_REPO_PATH = r"C:\GitHub\Project-Poiesis-Work"

_ALIAS_LINE_PATTERN = re.compile(r"aliases:\s*\[([^\]]*)\]")


def _resolve_pp_repo(pp_repo_path: Optional[str]) -> Path:
    if pp_repo_path:
        return Path(pp_repo_path)
    env_path = os.environ.get("PP_REPO_PATH")
    if env_path:
        return Path(env_path)
    return Path(DEFAULT_PP_REPO_PATH)


def _import_persona_chatter_prompt(pp_repo: Path):
    """Imports the EXISTING CHATTER-002 reader module rather than
    re-parsing persona-ontology-registry.ts / personas.yaml a second
    time. Returns the imported module."""
    scripts_dir = str(pp_repo / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import persona_chatter_prompt as _pcp  # noqa: E402

    return _pcp


def _read_personas_yaml_aliases(pp_repo: Path, canonical_id_pattern: re.Pattern) -> dict[str, list[str]]:
    """READ ONLY. Narrow, single-line extraction of personas.yaml's
    `aliases: [...]` field per canonical_id -- mirrors
    persona_chatter_prompt.py's own narrow single-line-oriented
    convention (same file, same line-based read). Fails closed to {}
    on any read problem, never raises."""
    path = pp_repo / "config" / "virtual-office" / "personas.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    result: dict[str, list[str]] = {}
    for line in text.splitlines():
        cid_match = canonical_id_pattern.search(line)
        if not cid_match:
            continue
        alias_match = _ALIAS_LINE_PATTERN.search(line)
        if not alias_match:
            continue
        aliases = [a.strip() for a in alias_match.group(1).split(",") if a.strip()]
        result[cid_match.group(1)] = aliases
    return result


def _normalize(s: str) -> str:
    return s.strip().lower()


def resolve_persona_id(query: str, pp_repo_path: Optional[str] = None) -> Optional[str]:
    """Resolves a free-text Persona reference (canonical id, full
    display name, or an alias/nickname already present in
    personas.yaml) to a canonical_id. Never invents an alias that
    isn't already recorded in personas.yaml or the display name in
    persona-ontology-registry.ts -- an unmatched query returns None
    (fail closed), it is never guessed."""
    if not query or not query.strip():
        return None
    pp_repo = _resolve_pp_repo(pp_repo_path)
    pcp = _import_persona_chatter_prompt(pp_repo)
    nodes = pcp._read_ontology_persona_nodes()  # {canonical_id: {canonical_id, display_name, role}}
    aliases_map = _read_personas_yaml_aliases(pp_repo, pcp._CANONICAL_ID_PATTERN)

    q = _normalize(query)

    # 1. Exact canonical_id (e.g. "PP-PER-007").
    for cid in nodes:
        if _normalize(cid) == q:
            return cid

    # 2. Exact full display name (e.g. "Police Horitius").
    for cid, node in nodes.items():
        if _normalize(node["display_name"]) == q:
            return cid

    # 3. An alias/nickname already recorded in personas.yaml (e.g.
    #    "police", "ポリス", "ホリティウス").
    for cid, aliases in aliases_map.items():
        if q in (_normalize(a) for a in aliases):
            return cid

    # 4. First word of the display name (e.g. "Police" -> "Police Horitius",
    #    "Verba" -> "Verba Clarion"). Still grounded in the canonical
    #    display name, not a fabricated nickname table.
    for cid, node in nodes.items():
        first_word = _normalize(node["display_name"]).split(" ")[0]
        if first_word == q:
            return cid

    return None


def list_personas(pp_repo_path: Optional[str] = None) -> dict[str, Any]:
    """READ ONLY. Returns every Persona-type node from
    persona-ontology-registry.ts (via persona_chatter_prompt.py's own
    reader -- Gate/Process/Capability/Route nodes are excluded, same
    filter CHATTER already applies)."""
    pp_repo = _resolve_pp_repo(pp_repo_path)
    pcp = _import_persona_chatter_prompt(pp_repo)
    nodes = pcp._read_ontology_persona_nodes()
    personas = [
        {"id": cid, "name": node["display_name"], "role": node["role"]}
        for cid, node in sorted(nodes.items())
    ]
    return {"personas": personas}


def get_persona(persona_id_or_alias: str, pp_repo_path: Optional[str] = None) -> dict[str, Any]:
    """READ ONLY. Resolves an id/name/alias then returns
    {id, name, role, description}. `description` is personas.yaml's
    existing (non-legacy) `system_prompt` field for that canonical_id
    -- read via persona_chatter_prompt.py's own reader, not a new
    text. Fails closed to an explicit not-found shape; never guesses a
    Persona's identity."""
    pp_repo = _resolve_pp_repo(pp_repo_path)
    cid = resolve_persona_id(persona_id_or_alias, str(pp_repo))
    if cid is None:
        return {"error": "PERSONA_NOT_FOUND", "query": persona_id_or_alias}
    pcp = _import_persona_chatter_prompt(pp_repo)
    nodes = pcp._read_ontology_persona_nodes()
    node = nodes[cid]
    description = pcp._read_runtime_system_prompt_core(cid) or ""
    return {
        "id": cid,
        "name": node["display_name"],
        "role": node["role"],
        "description": description,
    }


def start_chatter(
    persona_a: str,
    persona_b: str,
    max_turns: int = 2,
    pp_repo_path: Optional[str] = None,
) -> dict[str, Any]:
    """ACTION (the only action this module performs): resolves both
    Persona references then dispatches to the EXISTING
    scripts/persona_chatter.py::run_chatter() entry point -- no chat
    logic is reimplemented here. Growth/Council/VERIFY/Discord/write
    modules are never imported (matches run_chatter()'s own
    guarantees, see scripts/persona_chatter.py's module docstring)."""
    pp_repo = _resolve_pp_repo(pp_repo_path)
    cid_a = resolve_persona_id(persona_a, str(pp_repo))
    cid_b = resolve_persona_id(persona_b, str(pp_repo))
    if cid_a is None or cid_b is None:
        return {
            "status": "PERSONA_NOT_FOUND",
            "persona_a": persona_a if cid_a is None else cid_a,
            "persona_b": persona_b if cid_b is None else cid_b,
            "turns": [],
            "external_api_calls": 0,
        }

    scripts_dir = str(pp_repo / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import persona_chatter as _chatter  # noqa: E402 -- the existing CHATTER runtime

    result = _chatter.run_chatter(
        persona_a_canonical_id=cid_a,
        persona_b_canonical_id=cid_b,
        max_turns=max_turns,
    )
    return {
        "status": result["final_status"],
        "persona_a": cid_a,
        "persona_b": cid_b,
        "turns": result["turns"],
        "external_api_calls": 0,
    }
