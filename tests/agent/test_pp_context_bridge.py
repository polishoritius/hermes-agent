"""PP Context Bridge tests -- PP-CONTEXT-BRIDGE-001.

list_personas()/get_persona() read the real Project Poiesis repo
files (READ ONLY, no network). start_chatter() and the "/pp chatter"
route are tested with persona_chatter.run_chatter mocked out, so this
suite makes zero Ollama/network calls.
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from agent import light_chat, pp_context_bridge

PP_REPO = r"C:\GitHub\Project-Poiesis-Work"

POLICE_ID = "PP-PER-007"
VERBA_ID = "PP-PER-009"


class ResolvePersonaIdTests(unittest.TestCase):
    def test_resolves_canonical_id(self):
        self.assertEqual(pp_context_bridge.resolve_persona_id("PP-PER-007", PP_REPO), POLICE_ID)

    def test_resolves_full_display_name(self):
        self.assertEqual(pp_context_bridge.resolve_persona_id("Police Horitius", PP_REPO), POLICE_ID)

    def test_resolves_personas_yaml_alias(self):
        self.assertEqual(pp_context_bridge.resolve_persona_id("police", PP_REPO), POLICE_ID)

    def test_resolves_first_word_of_display_name(self):
        self.assertEqual(pp_context_bridge.resolve_persona_id("Police", PP_REPO), POLICE_ID)
        self.assertEqual(pp_context_bridge.resolve_persona_id("Verba", PP_REPO), VERBA_ID)

    def test_resolves_verba_clarion_full_name(self):
        self.assertEqual(pp_context_bridge.resolve_persona_id("Verba Clarion", PP_REPO), VERBA_ID)

    def test_unknown_query_fails_closed(self):
        self.assertIsNone(pp_context_bridge.resolve_persona_id("unknownxyz", PP_REPO))

    def test_bell_resolves_to_verba_clarion(self):
        """"Bell"/"bell"/"ベル" were added as canonical aliases for
        Verba Clarion (PP-PER-009-ALIAS-001): docs/03_pps/
        Persona_Ontology_vNext.md's Persona Canonical Definition table
        (愛称 column) and config/virtual-office/personas.yaml's
        `aliases:` list for `id: verba`. No Bridge-side hardcoding was
        added -- this resolves purely from the updated PP data via the
        same resolve_persona_id() logic as every other alias."""
        for query in ("Bell", "bell", "ベル"):
            with self.subTest(query=query):
                self.assertEqual(pp_context_bridge.resolve_persona_id(query, PP_REPO), VERBA_ID)


class ListPersonasTests(unittest.TestCase):
    def test_returns_all_nine_personas(self):
        result = pp_context_bridge.list_personas(PP_REPO)
        self.assertEqual(len(result["personas"]), 9)
        ids = {p["id"] for p in result["personas"]}
        self.assertIn(POLICE_ID, ids)
        self.assertIn(VERBA_ID, ids)
        # Only Persona-type nodes -- Gate/Process/Capability/Route excluded.
        for p in result["personas"]:
            self.assertTrue(p["id"].startswith("PP-PER-"))

    def test_each_entry_has_id_name_role_only(self):
        result = pp_context_bridge.list_personas(PP_REPO)
        for p in result["personas"]:
            self.assertEqual(set(p.keys()), {"id", "name", "role"})


class GetPersonaTests(unittest.TestCase):
    def test_get_persona_by_alias(self):
        result = pp_context_bridge.get_persona("Police", PP_REPO)
        self.assertEqual(result["id"], POLICE_ID)
        self.assertEqual(result["name"], "Police Horitius")
        self.assertEqual(result["role"], "Fact / Observation")
        self.assertTrue(result["description"])

    def test_get_persona_verba_by_alias(self):
        result = pp_context_bridge.get_persona("Verba", PP_REPO)
        self.assertEqual(result["id"], VERBA_ID)
        self.assertEqual(result["name"], "Verba Clarion")

    def test_get_persona_unknown_is_explicit_not_found(self):
        result = pp_context_bridge.get_persona("unknownxyz", PP_REPO)
        self.assertEqual(result, {"error": "PERSONA_NOT_FOUND", "query": "unknownxyz"})

    def test_get_persona_bell_resolves_to_verba_clarion(self):
        result = pp_context_bridge.get_persona("Bell", PP_REPO)
        self.assertEqual(result["id"], VERBA_ID)
        self.assertEqual(result["name"], "Verba Clarion")

    def test_get_persona_jp_bell_resolves_to_verba_clarion(self):
        result = pp_context_bridge.get_persona("ベル", PP_REPO)
        self.assertEqual(result["id"], VERBA_ID)
        self.assertEqual(result["name"], "Verba Clarion")


class StartChatterTests(unittest.TestCase):
    def test_start_chatter_resolves_aliases_and_calls_existing_runtime(self):
        fake_result = {
            "final_status": "COMPLETED_MAX_TURNS",
            "turns": [{"speaker_canonical_id": POLICE_ID, "text": "..."}],
        }
        with patch("persona_chatter.run_chatter", return_value=fake_result) as mock_run:
            # persona_chatter isn't imported until start_chatter() runs,
            # so make sure it's importable from the PP repo's scripts dir
            # first (mirrors what start_chatter() itself does).
            sys.path.insert(0, str(__import__("pathlib").Path(PP_REPO) / "scripts"))
            result = pp_context_bridge.start_chatter("Police", "Verba", max_turns=2, pp_repo_path=PP_REPO)

        mock_run.assert_called_once_with(
            persona_a_canonical_id=POLICE_ID, persona_b_canonical_id=VERBA_ID, max_turns=2
        )
        self.assertEqual(result["status"], "COMPLETED_MAX_TURNS")
        self.assertEqual(result["persona_a"], POLICE_ID)
        self.assertEqual(result["persona_b"], VERBA_ID)
        self.assertEqual(result["external_api_calls"], 0)

    def test_start_chatter_unknown_persona_never_calls_runtime(self):
        with patch("persona_chatter.run_chatter") as mock_run:
            result = pp_context_bridge.start_chatter("unknownxyz", "Verba", pp_repo_path=PP_REPO)
        mock_run.assert_not_called()
        self.assertEqual(result["status"], "PERSONA_NOT_FOUND")

    def test_start_chatter_police_bell_now_resolves_and_calls_runtime(self):
        """Regression for PP-PER-009-ALIAS-001: "Bell" now resolves, so
        /pp chatter Police Bell must reach the existing CHATTER runtime
        exactly like /pp chatter Police Verba does."""
        fake_result = {"final_status": "COMPLETED_MAX_TURNS", "turns": []}
        with patch("persona_chatter.run_chatter", return_value=fake_result) as mock_run:
            result = pp_context_bridge.start_chatter("Police", "Bell", max_turns=2, pp_repo_path=PP_REPO)
        mock_run.assert_called_once_with(
            persona_a_canonical_id=POLICE_ID, persona_b_canonical_id=VERBA_ID, max_turns=2
        )
        self.assertEqual(result["status"], "COMPLETED_MAX_TURNS")
        self.assertEqual(result["persona_b"], VERBA_ID)
        self.assertEqual(result["turns"], [])
        self.assertEqual(result["external_api_calls"], 0)


class LightChatCommandRoutingTests(unittest.TestCase):
    """Confirms the "/pp ..." prefix routing in agent/light_chat.py:
    explicit commands only, ordinary text never reaches the Bridge."""

    def test_pp_personas_routes_to_bridge(self):
        with patch("agent.pp_context_bridge.list_personas", return_value={"personas": []}) as mock_list:
            result = light_chat.handle_light_input("/pp personas")
        mock_list.assert_called_once()
        self.assertEqual(result["kind"], "pp_bridge")
        self.assertEqual(result["command"], "personas")
        self.assertEqual(result["external_api_calls"], 0)

    def test_pp_persona_routes_to_bridge_with_query(self):
        with patch("agent.pp_context_bridge.get_persona", return_value={"id": POLICE_ID}) as mock_get:
            result = light_chat.handle_light_input("/pp persona Police")
        mock_get.assert_called_once_with("Police")
        self.assertEqual(result["kind"], "pp_bridge")
        self.assertEqual(result["command"], "persona")

    def test_pp_chatter_routes_to_bridge_with_both_names(self):
        with patch("agent.pp_context_bridge.start_chatter", return_value={"status": "COMPLETED_MAX_TURNS"}) as mock_chatter:
            result = light_chat.handle_light_input("/pp chatter Police Bell")
        mock_chatter.assert_called_once_with("Police", "Bell")
        self.assertEqual(result["kind"], "pp_bridge")
        self.assertEqual(result["command"], "chatter")

    def test_pp_chatter_wrong_arg_count_is_explicit_usage_error(self):
        with patch("agent.pp_context_bridge.start_chatter") as mock_chatter:
            result = light_chat.handle_light_input("/pp chatter Police")
        mock_chatter.assert_not_called()
        self.assertIn("error", result["data"])

    def test_ordinary_greeting_never_touches_bridge(self):
        with patch("agent.pp_context_bridge.list_personas") as mock_list, patch(
            "agent.pp_context_bridge.get_persona"
        ) as mock_get, patch("agent.pp_context_bridge.start_chatter") as mock_chatter, patch(
            "agent.light_chat.run_light_chat",
            return_value={"text": "こんにちは!", "tool_search_calls": 0, "tool_calls": 0, "external_api_calls": 0},
        ) as mock_run:
            result = light_chat.handle_light_input("こんにちは")

        mock_list.assert_not_called()
        mock_get.assert_not_called()
        mock_chatter.assert_not_called()
        mock_run.assert_called_once()
        self.assertEqual(result["kind"], "light_chat")

    def test_natural_language_mentioning_pp_is_not_routed(self):
        """Non-goal check: text that merely mentions "pp" or a Persona
        name in prose must NOT be classified as a Bridge command --
        only an exact "/pp ..." prefix routes there."""
        with patch("agent.pp_context_bridge.get_persona") as mock_get, patch(
            "agent.light_chat.run_light_chat",
            return_value={"text": "...", "tool_search_calls": 0, "tool_calls": 0, "external_api_calls": 0},
        ) as mock_run:
            result = light_chat.handle_light_input("Police呼んで")
        mock_get.assert_not_called()
        mock_run.assert_called_once()
        self.assertEqual(result["kind"], "light_chat")
