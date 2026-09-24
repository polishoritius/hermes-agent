"""Fast Router end-to-end integration tests via handle_light_input() --
HERMES-FAST-ROUTER-001 PHASE 6 (14 cases, numbered to match the work
order).

Cases 2/3/4/5/6/9/10/11 exercise the real PP repo files (list/lookup
are pure local file reads -- no LLM, no network). Cases 1/7/8/12 mock
the Ollama-touching calls (run_light_chat / persona_chatter.run_chatter)
so this suite makes zero real network calls.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agent import light_chat

PP_REPO = r"C:\GitHub\Project-Poiesis-Work"


class Case1LightChatDefault(unittest.TestCase):
    """1. "こんにちは" -> route=LIGHT_CHAT, Bridge unused, tools=0,
    tool_search=0, external=0."""

    def test_hello_routes_to_light_chat_bridge_unused(self):
        fake = {
            "text": "こんにちは!", "model": "llama3.2:3b", "provider": "custom",
            "base_url": "http://localhost:11434/v1", "prompt_tokens": 200, "output_tokens": 10,
            "latency_s": 1.0, "tool_search_calls": 0, "tool_calls": 0, "external_api_calls": 0,
        }
        with patch("agent.pp_context_bridge.list_personas") as mock_list, patch(
            "agent.pp_context_bridge.get_persona"
        ) as mock_get, patch("agent.pp_context_bridge.start_chatter") as mock_chatter, patch(
            "agent.light_chat.run_light_chat", return_value=fake
        ) as mock_run:
            result = light_chat.handle_light_input("こんにちは", pp_repo_path=PP_REPO)

        mock_list.assert_not_called()
        mock_get.assert_not_called()
        mock_chatter.assert_not_called()
        mock_run.assert_called_once()
        self.assertEqual(result["kind"], "light_chat")
        self.assertEqual(result["tool_search_calls"], 0)
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result["external_api_calls"], 0)


class Case2And3PersonaList(unittest.TestCase):
    """2. "今いる人格は？" / 3. "人格一覧を見せて" -> PERSONA_LIST, count=9,
    LLM call=0 (real file read, no client_factory/network involved)."""

    def test_ima_iru_jinkaku(self):
        result = light_chat.handle_light_input("今いる人格は？", pp_repo_path=PP_REPO)
        self.assertEqual(result["kind"], "pp_bridge_natural")
        self.assertEqual(result["command"], "personas")
        self.assertEqual(len(result["data"]["personas"]), 9)

    def test_jinkaku_ichiran_wo_misete(self):
        result = light_chat.handle_light_input("人格一覧を見せて", pp_repo_path=PP_REPO)
        self.assertEqual(result["kind"], "pp_bridge_natural")
        self.assertEqual(len(result["data"]["personas"]), 9)


class Case4And5And6PersonaLookup(unittest.TestCase):
    """4. "Bellは誰？" -> PP-PER-009. 5. "ベルって誰？" -> PP-PER-009.
    6. "Policeについて教えて" -> PP-PER-007. All LLM call=0."""

    def test_bell_wa_dare(self):
        result = light_chat.handle_light_input("Bellは誰？", pp_repo_path=PP_REPO)
        self.assertEqual(result["kind"], "pp_bridge_natural")
        self.assertEqual(result["data"]["id"], "PP-PER-009")
        self.assertEqual(result["data"]["name"], "Verba Clarion")

    def test_jp_bell_tte_dare(self):
        result = light_chat.handle_light_input("ベルって誰？", pp_repo_path=PP_REPO)
        self.assertEqual(result["data"]["id"], "PP-PER-009")

    def test_police_ni_tsuite_oshiete(self):
        result = light_chat.handle_light_input("Policeについて教えて", pp_repo_path=PP_REPO)
        self.assertEqual(result["data"]["id"], "PP-PER-007")
        self.assertEqual(result["data"]["name"], "Police Horitius")


class Case7And8Chatter(unittest.TestCase):
    """7. "PoliceとBellで話して" -> CHATTER dispatch, max_turns=2.
    8. "Policeとベるで会話して" -> CHATTER dispatch. persona_chatter's
    real run_chatter is mocked so this makes zero Ollama calls."""

    def test_police_to_bell_de_hanashite(self):
        fake = {"final_status": "COMPLETED_MAX_TURNS", "turns": []}
        with patch("persona_chatter.run_chatter", return_value=fake) as mock_run:
            import sys
            sys.path.insert(0, str(__import__("pathlib").Path(PP_REPO) / "scripts"))
            result = light_chat.handle_light_input("PoliceとBellで話して", pp_repo_path=PP_REPO)

        mock_run.assert_called_once_with(
            persona_a_canonical_id="PP-PER-007", persona_b_canonical_id="PP-PER-009", max_turns=2
        )
        self.assertEqual(result["kind"], "pp_bridge_natural")
        self.assertEqual(result["command"], "chatter")

    def test_police_to_jp_bell_de_kaiwa_shite(self):
        fake = {"final_status": "COMPLETED_MAX_TURNS", "turns": []}
        with patch("persona_chatter.run_chatter", return_value=fake) as mock_run:
            result = light_chat.handle_light_input("Policeとベルで会話して", pp_repo_path=PP_REPO)
        mock_run.assert_called_once_with(
            persona_a_canonical_id="PP-PER-007", persona_b_canonical_id="PP-PER-009", max_turns=2
        )
        self.assertEqual(result["kind"], "pp_bridge_natural")


class Case9And10NoMisfire(unittest.TestCase):
    """9. "明日の昼飯どうしよう？" -> LIGHT_CHAT, no Persona router misfire.
    10. "警察について教えて" -> LIGHT_CHAT (not confused with Police
    Horitius)."""

    def test_lunch_question(self):
        with patch("agent.light_chat.run_light_chat", return_value={"text": "..."}) as mock_run:
            result = light_chat.handle_light_input("明日の昼飯どうしよう？", pp_repo_path=PP_REPO)
        mock_run.assert_called_once()
        self.assertEqual(result["kind"], "light_chat")

    def test_keisatsu_not_confused_with_police(self):
        with patch("agent.light_chat.run_light_chat", return_value={"text": "..."}) as mock_run, patch(
            "agent.pp_context_bridge.get_persona"
        ) as mock_get:
            result = light_chat.handle_light_input("警察について教えて", pp_repo_path=PP_REPO)
        mock_get.assert_not_called()
        mock_run.assert_called_once()
        self.assertEqual(result["kind"], "light_chat")


class Case11UnknownPersonaNoGuess(unittest.TestCase):
    """11. Unknown persona -> never guessed/mapped."""

    def test_unknown_persona_lookup_is_explicit_not_found(self):
        with patch("agent.light_chat.run_light_chat", return_value={"text": "..."}) as mock_run, patch(
            "agent.pp_context_bridge.get_persona"
        ) as mock_get:
            result = light_chat.handle_light_input("Xyzzyについて教えて", pp_repo_path=PP_REPO)
        # "Xyzzyについて教えて" structurally matches the lookup pattern,
        # but "Xyzzy" resolves to nothing in PP data -- resolve_persona_id()
        # inside fast_router.route() already filtered it out, so
        # get_persona() (the Bridge) is never even called, and this
        # falls through to ordinary Light Chat rather than guessing.
        mock_get.assert_not_called()
        mock_run.assert_called_once()
        self.assertEqual(result["kind"], "light_chat")


class Case12OllamaUnavailable(unittest.TestCase):
    """12. Ollama unavailable -> explicit error, external fallback=0
    (no fallback code path exists in this module at all)."""

    def test_unavailable_raises_explicit_error_not_swallowed(self):
        with patch(
            "agent.light_chat.run_light_chat",
            side_effect=light_chat.LightChatUnavailableError("Light Chat: local endpoint unavailable (...)"),
        ):
            with self.assertRaises(light_chat.LightChatUnavailableError) as ctx:
                light_chat.handle_light_input("こんにちは", pp_repo_path=PP_REPO)
        self.assertNotIn("openrouter", str(ctx.exception).lower())
        self.assertNotIn("anthropic", str(ctx.exception).lower())


class Case13ExplicitCommandsStillPass(unittest.TestCase):
    """13. Existing explicit /pp commands unaffected by the Fast
    Router addition (priority order: /pp prefix checked first)."""

    def test_pp_personas_unaffected(self):
        with patch("agent.pp_context_bridge.list_personas", return_value={"personas": []}) as mock_list:
            result = light_chat.handle_light_input("/pp personas", pp_repo_path=PP_REPO)
        mock_list.assert_called_once()
        self.assertEqual(result["kind"], "pp_bridge")

    def test_pp_persona_police_unaffected(self):
        with patch("agent.pp_context_bridge.get_persona", return_value={"id": "PP-PER-007"}) as mock_get:
            result = light_chat.handle_light_input("/pp persona Police", pp_repo_path=PP_REPO)
        mock_get.assert_called_once_with("Police")
        self.assertEqual(result["kind"], "pp_bridge")

    def test_pp_chatter_police_bell_unaffected(self):
        with patch("agent.pp_context_bridge.start_chatter", return_value={"status": "COMPLETED_MAX_TURNS"}) as mock_chatter:
            result = light_chat.handle_light_input("/pp chatter Police Bell", pp_repo_path=PP_REPO)
        mock_chatter.assert_called_once_with("Police", "Bell")
        self.assertEqual(result["kind"], "pp_bridge")


# Case 14 (Normal Full Agent: no structural impact) is covered by
# tests/agent/test_light_chat.py::CliLightFlagStructuralTests, which
# asserts the "if light:" branch in cli.py's main() still returns
# before HermesCLI() construction -- unaffected by this task's changes
# (only agent/light_chat.py's internals and two new modules were
# touched; cli.py's --light branch dispatch point itself is the same
# one line calling handle_light_input()).
