"""Fast Router unit tests -- HERMES-FAST-ROUTER-001 PHASE 2.

Uses the real PP repo (read-only) for resolve_persona_id() lookups so
these tests exercise the actual current canonical/runtime data, same
convention as tests/agent/test_pp_context_bridge.py. No LLM, no
network.
"""
from __future__ import annotations

import unittest

from agent import fast_router

PP_REPO = r"C:\GitHub\Project-Poiesis-Work"


class ChatterRoutingTests(unittest.TestCase):
    def test_police_and_bell_de_hanashite(self):
        d = fast_router.route("PoliceとBellで話して", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_CHATTER)
        self.assertEqual(d["persona_a"], "Police")
        self.assertEqual(d["persona_b"], "Bell")

    def test_police_and_verba_wo_hanasasete(self):
        d = fast_router.route("PoliceとVerbaを話させて", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_CHATTER)

    def test_police_and_jp_bell_de_kaiwa_shite(self):
        d = fast_router.route("Policeとベルで会話して", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_CHATTER)
        self.assertEqual(d["persona_b"], "ベル")

    def test_unrelated_names_fall_through_to_light_chat(self):
        d = fast_router.route("田中と鈴木で話して", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_LIGHT_CHAT)


class PersonaLookupRoutingTests(unittest.TestCase):
    def test_bell_wa_dare(self):
        d = fast_router.route("Bellは誰？", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LOOKUP)
        self.assertEqual(d["query"], "Bell")

    def test_bell_tte_dare(self):
        d = fast_router.route("ベルって誰？", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LOOKUP)
        self.assertEqual(d["query"], "ベル")

    def test_police_ni_tsuite_oshiete(self):
        d = fast_router.route("Policeについて教えて", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LOOKUP)
        self.assertEqual(d["query"], "Police")

    def test_verba_ni_tsuite(self):
        d = fast_router.route("Verbaについて", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LOOKUP)

    def test_police_no_yakuwari_wa(self):
        d = fast_router.route("Policeの役割は？", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LOOKUP)

    def test_keisatsu_ni_tsuite_oshiete_is_not_misrouted(self):
        """"警察" (the generic Japanese word for "police") structurally
        matches the "...について教えて" pattern, but it is not a
        registered alias/name for any Persona -- resolve_persona_id()
        must return None for it, so this falls through to LIGHT_CHAT."""
        d = fast_router.route("警察について教えて", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_LIGHT_CHAT)

    def test_unrelated_topic_ni_tsuite_oshiete_is_not_misrouted(self):
        d = fast_router.route("野球について教えて", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_LIGHT_CHAT)


class PersonaListRoutingTests(unittest.TestCase):
    def test_ima_iru_jinkaku_wa(self):
        d = fast_router.route("今いる人格は？", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LIST)

    def test_jinkaku_ichiran(self):
        d = fast_router.route("人格一覧", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LIST)

    def test_jinkaku_wo_misete(self):
        d = fast_router.route("人格を見せて", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LIST)

    def test_persona_ichiran_english(self):
        d = fast_router.route("Persona一覧", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LIST)

    def test_perusona_ichiran(self):
        d = fast_router.route("ペルソナ一覧", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LIST)

    def test_dare_ga_iru_exact_match(self):
        d = fast_router.route("誰がいる？", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LIST)

    def test_dare_ga_iru_no_exact_match(self):
        d = fast_router.route("誰がいるの？", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_PERSONA_LIST)

    def test_dare_ga_iru_embedded_in_longer_sentence_not_routed(self):
        """"誰がいる" is only routed as an EXACT whole-message match
        (see module docstring) -- embedded in a longer, unrelated
        sentence it must not misfire."""
        d = fast_router.route("今日のオフィスに誰がいるのか気になる", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_LIGHT_CHAT)


class LightChatDefaultRoutingTests(unittest.TestCase):
    def test_greeting(self):
        d = fast_router.route("こんにちは", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_LIGHT_CHAT)

    def test_general_question(self):
        d = fast_router.route("今日は何をしようか", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_LIGHT_CHAT)

    def test_lunch_question_not_misrouted(self):
        d = fast_router.route("明日の昼飯どうしよう？", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_LIGHT_CHAT)

    def test_empty_string(self):
        d = fast_router.route("", PP_REPO)
        self.assertEqual(d["route"], fast_router.ROUTE_LIGHT_CHAT)


class NoAliasHardcodingTests(unittest.TestCase):
    def test_module_source_has_no_persona_ids_or_display_names(self):
        """Structural check: agent/fast_router.py's own source text
        must not contain a hardcoded Persona canonical ID or display
        name -- it must resolve everything through
        agent.pp_context_bridge.resolve_persona_id()."""
        import inspect

        src = inspect.getsource(fast_router)
        for forbidden in ("PP-PER-", "Police Horitius", "Verba Clarion", "\"Bell\"", "'Bell'"):
            self.assertNotIn(forbidden, src, f"fast_router.py must not hardcode {forbidden!r}")
