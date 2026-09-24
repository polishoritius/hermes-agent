"""Deterministic formatter tests -- HERMES-FAST-ROUTER-001 PHASE 3."""
from __future__ import annotations

import unittest

from agent import pp_bridge_formatter as fmt


class FormatPersonaListTests(unittest.TestCase):
    def test_lists_all_names(self):
        data = {"personas": [{"id": "PP-PER-007", "name": "Police Horitius", "role": "Fact"}]}
        text = fmt.format_persona_list(data)
        self.assertIn("1名です", text)
        self.assertIn("- Police Horitius", text)


class FormatPersonaTests(unittest.TestCase):
    def test_includes_id_role_description(self):
        data = {"id": "PP-PER-009", "name": "Verba Clarion", "role": "Definition Architect", "description": "..."}
        text = fmt.format_persona(data, query="Bell")
        self.assertIn("Verba Clarion（Bell）", text)
        self.assertIn("PP-PER-009", text)
        self.assertIn("Definition Architect", text)

    def test_no_parenthetical_when_query_equals_name(self):
        data = {"id": "PP-PER-009", "name": "Verba Clarion", "role": "Definition Architect", "description": ""}
        text = fmt.format_persona(data, query="Verba Clarion")
        self.assertNotIn("（", text)

    def test_not_found(self):
        text = fmt.format_persona({"error": "PERSONA_NOT_FOUND", "query": "xyz"}, query="xyz")
        self.assertIn("見つかりませんでした", text)


class FormatChatterTests(unittest.TestCase):
    def test_renders_turns(self):
        data = {
            "status": "COMPLETED_MAX_TURNS",
            "turns": [
                {"speaker_display_name": "Police Horitius", "text": "hello"},
                {"speaker_display_name": "Verba Clarion", "text": "hi"},
            ],
        }
        text = fmt.format_chatter(data)
        self.assertIn("Police Horitius:", text)
        self.assertIn("hello", text)
        self.assertIn("Verba Clarion:", text)

    def test_not_found(self):
        data = {"status": "PERSONA_NOT_FOUND", "persona_a": "Police", "persona_b": "xyz", "turns": []}
        text = fmt.format_chatter(data)
        self.assertIn("失敗", text)
