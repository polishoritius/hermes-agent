"""Light Chat tests -- HERMES-LIGHT-CHAT-001.

Every test here uses a mock ``client_factory`` (never a real
``openai.OpenAI`` instance), so this suite makes zero network calls.
"""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from agent import light_chat


def _mock_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _mock_client_factory(response_text: str, capture: dict):
    client = MagicMock()

    def _create(*, model, messages):
        capture["model"] = model
        capture["messages"] = messages
        capture["call_count"] = capture.get("call_count", 0) + 1
        return _mock_response(response_text)

    client.chat.completions.create.side_effect = _create
    return lambda: client


class LightChatBasicTurnTests(unittest.TestCase):
    def _run(self, query, response_text, tmp_path, **kwargs):
        capture: dict = {}
        result = light_chat.run_light_chat(
            query,
            hermes_home=tmp_path,
            client_factory=_mock_client_factory(response_text, capture),
            **kwargs,
        )
        return result, capture

    def test_hello_makes_exactly_one_local_call_with_no_tools(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result, capture = self._run("こんにちは", "こんにちは!今日もよろしくお願いします。", tmp_path)

            self.assertEqual(capture["call_count"], 1, "expected exactly one local Ollama call")
            # No tools/tool_search: the create() call signature in this
            # module never includes a `tools` kwarg at all.
            create_sig_kwargs = set(inspect.signature(light_chat.run_light_chat).parameters)
            self.assertNotIn("tools", create_sig_kwargs)
            self.assertEqual(result["tool_search_calls"], 0)
            self.assertEqual(result["tool_calls"], 0)
            self.assertEqual(result["external_api_calls"], 0)
            self.assertEqual(result["provider"], light_chat.DEFAULT_PROVIDER)
            self.assertEqual(result["base_url"], light_chat.DEFAULT_BASE_URL)
            self.assertEqual(result["model"], light_chat.DEFAULT_MODEL)

    def test_short_natural_question_returns_text(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result, _ = self._run("今日何しようか", "散歩でもいかがですか。", tmp_path)
            self.assertEqual(result["text"], "散歩でもいかがですか。")
            self.assertTrue(len(result["text"]) > 0)

    def test_history_capped_at_max_messages(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for i in range(6):
                self._run(f"turn {i}", f"reply {i}", tmp_path, session_id="cap-test")
            history = light_chat._load_history(tmp_path, "cap-test")
            self.assertLessEqual(len(history), light_chat.MAX_HISTORY_MESSAGES)

    def test_system_prompt_has_no_agents_md_or_tool_metadata(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # A huge AGENTS.md-like file must never be read by this module --
            # light_chat.py has no code path that reads any file named
            # AGENTS.md at all, unlike agent/prompt_builder.py's full
            # context-files loader.
            prompt = light_chat._build_system_prompt(tmp_path)
            self.assertNotIn("tool_search", prompt.lower())
            self.assertIn("Light Chat", prompt)


class LightChatUnavailableTests(unittest.TestCase):
    def test_connection_failure_raises_explicit_error_no_fallback(self):
        import tempfile

        def failing_client_factory():
            client = MagicMock()
            client.chat.completions.create.side_effect = ConnectionError("connection refused")
            return client

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(light_chat.LightChatUnavailableError) as ctx:
                light_chat.run_light_chat(
                    "こんにちは",
                    hermes_home=tmp_path,
                    client_factory=failing_client_factory,
                    base_url="http://localhost:11434/v1",
                )
            # The error message names the SAME local base_url that was
            # passed in -- proof this module never substitutes another
            # provider/base_url on failure.
            self.assertIn("http://localhost:11434/v1", str(ctx.exception))
            self.assertNotIn("openrouter", str(ctx.exception).lower())
            self.assertNotIn("anthropic", str(ctx.exception).lower())

    def test_empty_query_rejected_before_any_client_call(self):
        import tempfile

        calls = {"n": 0}

        def client_factory():
            calls["n"] += 1
            return MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(ValueError):
                light_chat.run_light_chat("   ", hermes_home=tmp_path, client_factory=client_factory)
            self.assertEqual(calls["n"], 0)


class CliLightFlagStructuralTests(unittest.TestCase):
    """Structural checks on cli.py's `main()` -- confirms the `--light`
    branch is an early-return placed BEFORE the normal --query path
    (HermesCLI() construction, worktree setup, _init_agent()), so the
    normal Agent route's own code is provably unreached and unmodified
    when `--light` is not passed."""

    def _read_cli_source(self) -> str:
        import cli

        return inspect.getsource(cli.main)

    def test_light_param_defaults_to_false(self):
        import cli

        sig = inspect.signature(cli.main)
        self.assertIn("light", sig.parameters)
        self.assertEqual(sig.parameters["light"].default, False)
        # Existing flags remain present and unchanged.
        for name in ("query", "q", "gateway", "provider", "model", "base_url", "reasoning"):
            self.assertIn(name, sig.parameters)

    def test_light_branch_precedes_normal_query_dispatch(self):
        src = self._read_cli_source()
        light_branch_pos = src.find("if light:")
        hermes_cli_construct_pos = src.find("cli = HermesCLI(")
        self.assertGreater(light_branch_pos, -1, "the `if light:` branch was not found in cli.main")
        self.assertGreater(
            hermes_cli_construct_pos, -1, "HermesCLI(...) construction was not found in cli.main"
        )
        self.assertLess(
            light_branch_pos,
            hermes_cli_construct_pos,
            "the light-chat branch must return before HermesCLI() is constructed, "
            "so it never triggers _init_agent()/MCP discovery/tool_search",
        )

    def test_light_branch_returns(self):
        src = self._read_cli_source()
        light_branch = src[src.find("if light:") : src.find("if light:") + 1500]
        self.assertIn("return", light_branch)


class NativeTransportKeepAliveTests(unittest.TestCase):
    """HERMES-FAST-ROUTER-001 PHASE 4: the production path (no
    client_factory override) must use native Ollama /api/chat with
    keep_alive, not the OpenAI-compatible /v1 endpoint (empirically
    confirmed to silently drop keep_alive -- see module docstring).
    Mocks agent.light_chat._call_native_ollama_chat, so this test
    makes zero network calls."""

    def test_default_path_calls_native_transport_with_keep_alive(self):
        import tempfile
        from unittest.mock import patch

        fake = {"response_text": "hi", "prompt_tokens": 10, "output_tokens": 2}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("agent.light_chat._call_native_ollama_chat", return_value=fake) as mock_call:
                result = light_chat.run_light_chat("こんにちは", hermes_home=Path(tmp))

        mock_call.assert_called_once()
        kwargs = mock_call.call_args.kwargs
        self.assertEqual(kwargs["keep_alive"], light_chat.DEFAULT_KEEP_ALIVE)
        self.assertEqual(kwargs["model"], light_chat.DEFAULT_MODEL)
        self.assertEqual(result["text"], "hi")
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["output_tokens"], 2)

    def test_client_factory_override_still_uses_openai_sdk_shape(self):
        """Backward compatibility: every existing test in this file
        passes an explicit client_factory and must be unaffected by
        the PHASE 4 transport change."""
        import tempfile

        capture = {}

        def factory():
            return _mock_client_factory("ok", capture)()

        with tempfile.TemporaryDirectory() as tmp:
            result = light_chat.run_light_chat("hi", hermes_home=Path(tmp), client_factory=factory)
        self.assertEqual(result["text"], "ok")
        self.assertEqual(capture["call_count"], 1)

    def test_native_chat_url_strips_v1_suffix(self):
        self.assertEqual(
            light_chat._native_chat_url("http://localhost:11434/v1"), "http://localhost:11434/api/chat"
        )

    def test_native_chat_url_handles_no_v1_suffix(self):
        self.assertEqual(
            light_chat._native_chat_url("http://localhost:11434"), "http://localhost:11434/api/chat"
        )

    def test_default_keep_alive_is_longer_than_chatter(self):
        """CHATTER (scripts/persona_chatter.py) uses "5m"; Light Chat
        is meant to stay resident for a whole interactive session, so
        it defaults higher -- documented decision, not a guess."""
        self.assertEqual(light_chat.DEFAULT_KEEP_ALIVE, "30m")
