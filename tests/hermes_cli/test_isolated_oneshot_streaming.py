"""Streaming-only completion tests for the isolated one-shot contract.

Uses unittest and an in-process fake SSE transport.  No real socket or
credential is needed, and the file remains runnable when pytest is absent.
"""

from __future__ import annotations

import hashlib
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


FAKE_API_KEY = "sk-fake-DUMMY-not-a-real-credential"
FAKE_BASE_URL = "https://openrouter.ai/api/v1"
FAKE_MODEL = "vendor/test-model-v1"


def _chunk(content: str = "FAKE_STREAM_OK"):
    delta = SimpleNamespace(
        role="assistant",
        content=content,
        reasoning=None,
        reasoning_content=None,
        tool_calls=None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason="stop", index=0)
    usage = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return SimpleNamespace(
        choices=[choice], usage=usage, model=FAKE_MODEL, id="fake-stream-id"
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.response = None
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


class _FakeCompletions:
    def __init__(self, *, error=None):
        self.calls = []
        self.error = error

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return _FakeStream([_chunk()])


class _FakeClient:
    def __init__(self, *, error=None):
        self.completions = _FakeCompletions(error=error)
        self.chat = SimpleNamespace(completions=self.completions)

    def close(self):
        return None


class _FakeHTTPError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"simulated HTTP {status_code}")
        self.status_code = status_code


def _isolated_agent(fake_client):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key=FAKE_API_KEY,
        base_url=FAKE_BASE_URL,
        provider="openrouter",
        api_mode="chat_completions",
        model=FAKE_MODEL,
        enabled_toolsets=[],
        quiet_mode=True,
        platform="cli",
        session_db=None,
        fallback_model=None,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        persist_session=False,
        minimal_system_prompt=True,
        api_max_attempts=1,
        isolated_runtime=True,
    )
    agent.client = fake_client
    agent._create_request_openai_client = lambda **_kwargs: fake_client
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None
    return agent


class IsolatedOneShotStreamingTests(unittest.TestCase):
    def setUp(self):
        self._agents = []
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.home = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"HERMES_HOME": str(self.home)}, clear=False)
        self._env.start()
        # Build the ordinary profile skeleton before the runtime snapshot.  It
        # is profile bootstrap state, not a one-shot artifact; snapshots below
        # then prove the isolated execution adds neither files nor directories.
        from hermes_cli.config import ensure_hermes_home

        ensure_hermes_home()
        self._network = patch.multiple(
            socket.socket,
            connect=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("real network attempted")
            ),
            connect_ex=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("real network attempted")
            ),
        )
        self._network.start()

    def tearDown(self):
        for agent in reversed(self._agents):
            agent.close()
        self._network.stop()
        self._env.stop()
        self._tmp.cleanup()

    def _agent(self, client):
        agent = _isolated_agent(client)
        self._agents.append(agent)
        return agent

    def _snapshot(self):
        return {
            ("dir" if path.is_dir() else "file", path.relative_to(self.home))
            for path in self.home.rglob("*")
        }

    def test_streaming_uses_isolated_agent_without_session_or_memory(self):
        client = _FakeClient()
        agent = self._agent(client)
        result = agent.run_conversation("fictional input")

        self.assertEqual(result["final_response"], "FAKE_STREAM_OK")
        self.assertIsNone(agent._session_db)
        self.assertTrue(agent._persist_disabled)
        self.assertIsNone(agent._memory_store)
        self.assertIsNone(agent._memory_manager)
        self.assertEqual(len(client.completions.calls), 1)

    def test_streaming_request_has_minimal_prompt_and_no_capabilities(self):
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

        client = _FakeClient()
        agent = self._agent(client)
        agent.run_conversation("fictional input")
        body = client.completions.calls[0]

        systems = [m for m in body["messages"] if m.get("role") == "system"]
        self.assertEqual([m["content"].strip() for m in systems], [DEFAULT_AGENT_IDENTITY.strip()])
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)
        self.assertTrue(body["stream"])
        self.assertEqual(agent.valid_tool_names, set())

    def test_streaming_never_activates_fallback(self):
        client = _FakeClient(error=RuntimeError("simulated provider error"))
        agent = self._agent(client)
        fallback_calls = []
        agent._try_activate_fallback = lambda *_a, **_k: fallback_calls.append(True) or False

        agent.run_conversation("fictional input")

        self.assertEqual(len(client.completions.calls), 1)
        self.assertEqual(fallback_calls, [])
        self.assertTrue(agent._fallback_disabled)

    def test_streaming_internal_retry_is_capped_at_one_total_attempt(self):
        client = _FakeClient(error=ConnectionError("simulated stream drop"))
        with patch.dict(os.environ, {"HERMES_STREAM_RETRIES": "9"}, clear=False):
            agent = self._agent(client)
            agent.run_conversation("fictional input")

        self.assertEqual(len(client.completions.calls), 1)
        self.assertEqual(agent._api_max_retries, 1)

    def test_streaming_does_not_persist_messages_or_artifacts(self):
        auth = self.home / "auth.json"
        dotenv = self.home / ".env"
        auth.write_text('{"fake":"value"}', encoding="utf-8")
        dotenv.write_text("FAKE_KEY=value\n", encoding="utf-8")

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        before = (digest(auth), digest(dotenv))
        files_before = {p.relative_to(self.home) for p in self.home.rglob("*") if p.is_file()}
        agent = self._agent(_FakeClient())
        agent.run_conversation("fictional input")
        files_after = {p.relative_to(self.home) for p in self.home.rglob("*") if p.is_file()}

        self.assertEqual((digest(auth), digest(dotenv)), before)
        new_files = files_after - files_before - {Path("SOUL.md")}
        self.assertFalse(any("state.db" in str(p) for p in new_files))
        self.assertFalse(any("session" in str(p).lower() for p in new_files))
        self.assertFalse(any("memory" in str(p).lower() for p in new_files))

    def test_isolated_failure_matrix_has_zero_filesystem_delta(self):
        """Every provider failure shape is denied at the dump boundary."""
        (self.home / "SOUL.md").write_text("fixture", encoding="utf-8")
        failures = (
            ("http_404", _FakeHTTPError(404)),
            ("http_400", _FakeHTTPError(400)),
            ("http_401", _FakeHTTPError(401)),
            ("timeout", TimeoutError("simulated timeout")),
            ("transport", ConnectionError("simulated transport exception")),
            ("streaming", RuntimeError("simulated streaming failure")),
        )
        for label, error in failures:
            with self.subTest(label=label):
                before = self._snapshot()
                agent = self._agent(_FakeClient(error=error))
                agent.run_conversation("fictional input")
                self.assertEqual(self._snapshot(), before)
                self.assertEqual(list((self.home / "sessions").glob("request_dump_*.json")), [])

    def test_isolated_nonstreaming_failure_has_zero_filesystem_delta(self):
        (self.home / "SOUL.md").write_text("fixture", encoding="utf-8")
        before = self._snapshot()
        agent = self._agent(_FakeClient())
        agent._disable_streaming = True

        def fail_nonstreaming(*_args, **_kwargs):
            raise _FakeHTTPError(404)

        with patch(
            "agent.chat_completion_helpers._dispatch_nonstreaming_api_request",
            fail_nonstreaming,
        ):
            agent.run_conversation("fictional input")

        self.assertEqual(self._snapshot(), before)
        self.assertEqual(list((self.home / "sessions").glob("request_dump_*.json")), [])

    def test_isolation_overrides_dump_requests_environment(self):
        (self.home / "SOUL.md").write_text("fixture", encoding="utf-8")
        before = self._snapshot()
        with patch.dict(os.environ, {"HERMES_DUMP_REQUESTS": "1"}, clear=False):
            agent = self._agent(_FakeClient())
            agent.run_conversation("fictional input")
        self.assertEqual(self._snapshot(), before)
        self.assertEqual(list((self.home / "sessions").glob("request_dump_*.json")), [])

    def test_isolated_dump_boundary_never_persists_headers(self):
        agent = self._agent(_FakeClient())
        with patch(
            "agent.agent_runtime_helpers.atomic_json_write",
            side_effect=AssertionError("isolated dump reached filesystem writer"),
        ):
            result = agent._dump_api_request_debug(
                {
                    "model": FAKE_MODEL,
                    "messages": [{"role": "user", "content": "fictional input"}],
                },
                reason="test",
            )
        self.assertIsNone(result)
        self.assertEqual(list((self.home / "sessions").glob("request_dump_*.json")), [])

    def test_isolated_agent_does_not_create_missing_logs_directory(self):
        sessions = self.home / "sessions"
        sessions.rmdir()
        self.assertFalse(sessions.exists())
        self._agent(_FakeClient())
        self.assertFalse(sessions.exists())

    def test_normal_mode_debug_dump_behavior_is_unchanged(self):
        agent = self._agent(_FakeClient())
        agent._isolated_runtime = False
        agent.logs_dir = self.home / "normal-sessions"
        dump = agent._dump_api_request_debug(
            {
                "model": FAKE_MODEL,
                "messages": [{"role": "user", "content": "fictional input"}],
            },
            reason="test",
        )
        self.assertIsNotNone(dump)
        self.assertTrue(dump.exists())

    def test_streaming_and_nonstreaming_share_isolation_request(self):
        stream_client = _FakeClient()
        stream_agent = self._agent(stream_client)
        stream_agent.run_conversation("fictional input")
        stream_body = dict(stream_client.completions.calls[0])

        nonstream_agent = self._agent(_FakeClient())
        nonstream_agent._disable_streaming = True
        captured = []

        def fake_dispatch(_agent, kwargs, *, make_client):
            captured.append(dict(kwargs))
            message = SimpleNamespace(
                role="assistant", content="FAKE_STREAM_OK", tool_calls=None,
                reasoning=None, refusal=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop", index=0)],
                usage=None, model=FAKE_MODEL, id="fake-nonstream-id",
            )

        with patch("agent.chat_completion_helpers._dispatch_nonstreaming_api_request", fake_dispatch):
            nonstream_agent.run_conversation("fictional input")

        for key in ("stream", "stream_options", "timeout"):
            stream_body.pop(key, None)
        nonstream_body = captured[0]
        for key in ("stream", "stream_options", "timeout"):
            nonstream_body.pop(key, None)
        self.assertEqual(stream_body, nonstream_body)


if __name__ == "__main__":
    unittest.main()
