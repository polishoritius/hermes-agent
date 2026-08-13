from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import oneshot as oneshot_mod


REQUESTED_MODEL = "test/requested-model"
RESPONSE_MODEL = "test/provider-model"


def _fake_response(content: str = "HERMES_P5_PROVIDER_OK"):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=None,
        reasoning=None,
        refusal=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop", index=0)],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        model=RESPONSE_MODEL,
        id="fake-response-id",
    )


class IsolatedResponseMetadataTests(unittest.TestCase):
    def _build_agent(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="SUPER_SECRET_TEST_KEY_DO_NOT_PRINT",
            base_url="http://127.0.0.1:9/v1",
            provider="openrouter",
            api_mode="chat_completions",
            model=REQUESTED_MODEL,
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
        agent._disable_streaming = True
        return agent

    def test_fake_provider_preserves_actual_model_without_extra_network_or_write(self):
        import agent.chat_completion_helpers as completion_helpers
        import agent.model_metadata as model_metadata

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ, {"HERMES_HOME": temp_dir}, clear=False
        ):
            provider_calls: list[dict] = []

            def fake_dispatch(agent, api_kwargs, *, make_client):
                provider_calls.append(dict(api_kwargs))
                return _fake_response()

            with patch.object(
                completion_helpers,
                "_dispatch_nonstreaming_api_request",
                side_effect=fake_dispatch,
            ), patch.object(
                model_metadata.requests,
                "get",
                side_effect=AssertionError("metadata network request attempted"),
            ):
                agent = self._build_agent()
                before = {
                    p.relative_to(temp_dir).as_posix(): p.read_bytes()
                    for p in Path(temp_dir).rglob("*")
                    if p.is_file()
                }
                result = agent.run_conversation("SECRET_PROMPT_SENTINEL_DO_NOT_PRINT")
                after = {
                    p.relative_to(temp_dir).as_posix(): p.read_bytes()
                    for p in Path(temp_dir).rglob("*")
                    if p.is_file()
                }
                agent.close()

        self.assertEqual(len(provider_calls), 1)
        self.assertEqual(result["model"], REQUESTED_MODEL)
        self.assertEqual(result["response_model"], RESPONSE_MODEL)
        self.assertEqual(after, before)

    def test_fake_failures_make_one_provider_attempt_and_zero_metadata_requests(self):
        import agent.chat_completion_helpers as completion_helpers
        import agent.model_metadata as model_metadata

        failures = (
            RuntimeError("simulated 400 Bad Request"),
            RuntimeError("simulated 401 Unauthorized"),
            RuntimeError("simulated 404 Not Found"),
            TimeoutError("simulated request timeout"),
        )
        for failure in failures:
            with self.subTest(failure=str(failure)), tempfile.TemporaryDirectory() as temp_dir, patch.dict(
                os.environ, {"HERMES_HOME": temp_dir}, clear=False
            ):
                provider_calls: list[dict] = []

                def fake_dispatch(agent, api_kwargs, *, make_client):
                    provider_calls.append(dict(api_kwargs))
                    raise failure

                with patch.object(
                    completion_helpers,
                    "_dispatch_nonstreaming_api_request",
                    side_effect=fake_dispatch,
                ), patch.object(
                    model_metadata.requests,
                    "get",
                    side_effect=AssertionError("metadata network request attempted"),
                ):
                    agent = self._build_agent()
                    before = {
                        p.relative_to(temp_dir).as_posix(): p.read_bytes()
                        for p in Path(temp_dir).rglob("*")
                        if p.is_file()
                    }
                    agent.run_conversation("SECRET_PROMPT_SENTINEL_DO_NOT_PRINT")
                    after = {
                        p.relative_to(temp_dir).as_posix(): p.read_bytes()
                        for p in Path(temp_dir).rglob("*")
                        if p.is_file()
                    }
                    agent.close()

                self.assertEqual(len(provider_calls), 1)
                self.assertEqual(after, before)

    def test_streaming_response_stub_preserves_chunk_model_identity(self):
        import agent.chat_completion_helpers as completion_helpers

        response = completion_helpers._build_partial_stream_stub(
            "assistant",
            "HERMES_P5_PROVIDER_OK",
            None,
            RESPONSE_MODEL,
            None,
        )
        self.assertEqual(response.model, RESPONSE_MODEL)

    def test_opt_in_diagnostic_is_stderr_only_and_secret_safe(self):
        secret_key = "SUPER_SECRET_TEST_KEY_DO_NOT_PRINT"
        secret_prompt = "SECRET_PROMPT_SENTINEL_DO_NOT_PRINT"
        stdout = io.StringIO()
        stderr = io.StringIO()

        def fake_run_agent(prompt, **kwargs):
            self.assertEqual(prompt, secret_prompt)
            self.assertNotIn(secret_key, prompt)
            return (
                "HERMES_P5_PROVIDER_OK",
                {
                    "model": REQUESTED_MODEL,
                    "response_model": RESPONSE_MODEL,
                    "provider": "openrouter",
                },
            )

        with patch.object(oneshot_mod, "_run_agent", side_effect=fake_run_agent), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = oneshot_mod.run_oneshot(
                secret_prompt,
                isolated=True,
                show_response_metadata=True,
            )

        self.assertEqual(rc, 0)
        self.assertEqual(stdout.getvalue(), "HERMES_P5_PROVIDER_OK\n")
        self.assertEqual(
            stderr.getvalue(),
            "HTTP_STATUS=UNKNOWN\n"
            f"REQUESTED_MODEL={REQUESTED_MODEL}\n"
            f"RESPONSE_MODEL={RESPONSE_MODEL}\n"
            "ROUTING_PROVIDER=openrouter\n",
        )
        combined = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(secret_key, combined)
        self.assertNotIn(secret_prompt, combined)
        self.assertNotIn("Authorization", combined)

    def test_default_oneshot_output_is_unchanged(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        result = {
            "model": REQUESTED_MODEL,
            "response_model": RESPONSE_MODEL,
            "provider": "openrouter",
        }
        with patch.object(
            oneshot_mod,
            "_run_agent",
            return_value=("HERMES_P5_PROVIDER_OK", result),
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = oneshot_mod.run_oneshot("prompt", isolated=True)

        self.assertEqual(rc, 0)
        self.assertEqual(stdout.getvalue(), "HERMES_P5_PROVIDER_OK\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_diagnostic_requires_isolated_and_rejects_multiline_identity(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = oneshot_mod.run_oneshot(
                "prompt",
                show_response_metadata=True,
            )
        self.assertEqual(rc, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("requires --isolated", stderr.getvalue())
        self.assertEqual(oneshot_mod._diagnostic_slug("model\nSECRET"), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
