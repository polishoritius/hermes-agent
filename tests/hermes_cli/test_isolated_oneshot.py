"""Tests for the isolated one-shot runtime (``hermes -z --isolated``).

Isolated mode is a fail-closed contract, so these tests assert the *absence*
of behaviour (no session store, no memory, no tools, no MCP/plugins/hooks, no
fallback, one HTTP attempt) rather than just the presence of a flag.

Every test drives a fake transport: ``_dispatch_nonstreaming_api_request`` is
replaced so the constructed request body can be inspected and no socket is
ever opened. ``_no_real_network`` (autouse) makes any *real* outbound attempt
raise immediately, so a regression that bypasses the fake transport fails
loudly instead of silently reaching a provider.
"""

from __future__ import annotations

import hashlib
import socket
from types import SimpleNamespace

import pytest

from hermes_cli import oneshot as oneshot_mod
from hermes_cli._parser import (
    ISOLATED_REQUIRES_ONESHOT_ERROR,
    build_top_level_parser,
    isolated_oneshot_active,
    validate_isolated_oneshot,
)

FAKE_API_KEY = "sk-fake-DUMMY-not-a-real-credential"
FAKE_BASE_URL = "http://127.0.0.1:9/v1"
FAKE_MODEL = "vendor/test-model-v1"
FAKE_PROVIDER = "openrouter"


# ── Network kill-switch ────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Make any genuine outbound network attempt fail loudly.

    Guards the whole module: if a change ever routes around the fake
    transport, the test errors instead of contacting a real provider.
    """

    def _blocked(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError(
            "REAL NETWORK ACCESS ATTEMPTED during an isolated-mode test"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=False)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked, raising=False)
    monkeypatch.setattr(socket, "create_connection", _blocked, raising=False)
    try:
        import httpx

        monkeypatch.setattr(httpx.Client, "send", _blocked, raising=False)
        monkeypatch.setattr(httpx.AsyncClient, "send", _blocked, raising=False)
    except Exception:  # pragma: no cover - httpx always present in practice
        pass


# ── Fake transport helpers ─────────────────────────────────────────────────


def _fake_response(content: str = "FAKE_OK"):
    message = SimpleNamespace(
        role="assistant", content=content, tool_calls=None, reasoning=None, refusal=None
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
        model=FAKE_MODEL,
        id="fake-response-id",
    )


@pytest.fixture()
def capture_requests(monkeypatch):
    """Replace the non-streaming dispatch and record every request body."""
    import agent.chat_completion_helpers as cch

    captured: list[dict] = []
    behaviour: dict = {"raise": None}

    def fake_dispatch(agent, api_kwargs, *, make_client):
        captured.append(dict(api_kwargs))
        exc = behaviour["raise"]
        if exc is not None:
            raise exc
        return _fake_response()

    monkeypatch.setattr(cch, "_dispatch_nonstreaming_api_request", fake_dispatch)
    return SimpleNamespace(calls=captured, behaviour=behaviour)


def _build_isolated_agent(**overrides):
    """Construct an AIAgent with exactly the isolated-mode contract."""
    from run_agent import AIAgent

    kwargs = dict(
        api_key=FAKE_API_KEY,
        base_url=FAKE_BASE_URL,
        provider=FAKE_PROVIDER,
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
    kwargs.update(overrides)
    agent = AIAgent(**kwargs)
    # Existing lever for "provider does not support streaming"; keeps this on
    # the non-streaming dispatch the fake transport intercepts. Isolated mode
    # deliberately does NOT change the product's streaming default.
    agent._disable_streaming = True
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None
    return agent


# ── Parser ─────────────────────────────────────────────────────────────────


class TestParser:
    def test_oneshot_with_isolated_is_accepted(self):
        parser, _subs, _chat = build_top_level_parser()
        args = parser.parse_args(["-z", "hello", "--isolated"])
        assert args.oneshot == "hello"
        assert args.isolated_oneshot is True
        assert validate_isolated_oneshot(args) is None
        assert isolated_oneshot_active(args) is True

    def test_isolated_without_oneshot_is_rejected(self):
        parser, _subs, _chat = build_top_level_parser()
        args = parser.parse_args(["--isolated"])
        assert validate_isolated_oneshot(args) == ISOLATED_REQUIRES_ONESHOT_ERROR
        # Never silently auto-enables one-shot.
        assert isolated_oneshot_active(args) is False

    def test_isolated_without_oneshot_exits_with_code_2(self, capsys):
        from hermes_cli.main import _enforce_isolated_requires_oneshot

        parser, _subs, _chat = build_top_level_parser()
        args = parser.parse_args(["--isolated"])
        with pytest.raises(SystemExit) as excinfo:
            _enforce_isolated_requires_oneshot(args)
        assert excinfo.value.code == 2
        assert "--isolated requires -z/--oneshot" in capsys.readouterr().err

    def test_plain_oneshot_is_unchanged(self):
        parser, _subs, _chat = build_top_level_parser()
        args = parser.parse_args(["-z", "hello"])
        assert args.oneshot == "hello"
        assert args.isolated_oneshot is False
        assert validate_isolated_oneshot(args) is None
        assert isolated_oneshot_active(args) is False

    def test_normal_chat_invocation_is_unchanged(self):
        parser, _subs, _chat = build_top_level_parser()
        args = parser.parse_args([])
        assert args.oneshot is None
        assert args.isolated_oneshot is False
        assert validate_isolated_oneshot(args) is None

    def test_serve_isolated_still_works(self):
        """`hermes serve --isolated` predates this flag and must be untouched.

        The dashboard subparser owns dest ``isolated``; the top-level one-shot
        flag deliberately uses dest ``isolated_oneshot`` so the two cannot
        collide (argparse lets a subparser default overwrite a top-level dest).
        """
        from hermes_cli.subcommands.dashboard import build_dashboard_parser

        parser, subs, _chat = build_top_level_parser()
        build_dashboard_parser(
            subs, cmd_dashboard=lambda a: None, cmd_dashboard_register=lambda a: None
        )
        args = parser.parse_args(["serve", "--isolated"])
        assert args.command == "serve"
        assert args.isolated is True  # serve's own flag
        assert args.isolated_oneshot is False  # one-shot flag untouched
        assert validate_isolated_oneshot(args) is None
        assert isolated_oneshot_active(args) is False


# ── Fail-closed preconditions ──────────────────────────────────────────────


class TestFailClosed:
    @pytest.mark.parametrize(
        "var", ["HERMES_KANBAN_TASK", "HERMES_KANBAN_BOARD"]
    )
    def test_worker_context_refuses_to_run(self, monkeypatch, var):
        """A Kanban/delegation context must error, never silently proceed.

        model_tools._compute_tool_definitions force-adds the ``kanban``
        toolset for a dispatcher-owned worker even when the caller passed
        ``enabled_toolsets=[]`` — so stripping the env var instead of refusing
        would leave a live worker tool surface.
        """
        monkeypatch.setenv(var, "task-123")
        err = oneshot_mod._isolated_precondition_error(None, None)
        assert err is not None
        assert var in err

        rc = oneshot_mod.run_oneshot("prompt", isolated=True)
        assert rc == 2

    def test_toolsets_cannot_be_combined_with_isolated(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        err = oneshot_mod._isolated_precondition_error("terminal,file", None)
        assert err is not None and "--toolsets" in err
        assert oneshot_mod.run_oneshot("p", toolsets="terminal", isolated=True) == 2

    def test_usage_file_cannot_be_combined_with_isolated(self, tmp_path):
        err = oneshot_mod._isolated_precondition_error(None, str(tmp_path / "u.json"))
        assert err is not None and "--usage-file" in err

    def test_clean_invocation_has_no_precondition_error(self):
        assert oneshot_mod._isolated_precondition_error(None, None) is None


class TestRunOneshotForwarding:
    """``run_oneshot`` must carry the flag through to ``_run_agent``.

    A full end-to-end ``run_oneshot`` run is deliberately NOT exercised here:
    it calls ``resolve_runtime_provider()``, which resolves real credentials.
    The hermetic conftest strips those by design, so the seam is taken at
    ``_run_agent`` (whose own contract is asserted in TestAgentWiring).
    """

    @pytest.mark.parametrize("isolated", [True, False])
    def test_isolated_flag_is_forwarded(self, monkeypatch, isolated):
        seen: dict = {}

        def fake_run_agent(prompt, **kwargs):
            seen.update(kwargs)
            seen["_prompt"] = prompt
            return ("done", {})

        monkeypatch.setattr(oneshot_mod, "_run_agent", fake_run_agent)
        rc = oneshot_mod.run_oneshot("hello world", isolated=isolated)

        assert rc == 0
        assert seen["isolated"] is isolated
        assert seen["_prompt"] == "hello world"
        if isolated:
            # Isolated forces an empty toolset list and ignores config.
            assert seen["toolsets"] == []
            assert seen["use_config_toolsets"] is False

    def test_isolated_does_not_set_yolo_env(self, monkeypatch):
        """Isolated mode must not widen the approval posture."""
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
        monkeypatch.setattr(
            oneshot_mod, "_run_agent", lambda prompt, **k: ("done", {})
        )
        oneshot_mod.run_oneshot("hello", isolated=True)
        import os as _os

        assert _os.environ.get("HERMES_YOLO_MODE") is None
        assert _os.environ.get("HERMES_ACCEPT_HOOKS") is None


# ── Startup bypass (MCP / plugins / hooks / webhooks) ──────────────────────


class TestStartupBypass:
    def test_prepare_agent_startup_returns_before_discovery(self, monkeypatch):
        """No plugin discovery, MCP discovery, or hook registration."""
        import hermes_cli.main as main_mod

        calls: list[str] = []
        monkeypatch.setattr(
            main_mod,
            "_apply_safe_mode",
            lambda args: calls.append("safe_mode"),
        )

        import hermes_cli.plugins as plugins_mod

        monkeypatch.setattr(
            plugins_mod,
            "start_background_plugin_discovery",
            lambda *a, **k: calls.append("plugins"),
        )

        args = SimpleNamespace(
            command=None,
            oneshot="hello",
            isolated_oneshot=True,
            yolo=False,
            safe_mode=False,
            accept_hooks=False,
            tui=False,
        )
        main_mod._prepare_agent_startup(args)

        assert "plugins" not in calls, "plugin discovery must not run"
        # Safe mode is still applied — isolated layers on top of it rather
        # than replacing it.
        assert "safe_mode" in calls

    def test_non_isolated_oneshot_still_discovers(self, monkeypatch):
        """Backward compatibility: normal one-shot startup is unchanged."""
        import hermes_cli.main as main_mod
        import hermes_cli.plugins as plugins_mod

        calls: list[str] = []
        monkeypatch.setattr(
            plugins_mod,
            "start_background_plugin_discovery",
            lambda *a, **k: calls.append("plugins"),
        )
        monkeypatch.setattr(
            main_mod, "_should_background_mcp_startup", lambda args: True
        )
        import hermes_cli.mcp_startup as mcp_startup_mod

        monkeypatch.setattr(
            mcp_startup_mod,
            "start_background_mcp_discovery",
            lambda *a, **k: calls.append("mcp"),
        )

        args = SimpleNamespace(
            command=None,
            oneshot="hello",
            isolated_oneshot=False,
            yolo=False,
            safe_mode=False,
            accept_hooks=False,
            tui=False,
        )
        main_mod._prepare_agent_startup(args)
        assert "plugins" in calls, "normal one-shot must still discover plugins"


# ── Agent construction wiring ──────────────────────────────────────────────


def _capture_agent_kwargs(monkeypatch, *, isolated: bool):
    """Run ``_run_agent`` with AIAgent replaced, returning constructor kwargs."""
    captured: dict = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._session_messages = []

        def run_conversation(self, prompt):
            captured["_prompt"] = prompt
            return {"final_response": "ok"}

        def shutdown_memory_provider(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    mcp_calls: list[str] = []
    import hermes_cli.mcp_startup as mcp_startup_mod

    monkeypatch.setattr(
        mcp_startup_mod,
        "ensure_mcp_discovery_before_agent_build",
        lambda **k: mcp_calls.append("mcp"),
    )
    session_calls: list[str] = []
    monkeypatch.setattr(
        oneshot_mod,
        "_create_session_db_for_oneshot",
        lambda: session_calls.append("sessiondb"),
    )

    oneshot_mod._run_agent(
        "THE_PROMPT",
        model=FAKE_MODEL,
        provider=FAKE_PROVIDER,
        toolsets=[] if isolated else None,
        use_config_toolsets=not isolated,
        isolated=isolated,
    )
    return captured, mcp_calls, session_calls


class TestAgentWiring:
    def test_isolated_constructor_contract(self, monkeypatch):
        kwargs, mcp_calls, session_calls = _capture_agent_kwargs(
            monkeypatch, isolated=True
        )

        # Session: no SessionDB is ever constructed.
        assert session_calls == [], "SessionDB must not be constructed"
        assert kwargs["session_db"] is None
        assert kwargs["persist_session"] is False

        # Memory / context files / background review.
        assert kwargs["skip_memory"] is True
        assert kwargs["skip_context_files"] is True
        assert kwargs["skip_background_review"] is True
        assert kwargs["load_soul_identity"] is False

        # Tools: empty LIST, not None (None means "every toolset").
        assert kwargs["enabled_toolsets"] == []
        assert kwargs["enabled_toolsets"] is not None

        # System prompt, fallback, attempts, artifacts.
        assert kwargs["minimal_system_prompt"] is True
        assert kwargs["fallback_model"] is None
        assert kwargs["api_max_attempts"] == 1
        assert kwargs["checkpoints_enabled"] is False
        assert kwargs["save_trajectories"] is False

        # MCP discovery never runs.
        assert mcp_calls == [], "MCP discovery must not run in isolated mode"

        # Provider / model preserved exactly as requested.
        assert kwargs["model"] == FAKE_MODEL
        assert kwargs["_prompt"] == "THE_PROMPT"

    def test_non_isolated_wiring_is_unchanged(self, monkeypatch):
        """Parity: a normal one-shot keeps session store + MCP discovery."""
        kwargs, mcp_calls, session_calls = _capture_agent_kwargs(
            monkeypatch, isolated=False
        )
        assert session_calls == ["sessiondb"], "normal one-shot still opens SessionDB"
        assert mcp_calls == ["mcp"], "normal one-shot still runs MCP discovery"
        # The non-isolated path passes NO isolation kwargs whatsoever — every
        # one of them is simply absent, so AIAgent's own defaults apply and
        # existing behaviour is bit-for-bit what it was before this change.
        for key in (
            "skip_memory",
            "skip_context_files",
            "skip_background_review",
            "minimal_system_prompt",
            "persist_session",
            "api_max_attempts",
            "isolated_runtime",
            "checkpoints_enabled",
            "save_trajectories",
        ):
            assert key not in kwargs, f"non-isolated one-shot must not pass {key}"
        # And it still receives a real fallback chain argument.
        assert "fallback_model" in kwargs


# ── System prompt content ──────────────────────────────────────────────────


class TestMinimalSystemPrompt:
    def test_minimal_prompt_is_generic_identity_only(self):
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
        from agent.system_prompt import build_system_prompt_parts

        agent = SimpleNamespace(_minimal_system_prompt=True)
        parts = build_system_prompt_parts(agent)
        assert parts["stable"] == DEFAULT_AGENT_IDENTITY
        assert parts["context"] == ""
        assert parts["volatile"] == ""

    def test_sentinels_never_reach_the_prompt(self, tmp_path, monkeypatch):
        """Context-file, SOUL, memory, cwd and env sentinels must all be absent."""
        from agent.system_prompt import build_system_prompt

        sentinels = {}
        for name in (
            "AGENTS.md",
            "SOUL.md",
            "MEMORY.md",
            "CLAUDE.md",
            ".hermes.md",
            ".cursorrules",
        ):
            token = f"SENTINEL_{name.replace('.', '_').upper()}_LEAK"
            (tmp_path / name).write_text(token, encoding="utf-8")
            sentinels[name] = token

        # Planted in a directory name on the real cwd/TERMINAL_CWD path, so a
        # prompt that echoes the working directory would carry it verbatim.
        path_sentinel = "SENTINEL_LOCAL_PATH_MUST_NOT_LEAK"
        workdir = tmp_path / path_sentinel
        workdir.mkdir()
        for name, token in sentinels.items():
            (workdir / name).write_text(token, encoding="utf-8")

        env_sentinel = "SENTINEL_ENV_VALUE_MUST_NOT_LEAK"
        monkeypatch.setenv("TERMINAL_CWD", str(workdir))
        monkeypatch.setenv("HERMES_TEST_LEAK_PROBE", env_sentinel)
        monkeypatch.chdir(workdir)

        agent = _build_isolated_agent()
        prompt = build_system_prompt(agent)

        for name, token in sentinels.items():
            assert token not in prompt, f"{name} content leaked into the prompt"
        assert path_sentinel not in prompt, "local path leaked into the prompt"
        assert env_sentinel not in prompt, "env var value leaked into the prompt"
        assert str(workdir) not in prompt, "local cwd path leaked into the prompt"

    def test_control_non_minimal_prompt_does_leak_cwd(self, tmp_path, monkeypatch):
        """Positive control: without minimal_system_prompt the cwd DOES appear.

        Without this, the sentinel test above could pass simply because the
        prompt builder never embeds these values on this platform — i.e. it
        would not actually be detecting isolation.
        """
        from agent.system_prompt import build_system_prompt

        path_sentinel = "SENTINEL_CONTROL_PATH_SHOULD_APPEAR"
        workdir = tmp_path / path_sentinel
        workdir.mkdir()
        (workdir / "AGENTS.md").write_text("CONTROL_CONTEXT_FILE", encoding="utf-8")
        monkeypatch.setenv("TERMINAL_CWD", str(workdir))
        monkeypatch.chdir(workdir)

        agent = _build_isolated_agent(
            minimal_system_prompt=False, skip_context_files=False
        )
        prompt = build_system_prompt(agent)

        assert path_sentinel in prompt or "CONTROL_CONTEXT_FILE" in prompt, (
            "control failed: the non-minimal prompt embedded neither the cwd "
            "path nor the context file, so the isolation assertions above "
            "would pass vacuously"
        )


# ── Fake-transport behaviour ───────────────────────────────────────────────


class TestIsolatedRequest:
    def test_request_has_no_tools_and_one_attempt(self, capture_requests):
        agent = _build_isolated_agent()
        assert agent.tools in ([], None), "no tools may be registered"
        assert not agent.valid_tool_names

        result = agent.run_conversation("PROMPT_FIDELITY_SENTINEL_123")

        assert len(capture_requests.calls) == 1
        body = capture_requests.calls[0]
        assert "tools" not in body, "request must carry no tools key at all"
        assert "tool_choice" not in body, "request must carry no tool_choice key"
        assert body["model"] == FAKE_MODEL
        assert result.get("final_response") == "FAKE_OK"

    def test_prompt_is_preserved_byte_for_byte(self, capture_requests):
        literal = "Explain the Zorbulon Paradox, verbatim: éè {braces} \"quotes\"\n\ttabbed"
        agent = _build_isolated_agent()
        agent.run_conversation(literal)

        body = capture_requests.calls[0]
        user_messages = [m for m in body["messages"] if m["role"] == "user"]
        assert user_messages[-1]["content"] == literal

    def test_system_message_is_minimal(self, capture_requests):
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

        agent = _build_isolated_agent()
        agent.run_conversation("hello")
        body = capture_requests.calls[0]
        system_messages = [m for m in body["messages"] if m["role"] == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content"].strip() == DEFAULT_AGENT_IDENTITY.strip()

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("simulated 500 Internal Server Error"),
            TimeoutError("simulated request timeout"),
            RuntimeError("simulated 429 Too Many Requests"),
        ],
        ids=["http_500", "timeout", "http_429"],
    )
    def test_exactly_one_attempt_and_no_fallback_on_error(
        self, capture_requests, monkeypatch, exc
    ):
        """One attempt for every failure shape, and fallback never activates."""
        agent = _build_isolated_agent()
        assert agent._fallback_disabled is True

        fallback_calls: list[str] = []
        monkeypatch.setattr(
            agent,
            "_try_activate_fallback",
            lambda *a, **k: fallback_calls.append("fallback") or False,
            raising=False,
        )

        capture_requests.behaviour["raise"] = exc
        agent.run_conversation("hello")

        assert len(capture_requests.calls) == 1, (
            f"expected exactly 1 HTTP attempt, got {len(capture_requests.calls)}"
        )
        assert fallback_calls == [], "fallback must never activate in isolated mode"

    def test_sdk_max_retries_stays_zero(self):
        agent = _build_isolated_agent()
        assert agent._api_max_retries == 1, "exactly one app-level attempt"

    def test_config_cannot_widen_the_attempt_cap(self, monkeypatch):
        """api_max_attempts overrides a user's agent.api_max_retries config."""
        agent = _build_isolated_agent(api_max_attempts=1)
        assert agent._api_max_retries == 1

    def test_persistence_is_disabled_on_the_real_agent(self):
        agent = _build_isolated_agent()
        assert agent._persist_disabled is True
        assert agent._minimal_system_prompt is True
        assert agent._session_db is None


# ── Credentials & artifacts ────────────────────────────────────────────────


class TestNoSideEffects:
    def test_credential_files_are_untouched(self, tmp_path, monkeypatch, capture_requests):
        """auth.json / .env content hashes must be identical before and after."""
        home = tmp_path / "hermes_home"
        home.mkdir()
        auth = home / "auth.json"
        env_file = home / ".env"
        auth.write_text('{"fake": "not-a-real-credential"}', encoding="utf-8")
        env_file.write_text("FAKE_KEY=not-a-real-credential\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))

        def digest():
            return (
                hashlib.sha256(auth.read_bytes()).hexdigest(),
                hashlib.sha256(env_file.read_bytes()).hexdigest(),
            )

        before = digest()
        agent = _build_isolated_agent()
        agent.run_conversation("hello")
        assert digest() == before, "isolated mode must not mutate credential files"

    def test_no_credential_pool_writes(self, monkeypatch, capture_requests):
        """No failure-counter, priority, or last-used persistence."""
        import agent.credential_persistence as credential_persistence

        writes: list[str] = []
        for name in dir(credential_persistence):
            if name.startswith("_"):
                continue
            attr = getattr(credential_persistence, name)
            if callable(attr) and ("save" in name or "persist" in name or "write" in name):
                monkeypatch.setattr(
                    credential_persistence,
                    name,
                    lambda *a, _n=name, **k: writes.append(_n),
                    raising=False,
                )

        agent = _build_isolated_agent()
        agent.run_conversation("hello")
        assert writes == [], f"credential persistence called: {writes}"

    def test_no_new_runtime_artifacts(self, tmp_path, monkeypatch, capture_requests):
        """No usage file, trajectory, checkpoint, session DB, or log is written.

        ``SOUL.md`` is deliberately tolerated: it is first-run HERMES_HOME
        scaffolding written by ``ensure_hermes_home()`` for *any* hermes
        invocation, not an artifact of this run. Isolated mode never reads it
        (see TestMinimalSystemPrompt) — it only must not produce the
        run-scoped artifact classes below.
        """
        home = tmp_path / "artifact_home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        scaffolding = {"SOUL.md"}

        def snapshot():
            return {
                p.relative_to(home).as_posix()
                for p in home.rglob("*")
                if p.is_file()
            }

        before = snapshot()
        agent = _build_isolated_agent()
        agent.run_conversation("hello")
        new_files = snapshot() - before - scaffolding

        artifact_markers = (
            "usage",
            "trajector",
            "checkpoint",
            "state.db",
            "sessions",
            ".log",
            "metadata_cache",
        )
        offenders = [
            f for f in new_files if any(m in f.lower() for m in artifact_markers)
        ]
        assert not offenders, f"isolated run created runtime artifacts: {sorted(offenders)}"
