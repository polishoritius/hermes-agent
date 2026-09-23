"""Regression tests: model picker must resolve provider aliases the same
way runtime_provider.py already does before deciding whether to surface
the current custom/local endpoint row.

Root cause: list_authenticated_providers() compared the raw
`current_provider` string against the literal "custom" everywhere it
decided to build the "Active bare custom endpoint from model config"
row (section 3b) -- so `provider: "ollama"` (a documented alias for
"custom" in providers.py's own ALIASES table, and already resolved
correctly by runtime_provider.py at actual inference time) never
matched, and the local endpoint's models never appeared in the picker.

Fix: normalize `current_provider` through providers.py's existing
`normalize_provider()` (the same ALIASES-table-backed helper other call
sites in this file already use) once, at the top of
list_authenticated_providers(), instead of comparing the raw string.
"""

import pytest

from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.providers import normalize_provider


@pytest.fixture(autouse=True)
def _disable_live_custom_provider_model_probe(monkeypatch):
    """Keep this deterministic and offline -- no real network/Ollama call."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids", lambda *_a, **_kw: []
    )
    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids", lambda *_a, **_kw: []
    )
    monkeypatch.setattr(
        "hermes_cli.models.cached_fetch_api_models",
        lambda *_a, **_kw: ["llama3.2:3b"],
    )


def _custom_row(rows: list[dict]) -> dict | None:
    return next((r for r in rows if r.get("slug") == "custom"), None)


class TestNormalizeProviderHelperIsTheExistingSsot:
    def test_ollama_normalizes_to_custom(self):
        """providers.py's own alias table already documents this mapping;
        this asserts the fix reuses it rather than a new one."""
        assert normalize_provider("ollama") == "custom"

    def test_vllm_and_llamacpp_normalize_consistently(self):
        assert normalize_provider("vllm") == "local"
        assert normalize_provider("llamacpp") == "local"


class TestA_OllamaAliasSurfacesLocalEndpoint:
    def test_ollama_alias_produces_current_custom_row(self):
        rows = list_authenticated_providers(
            current_provider="ollama",
            current_base_url="http://localhost:11434/v1",
            current_model="llama3.2:3b",
            user_providers={},
            custom_providers=[],
        )
        row = _custom_row(rows)
        assert row is not None, "expected a 'custom' row for provider='ollama'"
        assert row["is_current"] is True
        assert "llama3.2:3b" in row["models"]

    def test_llama_model_present_in_full_options_payload(self):
        rows = list_authenticated_providers(
            current_provider="ollama",
            current_base_url="http://localhost:11434/v1",
            current_model="llama3.2:3b",
            user_providers={},
            custom_providers=[],
        )
        all_models = [m for r in rows for m in (r.get("models") or [])]
        assert "llama3.2:3b" in all_models


class TestB_LiteralCustomProviderUnchanged:
    def test_literal_custom_provider_still_works(self):
        """Baseline behavior (already correct before this fix) must be preserved."""
        rows = list_authenticated_providers(
            current_provider="custom",
            current_base_url="http://localhost:11434/v1",
            current_model="llama3.2:3b",
            user_providers={},
            custom_providers=[],
        )
        row = _custom_row(rows)
        assert row is not None
        assert row["is_current"] is True
        assert "llama3.2:3b" in row["models"]


class TestC_AliasAndLiteralProduceEquivalentPayloads:
    def test_ollama_and_custom_produce_the_same_row_shape(self):
        common_kwargs = dict(
            current_base_url="http://localhost:11434/v1",
            current_model="llama3.2:3b",
            user_providers={},
            custom_providers=[],
        )
        rows_alias = list_authenticated_providers(current_provider="ollama", **common_kwargs)
        rows_literal = list_authenticated_providers(current_provider="custom", **common_kwargs)

        row_alias = _custom_row(rows_alias)
        row_literal = _custom_row(rows_literal)
        assert row_alias is not None and row_literal is not None
        assert row_alias == row_literal


class TestD_RefreshModelsKeepsLocalRow:
    def test_refresh_true_still_surfaces_the_local_row(self):
        rows = list_authenticated_providers(
            current_provider="ollama",
            current_base_url="http://localhost:11434/v1",
            current_model="llama3.2:3b",
            user_providers={},
            custom_providers=[],
            refresh=True,
        )
        row = _custom_row(rows)
        assert row is not None
        assert "llama3.2:3b" in row["models"]


class TestE_OtherProviderRowsUnaffected:
    def test_openrouter_row_unaffected_by_ollama_current_provider(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fictional-test-value")
        rows = list_authenticated_providers(
            current_provider="ollama",
            current_base_url="http://localhost:11434/v1",
            current_model="llama3.2:3b",
            user_providers={},
            custom_providers=[],
        )
        openrouter_row = next((r for r in rows if r.get("slug") == "openrouter"), None)
        assert openrouter_row is not None
        assert openrouter_row["is_current"] is False

    def test_vllm_alias_does_not_produce_a_custom_slug_row(self):
        """vllm/llamacpp alias to "local", a different canonical id than
        "custom" -- this fix must not conflate the two."""
        rows = list_authenticated_providers(
            current_provider="vllm",
            current_base_url="http://localhost:8000/v1",
            current_model="some-model",
            user_providers={},
            custom_providers=[],
        )
        assert _custom_row(rows) is None
