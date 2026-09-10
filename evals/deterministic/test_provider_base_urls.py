"""Regional Base URL choices for providers with separate domestic/global APIs."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tini import integrations
from tini.config import Settings
from tini.loop import models
from tini.loop.models import PROVIDERS
from tini.ops import catalog

EXPECTED_ENDPOINTS = {
    "minimax": [
        (
            "China",
            "https://api.minimaxi.com/anthropic",
            "https://api.minimaxi.com/anthropic/v1/models",
        ),
        (
            "Global",
            "https://api.minimax.io/anthropic",
            "https://api.minimax.io/anthropic/v1/models",
        ),
    ],
    "glm": [
        ("Global", "https://api.z.ai/api/anthropic", None),
        ("China", "https://open.bigmodel.cn/api/anthropic", None),
    ],
    "kimi": [
        ("Global", "https://api.moonshot.ai/anthropic", "https://api.moonshot.ai/v1/models"),
        ("China", "https://api.moonshot.cn/anthropic", "https://api.moonshot.cn/v1/models"),
    ],
}


def test_regional_provider_endpoint_choices_match_official_urls():
    for name, expected in EXPECTED_ENDPOINTS.items():
        provider = PROVIDERS[name]
        assert [(e.label, e.base_url, e.catalog_url) for e in provider.endpoints] == expected
        assert provider.base_url_env


@pytest.mark.parametrize(
    ("provider_name", "base_url_env", "selected_url"),
    [
        ("minimax", "MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"),
        ("glm", "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/anthropic"),
        ("kimi", "MOONSHOT_BASE_URL", "https://api.moonshot.cn/anthropic"),
    ],
)
def test_client_uses_provider_specific_base_url(
    monkeypatch, tmp_path, provider_name, base_url_env, selected_url
):
    captured = {}

    class StubAnthropicClient:
        def __init__(self, *, api_key, base_url, timeout):
            captured.update(api_key=api_key, base_url=base_url, timeout=timeout)

    provider = PROVIDERS[provider_name]
    monkeypatch.setenv(provider.key_env, "test-key")
    monkeypatch.setenv(base_url_env, selected_url)
    monkeypatch.delenv("TINI_BASE_URL", raising=False)
    monkeypatch.setitem(
        __import__("sys").modules,
        "anthropic",
        SimpleNamespace(Anthropic=StubAnthropicClient),
    )

    client = models.get_client(
        Settings(
            provider=provider_name,
            api_key="",
            base_url=None,
            model="",
            small_model="",
            home=tmp_path,
        )
    )

    assert isinstance(client, StubAnthropicClient)
    assert captured["base_url"] == selected_url


@pytest.mark.parametrize(
    ("provider_name", "base_url_env", "selected_url", "catalog_url"),
    [
        (
            "minimax",
            "MINIMAX_BASE_URL",
            "https://api.minimax.io/anthropic",
            "https://api.minimax.io/anthropic/v1/models",
        ),
        (
            "kimi",
            "MOONSHOT_BASE_URL",
            "https://api.moonshot.cn/anthropic",
            "https://api.moonshot.cn/v1/models",
        ),
    ],
)
def test_catalog_follows_selected_provider_region(
    monkeypatch, provider_name, base_url_env, selected_url, catalog_url
):
    captured = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        return Response(json.dumps({"data": [{"id": "model-one"}]}).encode())

    provider = PROVIDERS[provider_name]
    monkeypatch.setenv("TINI_PROVIDER", provider_name)
    monkeypatch.setenv(provider.key_env, "test-key")
    monkeypatch.setenv(base_url_env, selected_url)
    monkeypatch.delenv("TINI_BASE_URL", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    catalog._models_cache.clear()

    result = catalog.list_models(provider_name, use_cache=False)

    assert result["listed"] is True
    assert captured["url"] == catalog_url


def test_provider_view_exposes_base_url_choice(monkeypatch):
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")

    view = next(v for v in integrations.list_providers() if v.key == "minimax")
    base = next(field for field in view.fields if field.name == "MINIMAX_BASE_URL")

    assert base.kind is integrations.FieldKind.CHOICE
    assert base.value == "https://api.minimax.io/anthropic"
    assert base.options == (
        "https://api.minimaxi.com/anthropic",
        "https://api.minimax.io/anthropic",
    )
    assert base.option_labels == ("China", "Global")


@pytest.mark.parametrize(
    ("provider_name", "base_url_env"),
    [
        ("openai", "OPENAI_BASE_URL"),
        ("deepseek", "DEEPSEEK_BASE_URL"),
        ("custom_openai", "CUSTOM_OPENAI_BASE_URL"),
    ],
)
def test_openai_compatible_provider_uses_scoped_base_url(
    monkeypatch, tmp_path, provider_name, base_url_env
):
    captured = {}

    class StubOpenAICompatClient:
        def __init__(self, *, api_key, base_url, timeout, model_prefix=""):
            captured.update(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                model_prefix=model_prefix,
            )

    provider = PROVIDERS[provider_name]
    monkeypatch.setenv(provider.key_env, "test-key")
    monkeypatch.setenv(base_url_env, "https://compatible.example/v1")
    monkeypatch.delenv("TINI_BASE_URL", raising=False)
    monkeypatch.setattr(models, "OpenAICompatClient", StubOpenAICompatClient)

    client = models.get_client(
        Settings(
            provider=provider_name,
            api_key="",
            base_url=None,
            model="compatible-model" if provider_name == "custom_openai" else "",
            small_model="",
            home=tmp_path,
        )
    )

    assert isinstance(client, StubOpenAICompatClient)
    assert captured["base_url"] == "https://compatible.example/v1"


def test_custom_provider_uses_openai_wire_and_main_model_for_small_calls(monkeypatch, tmp_path):
    captured = {}

    class StubOpenAICompatClient:
        def __init__(self, *, api_key, base_url, timeout, model_prefix=""):
            captured.update(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                model_prefix=model_prefix,
            )

    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "custom-key")
    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "https://llm.internal.example/v1")
    monkeypatch.setenv("CUSTOM_OPENAI_UPSTREAM_PROVIDER", "openai")
    monkeypatch.setenv("TINI_BASE_URL", "https://stale-global.example/v1")
    monkeypatch.setattr(models, "OpenAICompatClient", StubOpenAICompatClient)
    settings = Settings(
        provider="custom_openai",
        api_key="",
        base_url=None,
        model="internal-chat-model",
        small_model="",
        home=tmp_path,
    )

    client = models.get_client(settings)

    assert isinstance(client, StubOpenAICompatClient)
    assert captured == {
        "api_key": "custom-key",
        "base_url": "https://llm.internal.example/v1",
        "timeout": 120.0,
        "model_prefix": "openai",
    }
    assert settings.small_model == "internal-chat-model"


def test_custom_provider_requires_base_url_and_model(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "custom-key")
    monkeypatch.delenv("CUSTOM_OPENAI_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="CUSTOM_OPENAI_BASE_URL"):
        models.get_client(
            Settings(
                provider="custom_openai",
                api_key="",
                base_url=None,
                model="model-a",
                small_model="",
                home=tmp_path,
            )
        )

    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "https://llm.example/v1")
    with pytest.raises(SystemExit, match="TINI_MODEL"):
        models.get_client(
            Settings(
                provider="custom_openai",
                api_key="",
                base_url=None,
                model="",
                small_model="",
                home=tmp_path,
            )
        )


def test_provider_view_exposes_custom_base_url_input(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://compatible.example/v1")

    view = next(v for v in integrations.list_providers() if v.key == "openai")
    base = next(field for field in view.fields if field.name == "OPENAI_BASE_URL")

    assert base.kind is integrations.FieldKind.TEXT
    assert base.value == "https://compatible.example/v1"
    assert base.options == ()


def test_custom_provider_view_requires_compatible_base_url(monkeypatch):
    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "https://llm.internal.example/v1")
    monkeypatch.setenv("CUSTOM_OPENAI_UPSTREAM_PROVIDER", "openai")

    view = next(v for v in integrations.list_providers() if v.key == "custom_openai")
    base = next(field for field in view.fields if field.name == "CUSTOM_OPENAI_BASE_URL")
    upstream = next(
        field for field in view.fields if field.name == "CUSTOM_OPENAI_UPSTREAM_PROVIDER"
    )

    assert view.name == "Custom OpenAI-Compatible"
    assert base.required is True
    assert base.value == "https://llm.internal.example/v1"
    assert upstream.value == "openai"


def test_custom_openai_is_distinct_from_official_openai():
    official = PROVIDERS["openai"]
    compatible = PROVIDERS["custom_openai"]
    views = {view.key: view for view in integrations.list_providers()}

    assert official.key_env == "OPENAI_API_KEY"
    assert official.requires_base_url is False
    assert compatible.key_env == "CUSTOM_OPENAI_API_KEY"
    assert compatible.requires_base_url is True
    assert compatible.model_prefix_env == "CUSTOM_OPENAI_UPSTREAM_PROVIDER"
    assert views["openai"].name == "OpenAI"
    assert views["custom_openai"].name == "Custom OpenAI-Compatible"


def test_custom_provider_modal_submits_models_before_activation():
    source = Path("tini/ops/static/js/models.js").read_text()

    assert 'const needsModels = current || provider === "custom_openai"' in source
    assert "if (needsModels)" in source
    assert 'payload.model = document.getElementById("pm-model")?.value' in source


def test_apply_custom_provider_persists_isolated_credentials_and_models(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("")
    monkeypatch.setenv("TINI_HOME", str(tmp_path / ".tini"))
    monkeypatch.setenv("TINI_PROVIDER", "anthropic")
    monkeypatch.delenv("CUSTOM_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("CUSTOM_OPENAI_UPSTREAM_PROVIDER", raising=False)
    monkeypatch.delenv("TINI_BASE_URL", raising=False)
    monkeypatch.setattr(integrations, "_provider_probe", lambda values: None)

    from tini.ops import browser_agent

    monkeypatch.setattr(browser_agent, "current", lambda: None)
    result = integrations.apply_provider(
        "custom_openai",
        key="custom-key",
        base_url="https://llm.internal.example/v1",
        upstream_provider="openai",
        model="main-model",
        small_model="small-model",
    )

    assert result.ok
    assert os.environ["TINI_PROVIDER"] == "custom_openai"
    assert os.environ["CUSTOM_OPENAI_API_KEY"] == "custom-key"
    assert os.environ["CUSTOM_OPENAI_BASE_URL"] == "https://llm.internal.example/v1"
    assert os.environ["CUSTOM_OPENAI_UPSTREAM_PROVIDER"] == "openai"
    assert os.environ["TINI_MODEL"] == "main-model"
    assert os.environ["TINI_SMALL_MODEL"] == "small-model"
    assert os.environ.get("TINI_BASE_URL", "") == ""
    contents = (tmp_path / ".env").read_text()
    assert "CUSTOM_OPENAI_BASE_URL='https://llm.internal.example/v1'" in contents
    assert "CUSTOM_OPENAI_UPSTREAM_PROVIDER='openai'" in contents


def test_custom_openai_endpoint_owns_its_model_catalog(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://compatible.example/v1")

    provider = PROVIDERS["openai"]

    assert provider.catalog_for(provider.configured_base_url()) is None


def test_apply_provider_persists_scoped_base_url_and_clears_legacy_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TINI_BASE_URL='https://legacy.example'\n")
    monkeypatch.setenv("TINI_HOME", str(tmp_path / ".tini"))
    monkeypatch.setenv("TINI_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("TINI_BASE_URL", "https://legacy.example")
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
    monkeypatch.setattr(integrations, "_provider_probe", lambda values: None)

    from tini.ops import browser_agent

    monkeypatch.setattr(browser_agent, "rebuild", lambda: None)
    tracer = SimpleNamespace(event=lambda *args: None)
    monkeypatch.setattr(browser_agent, "current", lambda: SimpleNamespace(tracer=tracer))

    result = integrations.apply_provider("minimax", base_url="https://api.minimax.io/anthropic")

    assert result.ok
    assert os.environ["MINIMAX_BASE_URL"] == "https://api.minimax.io/anthropic"
    assert os.environ.get("TINI_BASE_URL", "") == ""
    contents = Path(".env").read_text()
    assert "MINIMAX_BASE_URL='https://api.minimax.io/anthropic'" in contents


def test_models_edit_modal_posts_selected_base_url():
    source = Path("tini/ops/static/js/models.js").read_text()

    assert 'id="pm-base-url"' in source
    assert 'id="pm-upstream-provider"' in source
    assert "payload.base_url = baseUrl" in source
    assert "payload.upstream_provider = upstream.value" in source
    assert "payload.activate = false" in source


@pytest.mark.parametrize(
    ("current_provider", "expected_rebuilds"),
    [
        ("kimi", 0),
        ("minimax", 1),
    ],
)
def test_reusing_saved_provider_endpoint_skips_remote_probe_and_noop_rebuild(
    monkeypatch, tmp_path, current_provider, expected_rebuilds
):
    """Clicking either modal action with unchanged credentials stays local."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("")
    monkeypatch.setenv("TINI_HOME", str(tmp_path / ".tini"))
    monkeypatch.setenv("TINI_PROVIDER", current_provider)
    monkeypatch.setenv("MOONSHOT_API_KEY", "saved-key")
    monkeypatch.setenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/anthropic")
    monkeypatch.delenv("TINI_BASE_URL", raising=False)

    probes = []
    monkeypatch.setattr(integrations, "_provider_probe", lambda values: probes.append(values))

    from tini.ops import browser_agent

    rebuilds = []
    monkeypatch.setattr(browser_agent, "rebuild", lambda: rebuilds.append(True))
    tracer = SimpleNamespace(event=lambda *args: None)
    monkeypatch.setattr(browser_agent, "current", lambda: SimpleNamespace(tracer=tracer))

    result = integrations.apply_provider("kimi", base_url="https://api.moonshot.cn/anthropic")

    assert result.ok
    assert probes == []
    assert len(rebuilds) == expected_rebuilds


def test_saving_noncurrent_provider_does_not_activate_or_rebuild(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("")
    monkeypatch.setenv("TINI_HOME", str(tmp_path / ".tini"))
    monkeypatch.setenv("TINI_PROVIDER", "minimax")
    monkeypatch.setenv("MOONSHOT_API_KEY", "saved-key")
    monkeypatch.setenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/anthropic")
    monkeypatch.delenv("TINI_BASE_URL", raising=False)
    monkeypatch.setattr(integrations, "_provider_probe", lambda values: None)

    from tini.ops import browser_agent

    rebuilds = []
    monkeypatch.setattr(browser_agent, "rebuild", lambda: rebuilds.append(True))

    result = integrations.apply_provider(
        "kimi",
        base_url="https://api.moonshot.cn/anthropic",
        activate=False,
    )

    assert result.ok
    assert os.environ["TINI_PROVIDER"] == "minimax"
    assert os.environ["MOONSHOT_BASE_URL"] == "https://api.moonshot.cn/anthropic"
    assert rebuilds == []
