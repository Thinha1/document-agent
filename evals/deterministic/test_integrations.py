"""Deterministic coverage for the shared Connections registry."""

from __future__ import annotations

import os
import socket

import pytest

from tini import integrations
from tini.integrations import IntegrationState, IntegrationStatus
from tini.loop.models import PROVIDERS


def _isolate(monkeypatch, tmp_path):
    # Keep find_dotenv(usecwd=True) from walking up into the repository .env.
    (tmp_path / ".env").write_text("")
    monkeypatch.setenv("TINI_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    integrations._HEALTH = None
    integrations._reset_import_cache()


def test_registry_contract():
    items = integrations.registry()
    assert len({item.key for item in items}) == len(items)
    assert {item.key for item in items if item.group == "AI Providers"} == set(PROVIDERS)
    assert {"searxng", "otel", "notion"} <= {item.key for item in items}
    assert not {"google_calendar", "apple_calendar", "apple_tools", "youtube"} & {
        item.key for item in items
    }
    for item in items:
        assert callable(item.enabled)
        for field in item.env:
            if field.kind is integrations.FieldKind.CHOICE:
                assert field.options
            if field.secret:
                assert field.kind is integrations.FieldKind.TEXT


def test_status_masking_health_persistence_and_invalidation(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-1234")
    view = next(view for view in integrations.list_integrations() if view.key == "openai")
    assert view.status.state is IntegrationState.CONFIGURED
    assert view.fields[0].value == ""
    assert view.fields[0].last4 == "1234"
    integrations.record_health("openai", IntegrationStatus(IntegrationState.CONNECTED))
    integrations._HEALTH = None
    current = next(view for view in integrations.list_integrations() if view.key == "openai")
    assert current.status.state is IntegrationState.CONNECTED
    integrations.invalidate_health("openai")
    after = next(view for view in integrations.list_integrations() if view.key == "openai")
    assert after.status.state is IntegrationState.CONFIGURED


def test_otel_probe_checks_configured_collector_tcp_port(monkeypatch):
    captured = {}

    class Connection:
        def close(self):
            captured["closed"] = True

    def connect(address, timeout):
        captured.update(address=address, timeout=timeout)
        return Connection()

    monkeypatch.setattr(socket, "create_connection", connect)
    integrations._otel_probe({"OTEL_EXPORTER_OTLP_ENDPOINT": "localhost:4317"})
    assert captured == {"address": ("localhost", 4317), "timeout": 3, "closed": True}


@pytest.mark.parametrize("endpoint", ["localhost", "localhost:4317/v1/traces"])
def test_otel_probe_rejects_invalid_endpoint(endpoint):
    with pytest.raises(ValueError, match="host:port"):
        integrations._otel_probe({"OTEL_EXPORTER_OTLP_ENDPOINT": endpoint})


def test_apply_rejects_unknown_and_can_clear_searxng(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("SEARXNG_BASE_URL=http://localhost:8080\n")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8080")
    assert not integrations.apply_integration("searxng", {"NOPE": "x"}).ok
    result = integrations.apply_integration(
        "searxng", {}, ("SEARXNG_BASE_URL",), force=True
    )
    assert result.ok
    assert "SEARXNG_BASE_URL" not in os.environ
    assert "SEARXNG_BASE_URL" not in (tmp_path / ".env").read_text()


def test_configured_searxng_is_not_reported_as_needing_setup(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8080")
    searxng = next(v for v in integrations.list_integrations() if v.key == "searxng")
    assert searxng.status.state is IntegrationState.CONFIGURED

    monkeypatch.delenv("SEARXNG_BASE_URL")
    integrations._HEALTH = None
    empty = next(v for v in integrations.list_integrations() if v.key == "searxng")
    assert empty.status.state is IntegrationState.NOT_CONFIGURED


def test_missing_required_field_reports_incomplete_setup(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(integrations, "_extra_installed", lambda name: True)
    notion = next(i for i in integrations.registry() if i.key == "notion")
    required = [field.name for field in notion.env if field.required]
    monkeypatch.setenv(required[0], "set")
    for name in required[1:]:
        monkeypatch.delenv(name, raising=False)
    view = next(v for v in integrations.list_integrations() if v.key == "notion")
    assert view.status.state is IntegrationState.INSTALLED_BUT_UNCONFIGURED
    assert required[1] in view.status.message
