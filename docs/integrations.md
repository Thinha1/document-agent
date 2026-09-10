# Integrations

Tini separates model providers, retrieval sources, storage, observability, and
chat gateways. Configure them in `.env`; the dashboard's Connections page uses
the same registry as `.env.example`.

## Model providers

Set `TINI_PROVIDER` and that provider's key. Use `openai` for OpenAI and
`deepseek` for DeepSeek. For a proxy, local model server, or another compatible
vendor, select `custom_openai` and configure `CUSTOM_OPENAI_API_KEY`,
`CUSTOM_OPENAI_BASE_URL`, and `TINI_MODEL`. `TINI_SMALL_MODEL` is optional and
falls back to the main model. `TINI_BASE_URL` remains a legacy global override.
The custom adapter expects Chat Completions tool calling; `/models` is used only
for dashboard discovery. For a routing gateway such as Bifrost, set
`CUSTOM_OPENAI_UPSTREAM_PROVIDER` to its configured provider name. With
`TINI_MODEL=softdreams` and an upstream provider of `openai`, the request uses
`openai/softdreams`; direct compatible endpoints leave the setting blank.

## Retrieval

`search_web` requires `SEARXNG_BASE_URL`. It sends JSON searches only to that
self-hosted endpoint and has no public-provider fallback. Enable `json` in the
SearXNG `search.formats` setting.

For internal indexes, install `.[mcp]` and define servers in `.tini/mcp.json`.
Each server tool is registered as `<server>_<tool>`. Scope filesystem and
database servers to the minimum documents needed by the agent.

## Memory and traces

SQLite and FTS5 work without extra dependencies. Optional semantic backends
include Supabase and the memory-arena adapters; episodic memory can use Notion.
Set `OTEL_EXPORTER_OTLP_ENDPOINT=host:port` to export traces.

## Gateways

Telegram, Discord, WhatsApp, and voice are transport adapters around the same
retrieval agent. They do not add model-facing tools. Configure allowlists and
signature verification before exposing a gateway.
