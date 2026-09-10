# Tini Agent

Tini is a transparent, local-first agent harness focused on internal-document
retrieval. The model-facing tool surface stays intentionally small:

- `search_web` queries only an operator-controlled SearXNG instance.
- MCP servers add document stores and other retrieval adapters.
- The local memory layer provides SQLite/FTS5 retrieval and optional backends.

Write/action tools are outside the runtime scope; retrieval behavior stays
read-only and auditable.

## Quickstart

Tini requires Python 3.11 or newer.

```bash
git clone https://github.com/tini-coder/tini-agent
cd tini-agent
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
python -m tini
```

Configure one provider in `.env`. For OpenAI itself, use the named provider:

```dotenv
TINI_PROVIDER=openai
OPENAI_API_KEY=your-key
```

For a local server, proxy, or any other OpenAI-compatible service, use the
dedicated custom provider:

```dotenv
TINI_PROVIDER=custom_openai
CUSTOM_OPENAI_API_KEY=your-key
CUSTOM_OPENAI_BASE_URL=https://provider.example/v1
TINI_MODEL=your-model-id
# Only for routing gateways: the provider name configured in the gateway.
CUSTOM_OPENAI_UPSTREAM_PROVIDER=openai
# Optional; defaults to TINI_MODEL.
TINI_SMALL_MODEL=your-small-model-id
```

`custom_openai` is separate from the official `openai` provider. For a routing
gateway such as Bifrost, keep `TINI_MODEL` as the bare model ID and set
`CUSTOM_OPENAI_UPSTREAM_PROVIDER` to the provider configured in the gateway.
Tini then sends `provider/model` (for example `openai/softdreams`) on the wire.

The endpoint must implement OpenAI-style Chat Completions and function tool
calls. The dashboard also tries `GET <base-url>/models` to populate its picker;
manual model IDs still work when the catalog endpoint is unavailable.

DeepSeek remains a named provider with its own isolated settings:

```dotenv
TINI_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## Self-hosted search

Set the URL of your SearXNG deployment:

```dotenv
SEARXNG_BASE_URL=http://localhost:8080
```

The instance must allow JSON search responses. Tini calls
`GET /search?q=...&format=json` and has no public-provider fallback.

## MCP document sources

Install the optional dependency and create `.tini/mcp.json`:

```bash
pip install -e '.[mcp]'
```

```json
{
  "servers": [
    {
      "name": "docs",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/internal-docs"]
    }
  ]
}
```

MCP tools are namespaced as `<server>_<tool>` and join `search_web` in the
registry when the agent starts.

## Dashboard and graph routing

Run `python -m tini dashboard` (or `make dashboard`) and open
`http://localhost:9000`. The dashboard shows conversations, retrieval/tool
calls, memory, traces, model comparisons, integration health, and the live
triage topology.

Set `TINI_GRAPH_WORKFLOWS=1` to classify each message first. Greetings can use
the small model; document questions enter the normal retrieval loop. The graph
fails open to that loop if classification or execution fails.

## Evaluation

```bash
make eval          # deterministic offline suite
make eval-judge    # LLM-as-judge checks; requires a configured provider
make gate          # release gate
make lint          # Ruff checks
```

Retrieval benchmark cases live in `evals/dataset.jsonl`. Each case records the
expected tool, argument fragments, or an explicit no-tool expectation, making
it straightforward to add domain-specific RAG tests.

## Runtime data

Tini writes local state under `.tini/`: `state.db`, `SOUL.md`, traces, model
catalog data, and optional `mcp.json`. Keep credentials in `.env`; do not commit
runtime data or internal documents.

## License

MIT.
