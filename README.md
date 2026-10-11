# Document Internal Agent

**English** · [Tiếng Việt](README.vi.md)

A local-first assistant that answers questions from your organisation's internal
documents, and tells you **which document and which page** each answer came from.
It is built to be read end to end: one request can be followed from the gateway
to the answer in an afternoon.

> **Naming note.** The project was previously called *Tini Agent*. The Python
> package, the `tini` CLI and the `TINI_*` settings keep the old name for now, so
> existing `.env` files and commands keep working.

## What the agent does today

| Capability | What it means in practice |
|---|---|
| **Ingests internal documents** | PDF, DOCX, Markdown, plain text, HTML, CSV, JSON and YAML, from a file, a folder or a dashboard upload. Re-ingesting is idempotent (SHA-256 content hash); a changed file replaces its old chunks atomically. |
| **Reads scanned PDFs** | Pages with almost no text are rendered and transcribed by a vision model (OCR), with a cache so each page is paid for once. |
| **Searches by meaning *and* by exact words** | Hybrid retrieval in Qdrant: dense `bge-m3` embeddings plus a sparse keyword vector, fused with reciprocal-rank fusion. |
| **Re-ranks locally** | A self-hosted `bge-reranker-v2-m3` (text-embeddings-inference) re-scores the top 50 candidates. Document text is only ever sent to a local address; any other reranker URL is refused at startup. |
| **Works across English, Japanese and Vietnamese** | Multilingual embeddings, CJK-aware chunking and keyword search, and an optional step that reads a Vietnamese question in English for the reranker. |
| **Keeps tables intact** | Tables are chunked by row with the heading, caption and column header repeated on every chunk, so a bare row of numbers still carries its meaning. |
| **Cites sources with page numbers** | Answers reference `[title#chunk, pp.A-B]`. Page ranges are stored for every PDF chunk. |
| **Hides stale documents** | Superseded or excluded documents are filtered out of search; a catalog tracks versions and status. |
| **Answers in the user's language** | The reply language follows the question. |
| **Searches the web only through your SearXNG** | `search_web` talks to the configured `SEARXNG_BASE_URL` and nothing else; it returns an honest error rather than falling back to a public provider. |
| **Plugs into other document systems** | Any MCP server (file systems, wikis, drives) is added to the tool registry. |
| **Remembers** | SQLite/FTS5 memory with optional Mem0, Zep, LangMem, Supabase or Notion backends, plus editable skills and a persona file. |
| **Talks through several channels** | CLI, Telegram, Discord, WhatsApp and voice gateways. Gateways only move text; retrieval policy stays in the runtime. |
| **Shows its work** | A dashboard with conversations, tool calls, traces, a knowledge manager (upload, inspect chunks, probe retrieval), model comparison and evaluation runs. |

**By design it does not do:** write to or act on your systems. The built-in tools
are retrieval-only, and retrieved text and community skills are treated as
untrusted input.

## Current results

Retrieval results measured on the DENSO corpus (workspace `denso`, 26 documents,
1,144 chunks), split into the two kinds of documents it holds. Setting: Qdrant hybrid (dense `bge-m3` + keyword vector, RRF,
keyword weight 0.5), local `bge-reranker-v2-m3` over 50 candidates.

| Question group | Questions | Right document found | Expected answer text in the returned chunks |
|---|---|---|---|
| **Administrative documents** (securities and annual reports, governance, compliance, human rights, supply chain, QA manual) | 46 | 40 (87%) | 37 (80%) |
| **Technical documents** (AUTOSAR CAN Driver specification, DENSO spark-plug manual) | 24 | 24 (100%) | 22 (92%) |
| **All, in English, Japanese and Vietnamese** | 70 | 64 (91%) | 59 (84%) |

By language, answer found: Japanese 29/30, English 22/24, Vietnamese 8/16. Administrative:
Japanese 27/28, English 4/6, Vietnamese 6/12. Technical: English 18/18, Japanese 2/2,
Vietnamese 2/4. The technical group is `evals/documents/denso_qa_tech.jsonl` (also included
in `denso_qa_multilingual.jsonl`, which is where the three rows come from), so the rows are
not independent runs. Median latency is about 1.2 s per query (p95 about 1.4 s), measured in
the dashboard on the 108-question synthetic set with the same configuration.

"Expected answer text in the returned chunks" is stricter than "right document found". The
sets are small (the technical group has only two Japanese and four Vietnamese questions) and
were written by the people who chose the documents, so read them as a direction, not as a
benchmark. Method and the full history of experiments:
[docs/document-ingestion.md](docs/document-ingestion.md), [evals/baselines/](evals/baselines/).

## Known limitations

- Vietnamese questions are the weak spot (answer found for 8 of 16 in the result
  above): the reranker scores them poorly against English chunks. Reading the question
  in English for the reranker (`TINI_RERANK_TRANSLATE=vi`, off by default, needs a fast
  small model) was re-tried on 2026-10-11 and gave the identical 8 / 16, so it is not
  part of the result above; an earlier run on a different question set had shown a gain.
- Rows of a table inside a text-layer PDF arrive single-space separated, so
  row-aware chunking does not apply to them.
- Page numbers exist for PDFs only.
- The evaluation sets are small and were written by the corpus authors; scale
  beyond a few dozen documents is unmeasured.
- The retrieval baselines in `evals/baselines/` predate the merged `denso`
  workspace and need re-running.
- Two question sets (`denso_qa_tech.jsonl` and `denso_qa_multilingual.jsonl`)
  overlap; running both double-counts 24 questions.

## How a question is answered

```text
channel (CLI / Telegram / Discord / WhatsApp / voice)
   -> Tini.respond()            tini/runtime/session.py   instructions, skills, memory, history
   -> triage (optional graph)   tini/graph/               greeting -> small model, knowledge -> full loop
   -> agent loop                tini/loop/agent.py        model <-> tool calls
        -> search_documents     tini/tools/documents.py   hybrid search -> rerank -> cited chunks
        -> search_web           tini/tools/search.py      your SearXNG only
        -> MCP tools            tini/tools/mcp_client.py
   -> answer with sources and page numbers
```

Ingestion path: discover files -> extract text (OCR for scanned pages, page markers
for PDFs) -> structure-aware chunking -> embed -> upsert to Qdrant (or SQLite) with
metadata. See [docs/document-ingestion.md](docs/document-ingestion.md).

## Quickstart

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Thinha1/document-agent
cd document-agent
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
python -m tini
```

Configure one chat provider in `.env`:

```dotenv
TINI_PROVIDER=openai
OPENAI_API_KEY=your-key
```

For a local server, proxy or any OpenAI-compatible service use the dedicated
provider:

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

`custom_openai` is separate from the official `openai` provider. With a routing
gateway such as Bifrost, keep `TINI_MODEL` as the bare model ID and set
`CUSTOM_OPENAI_UPSTREAM_PROVIDER`; the wire value becomes `provider/model`. The
endpoint must support Chat Completions with function tool calls. The dashboard
tries `GET <base-url>/models` to fill its model picker; manual IDs still work.
DeepSeek and other named providers keep their own isolated settings.

### Full self-hosted stack (Docker)

`docker-compose.yml` runs the agent with Qdrant (vector index), Ollama
(`bge-m3` embeddings), an optional local reranker and optional MinIO storage:

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec ollama ollama pull bge-m3          # first time only
docker compose --profile reranker up -d                # adds the bge-reranker (~2.2 GB first start)
docker compose run --rm tini python -m tini ingest examples/denso_mock_docs --workspace denso
```

The dashboard is at `http://localhost:9000` (change the published port in the
git-ignored `docker-compose.override.yml` if it is taken). An NVIDIA card is used
through the same override file; see the comments in `docker-compose.yml`.

## Document ingestion

```bash
python -m tini ingest ./internal-docs                        # recursive; --no-recursive to stop at one level
python -m tini ingest ./internal-docs --workspace engineering
python -m tini documents                                     # list the library
python -m tini documents --search "incident recovery"
```

For a visual workflow run `tini dashboard` and open **Knowledge**: drop files or a
folder, watch the real ingestion stages, inspect chunks and page ranges, probe
retrieval and run the retrieval and generation evals. Uploads are copied to
`.tini/documents/<workspace>/`; removing a library entry keeps that source so it
can be re-indexed.

Settings that matter most (all in `.env.example`; each is documented with its
measurements in [docs/document-ingestion.md](docs/document-ingestion.md)):

```dotenv
TINI_DOCUMENT_INDEX=qdrant            # sqlite is the lightweight default
TINI_DOCUMENT_HYBRID=1                # dense + keyword, fused with RRF
TINI_EMBEDDING_PROVIDER=openai_compatible   # e.g. bge-m3 served by Ollama
TINI_RERANK_PROVIDER=tei              # local bge-reranker-v2-m3; localhost only
TINI_RERANK_TRANSLATE=vi              # read Vietnamese questions in English for the reranker
TINI_TABLE_CHUNKING=1                 # repeat table headers on every chunk (default on)
```

The embedding model is configured separately from the chat model, so a custom
OpenAI-compatible chat model can use, for example, Gemini or `bge-m3` embeddings.
Re-index documents after changing the embedding model, its dimensions or the
chunking settings. If the embedding endpoint is down at query time, the SQLite
backend keeps working through FTS5.

## Self-hosted web search

```dotenv
SEARXNG_BASE_URL=http://localhost:8080
```

## MCP document sources

```bash
pip install -e '.[mcp]'
```

Create `.tini/mcp.json`:

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

MCP tools are namespaced `<server>_<tool>` and join the registry at startup.

## Dashboard and graph routing

Run `tini dashboard` (or `make dashboard`). It shows conversations, retrieval and
tool calls, memory, traces, model comparisons, integration health, the live triage
topology and the **Knowledge** manager. Set `TINI_GRAPH_WORKFLOWS=1` to classify
each message first: greetings use the small model, document questions enter the
full retrieval loop, and the graph fails open to that loop on any error.

## Evaluation

```bash
make eval          # deterministic, network-free suite
make eval-judge    # LLM-as-judge checks; needs a configured provider
make gate          # release gate
make lint          # Ruff
python scripts/document_baseline.py my-change --compare evals/baselines/<file>.json
python scripts/generation_eval.py run --label flash --provider <p> --model <m> --judge-provider <p2> --judge-model <m2>
```

Retrieval sets live in `evals/documents/`, generation sets in `evals/generation/`
and tool-use cases in `evals/dataset.jsonl`. A runtime bug needs a regression case
under `evals/deterministic/`; subjective answer quality belongs under `evals/judge/`.
Use a judge model different from the generator.

## Runtime data and security

Local state lives under `.tini/` (`state.db`, traces, OCR cache, managed uploads,
`SOUL.md`, optional `mcp.json`); credentials live in `.env`. Both are git-ignored.
Never commit internal documents. Retrieved text and community `SKILL.md` files are
untrusted input; MCP roots are not broadened and credentials are never exposed to
the model. See [SECURITY.md](SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

[docs/README.md](docs/README.md) indexes the architecture, ingestion, evaluation and
memory references.

## License

MIT.
