# Architecture

Tini keeps document retrieval, model orchestration, memory, and evaluation as
separate modules with narrow interfaces.

```mermaid
flowchart TB
    GW["Gateways<br/>CLI · Telegram · Discord · WhatsApp · Voice"]
    SESSION["Session assembly<br/>instructions · skills · memory · recent history"]
    TRIAGE{"Optional triage graph<br/>quick or full?"}
    LOOP["Agent loop<br/>model → tool → model"]
    QUICK["Small-model reply"]
    MODEL["Provider adapter<br/>Anthropic or OpenAI-compatible"]
    REGISTRY["Tool registry"]
    SEARX["Self-hosted SearXNG"]
    DOCS[("SQLite document index<br/>metadata · chunks · FTS5")]
    INGEST["Ingestion pipeline<br/>discover · parse · chunk · upsert"]
    FILES["Local documents<br/>text · Markdown · HTML · PDF · DOCX"]
    MCP["MCP document sources"]
    MEMORY[("SQLite/FTS5 or optional memory backends")]
    TRACE["JSONL / OpenTelemetry traces"]
    OPS["Dashboard · Compare · Eval · Release gate"]

    GW --> SESSION --> TRIAGE
    TRIAGE -->|quick| QUICK --> GW
    TRIAGE -->|full| LOOP
    LOOP <--> MODEL
    LOOP <--> REGISTRY
    REGISTRY --> SEARX
    REGISTRY --> DOCS
    REGISTRY --> MCP
    FILES --> INGEST --> DOCS
    SESSION <--> MEMORY
    LOOP --> GW
    SESSION -. events .-> TRACE
    LOOP -. events .-> TRACE --> OPS
```

## Module Boundaries

- `tini/gateway/` transports text and calls the same runtime entry point.
- `tini/runtime/session.py` is the only context-assembly path.
- `tini/loop/` owns open-ended query refinement and provider wire formats.
- `tini/tools/` exposes `search_web` and optional namespaced MCP tools.
- `tini/documents/` owns parsing, deterministic chunking, provenance, and the
  `DocumentIndex` seam used by both ingestion and `search_documents`.
- `tini/memory/` owns retrieval gating, stores, skills, and consolidation.
- `tini/graph/` owns inspectable routing around the loop.
- `tini/ops/` and `evals/` observe and score behavior without becoming runtime
  dependencies of the loop.

## Design Decisions

The tool surface is intentionally retrieval-only. SearXNG gives operators
control over federated search, while MCP is the extension seam for filesystems,
vector databases, and internal APIs. Missing configuration and empty retrieval
results are returned honestly; there is no public search fallback.

The loop remains the RAG core because the model may need an unknown number of
searches. The graph handles only stable control flow. Its shipped triage workflow
routes social messages to a small model and knowledge questions to the unchanged
loop, failing open to the full path on errors.

SQLite, FTS5, and JSONL provide zero-service defaults. The document index
interface keeps backend-specific query syntax hidden, so a PostgreSQL/pgvector
adapter can replace SQLite without changing the pipeline or model tool.
Optional backends and telemetry integrations sit behind extras or environment
configuration. Deterministic tests pin routing and tool behavior; judge
evaluations score grounded answer quality.

For state details, see [../ARCHITECHURE.md](../ARCHITECHURE.md). For prompt and
retrieval context, see [../CONTEXT-ARCHITECHURE.md](../CONTEXT-ARCHITECHURE.md).
