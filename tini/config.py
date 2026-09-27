"""Configuration — every knob is an env var, documented in .env.example.

No settings framework: a dataclass read once at startup. If you can read this
file, you know everything Tini can be configured to do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def _load_env() -> str:
    """Find the user's .env the way the user expects: from where they ARE.

    Bare `load_dotenv()` searches upward from the file that called it, not from
    the working directory. Inside a git checkout that is invisible — config.py
    lives in the project, so walking up from it lands on the project's .env and
    everything works. Installed from PyPI it walks up from site-packages,
    reaches the filesystem root, and finds nothing: the user stands in a folder
    holding a perfectly good .env and Tini reports "No API key". Reported from
    a clean install on 2026-07-31, from inside the repo folder itself.

    usecwd=True is the whole fix. The walk upward is kept on purpose, so
    running `tini` from a subdirectory of your project still finds the .env at
    its root — the same rule git, npm and pytest already taught everyone.

    Returns the path that was loaded (empty string if none) so `tini doctor`
    and the first-run error can say WHICH file was read, rather than leaving
    people guessing between three .env files.
    """
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)
    return path


DOTENV_PATH = _load_env()


def _embedding_api_key() -> str:
    """Resolve a dedicated embedding key before a provider-key fallback."""
    configured = os.getenv("TINI_EMBEDDING_API_KEY", "").strip()
    if configured:
        return configured
    provider = os.getenv("TINI_EMBEDDING_PROVIDER", "none").strip().lower()
    fallback = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    return os.getenv(fallback, "").strip() if fallback else ""


def _ocr_api_key() -> str:
    """Resolve a dedicated OCR key before a provider-key fallback — same
    shape as _embedding_api_key, for the same reason: OCR is a separate
    concern from chat generation and may use a different account/model."""
    configured = os.getenv("TINI_OCR_API_KEY", "").strip()
    if configured:
        return configured
    provider = os.getenv("TINI_OCR_PROVIDER", "none").strip().lower()
    fallback = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    return os.getenv(fallback, "").strip() if fallback else ""


# Known-provider default endpoints — the openai SDK defaults to OpenAI's own
# api.openai.com when base_url is left None, so a bare TINI_OCR_PROVIDER=
# deepseek with no base_url would silently send the request (and the
# provider's key) to OpenAI instead and fail loudly with an auth error there.
# Found exactly that way: DEEPSEEK_API_KEY sent to OpenAI, 401 from
# platform.openai.com. Mirrors PROVIDERS["deepseek"].base_url in
# tini/loop/models.py.
_OCR_PROVIDER_DEFAULT_BASE_URL = {"deepseek": "https://api.deepseek.com"}


def _ocr_base_url() -> str | None:
    configured = os.getenv("TINI_OCR_BASE_URL", "").strip()
    if configured:
        return configured
    provider = os.getenv("TINI_OCR_PROVIDER", "none").strip().lower()
    # DEEPSEEK_BASE_URL lets an operator already using DeepSeek for chat
    # override the endpoint (e.g. a proxy) once and have OCR follow it too.
    env_override = os.getenv("DEEPSEEK_BASE_URL", "").strip() if provider == "deepseek" else ""
    return env_override or _OCR_PROVIDER_DEFAULT_BASE_URL.get(provider)


def _rerank_api_key() -> str:
    """Dedicated key first, then the provider's own conventional variable.
    There is deliberately no base-URL setting: a reranker receives document
    text, so its destination is fixed in tini/documents/rerank.py rather than
    configurable."""
    configured = os.getenv("TINI_RERANK_API_KEY", "").strip()
    if configured:
        return configured
    provider = os.getenv("TINI_RERANK_PROVIDER", "none").strip().lower()
    fallback = {"jev": "TYPESAFE_API_KEY"}.get(provider)
    return os.getenv(fallback, "").strip() if fallback else ""


@dataclass
class Settings:
    # --- LLM: pick a provider, set its key. See tini/loop/models.py PROVIDERS.
    provider: str = field(default_factory=lambda: os.getenv("TINI_PROVIDER", "anthropic"))
    # Explicit overrides (optional): key, endpoint, and model ids. Left empty,
    # the provider's own key env var and default models are used.
    api_key: str = field(default_factory=lambda: os.getenv("TINI_API_KEY", ""))
    base_url: str | None = field(default_factory=lambda: os.getenv("TINI_BASE_URL") or None)
    model: str = field(default_factory=lambda: os.getenv("TINI_MODEL", ""))
    # Cheap model used by the retrieval gate and the consolidation summarizer.
    small_model: str = field(default_factory=lambda: os.getenv("TINI_SMALL_MODEL", ""))
    # Embeddings are a separate concern from generation: the chat model may be
    # DeepSeek/custom OpenAI-compatible while retrieval uses Gemini or another
    # OpenAI-compatible embedding endpoint. These settings are independent of
    # the chat provider and drive document indexing plus vector retrieval.
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("TINI_EMBEDDING_PROVIDER", "none").strip().lower()
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("TINI_EMBEDDING_MODEL", "").strip()
    )
    embedding_api_key: str = field(default_factory=_embedding_api_key)
    embedding_base_url: str | None = field(
        default_factory=lambda: os.getenv("TINI_EMBEDDING_BASE_URL", "").strip() or None
    )
    embedding_dimensions: int = field(
        default_factory=lambda: int(os.getenv("TINI_EMBEDDING_DIMENSIONS", "768"))
    )
    # OCR fallback for image-only/scanned PDF pages (tini/documents/ocr.py) —
    # same independent-of-chat-provider shape as embeddings above. Unset by
    # default: a page that looks scanned is left as-is (see loaders.py) rather
    # than silently calling an API nobody configured.
    ocr_provider: str = field(
        default_factory=lambda: os.getenv("TINI_OCR_PROVIDER", "none").strip().lower()
    )
    ocr_model: str = field(default_factory=lambda: os.getenv("TINI_OCR_MODEL", "").strip())
    ocr_api_key: str = field(default_factory=_ocr_api_key)
    ocr_base_url: str | None = field(default_factory=_ocr_base_url)
    # Chunk vectors are cached on disk by (provider, model, title, text), so
    # re-ingesting an unchanged or lightly edited document only embeds the
    # chunks whose text is new (tini/documents/embeddings.py). On by default;
    # TINI_EMBEDDING_CACHE=0 turns it off.
    embedding_cache: bool = field(
        default_factory=lambda: os.getenv("TINI_EMBEDDING_CACHE", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # A scanned page's OCR call is a multi-second network round trip; pages
    # within one document are independent, so running several at once is a
    # straightforward win. Rendering itself (pypdfium2) stays single-threaded
    # — libpdfium is not thread-safe for concurrent page access — only the
    # OCR request is parallelized. 4 is a conservative default against a
    # hosted vision API's own rate limits.
    ocr_concurrency: int = field(
        default_factory=lambda: int(os.getenv("TINI_OCR_CONCURRENCY", "4"))
    )
    # Off by default: not yet measured against the single-prompt OCR flow (see
    # docs/document-ingestion.md). When on, a scanned page's transcription skips
    # tables (leaving a `[[TABLE:n]]` marker) and a second, table-only call
    # extracts them as structured rows instead of a Markdown blob the chunker
    # cannot tell apart from prose — see tini/documents/ocr.py and chunking.py.
    # The second call only runs on pages a marker was actually found on.
    ocr_table_flow: bool = field(
        default_factory=lambda: os.getenv("TINI_OCR_TABLE_FLOW", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    # Where dashboard-uploaded ORIGINAL files are kept (tini/documents/
    # blobstore.py): 'local' (default, under TINI_HOME) or 'minio' (any
    # S3-compatible server the operator points at). The index itself is
    # unaffected — this only decides where the source file lives, and is
    # recorded on the document as its s3:// source_uri.
    document_storage: str = field(
        default_factory=lambda: os.getenv("TINI_DOCUMENT_STORAGE", "local").strip().lower()
    )
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("TINI_MINIO_ENDPOINT", "localhost:9000").strip()
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("TINI_MINIO_ACCESS_KEY", "").strip()
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("TINI_MINIO_SECRET_KEY", "").strip()
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("TINI_MINIO_BUCKET", "tini-documents").strip()
    )
    minio_secure: bool = field(
        default_factory=lambda: os.getenv("TINI_MINIO_SECURE", "").strip().lower()
        in {"1", "true", "yes"}
    )
    # Reranking (tini/documents/rerank.py): retrieve more candidates than
    # needed, then let a second model reorder them against the question.
    # Off by default. OFF means candidate chunks never leave this machine;
    # turning on a hosted provider (jev) sends them to that provider.
    rerank_provider: str = field(
        default_factory=lambda: os.getenv("TINI_RERANK_PROVIDER", "none").strip().lower()
    )
    rerank_model: str = field(default_factory=lambda: os.getenv("TINI_RERANK_MODEL", "").strip())
    rerank_api_key: str = field(default_factory=_rerank_api_key)
    rerank_candidates: int = field(
        default_factory=lambda: int(os.getenv("TINI_RERANK_CANDIDATES", "").strip() or "20")
    )
    # Providers the user turned off in the dashboard (comma-separated ids).
    # Disabled providers are hidden from pickers/switchers; the ACTIVE provider
    # can't be disabled (guarded in integrations.apply_provider_disabled).
    disabled_providers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            p.strip() for p in os.getenv("TINI_DISABLED_PROVIDERS", "").split(",") if p.strip()
        )
    )

    # --- Home: where Tini keeps its state (memory DB, config, traces).
    # Defaults to ./.tini next to where you run it, so you can open every file
    # it writes. Local-first means you can always look.
    home: Path = field(default_factory=lambda: Path(os.getenv("TINI_HOME", ".tini")))

    # --- Loop guardrails
    max_iterations: int = field(default_factory=lambda: int(os.getenv("TINI_MAX_ITERATIONS", "10")))
    # Headroom matters for REASONING models (kimi-k3, gpt-5.x, gemini-*-pro):
    # they spend output tokens thinking before the answer, so a low cap makes
    # them hit stop_reason=max_tokens mid-thought and return an EMPTY reply
    # (watched kimi-k3 do exactly that at 2048). 8192 leaves room to think AND
    # answer; it's a ceiling, not a target, so efficient models still cost the same.
    max_tokens: int = field(default_factory=lambda: int(os.getenv("TINI_MAX_TOKENS", "8192")))
    # Working memory is a SLIDING WINDOW (like context RAM): only the last N
    # turns go into the prompt. Older turns aren't lost — they're in state.db,
    # distilled into facts by consolidation, and pulled back by the retrieval
    # gate when relevant. Without this cap a long thread (esp. the always-on
    # Telegram session) resends its whole history every turn until it explodes.
    history_turns: int = field(default_factory=lambda: int(os.getenv("TINI_HISTORY_TURNS", "12")))

    # --- Memory
    # Consolidate (distill chats into durable facts) only after N new exchanges.
    consolidate_every: int = field(
        default_factory=lambda: int(os.getenv("TINI_CONSOLIDATE_EVERY", "6"))
    )
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("TINI_RETRIEVAL_TOP_K", "4"))
    )
    # 'sqlite' (default, zero setup) or 'supabase' (pgvector upgrade path — see launch-rag).
    semantic_store: str = field(default_factory=lambda: os.getenv("TINI_SEMANTIC_STORE", "sqlite"))
    # 'sqlite' (default, zero setup) or 'notion' (episodes live in a Notion database).
    episodic_store: str = field(default_factory=lambda: os.getenv("TINI_EPISODIC_STORE", "sqlite"))
    # Internal-document ingestion. The index is SQLite/FTS5 locally; its
    # interface allows a pgvector adapter without changing callers.
    document_workspace: str = field(
        default_factory=lambda: os.getenv("TINI_DOCUMENT_WORKSPACE", "").strip() or "default"
    )
    document_chunk_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("TINI_DOCUMENT_CHUNK_MAX_TOKENS", "512"))
    )
    document_chunk_min_tokens: int = field(
        default_factory=lambda: int(os.getenv("TINI_DOCUMENT_CHUNK_MIN_TOKENS", "120"))
    )
    document_chunk_overlap_tokens: int = field(
        default_factory=lambda: int(os.getenv("TINI_DOCUMENT_CHUNK_OVERLAP_TOKENS", "48"))
    )
    document_max_file_bytes: int = field(
        default_factory=lambda: int(os.getenv("TINI_DOCUMENT_MAX_FILE_BYTES", "26214400"))
    )
    document_max_text_chars: int = field(
        default_factory=lambda: int(os.getenv("TINI_DOCUMENT_MAX_TEXT_CHARS", "5000000"))
    )
    # 'sqlite' (default, zero setup) or 'qdrant' (self-hosted vector upgrade
    # path — see docker-compose.yml and tini/documents/qdrant_store.py).
    document_index: str = field(
        default_factory=lambda: os.getenv("TINI_DOCUMENT_INDEX", "sqlite").strip().lower()
        or "sqlite"
    )
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("TINI_QDRANT_URL", "").strip() or "http://localhost:6333"
    )
    qdrant_api_key: str | None = field(
        default_factory=lambda: os.getenv("TINI_QDRANT_API_KEY", "").strip() or None
    )
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("TINI_QDRANT_COLLECTION", "").strip()
        or "tini_documents"
    )
    # Hybrid retrieval on the Qdrant backend: fuse dense vector search with a
    # keyword (BM25-style sparse vector) ranking. Off by default: measured on
    # the DENSO set it did not beat dense-only (docs/document-ingestion.md).
    document_hybrid: bool = field(
        default_factory=lambda: os.getenv("TINI_DOCUMENT_HYBRID", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )
    # How the two rankings are fused: "rrf" (rank based; the keyword side's vote
    # is scaled by TINI_DOCUMENT_LEXICAL_WEIGHT) or "dbsf" (score based).
    document_fusion: str = field(
        default_factory=lambda: os.getenv("TINI_DOCUMENT_FUSION", "rrf").strip().lower()
    )
    # The languages the documents are written in, e.g. "ja,en,vi". When set, a
    # question is also searched in its translation into each of them (one
    # small-model call per new question). Empty = off. See
    # tini/documents/multilingual.py.
    query_languages: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            code.strip().lower()
            for code in os.getenv("TINI_QUERY_LANGUAGES", "").split(",")
            if code.strip()
        )
    )
    document_lexical_weight: float = field(
        default_factory=lambda: float(os.getenv("TINI_DOCUMENT_LEXICAL_WEIGHT", "").strip() or "1.0")
    )

    # --- Orchestration
    # Route every message through the triage graph workflow first (a small model
    # classifies it; trivial messages get a fast small-model reply, real tasks
    # run the normal loop as a graph node). Any failure anywhere fails open to
    # the plain loop, so this can never make Tini worse — only faster/cheaper.
    graph_workflows: bool = field(
        default_factory=lambda: os.getenv("TINI_GRAPH_WORKFLOWS", "") in ("1", "true", "yes")
    )

    # --- Optional gateway
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    whatsapp_token: str = field(default_factory=lambda: os.getenv("WHATSAPP_TOKEN", ""))
    whatsapp_phone_number_id: str = field(
        default_factory=lambda: os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    )

    # --- Tracing (JSONL always; OTel exports if an endpoint is set)
    otel_endpoint: str = field(default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", ""))

    def ensure_home(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "traces").mkdir(exist_ok=True)
        return self.home


def load_settings() -> Settings:
    return Settings()
