# Retrieval baseline — DENSO corpus

The measured quality of the document-RAG pipeline as of **2026-09-24**, saved
so later changes (hybrid search, reranking, new chunking, another embedding
model) can be compared against something real rather than against memory.
Raw per-question results: [`evals/baselines/bge-m3-dense-2026-09-24.json`](../evals/baselines/bge-m3-dense-2026-09-24.json).

Reproduce it, or snapshot a change and diff it against this one:

```bash
python scripts/document_baseline.py my-change --compare evals/baselines/bge-m3-dense-2026-09-24.json
```

The output names the questions that were fixed and the ones that regressed.
Every dashboard run (Knowledge → Retrieval eval) is also saved, with its
pipeline config, under `TINI_HOME/eval_runs/`.

## The pipeline that was measured

| Stage | Setting |
|---|---|
| Parse | `pypdf` text; PDFs with an empty user password are decrypted |
| Scanned pages (< 40 chars/page) | Render with `pypdfium2`, transcribe with `deepseek-v4-flash-vision-exp` |
| Chunking | Structure-aware, max 512 estimated tokens, min 120, overlap 48, CJK counted per character |
| Embedding | `bge-m3` via Ollama, 1024 dimensions |
| Index | Qdrant `tini_documents_bge3`, dense vectors, cosine, **no lexical/sparse leg, no reranker** |
| Search | top-k = 3, workspace-filtered |
| Corpus | 8 documents, 494 chunks: 6 English PDFs (one image-only, OCR'd), one 170-page Japanese securities report (395 chunks), and the folder's own `README.md`, which a folder ingest also picks up (2 chunks of noise; not excluded) |

## Results (retrieval only, k = 3, warm model, one run on a 12-thread CPU)

| Question set | Recall@3 | By language | By source format | Latency median / p95 |
|---|---|---|---|---|
| `denso_qa.jsonl` (38) | 36/38 (94.7%) | en 36/38 | text-pdf 29/30, scanned-pdf 7/8 | 70.7 / 95.4 ms |
| `denso_qa_multilingual.jsonl` (31) | 30/31 (96.8%) | ja 15/15, vi 11/12, en 4/4 | table-heavy-pdf 19/19, scanned-pdf 2/2, text-pdf 9/10 | 64.3 / 97.1 ms |

"Recall@3" means the expected **document** appeared among the top 3 chunks.
That is a generous test, so the table below matters more.

### Finding: a document hit is not an answer hit

`keyword coverage` counts how many of a question's expected answer facts
(names, dates, numbers) appear in the chunks returned from the right document.

| Source format | Keywords found |
|---|---|
| text-pdf | 69/83 (83%) |
| scanned-pdf (OCR) | 16/19 (84%) |
| **table-heavy-pdf** | **18/29 (62%)** |

Eight questions found the right document but returned **none** of its
expected facts, so the answer chunk itself was missed: `ja-founded`,
`ja-deming`, `vi-x-founded`, `vi-x-deming`, `vi-x-women`, `en-x-founded`,
`en-x-women` (seven from the 170-page Japanese report, mostly its company-history and
workforce sections) and `vi-qa-standard`. Two English questions did the same
(`code-antibribery`, `qa-traceability-change`). A chatbot answering from those
chunks would say the information is not there.

### Document-level misses (3)

| Question | Wanted | Got |
|---|---|---|
| `outside-directors` | governance policy **2021** | governance report **2025**, EDINET |
| `supplier-ghg` | supplier guidelines (OCR'd) | QA manual, EDINET |
| `vi-nomination` | governance report 2025 | EDINET ×3 (Vietnamese query) |

The 2021 and 2025 governance documents overlap heavily, so which one is "right"
depends on a version the question does not state.

## What this baseline does not tell you

- **Answer quality.** Nothing here calls a chat model. The dashboard's "Also
  answer with the chat model" checkbox measures that path, but no run of it
  is saved in this baseline yet.
- **Scale.** 494 chunks. Latency and precision at thousands of documents are
  unmeasured; do not extrapolate.
- **Independence.** The questions were written by the same people who chose
  the documents, so the numbers are likely optimistic.
- **Stability.** One run. Embeddings on CPU are not perfectly deterministic;
  a borderline question moved between runs earlier (17/31 vs 16/31).
- **Cold start.** The first query after a local model idles can take tens of
  seconds. The snapshot script discards a warm-up pass; a real deployment
  will hit it.

## History

| Date | Pipeline | English (38) | Multilingual (31) | Notes |
|---|---|---|---|---|
| 2026-09-24 | `nomic-embed-text` 768d, Japanese report chunked at ~2,040 chars | 36/38 | 16-17/31 | Japanese keywords found ≈ 0/25 |
| 2026-09-24 | same model, Japanese report re-chunked (~490 chars) | 36/38 | 17/31 | Japanese keywords found 6/25 |
| 2026-09-24 | **`bge-m3` 1024d (this baseline)** | 36/38 | 30/31 | Japanese keywords found 21/25 |

The first two rows are summary numbers from earlier the same day; their
per-question results were not saved.

## Candidate next steps

These are hypotheses tied to the failures above, **not results**. Each should
be tried one at a time and compared with `document_baseline.py --compare`.

| Failure | Candidate | Why it might help |
|---|---|---|
| Right document, answer chunk missed (dates, figures in tables) | Sparse/BM25 leg + rank fusion (Qdrant has no ranked full-text, so sparse vectors) | Exact numbers and dates are what dense embeddings blur |
| Same | Reranker over the top ~20 chunks | Reorders candidates by the actual question; costs latency per query |
| Same, table content | Table-aware chunking, or a title/section header prefixed to each chunk before embedding | A bare table row carries no hint of what it describes |
| 2021 vs 2025 governance | Document version as metadata, filter or boost | The question does not say which year |
| `vi-nomination`, cross-language misses | Translate the query into the corpus languages first | Cheap; adds an LLM call per query |
