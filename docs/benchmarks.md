# Retrieval Evaluation

Tini keeps evaluation beside the runtime so changes to prompts, models,
retrieval, and orchestration can be compared with the same harness.

## Deterministic completion

`evals/dataset.jsonl` contains RAG-facing cases. A case can require
`search_web`, assert a substring in its query arguments, set a minimum number
of tool calls, or require no tool. Run:

```bash
make eval
```

The offline suite scripts model responses and tests tool wiring without a
network. Live dataset cases run only when `TINI_RUN_LIVE_EVALS=1` and a provider
key are configured.

## Model comparison

The dashboard Compare view sends one prompt through several pinned models in
isolated temporary homes. It records retrieval-gate decisions, tool calls,
latency, token usage, cost, deterministic completion, and an optional LLM judge
score. Results are written to the compare history, not `state.db`.

Use retrieval prompts that can be scored independently of prose quality. For
example, require a lookup for an internal policy and separately judge whether
the final answer cites evidence and handles missing documents honestly.

## Release gate

```bash
make eval-judge
make gate
```

The release gate combines deterministic checks with the configured judge
threshold. Keep answer-quality rubrics separate from binary tool and argument
checks; this makes regressions attributable instead of hiding them in one
aggregate score.

The independent `evals/coding.jsonl` battery remains available for model
research, but it is not exposed as a runtime tool to the RAG agent.
