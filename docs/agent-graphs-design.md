# Graph Orchestration Design

Status: the in-repository graph engine and triage workflow are shipped behind
`TINI_GRAPH_WORKFLOWS=1`.

## Responsibility

The agent loop handles dynamic retrieval: the model decides which source to
query, inspects results, refines the query, and stops when it can answer. A graph
is useful only when the application has stable, inspectable control flow around
that loop.

The current workflow is deliberately small:

```text
START → classify ─┬→ quick_reply → END
                  └→ full_agent  → END
```

`classify` uses the small model and writes `route` plus `triage_reason`.
`quick_reply` handles greetings and acknowledgements without retrieval.
`full_agent` invokes the normal context assembly and loop. Invalid classifier
output, exceptions, or graph failures route to the full path.

## Engine Contract

[`tini/graph/engine.py`](../tini/graph/engine.py) implements `Graph`, `Node`, and
`run_graph`. State is a plain dictionary. A node reads the state and returns only
the keys it wants merged. Routers are Python functions over state, keeping branch
selection deterministic once a model has produced its classification.

The engine supports conditional edges, bounded visits, a global step limit,
observer events, and parallel-ready nodes. Node callables are injected into
workflow builders so tests can replace every provider or retrieval call with a
pure stub.

## Why the Repository Does Not Require LangGraph

The shipped triage workflow needs no durable checkpoints, distributed workers,
or human-resume protocol. A small stdlib engine keeps state transitions visible
and avoids adding LangChain as a core dependency. This does not prevent RAG
benchmarks or Langfuse integration: evaluation consumes traces and outputs, while
Langfuse-compatible observability can receive OpenTelemetry spans.

Reconsider LangGraph when the product genuinely needs several of these features:

- persistent checkpoint and resume across processes;
- long-running workflows with human approval;
- durable retries and recovery per node;
- a growing catalog of shared graph components;
- distributed execution or managed LangGraph deployment.

Adopt it for those runtime requirements, not merely to draw a flow diagram or
run evaluations.

## Extending the RAG Flow

A future document pipeline can be modeled explicitly when its stages become
stable:

```text
select corpus → retrieve → rerank → answer → citation check
```

Use named state keys such as `corpus_ids`, `query`, `chunks`, `ranked_chunks`,
and `citations`. Keep exploratory multi-query retrieval inside an agent node.
Expose node and retrieval events to the dashboard so it visualizes actual
execution rather than a manually maintained mock flow.

Every workflow needs deterministic tests for topology, routes, failure behavior,
and step bounds. Add model-scored tests separately for groundedness and answer
quality.
