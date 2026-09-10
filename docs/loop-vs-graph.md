# Loop and Graph Responsibilities

The agent loop and graph orchestration solve different problems.

## Loop: dynamic retrieval

The loop lets the model decide whether to call `search_web` or an MCP tool,
inspect results, refine the query, and stop when it can answer. This is the
right default for document questions because the number and order of retrieval
steps are not known in advance.

## Graph: explicit control flow

The current `triage` graph classifies each message and routes it to one of two
paths:

```text
START → classify ─┬→ quick_reply → END
                  └→ full_agent  → END
```

`full_agent` is the normal loop embedded as a graph node. Set
`TINI_GRAPH_WORKFLOWS=1` to enable this front door, or use
`/triage <message>` to inspect one routing decision.

## When to add another graph

Add a graph when the RAG process has stable stages that need observability,
retries, branching, or parallelism—for example:

```text
select corpus → retrieve → rerank → answer → citation check
```

Keep provider calls and retrievers behind injected callables so the workflow
can be evaluated offline. Do not turn every model/tool iteration into a graph
node; leave open-ended query refinement inside the loop.
