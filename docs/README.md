# Documentation Map

Use this page to choose the smallest reference that answers the current question.

## Reference — how the system works

| file | what it answers |
|---|---|
| [architecture.md](architecture.md) | the four pillars, and which file is which diagram box |
| [loop-vs-graph.md](loop-vs-graph.md) | when a turn needs shape, and why the loop never changes |
| [agent-graphs-design.md](agent-graphs-design.md) | the graph engine's design and its fail-open rule |
| [memory-backends-playbook.md](memory-backends-playbook.md) | seeing your memories in each provider's own console |
| [benchmarks.md](benchmarks.md) | what has been measured, and how |
| [integrations.md](integrations.md) | providers, SearXNG, MCP, memory, traces, and gateways |

## Write-ups — one topic, explained

| file | what it is |
|---|---|
| [kimi-k3-explained.md](kimi-k3-explained.md) | the K3 model, and what the arena found |
| [phan-tich-ky-thuat-tini-agent.md](phan-tich-ky-thuat-tini-agent.md) | a ten-minute Vietnamese code walkthrough |
| [stack-report.md](stack-report.md) | the dated stack decisions and verification notes |

## notes/ — working material, not reference

Working checklists and rundowns — including
[memory-video-rundown.md](notes/memory-video-rundown.md), the shot-by-shot for
the memory-layer comparison. Useful, dated, and **not** the place to learn
how anything works — they assume you already do.

## Whiteboards

[whiteboards/](whiteboards/) holds editable architecture sources. When a board
has a matching `tini/ops/whiteboard/build_*.py`, edit the builder and regenerate
the artifact.

Related, and deliberately not in `docs/`: [../examples/](../examples/) is
runnable teaching material — one folder per topic, nothing under `tini/`
imports from it. See `CLAUDE.md` for the four rules that keep it honest.
