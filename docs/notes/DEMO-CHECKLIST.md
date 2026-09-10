# RAG Demo Checklist

Use a throwaway `.tini` home and a SearXNG instance configured with safe demo
documents. Do not record real credentials or internal content.

| Step | Prompt or action | Expected evidence |
|---|---|---|
| 1 | Open Dashboard → Connections | SearXNG is configured or connected |
| 2 | “Find the worker restart runbook” | `search_web` appears in the Loop trace |
| 3 | Ask for a cited summary | Answer includes document title or URL |
| 4 | Ask for an undocumented value | Answer states that evidence is missing |
| 5 | Configure `.tini/mcp.json` | Namespaced MCP retrieval tools appear |
| 6 | Enable graph workflows | Graph tab shows classify → full-agent path |
| 7 | Say “thanks” | Quick path uses the small model, no retrieval |
| 8 | Run Compare with the same query | Cards show latency, cost, tools, and scores |
| 9 | Run `make eval` | Deterministic retrieval suite passes |

Before recording, check that the UI contains no private document snippets,
tokens, `.env` values, or production URLs.
