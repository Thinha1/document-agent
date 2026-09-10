# Security

tini runs on your own machine, with your own API keys, and can retrieve private
documents. That makes a few things worth stating plainly.

## What we consider in scope

- Anything that exfiltrates keys, `.env`, memory (`state.db`), traces, or
  message contents off the machine.
- Code executing at install time, or a dependency doing so.
- A gateway or webhook accepting instructions it shouldn't — an unsigned or
  unauthenticated inbound request that can drive the agent.
- Prompt injection that exfiltrates retrieved document content or credentials.

## What isn't a vulnerability

- **The agent can query configured document sources.** That's the product.
  Retrieval tools are listed in the dashboard.
- **Your own API keys in your own `.env`.** tini never sends them anywhere but
  the provider you configured.

## Running it safely

- Keep `.env` out of git — it's gitignored, and so are `credentials.json` and
  `*token*.json`.
- Inbound gateways (WhatsApp-style webhooks) must verify request signatures
  before acting. Outbound ones (Telegram, Discord) dial out and aren't exposed.
- Review `.tini/mcp.json` and scope each server to the minimum document roots.
- Review a community skill before installing it. Treat `SKILL.md` as code.
