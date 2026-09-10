---
name: your-skill-name
description: One sentence saying what this skill does AND when to use it — the loader matches user messages against these words, so include the words people actually say.
---

<!--
To contribute: copy this file to skills/community/<your-skill-name>/SKILL.md
and open a PR. CI checks the frontmatter (name + description required — the
official Anthropic Agent Skills format). Keep the body under ~60 lines:
skills are loaded into the prompt only when they match, but shorter is better.
-->

## Instructions

Step-by-step guidance for the model. Be concrete: name the tools to call
(`search_web` or a namespaced MCP retrieval tool), the defaults to assume, and
the tone to take.

## Edge cases

| Situation | Do |
|---|---|
| Something ambiguous | Ask one clarifying question |
