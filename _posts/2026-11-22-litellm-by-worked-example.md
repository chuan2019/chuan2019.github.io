---
title: "LiteLLM by Worked Example: A Hands-On Tutorial"
date: 2026-11-22 12:00:00 -0500
categories: [Machine Learning, LLM]
tags: [llm, litellm, ai-gateway, llm-proxy, aws-bedrock, tutorial]
description: "A 23-section, hands-on tutorial for the LiteLLM SDK and proxy, written for LiteLLM 1.109.1: every example runs, every output is real, and every section ends with exercises."
---

Every LLM provider has its own API, so an application that uses three providers carries three integrations, plus its own retries, failover, spend tracking and key management. [LiteLLM](https://github.com/BerriAI/litellm) puts one OpenAI-shaped interface in front of more than 100 providers. It comes as a Python SDK and as a proxy server that any OpenAI client can call.

I wrote a tutorial that covers both: **[LiteLLM by Worked Example](/tutorials/litellm/)**.

> **Update:** an edition for LiteLLM 1.110.2 is now available as well. The [editions page](/tutorials/litellm/) lists every edition, and the [new edition's introduction](/tutorials/litellm/1.110.2/) lists what changed.
{: .prompt-info }

## How it works

Each section is one practical use case. It comes with a runnable script, the output the script really produced, an explanation of what the output shows, and exercises with hints. Most examples run offline with mocked replies or small local fakes of the provider APIs, so you can work through them without an account or spending money. Where a section needs a real provider, it says so.

At the end there is a [self-test](/tutorials/litellm/1.109.1/exercises/exam.html) with 257 questions across 22 topics.

## What it covers

| Sections | Topics |
| --- | --- |
| 1–11 | The SDK: calls, streaming, async, structured output, tool calling, errors and fallbacks, the `Router`, cost, caching, callbacks, comparing models |
| 12–13 | The proxy, run with `uv` or Docker; reaching models through AWS Bedrock, Azure, Google Vertex AI, OpenAI, Anthropic, Ollama and vLLM; virtual keys, budgets and rate limits |
| 14 | Smart routers that choose a model per request |
| 15–18 | The MCP gateway, the skills and agent (A2A) gateways, prompt management, observability |
| 19 | Lens, which records and reviews agent runs |
| 20–21 | Licensing, access control, identity providers, secrets, guardrails and a hardened production deployment |
| 22 | Context and memory management |
| 23 | Other providers, for reference |

Along the way it records the gotchas I hit while writing it. They range from settings that silently do nothing to features that need an Enterprise licence.

## Tied to one LiteLLM version

LiteLLM releases very often, and behaviour changes between releases. So this tutorial is written for, and verified against, exactly one release: **LiteLLM 1.109.1**. The project pins it, the Docker commands use the matching image, and every claim about behaviour says which version it was observed on.

When I move the tutorial to a newer LiteLLM, the old edition stays available and keeps improving on its own branch. The [editions page](/tutorials/litellm/) lists them all.

## Get the code

Everything is on GitHub at [chuan2019/litellm-tutorial](https://github.com/chuan2019/litellm-tutorial). To run the first example:

```bash
git clone --branch v0.1.0-litellm1.109.1 https://github.com/chuan2019/litellm-tutorial.git
cd litellm-tutorial
uv sync
cd examples
export LITELLM_LOCAL_MODEL_COST_MAP=True
uv run python 01_first_call.py
```
