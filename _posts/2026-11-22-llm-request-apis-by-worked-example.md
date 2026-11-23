---
title: "One Task, Five APIs: A Worked-Example Guide to LLM Request APIs"
date: 2026-11-22 21:00:00 -0500
categories: [Machine Learning, LLM]
tags: [llm, llm-api, openai, anthropic, gemini, aws-bedrock, tool-calling, batch, tutorial]
description: "OpenAI, Anthropic, Google and AWS Bedrock each give you two or more ways to call a model. One small task run through every major request API: what each is for, where they differ, and which one to pick."
---

If you have called more than one LLM provider, you have probably written the same request several times in slightly different shapes. The system prompt is a message in one API and a top-level field in another. Tool arguments come back as a JSON string here and as a parsed object there. One API remembers the conversation for you; the next one forgets it the moment it replies.

None of this is hard. It is just easy to get wrong, and the differences only show up when something breaks.

In this post I take one small task and run it through every major request API, side by side. By the end you should know what each API looks like on the wire, which differences actually matter, and how to choose.

> **Versions.** Written in November 2026. The code was checked against these SDK versions: `openai` 3.57.0, `anthropic` 2.12.0, `google-genai` 3.6.1, `boto3` 1.44.9. I checked the method signatures and parameter names against those SDKs, but I did not run every snippet against a live endpoint. Model names change often, so swap in whatever is current when you read this.
{: .prompt-info }

---

## The one-paragraph version

Every major provider now offers two styles of API:

1. A **stateless message-array API.** You send the whole conversation every time and get one reply back. The server remembers nothing.
2. A newer **stateful, item-based API.** The server can keep the conversation, run built-in tools, and return a typed list of outputs: text, reasoning, tool calls.

Around those two sit **batch endpoints** for cheap offline work and **OpenAI-compatible shims** that let you point the OpenAI SDK at someone else's models.

Here is how that maps onto the providers:

| Provider | Stateless API | Stateful API | Notes |
|---|---|---|---|
| OpenAI | Chat Completions | Responses | Responses is the recommended API for new work. The old Assistants API has been shut down. |
| Anthropic | Messages | — | Messages is the only API. It is stateless and does tool use, thinking and server tools itself. |
| Google | `generateContent` | Interactions | Interactions became GA in June 2026 and is recommended for new projects. |
| AWS Bedrock | Converse, InvokeModel | Via `bedrock-mantle` | Converse is the portable one; InvokeModel takes each provider's native body. |

Now let's make it concrete.

---

## The running example: ticket triage

The task: read a customer support message and return a small JSON object.

```python
SYSTEM = """You triage customer support tickets.
Reply with JSON only, no prose, in exactly this shape:
{"category": "billing" | "shipping" | "account" | "other",
 "urgency": "low" | "medium" | "high",
 "summary": "<one sentence>"}"""

TICKET = """Hi, I ordered a standing desk two weeks ago (order A-10442).
Tracking hasn't moved in six days and I start a new job on Monday.
Can someone tell me what's going on?"""
```

A good answer looks something like this:

```json
{"category": "shipping", "urgency": "high",
 "summary": "Order A-10442 has had no tracking updates for six days and the customer needs it by Monday."}
```

Later we will add a follow-up turn ("draft a reply to the customer") and a tool (`get_order_status`). Those two additions are where the APIs really start to differ.

---

## 1. OpenAI Chat Completions

This is the API the rest of the industry copied. Messages in, one message out.

```python
from openai import OpenAI

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5.6",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TICKET},
    ],
    response_format={"type": "json_object"},
)

print(resp.choices[0].message.content)
```

Three things to notice:

- The system prompt is just a message with `role: "system"`.
- The answer lives at `choices[0].message.content`. There is a list because Chat Completions can return several alternative answers when you pass `n`.
- `response_format={"type": "json_object"}` asks for valid JSON. It requires the word "JSON" to appear somewhere in your messages, which our system prompt already does.

For a follow-up turn, you keep the history yourself:

```python
history = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": TICKET},
    {"role": "assistant", "content": resp.choices[0].message.content},
    {"role": "user", "content": "Now draft a two-sentence reply to the customer. Plain text, not JSON."},
]
followup = client.chat.completions.create(model="gpt-5.6", messages=history)
print(followup.choices[0].message.content)
```

Chat Completions is still supported, but OpenAI now recommends the Responses API for new projects. There is also a concrete reason to move if you use reasoning models: starting with GPT-5.4, Chat Completions does not support tool calling with any `reasoning_effort` other than `none`.

---

## 2. OpenAI Responses

Responses replaces the message list with **items**. A message is one kind of item; so are `reasoning`, `function_call` and `function_call_output`. You send input items, you get output items back.

```python
resp = client.responses.create(
    model="gpt-5.6",
    instructions=SYSTEM,
    input=TICKET,
    store=False,
)

print(resp.output_text)
```

That is shorter, and two of the differences are worth pointing out:

- `instructions` is a top-level field rather than a system message.
- `output_text` is a convenience helper that joins all the text output items. Under it, `resp.output` is a list of typed items, and it is worth printing once just to see what comes back.

### Letting the server keep the conversation

The headline feature is that you can chain turns without resending history:

```python
first = client.responses.create(
    model="gpt-5.6",
    instructions=SYSTEM,
    input=TICKET,
)

followup = client.responses.create(
    model="gpt-5.6",
    instructions="You write short, friendly customer support replies.",
    previous_response_id=first.id,
    input="Now draft a two-sentence reply to the customer. Plain text, not JSON.",
)
print(followup.output_text)
```

Two gotchas here:

1. **Chaining needs storage.** Responses are stored by default, and `previous_response_id` only works on a response the server kept. If you set `store=False`, you are back to managing history yourself.
2. **Instructions don't carry over.** The previous turn's *messages* come along with `previous_response_id`; its `instructions` do not. Pass them again on every call. I have seen this bite people who change the system prompt on turn two and then wonder why the model ignores it, and people who set it once and then wonder why it disappears.

For longer-lived conversations there is also a Conversations API (`conversation=...`) that holds state under a stable ID, instead of chaining response IDs.

### When storage matters

Server-side state is convenient for chat products. For a pipeline that processes documents, it means your inputs now live on the provider's side for a retention period. If that is not what you want, set `store=False` and pass history explicitly. You lose nothing for single-shot calls.

---

## 3. Anthropic Messages

Anthropic has one request API, and it is stateless.

```python
import anthropic

client = anthropic.Anthropic()

msg = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=1024,
    system=SYSTEM,
    messages=[{"role": "user", "content": TICKET}],
)

print(msg.content[0].text)
```

The differences from Chat Completions:

- `system` is a **top-level field**, not a message. There is one system prompt, at the start.
- `max_tokens` is **required**. Leave it out and the call fails.
- The reply is `msg.content`, a list of typed **content blocks**: `text`, `tool_use`, `thinking` and so on. For a plain text answer it is one `text` block, but code that blindly reads `content[0].text` will break as soon as you turn on thinking or tools. A safer habit:

```python
text = "".join(b.text for b in msg.content if b.type == "text")
```

A follow-up turn is plain history, just like Chat Completions:

```python
followup = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=512,
    system="You write short, friendly customer support replies.",
    messages=[
        {"role": "user", "content": TICKET},
        {"role": "assistant", "content": msg.content},
        {"role": "user", "content": "Now draft a two-sentence reply to the customer."},
    ],
)
```

Note that you can pass `msg.content` straight back as the assistant turn. That matters later: when the assistant turn contains `thinking` or `tool_use` blocks, you must send them back unchanged.

Anthropic does have an OpenAI SDK compatibility layer, but Anthropic positions it for evaluation. If you use it, all system and developer messages get concatenated into one system prompt, and the response does not include Claude's thinking. For production, call Messages directly.

---

## 4. Google Gemini: `generateContent` and Interactions

Google went through the same transition as OpenAI. The Interactions API is now recommended for new projects; `generateContent` is labelled legacy but remains fully supported.

### `generateContent`

```python
from google import genai
from google.genai import types

client = genai.Client()

resp = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=TICKET,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
    ),
)

print(resp.text)
```

Gemini's vocabulary is a little different. A conversation is a list of `Content` objects, each with a `role` (`user` or `model`, not `assistant`) and a list of `parts`. `system_instruction` lives in the config. `response_mime_type="application/json"` gives you JSON mode, and you can tighten it with a schema.

### Interactions

```python
first = client.interactions.create(
    model="gemini-3.8-flash",
    system_instruction=SYSTEM,
    input=TICKET,
)
print(first.output_text)

followup = client.interactions.create(
    model="gemini-3.8-flash",
    system_instruction="You write short, friendly customer support replies.",
    previous_interaction_id=first.id,
    input="Now draft a two-sentence reply to the customer.",
)
print(followup.output_text)
```

If this looks familiar, it should. It is the same idea as OpenAI Responses: stored by default, chained by ID, with typed output steps underneath and a text helper on top. As with OpenAI, I pass `system_instruction` on every call rather than assume it carries over.

### Not everything has moved yet

At the time of writing, a handful of features are still only on `generateContent`: the Batch API, explicit caching, custom safety settings, `video_metadata`, and automatic function calling in Python. So the honest answer for Gemini today is: **Interactions for new interactive work, `generateContent` for batch jobs and anything that relies on explicit caching.**

---

## 5. AWS Bedrock: Converse and InvokeModel

Bedrock hosts models from several vendors, so it offers two different ideas of what a request is.

**Converse** is Bedrock's own standard shape. The same call works for any chat model Bedrock hosts:

```python
import boto3

MODEL_ID = "<your Bedrock model ID or inference profile ID>"
brt = boto3.client("bedrock-runtime", region_name="us-east-1")

resp = brt.converse(
    modelId=MODEL_ID,
    system=[{"text": SYSTEM}],
    messages=[{"role": "user", "content": [{"text": TICKET}]}],
    inferenceConfig={"maxTokens": 1024},
)

print(resp["output"]["message"]["content"][0]["text"])
```

**InvokeModel** takes the model vendor's *native* request body as raw JSON. For a Claude model, that is the Anthropic Messages body plus an `anthropic_version` field:

```python
import json

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "system": SYSTEM,
    "messages": [{"role": "user", "content": TICKET}],
}

resp = brt.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
out = json.loads(resp["body"].read())
print(out["content"][0]["text"])
```

AWS recommends Converse when the model supports it, because it unifies requests across models. Use InvokeModel when you need a vendor-specific parameter that Converse does not expose, or for model types Converse does not cover.

There is a third option worth knowing about. Bedrock's newer `bedrock-mantle` endpoint speaks the OpenAI Responses and Chat Completions APIs and the Anthropic Messages API directly, including stateful conversations. If you already have OpenAI SDK code, you can point it at Bedrock by changing the base URL and key.

---

## Where it really diverges: tool calling

Single-shot text is easy to port. Tool calling is where every API has its own shape, and it is where most porting bugs live.

Here is our tool:

```python
def get_order_status(order_id: str) -> dict:
    # Stand-in for a real lookup
    return {"order_id": order_id, "status": "held_at_depot",
            "reason": "address verification", "eta": "2 business days"}
```

### A complete loop on Anthropic Messages

Let's write one full loop end to end, then compare the shapes.

```python
import json
import anthropic

client = anthropic.Anthropic()

tools = [{
    "name": "get_order_status",
    "description": "Look up the current shipping status of an order by its ID.",
    "input_schema": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}]

messages = [{"role": "user", "content": TICKET + "\n\nCheck the order, then summarize what's happening."}]

while True:
    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )
    messages.append({"role": "assistant", "content": msg.content})

    if msg.stop_reason != "tool_use":
        break

    results = []
    for block in msg.content:
        if block.type == "tool_use":
            output = get_order_status(**block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(output),
            })
    messages.append({"role": "user", "content": results})

print("".join(b.text for b in msg.content if b.type == "text"))
```

The loop is the same on every provider: call the model, run any tools it asks for, send the results back, repeat until it stops asking. What changes is the shape of the request and the reply.

### The same loop, five shapes

**How the model asks for a tool:**

| API | Where the call appears | Arguments arrive as |
|---|---|---|
| Chat Completions | `message.tool_calls[i]` with `id` and `function.name` | **a JSON string** in `function.arguments` |
| Responses | an output item with `type: "function_call"`, `call_id`, `name` | **a JSON string** in `arguments` |
| Anthropic Messages | a content block with `type: "tool_use"`, `id`, `name` | **a parsed object** in `input` |
| Gemini `generateContent` | a part with `function_call.name` | **a parsed object** in `function_call.args` |
| Bedrock Converse | a content block `{"toolUse": {...}}` with `toolUseId`, `name` | **a parsed object** in `input` |

**How you send the result back:**

| API | What you send |
|---|---|
| Chat Completions | a message with `role: "tool"` and `tool_call_id` |
| Responses | an input item with `type: "function_call_output"` and `call_id` |
| Anthropic Messages | a **user** message containing a `tool_result` block with `tool_use_id` |
| Gemini `generateContent` | a `user` content with a function-response part, built with `types.Part.from_function_response(name=..., response=...)` |
| Bedrock Converse | a **user** message containing `{"toolResult": {"toolUseId": ..., "content": [...]}}` |

Two differences cause most of the bugs I see:

1. **String versus object.** On both OpenAI APIs, the arguments are a JSON *string* you must `json.loads()`. Everywhere else they are already parsed. Code ported from OpenAI that calls `json.loads()` on an Anthropic `input` will fail with a `TypeError`; code going the other way passes a string where a dict was expected.
2. **Who owns the tool result.** OpenAI has a dedicated `tool` role or item type. Anthropic and Bedrock put the result inside a **user** turn. If you send it as an assistant turn, or as a separate role, the request is rejected.

And one rule that holds everywhere: **send the assistant's turn back unchanged.** That includes tool calls and any reasoning or thinking content. Rebuilding it from the text alone is the quickest way to get a "tool result without matching tool call" error.

---

## Cheap and slow: batch endpoints

If nobody is waiting for the answer, don't pay for speed. OpenAI, Anthropic, Google and Bedrock all offer batch processing that typically costs around half the synchronous price, in exchange for results that arrive within hours rather than seconds. (Check current pricing; it changes.)

Here is our triage task over many tickets with Anthropic's Message Batches:

```python
import time

tickets = [TICKET, "My invoice shows two charges for the same month.", "How do I change my email address?"]

batch = client.messages.batches.create(requests=[
    {
        "custom_id": f"ticket-{i}",
        "params": {
            "model": "claude-sonnet-5-5",
            "max_tokens": 512,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": t}],
        },
    }
    for i, t in enumerate(tickets)
])

while client.messages.batches.retrieve(batch.id).processing_status != "ended":
    time.sleep(30)

for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        print(result.custom_id, result.result.message.content[0].text)
    else:
        print(result.custom_id, "failed:", result.result.type)
```

Note that `params` is exactly a Messages request. That is the nice property of batch APIs in general: you write the request once and decide later whether to send it now or overnight.

Two practical notes. Results are not guaranteed to come back in order, so always key on `custom_id`. And on Gemini, remember that batch is still a `generateContent` feature.

---

## About those OpenAI-compatible endpoints

Anthropic, Google and Bedrock all accept requests from the OpenAI SDK. It is tempting to standardize on the OpenAI SDK and swap base URLs. For a prototype, that is fine. For production I would not, for one reason: **the shim only carries what the OpenAI shape can express.**

Things you typically lose or get degraded:

- Provider-specific features: Anthropic's prompt caching controls, citations and thinking output; Gemini's safety settings and caching.
- System prompt placement. OpenAI allows system and developer messages anywhere; Anthropic's compatibility layer concatenates them all into one prompt at the start.
- Exact error messages, which are reshaped to look like OpenAI's.

A thin adapter over each native API is a little more code and saves you from discovering missing features one at a time.

---

## A thin adapter

Here is the minimum I would write before reaching for a framework. One request type, one response type, one function per provider.

```python
from dataclasses import dataclass, field

@dataclass
class LLMRequest:
    system: str
    messages: list[dict]          # [{"role": "user" | "assistant", "text": "..."}]
    max_tokens: int = 1024
    model: str = ""

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    raw: object = field(repr=False)


def call_anthropic(client, req: LLMRequest) -> LLMResponse:
    msg = client.messages.create(
        model=req.model,
        max_tokens=req.max_tokens,
        system=req.system,
        messages=[{"role": m["role"], "content": m["text"]} for m in req.messages],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return LLMResponse(text, msg.usage.input_tokens, msg.usage.output_tokens, msg)


def call_openai(client, req: LLMRequest) -> LLMResponse:
    resp = client.responses.create(
        model=req.model,
        instructions=req.system,
        input=[{"role": m["role"], "content": m["text"]} for m in req.messages],
        max_output_tokens=req.max_tokens,
        store=False,
    )
    return LLMResponse(resp.output_text, resp.usage.input_tokens, resp.usage.output_tokens, resp)


def call_bedrock(brt, req: LLMRequest) -> LLMResponse:
    resp = brt.converse(
        modelId=req.model,
        system=[{"text": req.system}],
        messages=[{"role": m["role"], "content": [{"text": m["text"]}]} for m in req.messages],
        inferenceConfig={"maxTokens": req.max_tokens},
    )
    text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    u = resp["usage"]
    return LLMResponse(text, u["inputTokens"], u["outputTokens"], resp)
```

Each function is about ten lines, and each one is honest about its provider. When you need prompt caching on Anthropic or a safety setting on Gemini, you add it to that one function, instead of discovering that your abstraction cannot express it. The `raw` field is the escape hatch for everything the common type leaves out.

If you need many providers, retries, fallbacks and spend tracking, a gateway library such as LiteLLM gives you this layer ready-made (my [LiteLLM by Worked Example](/posts/litellm-by-worked-example/) tutorial covers it in depth). Just keep in mind it makes the same trade-off as the OpenAI shims: one common shape, with provider extras passed through on a best-effort basis.

---

## Which one should I use?

> **Rule of thumb:** use the stateless API unless you have a specific reason to let the server hold the conversation.

| Situation | Choice |
|---|---|
| Single-shot calls: classify, extract, score, summarize | Stateless: Messages, Responses with `store=False`, `generateContent`, Converse |
| Large offline jobs where nobody is waiting | The provider's **batch** API |
| Chat products with long conversations | Stateful: Responses with Conversations, Interactions |
| Agents that use hosted tools (web search, code execution, file search) | The provider's newest API: Responses, Interactions, Messages with server tools |
| Reasoning models with tool calling on OpenAI | Responses, not Chat Completions |
| One codebase, several providers | A thin adapter over the native APIs (or Converse, if everything runs on Bedrock) |
| Long, reused prompts where caching matters | The native API, never a compatibility shim |
| Prototyping a model comparison quickly | OpenAI-compatible endpoints are fine here |

---

## Gotchas worth pinning up

- **Anthropic requires `max_tokens`.** OpenAI and Gemini have sensible defaults; Anthropic does not.
- **Gemini says `model`, not `assistant`.** The role names differ, and so does the structure: `parts`, not `content`.
- **OpenAI tool arguments are a JSON string.** Everyone else gives you a parsed object.
- **Tool results go in a user turn** on Anthropic and Bedrock Converse.
- **Don't read `content[0].text` blindly.** Once thinking or tools are on, the first block may not be text. Filter by type.
- **`previous_response_id` and `previous_interaction_id` need stored responses.** With storage off, they have nothing to chain to.
- **Instructions don't ride along with the chain.** Send the system prompt on every call.
- **Stored by default means retained by default.** Both stateful APIs keep your requests unless you opt out. Decide on purpose.
- **Batch results arrive unordered.** Always key on your own `custom_id`.
- **Not every feature is on every API yet.** Gemini batch and explicit caching are still `generateContent`-only. Check before you migrate.

---

## Wrapping up

Underneath the naming differences, the major providers have converged on the same two ideas: a stateless request you fully control, and a stateful one where the server holds state and runs tools for you. Once you see that, the APIs stop looking like five unrelated designs and start looking like dialects of one.

My own default is boring on purpose: a stateless native API for anything that runs in a pipeline, the batch endpoint for anything that can wait, and the stateful APIs only where a conversation or hosted tools actually earn their keep.

---

*Further reading: each provider's official docs are the source of truth and change often. Start with the OpenAI "Responses vs. Chat Completions" guide, Anthropic's Messages API reference, Google's "Migrating to the Interactions API" guide, and the Bedrock "Converse API" user guide.*
