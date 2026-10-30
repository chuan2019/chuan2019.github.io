---
title: "Quick Notes: Context and Memory in Modern AI Systems"
date: 2026-10-29 06:40:00 +0000
categories: [Machine Learning, LLM]
tags: [ai-agents, llm, context-engineering, ai-memory, software-architecture]
description: "Model sees Context on each call, Memory persists between calls"
---

*Model sees Context on each call, Memory persists between calls*

![Context and memory in modern AI systems](https://miro.medium.com/v2/resize:fit:1400/1*oOf8IOlAJADraI-HHHmhCw.png)

When building LLM applications, such as chatbots, coding assistants, or agents running multi-step workflows, two terms come up all the time: **context** and **memory**. They are closely related, and in many discussions they are used as if they mean the same thing. They do not, and mixing them up usually leads to confusing designs.

This quick note explains the relationship between context and memory in modern AI systems, walks through a reference architecture, and demonstrates four working examples in Python, from a simple conversation buffer to an agent that manages its own memory. The focus is on system architecture, i.e., where each piece lives and how data flows between them.

## 1. Context and Memory

### 1.1 The Mental Model

**Context** is ***what the model sees on the current call***. It is a bounded sequence of tokens (*system prompt*, *conversation history*, *tool results*, etc.) assembled by the application for every single request. **Memory** is ***anything that survives after the call returns***.

The key point is that the model itself is **stateless**. Between two API calls it remembers nothing. So memory can only change the model’s behavior when the application persists something somewhere, and later loads part of it back into the context. In other words, every memory system can be reduced to two pipelines:

- **Write path**: *after a turn, decide what to persist, in what form, and in which store*.
- **Read path**: *before a turn, decide what to retrieve, and how to fit it into the token budget*.

The last step of the read path, i.e., *choosing*, *ordering* and *formatting* the tokens that actually go into the window, is what people nowadays call “**context engineering**”.

If you prefer an analogy, think of a computer. Model weights are the firmware (fixed skills and world knowledge), the ***context window*** is the ***RAM*** (fast, small, wiped after each call), the ***KV / prompt cache*** is the ***CPU cache*** over what is already in RAM, and ***external memory*** stores are the ***disk***. Nothing on disk is used until it is loaded into RAM.

### 1.2 Where Memory Lives

The word “memory” is used for quite different things in AI systems. Among them, only the context window is read directly by the attention mechanism. The others either shape the model before the call (weights), speed up the call (caches), or have to be loaded into the context window before they can make any difference (external stores). The table below summarizes the five layers.

![The five layers of memory in AI systems](https://miro.medium.com/v2/resize:fit:1400/1*rWSz2MpeId3L8C_t3MpiFA.png)

Long-term memory is usually further split into three types, borrowing the terms from cognitive science (see the [CoALA paper](https://arxiv.org/abs/2309.02427) in the references):

- Episodic memory records what happened, e.g., “*On Oct 1 the pipeline failed on an expired S3 credential.*” It is timestamped and mostly append-only.
- Semantic memory records what is true, e.g., “*User prefers Python; the pipeline runs on Airflow.*” These are consolidated facts, updated in place.
- Procedural memory records how to act, e.g., *system prompts*, *skills*, *tool definitions* and *playbooks*. It is usually versioned like code.

Two things are worth pointing out here. First, the ***KV cache*** and ***prompt cache*** are not memory. They reduce cost and latency, but never change what the model knows, i.e., a cache hit and a cache miss give the same output distribution. Second, a ***bigger context window*** is not memory either. It raises the ceiling of what a single call can see, but nothing persists once the call returns.

## 2. Reference Architecture

With the two pipelines in mind, a memory-augmented AI system can be described as a stateless LLM wrapped by a read path, which assembles the context before each call, and a write path, which persists what matters after the call. The diagram below shows how the pieces fit together.

![Reference architecture of a memory-augmented AI system](https://miro.medium.com/v2/resize:fit:2000/1*PSAz9K_i1Yn798F--7yuDw.png)

The easiest way to read the diagram is to follow its three loops:

1. The per-call loop (left to right). The orchestrator gathers candidates from the stores, the context assembler packs them into the token budget, and the LLM generates the response. This loop runs on every request and dominates the latency.
2. The agent loop (between the LLM and the tools). The model can read or write memory by itself in the middle of a turn. This is how MemGPT-style agents and agents using Claude’s memory tool work.
3. The background loop (the write path). Extraction and consolidation can run asynchronously after the response is returned, so they add no latency on the user side.

The table below lists what each component is responsible for, and how it is typically implemented.

![Components of the reference architecture](https://miro.medium.com/v2/resize:fit:1400/1*ZxkPb6HZ1ljm__32Qd_G5g.png)

One design boundary is worth highlighting: the retriever is optimized for recall, while the context assembler is optimized for precision. Keeping them as separate components allows one to retrieve, say, 50 candidates cheaply and only inject the best 5 into the context.

### 2.1 What the Model Actually Sees

The diagram above can still feel a bit abstract, so it helps to look at the tokens which actually reach the model on a single turn. The listing below shows one assembled context window, with each part labelled by the store it came from. The scenario is a user debugging a nightly Airflow pipeline, the same one used later in Examples 2 and 4, and the content is exactly what the `assemble_request` function in Example 4 produces for this turn.

![One assembled context window, labelled by source](https://miro.medium.com/v2/resize:fit:1400/1*X3u_SRxEPlNWT6OkNnDolQ.png)

A few things to notice from this listing:

- From the model’s point of view, there is no such thing as “memory”. There is only one flat sequence of tokens. The labels on the left exist only in the application, which is exactly why the application, not the model, owns the memory system.
- Out of everything stored about this user, only two long-term memories made it into this turn. The rest stayed on disk and had no effect on the answer, which is the read path doing its job.
- The parts are ordered from the most stable (tools and rules, unchanged for weeks) to the most volatile (the new user message). This ordering is what makes prompt caching effective, and Example 4 explains it in detail.
- Nothing from the parametric layer (the weights) shows up here, yet it is what lets the model write the rerun script at all. Weights provide the general skills; the context provides the user- and task-specific facts.

## 3. Working Examples

This section demonstrates four examples, each corresponding to a different part of the reference architecture above. All the Python code below was run locally, except the API call in Example 3, which requires an Anthropic API key.

### 3.1 Example 1: Short-Term Memory as a Token-Budgeted Buffer

The simplest form of memory is the chat transcript itself. The problem is that the transcript keeps growing, and sooner or later it does not fit in the context window. In this example, the buffer keeps the last few turns verbatim, and folds the evicted turns into a rolling summary which goes into the system prompt.

In terms of the architecture, only the session store is involved. The write path is “*append, then compact*”, and the read path is “*summary plus recent turns*”.

**Step 1. Implement the buffer**

The code below implements the buffer. The token counter here is a rough heuristic; in real projects one should use the token counter provided by the model provider.

```python
from dataclasses import dataclass, field
from typing import Callable

def count_tokens(text: str) -> int:
    """Rough heuristic (~4 chars/token for English). Use the provider's token counter in production."""
    return max(1, len(text) // 4)

@dataclass
class ConversationMemory:
    budget: int                                   # tokens reserved for history
    summarize: Callable[[str, list[dict]], str]   # (old_summary, evicted_turns) -> new_summary
    keep_last: int = 2                            # always keep this many turns verbatim
    summary: str = ""
    turns: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self._compact()

    def _size(self) -> int:
        return count_tokens(self.summary) + sum(count_tokens(t["content"]) for t in self.turns)

    def _compact(self) -> None:
        evicted = []
        while self._size() > self.budget and len(self.turns) > self.keep_last:
            evicted.append(self.turns.pop(0))
        # Most chat APIs require the first message to be from the user.
        while self.turns and self.turns[0]["role"] != "user":
            evicted.append(self.turns.pop(0))
        if evicted:
            self.summary = self.summarize(self.summary, evicted)

    def as_context(self, base_system: str) -> tuple[str, list[dict]]:
        system = base_system
        if self.summary:
            system += f"\n\n<conversation_summary>\n{self.summary}\n</conversation_summary>"
        return system, list(self.turns)
```

**Step 2. Run it with a stub summarizer**

In production, `summarize` is usually a call to a cheap LLM with a prompt like “*Update this running summary with the new turns. Keep decisions, open questions, names and numbers; drop pleasantries. Max 200 words.*” For testing purposes, a stub works just fine:

```python
import json

def fake_summarize(prev, evicted):
    gist = "; ".join(f"{t['role']}: {t['content'][:30]}" for t in evicted)
    return f"{prev} | {gist}".strip(" |")

mem = ConversationMemory(budget=60, summarize=fake_summarize)
for i in range(4):
    mem.add("user", f"Question {i}: explain topic {i} in some detail please")
    mem.add("assistant", f"Answer {i}: topic {i} works like this ...")
system, messages = mem.as_context("You are a helpful assistant.")
print(system)
print(json.dumps(messages, indent=2))
```

Running the code above prints the system prompt and the remaining messages, i.e., exactly what would be sent to the model on the next call:

```text
You are a helpful assistant.

<conversation_summary>
user: Question 0: explain topic 0 in; assistant: Answer 0: topic 0 works like t | user: Question 1: explain topic 1 in; assistant: Answer 1: topic 1 works like t | user: Question 2: explain topic 2 in; assistant: Answer 2: topic 2 works like t
</conversation_summary>
[
  {
    "role": "user",
    "content": "Question 3: explain topic 3 in some detail please"
  },
  {
    "role": "assistant",
    "content": "Answer 3: topic 3 works like this ..."
  }
]
```

From the output, we can see that the first three question-answer pairs no longer exist as messages. They have been folded into the `<conversation_summary>` block of the system prompt, while only the latest pair is kept verbatim. The model still “*knows*” about questions 0 to 2, but only through the summary, which is the compaction at work.

A few things to notice from this example:

- Summarization is lossy compression. Whatever the summarizer drops is gone for good, which is why long-term stores (Example 2) keep the original records.
- The summary itself has to be bounded too. The stub above simply concatenates, so eventually the summary alone exceeds the budget. A real summarizer rewrites the summary instead of appending to it.
- This is essentially what the provider-side “compaction” features do. Anthropic’s context editing (clearing old tool results) and the auto-compaction in agent harnesses apply the same idea on the server side.

### 3.2 Example 2: Long-Term Semantic Memory with Retrieval

Long-term memory is essentially a store plus a scoring function. Each memory item is embedded when it is written. When reading, the query is embedded as well, and the candidates are scored by similarity, recency and importance (the same idea as the Generative Agents paper), filtered by a relevance threshold, and then packed into a token budget.

In terms of the architecture, this example covers the long-term store, the retriever and the context assembler. The write path is “*embed, dedupe, store*”, and the read path is “*score, filter, budget, inject*”.

**Step 1. Implement the store and the retriever**

To keep the example runnable offline, the code below uses a toy hashed bag-of-words embedding. In real projects, it should be replaced with a real embedding model, such as the ones from OpenAI, Voyage, or sentence-transformers.

```python
import hashlib, re, time
from dataclasses import dataclass
import numpy as np

DIM = 512
STOP = {"the","a","an","to","of","in","on","for","and","me","my","is","i","it","due"}

def embed(text: str) -> np.ndarray:
    """Toy hashed bag-of-words embedding so the example runs offline.
    Replace with a real embedding model (OpenAI, Voyage, sentence-transformers)."""
    v = np.zeros(DIM)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w not in STOP:
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v

@dataclass
class MemoryItem:
    text: str
    kind: str          # "episodic" | "semantic"
    importance: float  # 0..1, set at write time (often by an LLM)
    created: float     # unix seconds
    vec: np.ndarray

class LongTermMemory:
    def __init__(self, half_life_days: float = 30):
        self.items: list[MemoryItem] = []
        self.half_life = half_life_days * 86400

    def write(self, text, kind="semantic", importance=0.5, now=None):
        now = now or time.time()
        v = embed(text)
        for it in self.items:                 # dedupe near-identical memories
            if float(it.vec @ v) > 0.95:
                it.created = now              # refresh instead of duplicating
                return it
        item = MemoryItem(text, kind, importance, now, v)
        self.items.append(item)
        return item

    def search(self, query, k=3, now=None, w_sim=1.0, w_rec=0.3, w_imp=0.3, min_sim=0.15):
        now = now or time.time()
        q = embed(query)
        scored = []
        for it in self.items:
            sim = float(it.vec @ q)
            if sim < min_sim:                 # relevance gate: never inject unrelated memories
                continue
            recency = 0.5 ** ((now - it.created) / self.half_life)
            scored.append((w_sim * sim + w_rec * recency + w_imp * it.importance, it))
        scored.sort(key=lambda s: -s[0])
        return [it for _, it in scored[:k]]

def build_system_prompt(query, ltm, budget_tokens=200) -> str:
    lines, used = [], 0
    for it in ltm.search(query, k=5):
        stamp = time.strftime("%Y-%m-%d", time.gmtime(it.created))
        line = f"- [{it.kind}, {stamp}] {it.text}"
        cost = max(1, len(line) // 4)
        if used + cost > budget_tokens:
            break
        lines.append(line); used += cost
    system = "You are a helpful engineering assistant."
    if lines:
        system += ("\n\nMemories about this user (may be outdated; use only if relevant):\n"
                   "<memories>\n" + "\n".join(lines) + "\n</memories>")
    return system
```

**Step 2. Write a few memories and build the prompt**

The code below writes four memories with different ages and importance, plus one duplicate, and then builds the system prompt for a query about rerunning the pipeline.

```python
DAY, now = 86400, time.time()
ltm = LongTermMemory()
ltm.write("User prefers Python for scripts and uses uv for packaging", "semantic", 0.8, now - 90*DAY)
ltm.write("Data pipeline runs on Airflow in us-east-1", "semantic", 0.6, now - 40*DAY)
ltm.write("Pipeline run failed on an expired S3 credential", "episodic", 0.7, now - 3*DAY)
ltm.write("Favourite colour is teal", "semantic", 0.2, now - 10*DAY)
ltm.write("User prefers Python for scripts and uses uv for packaging", "semantic", 0.8, now)  # dedupe + refresh

print(len(ltm.items))   # 4, the repeat refreshed instead of duplicating
print(build_system_prompt("Help me write scripts to rerun the pipeline", ltm))
```

Running the code above gives the output below:

```text
You are a helpful engineering assistant.

Memories about this user (may be outdated; use only if relevant):
<memories>
- [semantic, 2026-10-29] User prefers Python for scripts and uses uv for packaging
- [episodic, 2026-10-26] Pipeline run failed on an expired S3 credential
- [semantic, 2026-09-19] Data pipeline runs on Airflow in us-east-1
</memories>
```

A few things to notice from the output:

- The relevance threshold did its job. “Favourite colour is teal” never made it into the context. Without a threshold, top-k always returns k items, and irrelevant memories leak into the answers.
- Timestamps are included in the context. The model can only reason like “*this was 40 days ago, it may have changed*” if it can see the date.
- The framing line matters. Telling the model the memories “*may be outdated; use only if relevant*” reduces the chance that it forces a memory into every reply.
- For production, one would swap in real embeddings, a vector database with `user_id` / tenant filters, hybrid BM25 + vector search, and a reranker between the retriever and the assembler.

### 3.3 Example 3: Agent-Managed Memory via Tools

In Examples 1 and 2, the application decides what to remember. In this example, the model decides. Memory is exposed to the model as a tool, and the agent reads and writes memory files in its own reasoning loop. This is the idea behind MemGPT (the LLM managing its own memory), and Anthropic ships it in its API as the client-side memory tool `memory_20250818`.

In terms of the architecture, this is the agent loop in the reference diagram. The model emits `tool_use` blocks, the application executes them against a storage it controls, and returns `tool_result` blocks. Nothing is stored on the provider side. The sequence diagram below shows a typical session.

![Sequence diagram of an agent session using the memory tool](https://miro.medium.com/v2/resize:fit:1400/1*oaS8yTn5IIURMqF_eDrMVA.png)

**Step 1. Implement the storage backend**

Since the memory tool runs on the client side, protecting against path traversal is the application’s job, not the provider’s. The code below confines all paths under `<root>/memories`, and refuses to delete or rename the memory root itself, following the recommendations in Anthropic’s memory tool document.

```python
from pathlib import Path
import shutil

class FileMemoryStore:
    """Client-side backend for the memory tool. All paths are confined under <root>/memories."""
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        (self.root / "memories").mkdir(parents=True, exist_ok=True)

    def _safe(self, path: str) -> Path:
        p = (self.root / path.lstrip("/")).resolve()
        if not p.is_relative_to(self.root / "memories"):
            raise ValueError(f"path escapes memory dir: {path}")
        return p

    def handle(self, cmd: dict) -> str:
        op = cmd["command"]
        try:
            if op == "view":
                p = self._safe(cmd["path"])
                if p.is_dir():
                    files = sorted(str(c.relative_to(self.root)) for c in p.rglob("*") if c.is_file())
                    return "\n".join(files) or "(empty)"
                return p.read_text()
            if op == "create":
                p = self._safe(cmd["path"]); p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(cmd["file_text"]); return f"created {cmd['path']}"
            if op == "str_replace":
                p = self._safe(cmd["path"]); text = p.read_text()
                if text.count(cmd["old_str"]) != 1:
                    return "error: old_str must match exactly once"
                p.write_text(text.replace(cmd["old_str"], cmd.get("new_str", ""))); return "ok"
            if op == "insert":
                p = self._safe(cmd["path"]); lines = p.read_text().splitlines(keepends=True)
                lines.insert(cmd["insert_line"], cmd["insert_text"].rstrip("\n") + "\n")
                p.write_text("".join(lines)); return "ok"
            if op in ("delete", "rename"):
                p = self._safe(cmd.get("path") or cmd["old_path"])
                if p == self.root / "memories":
                    return "error: refusing to touch the memory root"
                if op == "delete":
                    shutil.rmtree(p) if p.is_dir() else p.unlink(); return "deleted"
                dst = self._safe(cmd["new_path"]); dst.parent.mkdir(parents=True, exist_ok=True)
                p.rename(dst); return "renamed"
            return f"error: unknown command {op}"
        except FileNotFoundError:
            return "error: not found"
        except ValueError as e:
            return f"error: {e}"
```

**Step 2. Run the agent loop**

The code below runs the agent loop. Here the memory tool is used together with context editing, so that the agent can save what matters to memory before old tool results get cleared from the context window. Note that the memory tool itself does not need a beta header; the `context-management-2025–06–27` header is for context editing.

```python
import anthropic

client = anthropic.Anthropic()
store = FileMemoryStore("./agent_state")

def run(task: str) -> str:
    messages = [{"role": "user", "content": task}]
    while True:
        resp = client.beta.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=4096,
            system="Check /memories before starting. Record decisions and progress there as you go.",
            tools=[{"type": "memory_20250818", "name": "memory"}],
            betas=["context-management-2025-06-27"],
            context_management={"edits": [{
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
                "exclude_tools": ["memory"],
            }]},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        results = [{"type": "tool_result", "tool_use_id": b.id, "content": store.handle(b.input)}
                   for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "user", "content": results})
```

**Step 3. Verify the storage backend**

The storage backend can be tested offline by feeding it the commands directly. The results are summarized below; the last two lines confirm the traversal and root-deletion guards work as expected.

![Results of testing the storage backend](https://miro.medium.com/v2/resize:fit:1354/1*kelNaH-UN7DAmR0GjbwWlA.png)

A few things to notice from this example:

- Memory becomes procedural. The system prompt (“check /memories first”) is effectively the write policy. Changing the prompt changes what gets remembered.
- Agent-managed vs. pipeline-managed. Here recall costs extra tool round-trips, but it is targeted. A pipeline like Example 2 adds no extra turns, but injects memories whether the model needs them or not. Many production systems combine both: always inject a small profile, and let the agent fetch the rest when needed.
- Plain files turn out to be a good format for memory. They are human-readable, diffable, easy to audit and easy for users to edit, and the model already knows how to work with them. Claude Code’s `CLAUDE.md` follows the same pattern.

### 3.4 Example 4: Assembling the Context for Prompt Caching

This example is where memory and the KV / prompt cache meet. Prompt caching reuses the computation for an exact token prefix, which means anything that changes in the prompt invalidates everything after it. So the rule of thumb is: order the context from the most stable part to the most volatile part, and put the cache breakpoints at the boundaries, as shown in the diagram below.

![Context ordered from most stable to most volatile, with cache breakpoints](https://miro.medium.com/v2/resize:fit:1140/1*Q3tWpdDfOi8hsaykK7W5_g.png)

With Anthropic’s API, the cached prefix follows the order tools → system → messages, so a breakpoint on the first system block also covers the tool definitions. The function below assembles a request following this rule.

```python
def assemble_request(system_rules: str, tools: list, memories: list[str],
                     summary: str, recent: list[dict], user_msg: str) -> dict:
    """Order blocks from most stable to most volatile so the provider's prefix cache hits."""
    system = [
        {"type": "text", "text": system_rules,
         "cache_control": {"type": "ephemeral"}},        # breakpoint 1: tools + rules (static)
    ]
    if memories or summary:
        dyn = ""
        if memories:
            dyn += "<memories>\n" + "\n".join(f"- {m}" for m in memories) + "\n</memories>\n"
        if summary:
            dyn += f"<conversation_summary>\n{summary}\n</conversation_summary>"
        system.append({"type": "text", "text": dyn})     # changes when memory changes
    messages = [dict(m) for m in recent]
    if messages:                                          # breakpoint 2: end of prior history
        last = messages[-1]
        last["content"] = [{"type": "text", "text": last["content"],
                            "cache_control": {"type": "ephemeral"}}]
    messages.append({"role": "user", "content": user_msg})  # volatile tail, never cached
    return {"model": "claude-sonnet-5-5", "max_tokens": 1024,
            "tools": tools, "system": system, "messages": messages}
```

To see what the function produces, the code below calls it with the Airflow scenario from Section 2.1: the memory tool as the only tool, two retrieved memories, a rolling summary, one prior exchange, and the new user message.

```python
import json

memories = ["[semantic, 2026-10-29] User prefers Python for scripts and uses uv for packaging",
            "[episodic, 2026-10-26] Pipeline run failed on an expired S3 credential"]
summary = ("User is debugging the nightly Airflow pipeline. S3 credential was rotated; "
           "DAG still fails at the export task.")
recent = [{"role": "user", "content": "The export task still fails after rotating the credential."},
          {"role": "assistant", "content": "Check that the Airflow connection aws_default picked up the new key."}]
tools = [{"type": "memory_20250818", "name": "memory"}]

req = assemble_request("You are a helpful engineering assistant.", tools, memories, summary, recent,
                       "Help me write a script to rerun the pipeline.")
print(json.dumps(req, indent=2))
```

Running it gives the request body below:

```json
{
  "model": "claude-sonnet-5-5",
  "max_tokens": 1024,
  "tools": [
    {
      "type": "memory_20250818",
      "name": "memory"
    }
  ],
  "system": [
    {
      "type": "text",
      "text": "You are a helpful engineering assistant.",
      "cache_control": {
        "type": "ephemeral"
      }
    },
    {
      "type": "text",
      "text": "<memories>\n- [semantic, 2026-10-29] User prefers Python for scripts and uses uv for packaging\n- [episodic, 2026-10-26] Pipeline run failed on an expired S3 credential\n</memories>\n<conversation_summary>\nUser is debugging the nightly Airflow pipeline. S3 credential was rotated; DAG still fails at the export task.\n</conversation_summary>"
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "The export task still fails after rotating the credential."
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Check that the Airflow connection aws_default picked up the new key.",
          "cache_control": {
            "type": "ephemeral"
          }
        }
      ]
    },
    {
      "role": "user",
      "content": "Help me write a script to rerun the pipeline."
    }
  ]
}
```

The two `cache_control` markers are where the two breakpoints land. The first one sits on the system rules, so the tool definitions and the rules form a prefix that stays the same across sessions. The second one sits on the last message of the prior history, so on the next turn everything up to and including that message can be read from the cache, and only the new user message (plus anything appended after it) has to be processed from scratch. The memory block between the two breakpoints is the part that changes per session, which is why it is placed after the static rules and not before them.

Where to put the memory in the prompt is a trade-off between cache hit rate and memory freshness. The table below compares the common choices.

![Where to put memory in the prompt: cache hit rate vs. freshness](https://miro.medium.com/v2/resize:fit:1400/1*TK2N0WobI516LV7Ny-Mf5w.png)

A common pattern in production is to put a fixed per-session memory profile in the system prompt (cached), and place the per-turn retrieved snippets in the volatile tail.

A few more things to keep in mind:

- Never put timestamps, request IDs, or lists in random order in the stable prefix. A single changed byte early in the prompt leads to a full cache miss.
- There are limits: at most 4 cache breakpoints per request, and a 20-block lookback window from each breakpoint, which can cause silent cache misses in long agentic loops if breakpoint 2 is not moved forward.
- Again, a cache hit is not recall. It makes the call cheaper and faster, but it never makes the model remember anything that is not in the context window.

## 4. Design Trade-offs and Pitfalls

In practice, most memory failures are actually context failures: the right fact was stored but never made it into the context window, or the wrong fact did. The table below lists the common ones, together with their symptoms and mitigations.

![Common memory failures, symptoms and mitigations](https://miro.medium.com/v2/resize:fit:1400/1*v_sC9f2oqrTOMCR2EIsdRg.png)

Besides, there are three design decisions which are better made explicitly, and early:

1. Who writes the memory? The pipeline (predictable and testable) or the agent (flexible but harder to bound). Hybrid approaches are common.
2. When is it written? In the hot path during the turn (available immediately, but adds latency), or in the background after the turn (no latency, but only available from the next turn).
3. How is it evaluated? Track recall@k on a labelled set of “should remember” questions, the rate of irrelevant memories injected, and the end-task success rate with and without memory. Without evaluation, adding memory tends to make things quietly worse.

## Conclusion

This quick note explained the relationship between context and memory in modern AI systems. **Context** is ***what the model sees on the current call***, and **memory** is ***what persists between calls***; since the model is stateless, memory only matters once it is loaded back into the context. With this in mind, every memory system boils down to a **write path** and a **read path** around the model, and the four examples demonstrated how the different parts of the reference architecture work, i.e., a *token-budgeted conversation buffer*, a *long-term semantic memory with retrieval*, an *agent* managing its own memory through tools, and a *cache-friendly way* to assemble the context. How to evaluate a memory system systematically is beyond the scope of this quick note, and could be the topic of a future one.

## References

- T.R. Sumers et al. (2023) *Cognitive Architectures for Language Agents (CoALA)*, [arXiv2309.02427](https://arxiv.org/abs/2309.02427)
- J.S. Packer et al. (2023) *MemGPT: Towards LLMs as Operating Systems,*
- *Generative Agents: Interactive Simulacra of Human Behavior*, [arXiv2304.03442](https://arxiv.org/abs/2304.03442)
- N.F. Liu et al., (2023) *Lost in the Middle: How Language Models Use Long Contexts*, [arXiv2307.03172](https://arxiv.org/abs/2307.03172)
- Anthropic (2025) *Managing context on the Claude Developer Platform*, [Claude Platform Doc](https://claude.com/resources/articles/context-management)
- *Anthropic API documentation* [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) and [Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- N. Alex (2025) *Memory & context management cookbook*, [Claude Cookbook](https://platform.claude.com/cookbook/tool-use-memory-cookbook)

---

*Originally published on [Medium](https://medium.com/@chuan-zhang/quick-notes-context-and-memory-in-modern-ai-systems-057ca9dfd337).*
