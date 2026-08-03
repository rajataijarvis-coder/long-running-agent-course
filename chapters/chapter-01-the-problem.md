# Chapter 01: The Problem with ChatGPT-style Agents

> Most AI agents are a single loop that dies the moment the process restarts, the context window fills up, or one API call fails.
> — Fareed Khan

## What We'll Cover

- Why chat-style agents don't survive real work
- The four failure modes that kill them
- What a durable agent looks like
- The architecture we'll build in this course

---

## The ChatGPT Loop

A typical ChatGPT-style agent works like this:

```
user asks → model thinks → model responds → conversation ends
```

For a single question, this is fine. For a week-long project, it is broken.

The model has no memory of what happened yesterday unless you paste the whole conversation into the context window. The context window is limited. The cost grows. The agent forgets details. And if the server restarts, the conversation is gone.

---

## Four Failure Modes

| Failure | Why it breaks a chat agent |
|---------|---------------------------|
| **Crash** | A process restart loses all in-memory state. There is no resume. |
| **Context limit** | Long missions exceed the model's token budget. Details get dropped. |
| **API failure** | One bad model call halts everything. There is no retry or checkpoint. |
| **No verification** | The model says the task is done, but the code doesn't compile or the tests fail. |

A real autonomous agent must survive all four.

---

## What Is a Long-Running Agent?

A long-running agent is a system that can:

1. Plan a multi-step mission
2. Work on it over hours, days, or weeks
3. Survive crashes and restarts
4. Verify its own work with real tests
5. Learn from failures
6. Ask a human only when necessary

The agent is not a conversation. It is a **stateful workflow**.

---

## The Architecture We Will Build

By the end of this course you will have this system:

```
┌─────────────────────────────────────┐
│         MissionWorkflow           │  ← Temporal durable workflow
│  (gather → act → verify → checkpoint) │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┬──────────────┐
    │                     │              │
┌───▼────┐           ┌────▼────┐   ┌────▼────┐
│  Lead  │           │Researchers│   │ Reviewer │
│Engineer│           │ (fan-out) │   │          │
└───┬────┘           └─────────┘   └─────────┘
    │
┌───▼─────────────────────────────────┐
│         Mission Anchor              │  ← git + four files
│ progress.md | checklist.json |      │
│ decisions.ndjson | events.ndjson    │
└─────────────────────────────────────┘
```

Every chapter adds one real component to this architecture. By Chapter 30 you will run a week-long mission end-to-end.

---

## What We'll Build in Code

This course ships with a working demo project in the `lra-demo/` folder. It starts as a tiny local runner and grows into a Temporal-backed, multi-agent system.

In this chapter, inspect the project:

```bash
cd lra-demo
ls -la src/lra/
```

You should see:

```
anchor.py
local_runner.py
loop.py
tools.py
verify.py
```

These are the foundation blocks. We will expand them as we go.

---

## Exercise

1. Read `lra-demo/src/lra/anchor.py`. Identify the four anchor files it manages.
2. Run the tests:
   ```bash
   cd lra-demo
   uv sync --extra dev
   uv run pytest
   ```
3. Note which tests pass and which fail. We will fix failures in later chapters.

---

## Key Takeaway

> A long-running agent is not a long chat. It is a durable workflow with externalized state, deterministic verification, and the ability to resume after failure.

---

## Next Chapter

**Chapter 02: Durability as an Engineering Property** — we will design every system component to survive interruption.
