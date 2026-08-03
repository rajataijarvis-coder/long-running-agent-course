# Chapter 01: The Problem with ChatGPT-style Agents

> Most AI agents are a single loop that dies the moment the process restarts, the context window fills up, or one API call fails.
> — Fareed Khan

## What We'll Cover

- Why most "agents" aren't really autonomous
- The three failure modes: crash, context, and API
- Why **durability** is an engineering problem, not a model problem
- Preview of the solution: externalized truth + durable execution + deterministic verification

---

## The ChatGPT Agent Pattern

A typical "agent" looks like this:

```python
while True:
    response = model(messages)
    action = parse(response)
    result = execute(action)
    messages.append(result)
```

Simple. Easy to demo. **Completely unreliable for real work.**

### Three Failure Modes

| Failure | What happens |
|---------|--------------|
| **Crash** | Process dies → all state lost → start over |
| **Context overflow** | `messages` grows until the model forgets the goal |
| **API failure** | One bad tool call kills the whole run |

A demo that runs for 5 minutes doesn't prove anything. A production agent needs to run for **days or weeks**.

---

## Why Smarter Models Don't Fix This

GPT-5 won't make your loop survive a reboot. A more capable model can reason better inside one turn, but it can't remember what happened before the crash.

**The fix is engineering:**
- Treat the context window as a cache, not a database
- Write state to disk/git/database every cycle
- Journal every action so it can be replayed
- Verify with real tests, not model self-assessment

---

## The LRA Answer

**LRA = Long-Running Agents** — an agent *organization* built for durability.

Three pillars:

### 1. Externalized Source of Truth
The real state lives outside the model:
- `progress.md` — what's been done
- `checklist.json` — what remains
- `decisions.ndjson` — why choices were made
- `events.ndjson` — what happened every cycle

The model re-reads this every turn. If you reboot on day 12, the agent reconstructs context in seconds.

### 2. Durable Execution
The loop runs inside a **Temporal workflow**. Every non-deterministic effect (LLM call, tool call, git commit) is journaled. A crash resumes exactly where it left off — no work lost, no tokens re-spent.

### 3. Deterministic Verification
A task is only "done" when tests/lint/build pass. The model doesn't get to say "looks good." Exit codes decide.

---

## What We'll Build

By the end of this course you'll have an agent that can:
1. Accept a multi-day software mission
2. Plan it into checkable items
3. Execute one item per cycle
4. Verify each item with real tests
5. Checkpoint to git
6. Resume after crashes
7. Improve its own prompts from failure traces
8. Run at **$0** on Ollama

---

## Exercise

Think of one long-horizon task you've tried to automate with an LLM.

Examples:
- Refactor a large codebase
- Build a full feature end-to-end
- Generate a 50-page report from research
- Migrate from one framework to another

**Write down three ways the simple loop would fail on that task.**

---

## Preview of Chapter 02

Next we'll look at **durability as an engineering property** — what it means for a system to be durable and how to design for interruption from the start.

---

## Key Takeaway

> The model thinks in short bursts. The system around it must run for weeks.
