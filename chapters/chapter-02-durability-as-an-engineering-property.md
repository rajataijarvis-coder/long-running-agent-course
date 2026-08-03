# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why durability belongs to the system, not the LLM
- The "Assume Interruption" design rule and how it changes every loop
- Volatile context vs. durable state: what lives in the window and what lives on disk/git
- The anatomy of a checkpoint using `CycleOutcome`
- A runnable crash-resume loop in plain Python
- How the durable spine (Temporal) extends this local idea to weeks-long missions

---

## From Chat Loop to Durable Loop

In Chapter 01 we saw that a ChatGPT-style agent is essentially one long conversation. The model holds the plan, the progress, and the intent inside its context window. When the process dies, the context window is truncated, or an API call fails, the agent loses the plot.

A durable agent does the opposite: it **assumes interruption**. The model is treated as a stateless worker that thinks in short bursts. The system around it owns the state, the plan, and the proof of progress. If the process restarts on day 12, the next cycle reconstructs situational awareness in seconds by reading the durable state, not by asking the model to remember.

This is why durability is an engineering property. The LLM does not get "better at long tasks" by changing the prompt. The system gets better by making every cycle idempotent, journaled, and recoverable.

## Assume Interruption

The design rule is simple: **every cycle must end with a durable checkpoint, and the next cycle must be able to continue by reading that checkpoint alone.**

This changes how you write every loop:

- No hidden state in Python variables.
- No "the model knows where we are" assumptions.
- Every decision, attempt, and result is written to durable storage before the cycle ends.
- If a cycle is killed halfway, the next cycle re-reads the checkpoint and may redo the last step, but it never loses the mission.

In the LRA codebase this rule is enforced by the `AgentLoop` in `src/lra/agent/loop.py`. It runs one cycle of **gather → act → verify → checkpoint** and returns a `CycleOutcome` that is the only thing the durable spine needs to know what happened.

## Volatile Context vs. Durable State

| Volatile (context window) | Durable (disk/git/database) |
|---|---|
| Model's working memory | Checklist, decision log, ownership map |
| Truncated by token limits | Git commits and file tree |
| Lost on restart | Cost ledger, event trace, skill library |
| Good for reasoning | Good for truth |

The context window is a **lossy cache**. The real state lives outside it. Every cycle starts by gathering the current truth from git and the structured logs, not by asking the model to recall.

## Anatomy of a Checkpoint

The checkpoint contract in LRA is `CycleOutcome`, defined in `src/lra/agent/loop.py`:

```python
@dataclass
class CycleOutcome:
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int
```

Each field has a specific job:

- `item_id`: which checklist item was worked on.
- `advanced`: did the cycle make forward progress (e.g., wrote code, ran tests).
- `verified`: did the deterministic verifier pass (we will build this in Chapter 06).
- `is_complete`: is the checklist item now done.
- `head_sha`: the git commit SHA after the checkpoint — the durable anchor.
- `tool_calls` / `turns`: telemetry for cost and loop detection.

The durable spine does not care *how* the model reasoned. It only cares about this outcome. That decoupling is what makes the system robust.

## A Runnable Crash-Resume Loop in Plain Python

Before we add Temporal, let's prove the idea with a tiny local loop. Save this as `lra-demo/ch02/crash_resume.py`:

```python
# lra-demo/ch02/crash_resume.py
"""Minimal crash-resume loop that demonstrates durability without Temporal."""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


# In the real LRA package this class lives in src/lra/agent/loop.py.
@dataclass
class CycleOutcome:
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


STATE_FILE = Path("lra-demo/ch02/state.json")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "items": [
            {"id": "item-1", "title": "Scaffold project", "done": False},
            {"id": "item-2", "title": "Add hello.py", "done": False},
            {"id": "item-3", "title": "Run tests", "done": False},
        ],
        "attempts": {},
        "head_sha": "0000000",
    }


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def do_work(item: dict) -> CycleOutcome:
    # Simulate non-determinism: sometimes we crash before saving.
    if random.random() < 0.25:
        raise SystemExit("Simulated crash mid-work")

    # Simulate verification: 80% pass after one attempt.
    verified = random.random() < 0.8
    return CycleOutcome(
        item_id=item["id"],
        advanced=True,
        verified=verified,
        is_complete=verified,
        head_sha=f"sha-{item['id']}",
        tool_calls=random.randint(1, 4),
        turns=1,
    )


def checkpoint(outcome: CycleOutcome, state: dict) -> None:
    state["head_sha"] = outcome.head_sha
    state["attempts"][outcome.item_id] = state["attempts"].get(outcome.item_id, 0) + 1
    if outcome.is_complete:
        for it in state["items"]:
            if it["id"] == outcome.item_id:
                it["done"] = True
    save_state(state)


def run_one_cycle(state: dict) -> bool:
    next_item = next((it for it in state["items"] if not it["done"]), None)
    if next_item is None:
        print("All items complete.")
        return False

    attempt = state["attempts"].get(next_item["id"], 0) + 1
    print(f"Working on {next_item['id']}: {next_item['title']} (attempt {attempt})")

    outcome = do_work(next_item)
    checkpoint(outcome, state)

    print(
        f"  outcome: advanced={outcome.advanced} verified={outcome.verified} "
        f"complete={outcome.is_complete} head={outcome.head_sha}"
    )
    return True


if __name__ == "__main__":
    random.seed(os.environ.get("SEED"))
    state = load_state()

    try:
        while run_one_cycle(state):
            pass
    except SystemExit as exc:
        print(f"CRASH: {exc}")
        print("Re-run the script. It will resume from the last checkpoint in state.json.")
        raise
```

Run it from the repo root:

```bash
python lra-demo/ch02/crash_resume.py
```

It will probably crash before finishing. When it does, inspect `lra-demo/ch02/state.json`. You will see the last checkpoint: which item was attempted, how many attempts, and the current `head_sha`. Run the script again. It resumes from the checkpoint, not from the beginning. Repeat until all three items are complete.

This is the local version of what Temporal does at scale: **journal the outcome, then continue from the journal.**

## How Temporal Extends This Idea

The local loop writes state to a JSON file. In production, LRA wraps the same `CycleOutcome`-producing cycle inside a Temporal workflow. The durable spine gives us:

- **Activity journaling**: every LLM call, tool call, and verification is recorded.
- **Replay-from-cache**: after a crash, already-completed activities are replayed from the journal; tokens are not re-spent.
- **Retries with backoff**: transient API failures are retried automatically.
- **Durable sleep**: the agent can sleep for hours or days at zero compute cost.

We will build the Temporal spine in Chapters 12–15. The important point now is that the *contract* — the `CycleOutcome` returned to the durable spine — stays the same whether the loop is local or distributed.

## Hands-On Exercise

1. Run `python lra-demo/ch02/crash_resume.py` until it completes. Count how many crashes and retry attempts occurred by reading `lra-demo/ch02/state.json`.
2. Modify `do_work` so that `verified` is `True` only when a specific file exists on disk, e.g.:
   ```python
   verified = Path(f"lra-demo/ch02/artifacts/{item['id']}.done").exists()
   ```
   Create the `artifacts` directory and manually create the `.done` files one at a time between runs. Observe that the loop cannot mark an item complete until the durable artifact exists.
3. Compare your modified script to `src/lra/agent/loop.py`. Identify where the real `AgentLoop` gathers state, dispatches tools, runs the verifier, and returns a `CycleOutcome`.

## Key Takeaway

> Durability is not a feature of the LLM; it is a property of the loop around it. Assume every cycle will be interrupted, write a checkpoint before you sleep, and make the next cycle start by reading the truth from durable storage.

## Next Chapter

In **Chapter 03: Externalizing Truth — Git as Memory**, we replace the JSON state file with a real git repository. We will see why git commits, branches, and diffs become the mission's source of truth, and how the `head_sha` in `CycleOutcome` anchors every checkpoint to an immutable history.