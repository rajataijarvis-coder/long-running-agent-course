# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why a smarter model cannot fix a loop that dies on restart
- The "Assume Interruption" design rule and how it changes every decision
- Volatile context vs. durable state: what belongs in the LLM window and what belongs on disk
- The anatomy of a checkpoint using `CycleOutcome` from `src/lra/agent/loop.py`
- A runnable crash-resume loop in plain Python (`lra-demo/ch02_crash_resume.py`)
- How this local idea extends into the Temporal durable spine used in later chapters

---

## Why a Smarter Model Is Not Enough

In Chapter 01 we saw that the ChatGPT-style pattern is a single, in-memory loop. It works while the process is alive, the API is responsive, and the context window is empty. As soon as one of those conditions breaks, the run is over.

You cannot fix that by using a smarter model. A bigger context window only delays the loss. Better reasoning does not create a record on disk. Lower API latency does not survive a host reboot. Durability is a property of the *system*, not the *model*.

The LRA design rule that follows is **Assume Interruption**. Every cycle, every activity, every write must be written as if the process could die in the next millisecond. The model is allowed to think in short bursts; the system is responsible for remembering what those bursts produced.

This chapter makes that rule concrete with a small, runnable Python program that survives a `kill -9`. The same ideas are later promoted into Temporal workflows, but the core concept is local and testable first.

---

## Volatile Context vs. Durable State

The LLM context window is a **lossy cache**. It is fast to read, limited in size, and disappears when the process ends. Anything that must survive a restart belongs in **durable state** outside the window.

| Volatile (context window) | Durable (disk / git / DB) |
|---|---|
| Current plan being reasoned about | Checklist of items and statuses |
| Tool output from the last minute | Event log of every attempt |
| Model's internal chain-of-thought | Decision log with rationale |
| Token budget scratchpad | Cost ledger with cumulative spend |
| File contents being edited | The actual files in git |

The agent loop in `src/lra/agent/loop.py` encodes this split. It returns a `CycleOutcome` that contains only the durable facts:

```python
@dataclass
class CycleOutcome:
    """The result of one agent cycle."""
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int
```

Notice what is *not* in `CycleOutcome`: the full conversation, the model's reasoning, or the raw tool output. Those are reconstructed from the durable anchor when the loop restarts. The durable record is intentionally small so it can be cheaply checkpointed every cycle.

---

## The Anatomy of a Checkpoint

A checkpoint in LRA has three responsibilities:

1. **Capture progress** — which checklist item was worked on and whether it advanced.
2. **Capture truth** — the git SHA of the repo after the work, so the state can be reproduced.
3. **Capture cost** — how many turns and tool calls were spent, so the budget governor can decide whether to continue.

The `CycleOutcome` fields map directly to these responsibilities:

- `item_id` — the checklist item being worked on.
- `advanced` — whether any forward progress happened this cycle.
- `verified` — whether the deterministic verifier passed.
- `is_complete` — whether the whole mission is finished.
- `head_sha` — the git commit SHA that contains the real state.
- `tool_calls` / `turns` — cost counters for the governor.

This is the contract between the agent loop and the durable spine. The loop does not know whether it is running inside Temporal or a unit test; it only promises to return a `CycleOutcome` that can be journaled.

---

## A Runnable Crash-Resume Loop

The file `lra-demo/ch02_crash_resume.py` demonstrates the idea without any external dependencies. It simulates an agent working through a checklist of three items. The process can be killed at any time and, when restarted, resumes exactly where it left off.

Create `lra-demo/ch02_crash_resume.py` with the following code:

```python
#!/usr/bin/env python3
"""lra-demo/ch02_crash_resume.py

A self-contained crash-resume agent loop. Run it, then press Ctrl-C or
send SIGKILL while it is "working". Run it again and it resumes from the
last durable checkpoint instead of starting over.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CHECKPOINT_PATH = Path(".lra-demo/ch02_checkpoint.json")


@dataclass
class Checkpoint:
    item_index: int = 0
    attempts: int = 0
    verified_count: int = 0
    total_turns: int = 0


def load_checkpoint() -> Checkpoint:
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[resume] loaded checkpoint: {data}")
        return Checkpoint(**data)
    return Checkpoint()


def save_checkpoint(cp: Checkpoint) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(asdict(cp), f, indent=2)
    print(f"[checkpoint] saved: {asdict(cp)}")


ITEMS = [
    "Create hello.py that prints 'hello'",
    "Add a pytest test for hello.py",
    "Run the test and verify it passes",
]


def do_work(cp: Checkpoint) -> bool:
    """Simulate one cycle of work on the current item.

    Returns True when the whole mission is complete.
    """
    if cp.item_index >= len(ITEMS):
        return True

    item = ITEMS[cp.item_index]
    cp.attempts += 1
    cp.total_turns += 1

    print(f"[work] item {cp.item_index}: {item} (attempt {cp.attempts}, turn {cp.total_turns})")

    # Simulate variable work duration and occasional failure.
    time.sleep(1.5)
    success = random.random() > 0.25

    if success:
        print(f"[work] item {cp.item_index} verified")
        cp.verified_count += 1
        cp.item_index += 1
        cp.attempts = 0
    else:
        print(f"[work] item {cp.item_index} not yet verified, will retry")

    save_checkpoint(cp)
    return cp.item_index >= len(ITEMS)


def main() -> int:
    cp = load_checkpoint()

    print(f"[start] mission state: {asdict(cp)}")
    complete = False

    try:
        while not complete:
            complete = do_work(cp)
    except KeyboardInterrupt:
        print("\n[interrupt] caught KeyboardInterrupt — state is already on disk")
        return 130

    print(f"[done] mission complete after {cp.total_turns} turns")
    CHECKPOINT_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Walkthrough

1. **`Checkpoint`** keeps only the durable facts: which item is active, how many attempts have been made, how many items are verified, and the total turn count.
2. **`load_checkpoint`** reads the JSON file if it exists. On first run there is no file, so it starts from zero.
3. **`do_work`** simulates one cycle. It increments counters, sleeps to represent real work, and randomly succeeds or fails.
4. **`save_checkpoint`** is called **inside** the cycle, before the loop decides what to do next. That is the critical placement: if the process dies after the work but before the save, the cycle is replayed; if it dies after the save, the next run resumes from the saved state.
5. **`main`** catches `KeyboardInterrupt` so a `Ctrl-C` does not corrupt the checkpoint. The state is already on disk, so the process can exit cleanly.

### Run It

```bash
cd lra-demo
python ch02_crash_resume.py
```

Let it run for a few seconds, then press `Ctrl-C`. You will see output like:

```
[resume] loaded checkpoint: {'item_index': 0, 'attempts': 0, 'verified_count': 0, 'total_turns': 0}
[start] mission state: {'item_index': 0, 'attempts': 0, 'verified_count': 0, 'total_turns': 0}
[work] item 0: Create hello.py that prints 'hello' (attempt 1, turn 1)
^C
[interrupt] caught KeyboardInterrupt — state is already on disk
```

Now run it again:

```bash
python ch02_crash_resume.py
```

It resumes from the saved `item_index`, not from the beginning. After enough successful cycles it deletes the checkpoint and prints:

```
[done] mission complete after N turns
```

This is the local version of the durable spine. The real LRA system does the same thing, but the checkpoint is a Temporal event and the state is a git commit.

---

## From Local Checkpoint to Temporal Durable Spine

The demo above is intentionally simple. In the full LRA package, the same responsibilities are handled by three layers:

1. **`GitMissionAnchor`** (`src/lra/state/mission_anchor.py`) — writes the real state to git after every verified cycle.
2. **`AgentLoop`** (`src/lra/agent/loop.py`) — produces a `CycleOutcome` that the anchor turns into a commit.
3. **`MissionWorkflow`** (`src/lra/durable/`) — runs the loop inside a Temporal workflow so that process crashes resume automatically.

The progression is:

- **This chapter:** durable checkpoint in a plain Python file.
- **Chapter 04:** the same cycle wrapped in git commits.
- **Chapter 12:** the same cycle wrapped in Temporal workflows.

The design does not change; only the durability mechanism gets stronger.

---

## Hands-On Exercise

1. Run `lra-demo/ch02_crash_resume.py` to completion once and note the total turn count.
2. Run it again and kill the process with `Ctrl-C` after one or two items complete. Verify that `.lra-demo/ch02_checkpoint.json` exists and contains the correct `item_index`.
3. Restart the script and confirm it resumes from the saved index, not from item 0.
4. Open the checkpoint file, manually set `attempts` to `10`, save it, and restart the script. Observe how the loop behaves when it reads a tampered checkpoint. (This is why the real system signs or commits checkpoints.)
5. Modify `do_work` to write a real file (for example, create `hello.py`) instead of just sleeping. Make the checkpoint include a SHA or file hash so the resume can detect whether the last cycle's work actually landed on disk.

---

## Key Takeaway

> Durability is not a side effect of good prompting; it is an engineering property built into every write. Assume interruption, keep the real state outside the model, and checkpoint after every verified cycle.

---

## Next Chapter

In Chapter 03 we externalize the source of truth completely. We replace the JSON checkpoint with a git repository and use commits as the durable memory of the mission.