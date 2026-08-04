# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why durability belongs to the *system*, not the LLM
- The "Assume Interruption" design rule and how it changes every loop
- Volatile context vs. durable state: what lives in the window and what lives on disk/git
- The anatomy of a checkpoint using `CycleOutcome`
- A runnable crash-resume loop in plain Python
- How the durable spine (Temporal) extends this local idea to weeks-long missions

---

## From Chat Loop to Durable Loop

In Chapter 01 we saw that a ChatGPT-style agent is basically one long conversation. If the process dies, the context window fills, or an API call flakes, the whole run is gone. The model itself is not the problem — it is doing exactly what it was designed to do: produce the next token given a prompt. The problem is that the *system* around the model treats the conversation as the database.

Durability fixes that by moving the database out of the conversation. The model still thinks in short bursts, but every burst is anchored to state that survives crashes. This chapter makes that concrete with a small, runnable Python loop you can kill and resume.

## Assume Interruption

The central design rule of a long-running agent is: **assume interruption**.

That means:

- Every cycle starts by re-reading the durable state.
- No progress is "real" until it has been checkpointed.
- In-memory context is a cache, not a source of truth.

If you design every loop with that rule, a host reboot on day 12 is a non-event. The process restarts, loads the last checkpoint, and reconstructs situational awareness in seconds.

## Volatile Context vs. Durable State

| Volatile (lives in the window) | Durable (lives on disk/git) |
|---|---|
| Prompt text | Checklist of items |
| Model reasoning trace | Decision log |
| Tool observations from this cycle | Git commits / file tree |
| Scratchpad notes | Cost ledger |
| Current `messages` list | Last verified checkpoint hash |

The context window is a **lossy cache**. It can be truncated, summarized, or rebuilt. The durable layer is the **source of truth**. The agent re-reads it at the top of every cycle, so even if the model has zero memory of yesterday, the system knows exactly where it stands.

## A Runnable Crash-Resume Loop

The file `lra-demo/ch02/durable_loop.py` is a minimal version of the same loop that lives in `src/lra/agent/loop.py`. It has no LLM and no Temporal server — just a JSON checkpoint and a deterministic verifier — so you can see durability in isolation.

```python
#!/usr/bin/env python3
"""lra-demo/ch02/durable_loop.py

A minimal crash-resume loop. It proves that durability is a property of the
loop, not the model: the same cheap stub "model" can keep making progress as
long as checkpoints are written to disk before the process dies.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

STATE_FILE = Path("lra-demo/ch02/state.json")
WORKSPACE = Path("lra-demo/ch02/workspace")

WORK_ITEMS = [
    {"id": "hello", "instruction": "Create hello.txt containing 'hello'"},
    {"id": "world", "instruction": "Create world.txt containing 'world'"},
    {"id": "done",  "instruction": "Create done.txt containing 'done'"},
]


@dataclass
class Checkpoint:
    item_id: str | None = None
    attempts: int = 0
    verified_count: int = 0
    last_sha: str = "init"


def load_checkpoint() -> Checkpoint:
    if STATE_FILE.exists():
        return Checkpoint(**json.loads(STATE_FILE.read_text()))
    return Checkpoint()


def save_checkpoint(cp: Checkpoint) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(cp), indent=2))


def select_next_item(cp: Checkpoint) -> dict | None:
    if cp.verified_count >= len(WORK_ITEMS):
        return None
    return WORK_ITEMS[cp.verified_count]


def do_work(item: dict) -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    path = WORKSPACE / f"{item['id']}.txt"
    expected = item["instruction"].split(" containing ")[1].strip("'\"")
    path.write_text(expected + "\n")
    return path


def verify(item: dict) -> bool:
    path = WORKSPACE / f"{item['id']}.txt"
    if not path.exists():
        return False
    expected = item["instruction"].split(" containing ")[1].strip("'\"")
    return path.read_text().strip() == expected


def maybe_crash(label: str) -> None:
    target = os.environ.get("CRASH_AFTER", "")
    if target == label:
        print(f"[INJECTED CRASH] killed after '{label}'", file=sys.stderr)
        sys.exit(1)


def run_cycle() -> bool:
    cp = load_checkpoint()

    item = select_next_item(cp)
    if item is None:
        print("All items verified. Mission complete.")
        return False

    # Journal that we are starting this item.
    cp.item_id = item["id"]
    cp.attempts += 1
    save_checkpoint(cp)

    print(f"Cycle start: item={item['id']} attempts={cp.attempts}")

    do_work(item)
    maybe_crash("work")

    ok = verify(item)
    maybe_crash("verify")

    if ok:
        cp.verified_count += 1
        cp.attempts = 0
        cp.last_sha = f"sha-{item['id']}"
        save_checkpoint(cp)
        maybe_crash("checkpoint")
        print(f"  verified -> {WORKSPACE / item['id']}.txt; checkpoint={cp.last_sha}")
    else:
        save_checkpoint(cp)
        print(f"  FAILED -> will retry on resume")

    return True


if __name__ == "__main__":
    while run_cycle():
        pass
```

## Code Walkthrough

1. **`Checkpoint`** — the durable record. It is the only object that must survive a crash.
2. **`load_checkpoint` / `save_checkpoint`** — every cycle starts by reading `state.json` and ends by writing it. No exception.
3. **`select_next_item`** — picks the next item based on `verified_count`, not on any in-memory pointer. This is what makes resumption trivial.
4. **`do_work` / `verify`** — the "agent" here is just file I/O, but the split is the same as in the real system: a writer produces something, and a deterministic verifier decides if progress happened.
5. **`maybe_crash`** — lets us simulate a kill at three different moments: after work, after verify, or after the checkpoint has been saved.

The important behavior is in the crash points:

- If `CRASH_AFTER=work`, the file is written but `verified_count` is not updated. On resume, the same item is retried.
- If `CRASH_AFTER=checkpoint`, the checkpoint is already saved. On resume, the loop skips to the next item.
- If `CRASH_AFTER=verify`, the verification passed but the count was not saved, so the item is retried once.

This is exactly the contract that `src/lra/agent/loop.py` enforces through `CycleOutcome`: an item is only `advanced` and `verified` when the durable layer has recorded it.

## Mapping to the Real LRA System

In the full project, the local JSON checkpoint becomes a git commit via `GitMissionAnchor`, and the durable spine is Temporal. The same ideas still apply:

- `AgentLoop.run_cycle()` returns a `CycleOutcome` with `advanced`, `verified`, `is_complete`, and `head_sha`.
- Every LLM call and tool call is executed inside a Temporal activity, which is journaled and replayed from cache.
- Idle time is spent in durable sleep, costing nothing.

The model is still a short-burst thinker. The system is what makes the mission long-running.

## Hands-On Exercise

1. Run the loop cleanly and inspect the durable state:

```bash
rm -rf lra-demo/ch02/state.json lra-demo/ch02/workspace
python lra-demo/ch02/durable_loop.py
cat lra-demo/ch02/state.json
ls lra-demo/ch02/workspace
```

2. Reset and inject a crash **before** the checkpoint is saved:

```bash
rm -rf lra-demo/ch02/state.json lra-demo/ch02/workspace
CRASH_AFTER=work python lra-demo/ch02/durable_loop.py
python lra-demo/ch02/durable_loop.py
```

Notice that `attempts` increments and the same item is completed on the second run.

3. Reset and inject a crash **after** the checkpoint is saved:

```bash
rm -rf lra-demo/ch02/state.json lra-demo/ch02/workspace
CRASH_AFTER=checkpoint python lra-demo/ch02/durable_loop.py
python lra-demo/ch02/durable_loop.py
```

This time the second run skips the already-verified item and starts on the next one.

4. Try `CRASH_AFTER=verify` and confirm the loop retries the item safely because the checkpoint was never updated.

> **Key Takeaway:** Durability is an engineering property, not a model capability. The model thinks in short bursts; the system runs for weeks by keeping real state outside the window, journaling every step, and replaying from cache.

## Next Chapter

In Chapter 03: **Externalizing Truth — Git as Memory**, we will replace the JSON checkpoint with real git commits and learn why a version-controlled file tree is the right long-term memory for a software-building agent.