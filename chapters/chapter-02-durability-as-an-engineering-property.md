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

## Why Smarter Models Do Not Help

In Chapter 01 we saw that chat-style agents collapse because they keep the mission state inside the model's context window. A bigger model, a better prompt, or a cleverer chain-of-thought does not change that. If the process dies, the window is gone. If the API call fails, the loop stops. If the context overflows, the agent forgets what it was doing.

Durability is not about reasoning quality. It is about whether the system can stop at any moment and resume correctly. That is an engineering property of the scaffolding around the model, not of the model itself.

This chapter makes that idea concrete. We will build a tiny but real crash-resume loop in plain Python, with no Temporal, no LLM, and no network. The goal is to prove the pattern: **the real state lives outside the process, and every cycle ends with a checkpoint.**

---

## The "Assume Interruption" Design Rule

The single most important rule in LRA is: **assume interruption.**

Every design decision is asked the same question: "What happens if the process is killed right here?" If the answer is "we lose work," the design is wrong.

This rule splits the world into two categories:

| Volatile (can be rebuilt) | Durable (must survive) |
|---|---|
| LLM context window | Checklist of remaining work |
| In-memory variables | Decision log |
| Model scratchpad | Git commits |
| Tool output cache | Verification results |
| Conversation history | Cost ledger |

The context window is a **lossy cache**. The durable files are the **source of truth**. When the agent wakes up after a crash, it does not ask the model to remember where things stand. It re-reads the durable files.

This is why `src/lra/agent/loop.py` ends every cycle with a `CycleOutcome` that is immediately written to git by the `GitMissionAnchor`. The model is allowed to think; the anchor is allowed to remember.

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

Notice what is durable here. `head_sha` is a git commit hash. `item_id` points to a checklist entry on disk. `verified` is a boolean produced by a deterministic verifier, not by the model's confidence. None of these live in the LLM window.

---

## A Crash-Resume Loop in Plain Python

To see the pattern without any framework, we will build `lra-demo/ch02_crash_resume.py`. It simulates an agent that must process a checklist of items. Each item requires one "work" step and one "verify" step. The process is killed at random moments. When it restarts, it re-reads its state file and continues exactly where it left off.

Create the file `lra-demo/ch02_crash_resume.py`:

```python
"""lra-demo/ch02_crash_resume.py

A minimal crash-resume loop that demonstrates durability as an engineering property.
No LLM, no Temporal, no network. Just a state file, a checkpoint function, and a loop
that assumes it can be killed at any moment.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_PATH = Path(".lra-demo/ch02_state.json")


@dataclass
class Item:
    id: str
    description: str
    done: bool = False
    attempts: int = 0


@dataclass
class MissionState:
    mission_id: str
    items: list[Item]
    cycle_count: int = 0
    last_action: str | None = None
    crashed_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "items": [asdict(item) for item in self.items],
            "cycle_count": self.cycle_count,
            "last_action": self.last_action,
            "crashed_at": self.crashed_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> MissionState:
        return cls(
            mission_id=raw["mission_id"],
            items=[Item(**item) for item in raw["items"]],
            cycle_count=raw.get("cycle_count", 0),
            last_action=raw.get("last_action"),
            crashed_at=raw.get("crashed_at"),
        )


def load_state() -> MissionState:
    if not STATE_PATH.exists():
        initial = MissionState(
            mission_id="demo-mission-001",
            items=[
                Item(id="item-1", description="Create project scaffold"),
                Item(id="item-2", description="Add hello.py"),
                Item(id="item-3", description="Add a test for hello.py"),
                Item(id="item-4", description="Run the test and verify it passes"),
            ],
        )
        save_state(initial)
        return initial
    with STATE_PATH.open() as f:
        return MissionState.from_dict(json.load(f))


def save_state(state: MissionState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w") as f:
        json.dump(state.to_dict(), f, indent=2)


def do_work(item: Item) -> None:
    """Simulate the model+tools doing work. Idempotent: safe to retry."""
    item.attempts += 1
    item.last_action = f"work_attempt_{item.attempts}"
    print(f"  → working on {item.id} (attempt {item.attempts})")
    time.sleep(0.2)


def verify(item: Item) -> bool:
    """Deterministic verifier. In a real system this runs tests/lint/build."""
    # Deterministic per item: item-1 and item-3 pass, item-2 needs two attempts, item-4 passes.
    if item.id == "item-2":
        return item.attempts >= 2
    return True


def maybe_crash(state: MissionState) -> None:
    """Simulate a host reboot or killed process."""
    if random.random() < 0.35:
        state.crashed_at = f"cycle_{state.cycle_count}"
        save_state(state)
        print(f"💥 CRASH at {state.crashed_at}!")
        sys.exit(1)


def run_cycle(state: MissionState) -> bool:
    """One durable cycle: pick next item, work, verify, checkpoint."""
    state.cycle_count += 1
    pending = [item for item in state.items if not item.done]

    if not pending:
        return False  # mission complete

    item = pending[0]
    state.last_action = f"start_{item.id}"
    save_state(state)

    do_work(item)

    state.last_action = f"verify_{item.id}"
    save_state(state)

    if verify(item):
        item.done = True
        state.last_action = f"done_{item.id}"
        print(f"  ✅ {item.id} verified and marked done")
    else:
        state.last_action = f"blocked_{item.id}"
        print(f"  ⛔ {item.id} blocked, will retry later")

    save_state(state)
    return True


def main() -> None:
    print("=" * 50)
    print("LRA Chapter 02 — Crash-Resume Demo")
    print("=" * 50)

    state = load_state()
    print(f"Loaded state: cycle={state.cycle_count}, "
          f"done={sum(1 for i in state.items if i.done)}/{len(state.items)}, "
          f"crashed_at={state.crashed_at}")

    while run_cycle(state):
        maybe_crash(state)

    print("\n🎉 Mission complete!")
    print(f"Total cycles: {state.cycle_count}")
    for item in state.items:
        print(f"  {item.id}: done={item.done}, attempts={item.attempts}")


if __name__ == "__main__":
    main()
```

### What the demo proves

1. **State is durable.** `MissionState` is written to `.lra-demo/ch02_state.json` before and after every meaningful action. If the process dies, the file tells the truth.
2. **Work is idempotent.** `do_work` increments `attempts` and can be retried safely. In the real system, tool calls are idempotent or journaled.
3. **Verification is deterministic.** `verify(item)` does not ask the model if the item looks done. It checks a concrete condition. Item `item-2` is deliberately designed to need two attempts, so you can watch the system retry without panicking.
4. **Resume is reconstruction, not recall.** When the script restarts, `load_state()` rebuilds situational awareness from the file. The model (here, just a function) does not need to remember anything.

Run it several times in a row. It will crash at random cycles, but each restart picks up exactly where it left off:

```bash
uv run python lra-demo/ch02_crash_resume.py
# 💥 CRASH at cycle_2!
uv run python lra-demo/ch02_crash_resume.py
# Loaded state: cycle=2, done=0/4, crashed_at=cycle_2
# ...
```

When it finally completes, inspect the state file:

```bash
cat .lra-demo/ch02_state.json
```

You will see a complete audit trail: every cycle, every attempt, every crash.

---

## From Local Checkpoint to Temporal Spine

The demo above is local. In later chapters we will replace the JSON file with git commits and the random crash with a real host reboot. Then we will wrap the same cycle inside a Temporal workflow.

The pattern does not change. The durable spine only makes it stronger:

- **Activities** journal every LLM call and tool call, with automatic retries.
- **Replay-from-cache** means a resumed workflow does not re-spend tokens on calls that already succeeded.
- **Durable sleep** lets the agent wait for hours or days at zero cost.
- **Continue-as-new** prevents the workflow history from growing forever.

But the mental model is the same: assume interruption, checkpoint after every cycle, and re-read the truth on wake-up.

---

## Hands-On Exercise

1. **Run the crash-resume demo** until it completes without crashing. Observe that `item-2` requires two attempts and the system retries it naturally.
2. **Introduce a bug** in `verify()` that makes it always return `False` for one item. Run the demo and confirm the system keeps retrying forever. This is the "blocked" state from `CycleOutcome`.
3. **Add a `max_attempts` field** to `Item` and modify the loop to skip an item after three failed attempts. Checkpoint the skip decision so the agent does not retry it on resume.
4. **Replace the JSON state file with a git commit** after every `save_state()`. Use `git -C .lra-demo log --oneline` to inspect the history. This previews Chapter 03.

---

> **Key Takeaway:** A durable agent does not rely on the model to remember. It writes the truth to disk after every cycle, assumes it will be interrupted, and reconstructs situational awareness by reading that truth on wake-up. The model thinks; the system remembers.

---

## Next Chapter

In **Chapter 03: Externalizing Truth — Git as Memory**, we replace the JSON state file with a real git repository. We will see why git is the perfect durable memory for an agent: commits are checkpoints, diffs are decision logs, and `HEAD` is the single source of truth.