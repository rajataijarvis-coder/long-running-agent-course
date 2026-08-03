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

In Chapter 01 we traced the ChatGPT-style agent: one process, one context window, one API session. It works until it doesn't — a restart, a full context window, or a failed call kills the run and the user has to start over.

A long-running agent cannot afford that. The model still thinks in short bursts; the *system* must remember where it is, what it has already verified, and what it still owes. Durability is not something the LLM provides. It is an engineering property we build into the loop.

The design rule that makes this concrete is **Assume Interruption**: at the start of every cycle, reconstruct situational awareness from durable storage. Never trust memory. If the process dies after this cycle, the next process must be able to pick up exactly where the last one left off.

---

## Volatile Context vs. Durable State

The context window is a lossy cache. It holds the prompt, recent observations, and tool outputs, but it is bounded, transient, and expensive to refill. The real state lives outside it.

In the `lra` codebase this separation is explicit:

- **Volatile context** — the messages passed to the model in `src/lra/agent/prompt.py`.
- **Durable state** — the git repo, the checklist, the decision log, and event records in `src/lra/state/` and `src/lra/contracts/state.py`.

`src/lra/agent/loop.py` imports two durable contracts:

```python
from lra.contracts.state import Checkpoint, EventRecord
```

These are the records that survive a reboot. The model window is rebuilt from them each cycle.

---

## Anatomy of a Checkpoint

At the end of every agent cycle the system records a `CycleOutcome`. This is the same shape returned by the real `AgentLoop.run_cycle` in `src/lra/agent/loop.py`:

```python
@dataclass
class CycleOutcome:
    item_id: str | None
    advanced: bool          # did the item move forward?
    verified: bool          # did the deterministic verifier pass?
    is_complete: bool       # is the whole mission done?
    head_sha: str           # git commit that captured the state
    tool_calls: int         # how many tool calls this cycle consumed
    turns: int              # how many model turns this cycle consumed
```

A checkpoint is not just "the model said it was done." It is a structured record that includes the git SHA of the saved state and the verifier's result. That is what lets the next cycle — or the next process — resume safely.

---

## A Plain-Python Crash-Resume Loop

Before we bring in Temporal, we can prove the idea with a tiny local script. Save this as `lra-demo/crash_resume.py` and run it. It keeps mission state in `lra-demo/state.json`, checkpoints before and after every cycle, and randomly simulates a process crash.

```python
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class CycleOutcome:
    item_id: str
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


@dataclass
class MissionState:
    task: str
    items: list[str]
    completed: set[str]
    attempts: dict[str, int]
    current: str | None
    cycle_count: int
    head_sha: str


STATE_PATH = Path("lra-demo/state.json")


def load_state(task: str, items: list[str]) -> MissionState:
    if STATE_PATH.exists():
        raw = json.loads(STATE_PATH.read_text())
        return MissionState(
            task=raw["task"],
            items=raw["items"],
            completed=set(raw["completed"]),
            attempts=raw["attempts"],
            current=raw["current"],
            cycle_count=raw["cycle_count"],
            head_sha=raw["head_sha"],
        )
    return MissionState(task, items, set(), {}, None, 0, "0000000")


def save_state(state: MissionState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    blob = asdict(state)
    blob["completed"] = sorted(state.completed)
    STATE_PATH.write_text(json.dumps(blob, indent=2))


def run_one_cycle(state: MissionState, verifier: Callable[[str], bool]) -> CycleOutcome:
    remaining = [i for i in state.items if i not in state.completed]
    if not remaining:
        return CycleOutcome("", False, False, True, state.head_sha, 0, 0)

    item = remaining[0]
    state.current = item
    state.attempts[item] = state.attempts.get(item, 0) + 1
    state.cycle_count += 1

    print(f"[cycle {state.cycle_count}] working on {item} (attempt {state.attempts[item]})")

    ok = verifier(item)
    if ok:
        state.completed.add(item)
        state.head_sha = f"{state.head_sha[:6]}{state.cycle_count:04x}"
    state.current = None

    return CycleOutcome(
        item_id=item,
        advanced=ok,
        verified=ok,
        is_complete=len(state.completed) == len(state.items),
        head_sha=state.head_sha,
        tool_calls=1,
        turns=1,
    )


def flaky_verifier(item: str) -> bool:
    """A deterministic-ish verifier: 'add tests' fails on the first attempt."""
    attempts = json.loads(STATE_PATH.read_text())["attempts"].get(item, 1)
    if item == "add tests" and attempts < 2:
        return False
    return True


def main() -> None:
    task = "Create a hello.py that prints hello and a test for it"
    items = ["create hello.py", "add tests", "run tests"]
    state = load_state(task, items)

    print(
        f"Resumed at cycle {state.cycle_count}, "
        f"completed={sorted(state.completed)}, head={state.head_sha}"
    )

    while True:
        save_state(state)  # checkpoint BEFORE work
        outcome = run_one_cycle(state, flaky_verifier)
        save_state(state)  # checkpoint AFTER work

        print(outcome)

        if outcome.is_complete:
            print("Mission complete.")
            break

        # Simulate a process crash 30% of the time.
        if random.random() < 0.3:
            print("!!! simulated crash !!!")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
```

The script is intentionally small, but it encodes the same pattern as `src/lra/agent/loop.py`:

1. Load durable state.
2. Checkpoint before doing work.
3. Run one verified cycle.
4. Checkpoint after work.
5. Repeat until the mission is complete.

Because the state file is written both before and after the cycle, a crash never loses more than one cycle's worth of progress, and the next run reconstructs everything from `lra-demo/state.json`.

---

## From Local Loop to Temporal Spine

The local script proves the pattern. The real `lra` system scales the same pattern to weeks by running the loop inside a Temporal workflow.

In `pyproject.toml` the durable execution backend is an optional extra:

```toml
[project.optional-dependencies]
durable = ["temporalio>=1.7"]
```

When installed, the code in `src/lra/durable/` wraps each LLM call, tool call, and verification step as a Temporal activity. Temporal journals every activity result, retries transient failures automatically, and replays from the journal after a crash. Idle time is spent in *durable sleep*, which costs nothing because no worker process is running.

The important insight is that Temporal does not replace the local loop — it *protects* it. The `AgentLoop` in `src/lra/agent/loop.py` is deliberately decoupled from Temporal so it can be unit tested on its own. The durable spine only adds process-level persistence and scheduling.

---

## Hands-On Exercise

1. Run the script above:

   ```bash
   python lra-demo/crash_resume.py
   ```

2. Let it crash at least once. Then run it again without changing anything. Verify that it resumes from the last completed item and does not redo work that was already verified.

3. Modify `flaky_verifier` so that a different item requires two attempts to pass. Confirm that the `attempts` counter in `lra-demo/state.json` tracks retries correctly.

4. Extra credit: add a `decision_log` field to `MissionState` and append one line per cycle. On resume, print the last three decisions to simulate reconstructing situational awareness from durable storage.

---

> **Key Takeaway:** Durability is not a prompt trick or a smarter model. It is the discipline of writing the real state to disk — git, JSON, or a Temporal journal — so that any cycle can be the last cycle and the next cycle can still continue.

---

## Next Chapter

State on disk is good. State that is versioned, diffable, and human-inspectable is better. In Chapter 03 we will make **Git the source of truth** for the agent's memory: checklists, decisions, and code all live in commits, and the model re-reads the repo every cycle.