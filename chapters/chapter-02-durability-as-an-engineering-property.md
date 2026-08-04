# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why durability is a *system* concern, not a smarter-model concern
- The "Assume Interruption" design rule and how it changes every loop
- Volatile context vs. durable state: what lives in the window and what lives on disk/git
- The anatomy of a checkpoint using `CycleOutcome` from `src/lra/agent/loop.py`
- A runnable crash-resume loop in plain Python (`lra-demo/ch02_crash_resume.py`)
- How the durable spine (Temporal) extends this local idea to week-long missions

---

## From Chat Loop to Durable Loop

In Chapter 01 we saw that a ChatGPT-style agent is a single, in-memory loop. It dies when the process restarts, the context window fills, or an API call fails. The obvious fix is to ask for a "smarter" model, but that does not solve any of those problems. A smarter model still runs inside the same fragile loop.

Durability is an engineering property. It means the *system* keeps the mission alive even when the model, the process, or the host is interrupted. The design rule that makes this possible is **Assume Interruption**: every cycle must be written as if the process could die immediately after it.

That rule changes how you build every part of the agent:

- The model's context window is treated as a **lossy cache**, not a database.
- The real state lives outside the model, on disk and in git.
- Every meaningful step is journaled before the next step begins.
- Progress is verified by deterministic checks, not by the model's opinion.

The model still does the thinking, but the system owns the memory.

---

## Volatile Context vs. Durable State

There are two kinds of state in a long-running agent.

**Volatile context** is what the model sees in its prompt. It is fast, convenient, and limited. It disappears when the process restarts and it degrades as the context window fills. It is good for one turn of reasoning, bad for a week of work.

**Durable state** is the real source of truth. In LRA it is:

- A git repo on disk (`src/lra/state/`)
- A structured mission/checklist file
- A decision log and event journal
- A durable execution history managed by Temporal

When the system resumes after a crash, it does not try to remember what it was doing. It re-reads the durable state and reconstructs situational awareness in seconds. This is the core idea behind the `GitMissionAnchor` we will use later: the anchor is the saved state, not the model's memory.

---

## Anatomy of a Checkpoint

A checkpoint is the atomic unit of durable progress. In LRA, one cycle of the inner agent loop returns a `CycleOutcome` defined in `src/lra/agent/loop.py`:

```python
@dataclass
class CycleOutcome:
    """The result of one agent cycle."""

    item_id: str | None
    advanced: bool          # did we move the mission forward?
    verified: bool          # did deterministic verification pass?
    is_complete: bool       # is the whole mission done?
    head_sha: str           # git commit that captured this state
    tool_calls: int         # how many tool calls this cycle
    turns: int              # how many model turns this cycle
```

This small object is the contract between the agent and the durable spine. It says: "Here is what I worked on, whether it passed real verification, and the exact git commit that captured the result." The durable spine writes this to the mission log before asking for the next cycle. If the process dies, the next process picks up from the last `CycleOutcome`, not from the model's fading memory.

The `AgentLoop` class in the same file is built around this contract:

```python
class AgentLoop:
    """Runs one verified cycle of work using a model + tools + sandbox + verifier + anchor."""

    def __init__(
        self,
        *,
        model: ModelProvider,
        dispatcher: ToolDispatcher,
        verifier: Verifier,
        anchor: GitMissionAnchor,
        ledger: CostLedger | None = None,
        recorder: TraceRecorder | None = None,
    ) -> None:
        ...
```

Notice that the loop takes a `GitMissionAnchor` and a `Verifier`. The anchor writes state; the verifier decides whether progress is real. The model is just one of several inputs.

---

## A Runnable Crash-Resume Loop

To make the idea concrete before we add Temporal, git, and sandboxes, here is a minimal durable loop in plain Python. It keeps its state in a JSON file and survives a simulated crash. Save it as `lra-demo/ch02_crash_resume.py`.

```python
# lra-demo/ch02_crash_resume.py
"""Minimal crash-resume loop: the state file is the source of truth."""

from __future__ import annotations

import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_PATH = Path(".lra-demo/ch02_state.json")


@dataclass
class CycleOutcome:
    """Same shape as the real CycleOutcome in src/lra/agent/loop.py."""

    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


@dataclass
class MissionState:
    mission_id: str
    items: list[str]
    done: list[bool]
    current: int
    cycles: int
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MissionState:
        return cls(
            mission_id=d["mission_id"],
            items=d["items"],
            done=d["done"],
            current=d["current"],
            cycles=d["cycles"],
            events=d.get("events", []),
        )


def load_or_init() -> MissionState:
    if STATE_PATH.exists():
        print(f"resuming from {STATE_PATH}")
        with open(STATE_PATH) as f:
            return MissionState.from_dict(json.load(f))

    return MissionState(
        mission_id=str(uuid.uuid4())[:8],
        items=[
            "scaffold project",
            "write hello.py",
            "write test_hello.py",
            "run pytest",
        ],
        done=[False, False, False, False],
        current=0,
        cycles=0,
    )


def save(state: MissionState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state.to_dict(), f, indent=2)


def do_work(state: MissionState) -> CycleOutcome:
    """Pretend to do one item. Verification succeeds 70% of the time."""
    if state.current >= len(state.items):
        return CycleOutcome(
            item_id=None,
            advanced=False,
            verified=True,
            is_complete=True,
            head_sha="fake-sha",
            tool_calls=0,
            turns=0,
        )

    item = state.items[state.current]
    print(f"[cycle {state.cycles}] working on item {state.current}: {item}")

    # Simulate real verification: sometimes it fails, so we do not advance.
    verified = random.random() > 0.3
    advanced = verified

    if advanced:
        state.done[state.current] = True
        state.current += 1

    state.cycles += 1

    outcome = CycleOutcome(
        item_id=item,
        advanced=advanced,
        verified=verified,
        is_complete=state.current >= len(state.items),
        head_sha=f"commit-{state.cycles:04d}",
        tool_calls=1,
        turns=1,
    )
    state.events.append(asdict(outcome))
    return outcome


def main() -> None:
    state = load_or_init()
    print(
        f"mission={state.mission_id} "
        f"progress={state.current}/{len(state.items)} "
        f"cycles={state.cycles}"
    )

    while state.current < len(state.items):
        outcome = do_work(state)
        save(state)  # <-- checkpoint BEFORE we do anything else
        print(f"  checkpoint: {outcome}")

        # Simulate a crash 20% of the time.
        if random.random() < 0.2:
            print("!!! simulated crash: process killed !!!")
            sys.exit(1)

        time.sleep(0.3)

    print("mission complete")
    print(f"final state written to {STATE_PATH}")


if __name__ == "__main__":
    main()
```

Run it several times. It will crash randomly, but each rerun resumes exactly where it left off:

```bash
cd lra-demo
python ch02_crash_resume.py
python ch02_crash_resume.py
python ch02_crash_resume.py
```

After a crash, inspect the state file:

```bash
cat .lra-demo/ch02_state.json
```

You will see a list of `events`, each one a `CycleOutcome`. The process does not remember the mission; the file does.

Key observations from this tiny example:

1. **State is loaded first.** The loop never assumes it knows where it is.
2. **The checkpoint happens immediately after the cycle.** `save(state)` is called before anything else.
3. **Verification gates advancement.** If `verified` is false, `current` does not move.
4. **A crash is harmless.** The next process reads the last saved state and continues.

This is the same contract the full LRA system uses, just without git and Temporal yet.

---

## How Temporal Extends This Local Idea

The demo above uses a JSON file as its durable spine. That is enough to survive process crashes on one machine, but it is not enough for a week-long mission across host reboots, container restarts, or API retries.

In LRA, the durable spine is Temporal. We will cover it in detail in later chapters, but the mapping is simple:

- The local `while` loop becomes a **Temporal Workflow**.
- One cycle of work becomes a **Temporal Activity**.
- Expensive or flaky calls (LLM, tools) are cached with **replay-from-cache**.
- Idle waiting becomes **durable sleep**, which costs nothing.

The mental model is the same: assume interruption, journal every step, verify before advancing. Temporal just makes the interruption boundary arbitrary—a host reboot on day 12 resumes exactly where the last activity completed.

---

## Hands-On Exercise

Make the crash-resume demo real.

1. Open `lra-demo/ch02_crash_resume.py`.
2. Replace the fake `do_work` function with a real one that writes actual files:
   - Item 0: create a directory `lra-demo/hello_project/`.
   - Item 1: write `hello_project/hello.py` with a `hello()` function that returns `"hello"`.
   - Item 2: write `hello_project/test_hello.py` with a pytest test.
   - Item 3: run `pytest hello_project/` and check the exit code.
3. Make `verified` depend on the actual pytest exit code: `True` only if pytest returns `0`.
4. Run the script, then `kill -9` the process mid-loop from another terminal. Rerun it and confirm it resumes from the last verified item.

If you do this correctly, the state file will contain a real history of verified progress, and a crash will never lose more than one unverified attempt.

---

> **Key Takeaway:** Durability is not a feature of the model; it is a property of the loop. Assume every cycle can be interrupted, write the real state to disk before doing anything else, and verify progress with deterministic checks. Do that, and the agent can survive for weeks. Fail to do that, and a smarter model only fails faster.

---

## Next Chapter

Now that we understand durability as an engineering property, we will look at the storage layer that makes it practical: **git as memory**. In Chapter 03 we will see why the mission state, the decision log, and the code itself all live in git commits, and how that turns `git log` into the agent's long-term memory.