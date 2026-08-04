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

In Chapter 01 we saw that a ChatGPT-style agent is essentially one long conversation. The model holds the plan, the memory, and the progress in its context window. When the process dies, the API rate-limits, or the context fills, the mission is over.

The fix is not to buy a bigger model or a bigger context window. The fix is to change the engineering contract:

> **Assume interruption.** Every cycle must be able to resume from durable state alone.

This single rule turns the agent from a conversation into a state machine. The model is still useful — it thinks in short bursts — but the *system* owns the mission. The model can be restarted, swapped, or downgraded without losing progress.

## Volatile Context vs. Durable State

There are two kinds of state in any agent system:

| Volatile (context window / RAM) | Durable (disk / git / database) |
|--------------------------------|---------------------------------|
| The current "train of thought" | The checklist and ownership map |
| Recent tool output | The git commit graph |
| Parsed action from the last turn | The journal of attempts and outcomes |
| Cost ledger in memory | The cost ledger on disk |

The context window is a **lossy cache**. It is fast to read and write, but it evaporates on restart. Durable state is the **source of truth**. It is slower to update, but it survives crashes, reboots, and API failures.

In `lra`, the durable state lives in:

- `src/lra/state/` — git-backed mission files
- `src/lra/durable/` — Temporal workflows and activities
- `src/lra/contracts/state.py` — typed state contracts like `Checkpoint` and `EventRecord`

The model layer in `src/lra/model/` is deliberately pluggable. You can run the whole system for \$0 with the `stub` model or a local Ollama instance. Durability does not depend on the model.

## The Anatomy of a Checkpoint

A checkpoint is the atomic unit of durable progress. In `src/lra/agent/loop.py`, one cycle produces a `CycleOutcome`:

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

Each field has a job:

- `item_id` — which checklist item was worked on
- `advanced` — did the agent change anything in the workspace?
- `verified` — did the deterministic verifier pass?
- `is_complete` — is the item allowed to move to `done`?
- `head_sha` — the git commit that captured the state after this cycle
- `tool_calls`, `turns` — telemetry for cost and replay

A checkpoint is written **before** the next cycle begins. If the process dies after the checkpoint, the next process reads the checkpoint and continues. No work is lost. No tokens are re-spent.

## A Runnable Crash-Resume Loop

The best way to prove durability is to build a loop that survives a `kill -9`. The file `lra-demo/ch02/crash_resume.py` is a minimal, self-contained version of the LRA inner loop. It uses a stub model, writes every checkpoint to disk, and can be killed and restarted without losing progress.

```python
# lra-demo/ch02/crash_resume.py
"""Minimal crash-resume loop: durability is a system property, not a model property.

It uses a $0 stub model, journals every cycle to disk, and resumes from the
journal alone. The in-memory context is intentionally disposable.
"""
from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


# ---------- durable contracts (simplified from src/lra/contracts/state.py) ----------

@dataclass
class Checkpoint:
    item_id: str
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class MissionState:
    task: str
    checklist: list[str]
    completed: set[str] = field(default_factory=set)
    attempts: dict[str, int] = field(default_factory=dict)
    journal: list[Checkpoint] = field(default_factory=list)

    def to_dict(self):
        return {
            "task": self.task,
            "checklist": self.checklist,
            "completed": sorted(self.completed),
            "attempts": self.attempts,
            "journal": [asdict(c) for c in self.journal],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MissionState":
        return cls(
            task=d["task"],
            checklist=d["checklist"],
            completed=set(d.get("completed", [])),
            attempts=d.get("attempts", {}),
            journal=[Checkpoint(**c) for c in d.get("journal", [])],
        )


STATE_PATH = Path(".lra/ch02/state.json")


def load_state() -> MissionState:
    if STATE_PATH.exists():
        return MissionState.from_dict(json.loads(STATE_PATH.read_text()))
    return MissionState(
        task="Create hello.py and a test for it",
        checklist=["write hello.py", "write test_hello.py", "run pytest"],
    )


def save_state(state: MissionState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state.to_dict(), indent=2, default=str))


# ---------- deterministic verifier (the only acceptable "done" signal) ----------

def verify(item_id: str, workdir: Path) -> bool:
    if item_id == "write hello.py":
        f = workdir / "hello.py"
        return f.exists() and "print(" in f.read_text()
    if item_id == "write test_hello.py":
        f = workdir / "test_hello.py"
        return f.exists() and "def test_" in f.read_text()
    if item_id == "run pytest":
        # In the real system this is a subprocess with an exit code.
        # Here we just confirm the test file exists.
        return (workdir / "test_hello.py").exists()
    return False


# ---------- stub model + tool dispatcher ----------

def model_think(item_id: str, context: str) -> str:
    """A $0 stub 'model' that emits a deterministic plan for each item."""
    plans = {
        "write hello.py": '{"tool":"write_file","arguments":{"path":"hello.py","content":"def main():\\n    print(\\"hello\\")\\n\\nif __name__ == \\"__main__\\":\\n    main()\\n"}}',
        "write test_hello.py": '{"tool":"write_file","arguments":{"path":"test_hello.py","content":"from hello import main\\n\\ndef test_hello():\\n    main()\\n"}}',
        "run pytest": '{"done":true,"summary":"pytest passed"}',
    }
    return plans.get(item_id, '{"done":true}')


def dispatch_tool(command: dict, workdir: Path) -> str:
    name = command.get("tool")
    args = command.get("arguments", {})
    if name == "write_file":
        path = workdir / args["path"]
        path.write_text(args["content"])
        return f"wrote {path}"
    return "unknown tool"


def parse_action(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"done": True, "summary": text}


# ---------- one durable cycle ----------

def run_one_cycle(state: MissionState, workdir: Path) -> Checkpoint | None:
    # Gather: rebuild context from durable state only.
    remaining = [i for i in state.checklist if i not in state.completed]
    if not remaining:
        return None

    item = remaining[0]
    state.attempts[item] = state.attempts.get(item, 0) + 1

    # Act: one model turn + optional tool call.
    context = f"task: {state.task}\ncompleted: {sorted(state.completed)}\nworking on: {item}"
    raw = model_think(item, context)
    action = parse_action(raw)

    tool_calls = 0
    if action.get("tool"):
        dispatch_tool(action, workdir)
        tool_calls += 1

    # Verify: never trust the model; trust the test.
    ok = verify(item, workdir)
    if ok:
        state.completed.add(item)

    cp = Checkpoint(
        item_id=item,
        advanced=tool_calls > 0,
        verified=ok,
        is_complete=ok,
        head_sha="local",  # Chapter 03 replaces this with a real git SHA.
        tool_calls=tool_calls,
        turns=1,
    )

    state.journal.append(cp)
    save_state(state)
    return cp


# ---------- main loop with simulated crash ----------

def main() -> None:
    workdir = Path(".lra/ch02/workspace")
    workdir.mkdir(parents=True, exist_ok=True)

    state = load_state()

    # Simulate a random kill after a cycle. In the exercise you will send SIGTERM yourself.
    def maybe_die(signum, frame):
        if random.random() < 0.25:
            print("\n>>> simulated crash <<<")
            sys.exit(137)

    signal.signal(signal.SIGTERM, maybe_die)

    print(f"resumed with {len(state.completed)}/{len(state.checklist)} items done")
    while True:
        cp = run_one_cycle(state, workdir)
        if cp is None:
            print("mission complete")
            break
        print(
            f"  cycle: {cp.item_id} "
            f"verified={cp.verified} "
            f"attempts={state.attempts[cp.item_id]}"
        )
        time.sleep(0.2)


if __name__ == "__main__":
    main()
```

### What the code proves

1. **Resume from durable state only.** On startup, `load_state()` reads `.lra/ch02/state.json`. The in-memory `MissionState` is reconstructed from disk, not from a previous conversation.
2. **Checkpoint before proceeding.** `save_state(state)` is called inside `run_one_cycle()` immediately after verification. If the process dies right after, the next process sees the completed item.
3. **Verification is the done signal.** The model says `"done": true` for `run pytest`, but the item is only marked complete if `verify()` passes. In this stub, verification checks the filesystem; in the real system it checks exit codes.
4. **The model is interchangeable.** `model_think()` is a stub. You could replace it with an LLM call, and the durability mechanics would not change.

## From Local Loop to Temporal Spine

The script above is a single-process loop. For a week-long mission, we need the same guarantees across process restarts, host reboots, and worker upgrades. That is where Temporal comes in.

In `src/lra/durable/`, the LRA package wraps the inner loop inside a Temporal workflow. The workflow is the scheduler; the inner loop runs inside a Temporal activity. Temporal journals every activity call, caches results, retries failures, and resumes exactly where the mission left off.

The `pyproject.toml` keeps the Temporal dependency in an optional extra:

```toml
[project.optional-dependencies]
durable = ["temporalio>=1.7"]
```

This means the core ideas — assume interruption, externalize state, verify deterministically — work without Temporal. Temporal just makes them industrial: durable sleep, replay-from-cache, and multi-worker execution.

The key insight is the same at every scale:

> The model does not need to be smarter. The system around it needs to be durable.

## Hands-On Exercise

1. Run the crash-resume demo:

   ```bash
   python lra-demo/ch02/crash_resume.py
   ```

   Let it finish once to see the happy path.

2. Run it again, but this time send `SIGTERM` from another terminal while it is working:

   ```bash
   # terminal 1
   python lra-demo/ch02/crash_resume.py

   # terminal 2
   pgrep -f crash_resume.py
   kill -TERM <pid>
   ```

3. Run the script again. Observe that it prints `resumed with X/3 items done` and continues from the last verified item.

4. Inspect the durable journal:

   ```bash
   cat .lra/ch02/state.json
   ```

   Notice that the journal contains every `Checkpoint`, including the one written just before the crash.

5. Delete `.lra/ch02/state.json` and restart. The mission starts over. This proves that the file, not any in-memory context, is the source of truth.

6. Optional: edit `model_think()` to return malformed JSON. Watch `parse_action()` fall back to `done=True`, and watch the verifier reject the item. The mission stalls safely instead of hallucinating progress.

## Key Takeaway

> Durability is an engineering property, not a model capability. The model thinks in short bursts; the system runs for weeks by keeping the real state outside the window, journaling every step, and verifying progress with real tests.
> — Fareed Khan

## Next Chapter

In Chapter 03: **Externalizing Truth — Git as Memory**, we replace the JSONL journal with a real git repository. Every checkpoint becomes a commit, the mission state lives in tracked files, and the agent reconstructs situational awareness by reading `git log` instead of asking the model to remember.