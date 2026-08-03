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

## From Chat-style to Durable

In Chapter 01 we saw how a ChatGPT-style agent dies when the process restarts, the context window fills, an API call fails, or the model hallucinates a "done" signal. The model itself is not the problem — it is still a powerful short-burst reasoner. The problem is that the *system* around it keeps the plan, the work, and the truth inside a volatile in-memory loop.

Durability fixes that by moving three responsibilities out of the model's head and into engineered components:

1. **State lives outside the context window.** The model re-reads the real state every cycle.
2. **Every step is journaled.** If the process dies, the next process can resume from the journal.
3. **Progress is verified deterministically.** A checklist item is done only when a real test/build/lint passes.

This chapter focuses on the first two. We will build a tiny but fully runnable crash-resume loop in plain Python, then map it to the real `lra` package.

---

## The "Assume Interruption" Design Rule

The single most important design rule in LRA is:

> **Assume interruption.** Every cycle must be able to start from durable state alone.

That means no hidden state in memory. No "the model remembers." No "we already did step 3." If the process is killed at any moment, the next process must reconstruct situational awareness by reading the journal, the git repo, and the checklist.

This rule changes how you write the loop:

- The loop does not *continue* from a conversation history. It *re-gathers* every cycle.
- The model is fed a fresh summary built from the durable anchor.
- Tool results are written to disk (or git) before the loop claims progress.

The context window becomes a **lossy cache**, not a database. The real state lives on disk.

---

## Volatile Context vs. Durable State

| Volatile (can die) | Durable (must survive) |
|---|---|
| Model context window | Git repo with code + tests |
| In-memory conversation | `checkpoints.jsonl` / event log |
| Python object state | Checklist, decision log, ownership map |
| API response cache | Replay journal from Temporal |

In the real `lra` package this is encoded in contracts:

- `src/lra/contracts/state.py` defines `Checkpoint`, `EventRecord`, and the mission state.
- `src/lra/state/mission_anchor.py` (`GitMissionAnchor`) is the durable anchor that reads and writes that state.
- `src/lra/agent/loop.py` implements the inner loop, but it is deliberately decoupled from Temporal so it can be unit tested on its own.

The `CycleOutcome` dataclass in `src/lra/agent/loop.py` is the contract for one cycle:

```python
@dataclass
class CycleOutcome:
    item_id: str | None
    advanced: bool          # did the cycle change the world?
    verified: bool           # did the deterministic verifier pass?
    is_complete: bool        # can we mark the item done?
    head_sha: str            # durable anchor pointer (git commit sha in production)
    tool_calls: int
    turns: int
```

Notice what is *not* in `CycleOutcome`: the model's opinion. The model proposes; the system records; the verifier decides.

---

## A Runnable Local Crash-Resume Loop

Before we add Temporal, retries, and git, let's prove the idea with a self-contained Python script. Create `lra-demo/scripts/local_durable_loop.py`:

```python
# lra-demo/scripts/local_durable_loop.py
"""A minimal crash-resume agent loop that proves durability in plain Python.

The model is a stub. The "world" is a text file. The verifier checks the file.
The anchor is a JSONL journal. A simulated crash kills the process once;
re-running the script resumes exactly where it left off.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Durable state contracts (simplified versions of src/lra/contracts/state.py)
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    cycle: int
    item_id: str
    head_sha: str           # in the real system this is a git commit sha
    verified: bool
    events: list[dict] = field(default_factory=list)


@dataclass
class CycleOutcome:
    item_id: str
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


# ---------------------------------------------------------------------------
# Anchor: the durable journal (simplified GitMissionAnchor)
# ---------------------------------------------------------------------------

class JsonlMissionAnchor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, cp: Checkpoint) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(cp), default=str) + "\n")

    def last(self) -> Checkpoint | None:
        if not self.path.exists():
            return None
        lines = [line for line in self.path.read_text().splitlines() if line.strip()]
        if not lines:
            return None
        return Checkpoint(**json.loads(lines[-1]))

    def history(self) -> list[Checkpoint]:
        if not self.path.exists():
            return []
        return [Checkpoint(**json.loads(line)) for line in self.path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Verifier: deterministic ground truth
# ---------------------------------------------------------------------------

class FileContainsVerifier:
    """A stand-in for the real Verifier in src/lra/verify/.
    In production this would run pytest / mypy / build and return a Check.
    """

    def __init__(self, path: Path, needle: str) -> None:
        self.path = path
        self.needle = needle

    def check(self) -> bool:
        if not self.path.exists():
            return False
        return self.needle in self.path.read_text()


# ---------------------------------------------------------------------------
# Model + tools (stubbed so the demo costs $0)
# ---------------------------------------------------------------------------

class StubModel:
    """A fake LLM that appends one line per cycle until the verifier is green."""

    def think(self, context: dict) -> dict:
        if context["verified"]:
            return {"done": True, "summary": "item complete"}
        return {
            "tool": "append_line",
            "arguments": {"line": f"step {context['cycle']}\n"},
        }


class ToolDispatcher:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    def run(self, action: dict) -> dict:
        if action.get("tool") == "append_line":
            target = self.workdir / "output.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as f:
                f.write(action["arguments"]["line"])
            return {"ok": True, "path": str(target)}
        return {"ok": False, "error": "unknown tool"}


# ---------------------------------------------------------------------------
# The agent loop (mirrors src/lra/agent/loop.py)
# ---------------------------------------------------------------------------

class AgentLoop:
    def __init__(
        self,
        model: StubModel,
        dispatcher: ToolDispatcher,
        verifier: FileContainsVerifier,
        anchor: JsonlMissionAnchor,
        should_crash: Callable[[int], bool] | None = None,
    ) -> None:
        self.model = model
        self.dispatcher = dispatcher
        self.verifier = verifier
        self.anchor = anchor
        self.should_crash = should_crash or (lambda _: False)

    def run_one_cycle(self, item_id: str) -> CycleOutcome:
        # 1. GATHER from durable state, not from memory
        last = self.anchor.last()
        cycle = (last.cycle + 1) if last else 0
        already_verified = self.verifier.check()

        context = {
            "item_id": item_id,
            "cycle": cycle,
            "verified": already_verified,
            "history": self.anchor.history(),
        }

        # 2. THINK (one short burst)
        action = self.model.think(context)

        # 3. ACT + VERIFY
        advanced = False
        tool_calls = 0
        if action.get("done"):
            is_complete = already_verified
            head_sha = f"cycle-{cycle}-done"
        else:
            result = self.dispatcher.run(action)
            tool_calls = 1
            advanced = result.get("ok", False)
            is_complete = self.verifier.check()
            head_sha = f"cycle-{cycle}-{'ok' if is_complete else 'pending'}"

        # 4. CHECKPOINT to durable anchor BEFORE claiming progress
        checkpoint = Checkpoint(
            cycle=cycle,
            item_id=item_id,
            head_sha=head_sha,
            verified=is_complete,
            events=[action],
        )
        self.anchor.append(checkpoint)

        # 5. (Demo only) simulate a crash to prove resume works
        if self.should_crash(cycle):
            raise SystemExit(f"Simulated crash after cycle {cycle}")

        return CycleOutcome(
            item_id=item_id,
            advanced=advanced,
            verified=is_complete,
            is_complete=is_complete,
            head_sha=head_sha,
            tool_calls=tool_calls,
            turns=1,
        )


# ---------------------------------------------------------------------------
# Main: run, crash once, then run again and watch it resume
# ---------------------------------------------------------------------------

def main() -> None:
    workdir = Path("lra-demo/.lra-demo-workdir")
    anchor_path = workdir / "checkpoints.jsonl"
    target_file = workdir / "output.txt"

    anchor = JsonlMissionAnchor(anchor_path)
    model = StubModel()
    dispatcher = ToolDispatcher(workdir)
    verifier = FileContainsVerifier(target_file, "step 3")

    # Crash once, on cycle 1, only if we have not already recovered past it.
    crash_flag = {"used": False}

    def should_crash(cycle: int) -> bool:
        last = anchor.last()
        if not crash_flag["used"] and (last is None or last.cycle < 1) and cycle == 1:
            crash_flag["used"] = True
            return True
        return False

    loop = AgentLoop(model, dispatcher, verifier, anchor, should_crash)

    item_id = "demo-item"
    print(f"Resuming from: {anchor.last()}")

    while True:
        outcome = loop.run_one_cycle(item_id)
        print(outcome)
        if outcome.is_complete:
            print("Mission item complete.")
            break
        time.sleep(0.1)


if __name__ == "__main__":
    main()
```

Run it:

```bash
cd lra-demo
python scripts/local_durable_loop.py
```

You will see it start at cycle 0, append `step 0`, then crash after writing the checkpoint for cycle 1:

```text
Resuming from: None
CycleOutcome(item_id='demo-item', advanced=True, verified=False, is_complete=False, head_sha='cycle-0-pending', tool_calls=1, turns=1)
CycleOutcome(item_id='demo-item', advanced=True, verified=False, is_complete=False, head_sha='cycle-1-pending', tool_calls=1, turns=1)
SystemExit: Simulated crash after cycle 1
```

Now run the exact same command again:

```bash
python scripts/local_durable_loop.py
```

It reads `lra-demo/.lra-demo-workdir/checkpoints.jsonl`, sees the last checkpoint was cycle 1, and continues from cycle 2. It stops only when `output.txt` contains `step 3`.

The critical line is this: **the crash happened after the checkpoint was written.** The model did not "remember" anything. The new process reconstructed state from the journal and the file on disk.

Inspect the journal:

```bash
cat lra-demo/.lra-demo-workdir/checkpoints.jsonl
```

Each line is a self-contained checkpoint. That is the durable spine in its simplest form.

---

## Mapping the Local Loop to the Real System

The script above is intentionally tiny, but every piece maps to the real `lra` package:

| Demo component | Real `lra` component | File |
|---|---|---|
| `JsonlMissionAnchor` | `GitMissionAnchor` | `src/lra/state/mission_anchor.py` |
| `Checkpoint` / `CycleOutcome` | Typed state contracts | `src/lra/contracts/state.py` |
| `AgentLoop` | The integrated inner loop | `src/lra/agent/loop.py` |
| `StubModel` | Pluggable model backends | `src/lra/model/` |
| `ToolDispatcher` | Real tool dispatcher + sandbox | `src/lra/execution/` |
| `FileContainsVerifier` | Deterministic verifier | `src/lra/verify/` |

In production the durable spine is Temporal:

- The `AgentLoop` runs inside a Temporal **activity**.
- Every LLM call, tool call, and verifier call is a separate journaled activity.
- Temporal caches results, retries failures, and **replays from cache** after a crash so no tokens are re-spent.
- Idle time is spent in **durable sleep**, which costs nothing.

We will build that spine in Chapters 12–15. The point of this chapter is that the *idea* of durability does not require Temporal. It requires the discipline to externalize state and journal checkpoints.

---

## Hands-On Exercise

1. Create `lra-demo/scripts/local_durable_loop.py` from the code above.
2. Run it once and let it crash.
3. Run it again and confirm it resumes from `lra-demo/.lra-demo-workdir/checkpoints.jsonl`.
4. Change the verifier needle from `"step 3"` to `"step 5"`, delete `output.txt`, and run again. Observe that the loop continues until the new ground truth is satisfied.
5. (Harder) Replace the `JsonlMissionAnchor` with a tiny git-based anchor: after each cycle, commit `output.txt` and the checkpoint file. Use the commit SHA as `head_sha`. Verify that `git log --oneline` shows one commit per cycle.

---

> **Key Takeaway:** The model thinks in short bursts; the system runs for weeks by keeping the real state outside the window, journaling every step, and resuming from the journal instead of from memory.

---

## Next Chapter Teaser

**Chapter 03: Externalizing Truth — Git as Memory.** We will replace the JSONL journal with git as the single source of truth, and see why a commit SHA is the perfect durable pointer for a long-running mission.