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

## Why Smarter Models Don't Help

In Chapter 01 we saw that a ChatGPT-style agent is basically one long conversation. The model holds the plan, the history, and the current step in its context window. That works until:

- the process restarts,
- the API rate-limits you,
- the host reboots,
- or the context window fills up and earlier turns are silently dropped.

A bigger model or a better prompt does not solve any of those problems. They are *system* problems. Durability is therefore an engineering property: it is something you build into the loop, not something you ask the model to remember.

The LRA system is built on one rule: **Assume Interruption**. Every cycle must be able to stop at any point and resume later without losing the plot. The model thinks in short bursts; the system runs for weeks by keeping the real state outside the model and journaling every step.

---

## Volatile Context vs. Durable State

There are two kinds of information in an agent run:

| Volatile (lives in the LLM window) | Durable (lives outside the model) |
|---|---|
| The reasoning for *this* turn | The checklist of work items |
| A summary of the current file | The actual files in git |
| The model's belief about progress | The verification results on disk |
| Tool output from the last call | The event log and cost ledger |

The context window is a lossy cache. It can be truncated, summarized, or rebuilt. The durable layer is the source of truth. When the system wakes up after a crash, it does not ask the model, "What were we doing?" It reads the durable state and asks the model, "Here is where things stand. What is the next action?"

This separation is what makes long-horizon autonomy possible.

---

## The Anatomy of a Checkpoint

In `src/lra/agent/loop.py`, one cycle of work returns a `CycleOutcome`:

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

A checkpoint is not just "the model said it was done." It records:

1. **What item was being worked on** (`item_id`)
2. **Whether the world changed** (`advanced`)
3. **Whether a real verifier passed** (`verified`)
4. **Whether the whole mission is finished** (`is_complete`)
5. **The exact git commit** that captured the state (`head_sha`)
6. **How much work and cost were spent** (`tool_calls`, `turns`)

The `head_sha` is the anchor. If the process dies and comes back, it can `git checkout` that commit and continue exactly from the last known-good state. We will build that anchor in detail in Chapter 03.

---

## A Runnable Crash-Resume Loop

To make the idea concrete without needing Temporal or a running LLM server, we will build a tiny durable loop in plain Python. Save it as `lra-demo/ch02_crash_resume.py`.

The script maintains a JSON state file. Each cycle:

1. Reads the durable state.
2. Picks the next unverified step.
3. Executes the step (writes a file or runs a shell command).
4. Verifies the step with a deterministic exit code.
5. Writes the updated state back to disk.

We also add a deliberate "crash" switch so you can kill the process mid-cycle and watch it resume.

```python
# lra-demo/ch02_crash_resume.py
"""A minimal durable agent loop that survives a mid-cycle crash.

Run:
    python lra-demo/ch02_crash_resume.py

Then kill it with Ctrl-C while it is sleeping to simulate a crash.
Run it again and it will resume from the last durable checkpoint.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


STATE_PATH = Path("lra-demo/ch02_state.json")
WORK_DIR = Path("lra-demo/ch02_workspace")
WORK_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Step:
    id: str
    description: str
    act: Callable[[], None]
    verify: Callable[[], bool]


@dataclass
class Checkpoint:
    task: str
    current_step_id: str | None
    completed_ids: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)
    last_verified: bool = False
    cost_usd: float = 0.0


def load_checkpoint() -> Checkpoint:
    if not STATE_PATH.exists():
        return Checkpoint(
            task="Create a hello.py and a passing test",
            current_step_id="write_hello",
        )
    with STATE_PATH.open() as f:
        raw = json.load(f)
    return Checkpoint(**raw)


def save_checkpoint(cp: Checkpoint) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w") as f:
        json.dump(asdict(cp), f, indent=2)


def deterministic_verify(command: list[str]) -> bool:
    """A step is done ONLY when the shell command exits 0."""
    result = subprocess.run(
        command, cwd=WORK_DIR, capture_output=True, text=True
    )
    print(f"  verify: {' '.join(command)} -> exit {result.returncode}")
    if result.returncode != 0 and result.stderr:
        print(f"  stderr: {result.stderr.strip()}")
    return result.returncode == 0


def write_hello() -> None:
    (WORK_DIR / "hello.py").write_text('print("hello")\n')


def write_test() -> None:
    (WORK_DIR / "test_hello.py").write_text(
        "from hello import hello\n\ndef test_hello():\n    assert hello() == 'hello'\n"
    )


def fix_hello() -> None:
    """The first test fails because hello.py only prints. Fix it."""
    (WORK_DIR / "hello.py").write_text('def hello():\n    return "hello"\n')


STEPS: list[Step] = [
    Step(
        id="write_hello",
        description="Write hello.py",
        act=write_hello,
        verify=lambda: deterministic_verify(["python", "hello.py"]),
    ),
    Step(
        id="write_test",
        description="Write a pytest test for hello()",
        act=write_test,
        verify=lambda: deterministic_verify(["python", "-m", "pytest", "test_hello.py", "-q"]),
    ),
    Step(
        id="fix_hello",
        description="Make hello.py importable and testable",
        act=fix_hello,
        verify=lambda: deterministic_verify(["python", "-m", "pytest", "test_hello.py", "-q"]),
    ),
]


def run_one_cycle(cp: Checkpoint) -> Checkpoint:
    step_id = cp.current_step_id
    if step_id is None:
        # Find next incomplete step
        remaining = [s for s in STEPS if s.id not in cp.completed_ids]
        if not remaining:
            cp.current_step_id = None
            return cp
        step_id = remaining[0].id
        cp.current_step_id = step_id

    step = next(s for s in STEPS if s.id == step_id)
    cp.attempts[step_id] = cp.attempts.get(step_id, 0) + 1
    cp.cost_usd += 0.01  # pretend each cycle costs a cent

    print(f"\n--- cycle ---")
    print(f"step: {step.id} | attempt: {cp.attempts[step_id]} | cost: ${cp.cost_usd:.2f}")

    # ACT
    print(f"  act: {step.description}")
    step.act()

    # Simulate a crash between act and verify on the second attempt of write_test
    # so the exercise is easy to reproduce. Remove this in production.
    if step.id == "write_test" and cp.attempts[step_id] == 1:
        print("  [simulated crash: process killed before verification]")
        os.kill(os.getpid(), 2)  # SIGINT

    # VERIFY
    passed = step.verify()
    cp.last_verified = passed

    if passed:
        cp.completed_ids.append(step.id)
        cp.current_step_id = None
        print(f"  -> verified; {len(cp.completed_ids)}/{len(STEPS)} done")
    else:
        print(f"  -> blocked; will retry on next cycle")

    return cp


def main() -> int:
    print("Loading durable checkpoint...")
    cp = load_checkpoint()
    print(f"checkpoint: {cp.current_step_id=} completed={cp.completed_ids}")

    if cp.current_step_id is None and len(cp.completed_ids) == len(STEPS):
        print("Mission already complete.")
        return 0

    cp = run_one_cycle(cp)
    save_checkpoint(cp)

    if len(cp.completed_ids) == len(STEPS):
        print("\n=== MISSION COMPLETE ===")
        print(f"final cost: ${cp.cost_usd:.2f}")
        print(f"state file: {STATE_PATH}")
        return 0

    print("\nRun again to continue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

A few things to notice:

- The state file (`lra-demo/ch02_state.json`) is written **after** verification, not before. If the process dies between `act()` and `verify()`, the next run will re-execute the same step because the step has not been added to `completed_ids`.
- The model is replaced by a hard-coded step list, but the durable structure is the same: gather state, act, verify, checkpoint.
- The verifier is a shell command with an exit code. The model does not get to declare success.

---

## Connecting to the Real LRA Loop

The real `AgentLoop` in `src/lra/agent/loop.py` does the same four phases, but with a model, a tool dispatcher, a sandbox, and a git anchor:

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

Instead of a JSON file, the durable checkpoint is a git commit. Instead of a hard-coded step list, the model chooses the next tool call. Instead of a local subprocess, the tool runs in a sandbox. But the contract is identical:

1. **Gather** durable state.
2. **Act** via one or more tool calls.
3. **Verify** with a deterministic check.
4. **Checkpoint** the result.

In later chapters we will wrap this loop inside a Temporal workflow so that even the *execution* of the loop itself is durable. For now, the important insight is that durability is a local property of the loop, not a property of the orchestrator.

---

## Hands-On Exercise

1. Create `lra-demo/ch02_crash_resume.py` and run it:

   ```bash
   python lra-demo/ch02_crash_resume.py
   ```

   The script will write `hello.py`, then simulate a crash before it can verify the test.

2. Inspect the durable state:

   ```bash
   cat lra-demo/ch02_state.json
   ```

   You should see `current_step_id` still set to `write_test` and `write_test` not in `completed_ids`.

3. Run the script again:

   ```bash
   python lra-demo/ch02_crash_resume.py
   ```

   It resumes from the checkpoint, verifies the test, discovers it fails, and moves to the `fix_hello` step.

4. Run it a third time to complete the mission.

5. **Stretch goal:** Remove the simulated `os.kill` line and add a real failure mode. For example, make `write_test` produce a test that fails, and confirm the loop marks the step as blocked and retries it on the next cycle without losing earlier completed steps.

---

> **Key Takeaway:** Durability is not something you get from a bigger model or a longer context window. It is a property of the loop: every cycle gathers durable state, acts, verifies with real tests, and checkpoints before moving on. If the process dies at any point, the next cycle can reconstruct exactly where things stand by reading the checkpoint, not by asking the model to remember.

---

## Next Chapter Teaser

In Chapter 03 we replace the JSON state file with the real source of truth: a git repository. We will see how **Git as Memory** makes every checkpoint auditable, branchable, and replayable, and how the `GitMissionAnchor` becomes the single source of ground truth for the whole mission.