# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why durability is a system property, not a model property
- The "Assume Interruption" design rule
- Volatile context vs. durable state
- The anatomy of a checkpoint (`CycleOutcome`)
- A runnable crash-resume loop you can kill and restart
- How the durable spine (Temporal) turns a checkpoint into weeks-long execution

---

## From Chat Loop to Durable System

In Chapter 01 we saw that a ChatGPT-style agent is basically one long conversation. If the process dies, the context window overflows, or an API call flakes, the run is over. The model has no memory of what it was doing except what fits in the chat history, and even that is reconstructed from a lossy summary on the next start.

Durability fixes this by moving the burden from the model to the system. The model still thinks in short bursts — maybe a few thousand tokens at a time — but the system around it keeps the real state on disk, journals every step, and resumes exactly where it left off after a crash.

This is an engineering problem, not a model-capability problem. You do not need a smarter LLM. You need:

1. **Externalized state** so the process can die and come back.
2. **Journaled steps** so work is not repeated or lost.
3. **Deterministic verification** so the system knows what is actually done.

The LRA demo project encodes this in `lra-demo/src/lra/agent/loop.py`. The `AgentLoop` class is deliberately decoupled from Temporal so you can unit-test the inner loop without a server, but it still produces a durable artifact: a `CycleOutcome`.

## The "Assume Interruption" Rule

The single most important design rule in LRA is: **assume interruption.**

Every cycle must be written as if the host will reboot, the container will be killed, or the API will rate-limit you right after the current line. That means:

- Never keep the only copy of progress in RAM.
- Never trust the model's memory of what happened.
- Always re-read the ground truth at the start of the next cycle.

In practice, the agent starts each cycle by **gathering** the current state from git and the structured mission log, not by asking the model to remember. This is what allows a week-long mission to survive a host reboot and reconstruct situational awareness in seconds.

## Volatile Context vs. Durable State

| Volatile context (the LLM window) | Durable state (the system) |
|---|---|
| Lives in RAM / API session | Lives in git + structured files |
| Limited size, lossy over time | Append-only, searchable, auditable |
| Can hallucinate past events | Grounded in commit SHAs and exit codes |
| Dies with the process | Survives reboots, crashes, redeploys |

In `lra-demo/src/lra/state/mission_anchor.py`, the `GitMissionAnchor` writes the mission state — checklist, decisions, events — into real files and commits them. The context window is only a scratchpad for the current thinking step. The repo is the source of truth.

## Anatomy of a Checkpoint

A checkpoint is the smallest durable unit of progress. In the demo repo it is represented by `CycleOutcome` in `lra-demo/src/lra/agent/loop.py`:

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

Each field has a specific job:

- `item_id` — which checklist item this cycle worked on.
- `advanced` — did the cycle make a concrete change (write a file, run a command, etc.).
- `verified` — did the deterministic verifier pass.
- `is_complete` — is the checklist item now done.
- `head_sha` — the git commit SHA that captured the state after this cycle.
- `tool_calls` / `turns` — cost and effort telemetry for the governor.

This object is what gets written to durable storage. The model does not need to remember it; the system does.

## A Runnable Crash-Resume Loop

Below is a minimal, self-contained crash-resume loop. It keeps its state in a JSON checkpoint file and only marks work as done after a deterministic verification. You can `Ctrl-C` it at any point, run it again, and it will resume safely.

Save this as `lra-demo/exercises/crash_resume.py`:

```python
"""Minimal crash-resume loop: externalized checkpoint + deterministic verifier."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_fixed

CHECKPOINT_PATH = Path(".lra/checkpoint.json")
OUTPUT_PATH = Path(".lra/output.txt")


@dataclass
class Checkpoint:
    item_id: str
    attempts: int
    output_path: str
    verified: bool
    done: bool


def load_checkpoint() -> Checkpoint:
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        return Checkpoint(**data)
    return Checkpoint(
        item_id="demo-task-1",
        attempts=0,
        output_path=str(OUTPUT_PATH),
        verified=False,
        done=False,
    )


def save_checkpoint(cp: Checkpoint) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(asdict(cp), indent=2))


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
def do_work(cp: Checkpoint) -> None:
    """Flaky work that is idempotent: writing the same file is safe to repeat."""
    cp.attempts += 1
    print(f"[attempt {cp.attempts}] writing output file...")
    Path(cp.output_path).write_text("hello durable world\n")
    # Simulate a flaky API or network failure.
    if random.random() < 0.4:
        raise RuntimeError("simulated flaky failure")


def verify(cp: Checkpoint) -> bool:
    p = Path(cp.output_path)
    if not p.exists():
        return False
    return p.read_text() == "hello durable world\n"


def run() -> None:
    cp = load_checkpoint()

    if cp.done:
        print("Mission item already complete. Nothing to do.")
        return

    try:
        do_work(cp)
    except Exception as exc:
        print(f"Work failed or was interrupted: {exc}")
        save_checkpoint(cp)
        sys.exit(1)

    cp.verified = verify(cp)
    if cp.verified:
        cp.done = True
        print(f"Verified. Marked {cp.item_id} as done after {cp.attempts} attempts.")
    else:
        print("Verification failed. Will retry on next run.")

    save_checkpoint(cp)


if __name__ == "__main__":
    run()
```

Run it a few times. Because `do_work` is idempotent and `verify` is deterministic, killing the process mid-run is harmless:

```bash
cd lra-demo
mkdir -p .lra
python exercises/crash_resume.py
# Ctrl-C during the sleep or after the file write but before verification
python exercises/crash_resume.py
```

The checkpoint file records exactly what happened. The next run does not start from scratch; it starts from the last saved state.

This is the same contract the full LRA system uses, just without Temporal. In the full repo, `AgentLoop.run_one_cycle()` returns a `CycleOutcome`, and the durable spine decides what to do with it.

## How Temporal Turns a Checkpoint into Weeks-Long Execution

The crash-resume script above uses a local file. In production, LRA wraps the same cycle inside a [Temporal](https://temporal.io) workflow. The durable spine gives you three things for free:

1. **Journaled activities.** Every LLM call, tool call, and verification is an activity. Temporal records the inputs and outputs. If a worker crashes, the next worker replays from the journal instead of re-executing.
2. **Replay-from-cache.** Once an activity succeeds, its result is cached. A crash on step 500 does not re-spend tokens for steps 1–499.
3. **Durable sleep.** When the agent is waiting for a human signal or a scheduled poll, it sleeps inside Temporal at zero compute cost.

The key architectural choice is that `AgentLoop` in `lra-demo/src/lra/agent/loop.py` knows nothing about Temporal. It just produces a `CycleOutcome`. The Temporal workflow in `lra-demo/src/lra/durable/` is a thin scheduler that calls the loop, saves the outcome, and decides whether to continue, sleep, or ask for human help.

That separation is what makes the system testable. You can run the inner loop with the `stub` model for $0, then drop the same code into a Temporal worker when you are ready for durability.

## Hands-On Exercise

1. Run `python exercises/crash_resume.py` until it completes. Inspect `.lra/checkpoint.json`.
2. Delete only `.lra/output.txt` and run again. Observe that the checkpoint still says `done: true`, but the verifier now fails. Fix the script so it resets `done` and `verified` when the output file is missing.
3. Add a second checklist item to the checkpoint and extend `run()` to process items sequentially, writing a new checkpoint after each one.
4. (Optional) Run the full LRA inner loop with the stub model:
   ```bash
   cd lra-demo
   uv run lra mission --task "Create hello.py that prints hello and a test for it" --workdir .lra/workspaces/demo
   ```
   Then inspect the durable state with:
   ```bash
   git -C .lra/workspaces/demo log --oneline
   ```

## Key Takeaway

> Durability is not something the model provides. It is something the system guarantees by keeping real state outside the process, journaling every step, and verifying progress with deterministic checks. A dumb model inside a durable system can outlast a smart model inside a chat loop.

## Next Chapter

In Chapter 03 we will externalize truth even further. We will make **git** the mission's memory: every checklist update, every decision, and every verified result becomes a commit. You will learn why `head_sha` is the most important field in `CycleOutcome`.