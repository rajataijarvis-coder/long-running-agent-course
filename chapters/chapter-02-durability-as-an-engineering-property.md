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

## From Chat Loop to Durable Loop

In Chapter 01 we saw that a ChatGPT-style agent is basically one long conversation. If the process dies, the context window fills, or an API call flakes, the run is over. The model has no memory of *what it actually did* versus *what it only thought about doing*.

Durability fixes this by making the **system** responsible for remembering. The model still thinks in short bursts, but every burst is:

1. **Gathered** from durable state, not from a long chat history.
2. **Executed** through journaled tool calls.
3. **Verified** by deterministic checks.
4. **Checkpointed** to durable storage before the next burst.

This is the "Assume Interruption" rule: design every cycle as if the process will be killed before the next one. If you can resume from disk, you can survive reboots, host migrations, API rate limits, and your own bugs.

## Volatile vs. Durable

| Volatile (can die) | Durable (must survive) |
|---|---|
| Model context window | Git repo with the real code |
| In-memory variables | Structured state files on disk |
| Chat history | Decision log + attempt journal |
| Process heap | Temporal event history |

The context window is a **lossy cache**. The real state lives outside it. In LRA that real state is a git repo plus a small set of saved-state files. When the system wakes up after a crash, it does not ask the model "what were we doing?" It re-reads the repo, the checklist, and the decision log.

## The Checkpoint Contract

The heart of the durable cycle is the checkpoint. In `src/lra/agent/loop.py` LRA defines `CycleOutcome` as the single source of truth for what happened in one cycle:

```python
# From src/lra/agent/loop.py
from dataclasses import dataclass

@dataclass
class CycleOutcome:
    """The result of one agent cycle."""

    item_id: str | None
    advanced: bool      # Did the item move forward at all?
    verified: bool       # Did the deterministic verifier pass?
    is_complete: bool    # Is the whole mission finished?
    head_sha: str        # Git commit SHA that captured the work
    tool_calls: int
    turns: int
```

Notice what is **not** in `CycleOutcome`: no chat history, no plan, no model reasoning. Those are inputs and side effects. The checkpoint only records *results* that can be re-read by a fresh process.

The `AgentLoop` class in the same file is deliberately decoupled from Temporal. It is unit-testable on its own, and the durable spine calls it from inside an activity:

```python
# src/lra/agent/loop.py (excerpt)
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

A later chapter will unpack every part of that constructor. For now, the important idea is: the loop produces a `CycleOutcome`, and the durable spine decides what to do with it.

## A Runnable Crash-Resume Loop

Before we bring in Temporal, let's prove the idea with plain Python and a JSON checkpoint file. Save this as `lra-demo/ch02/durable_loop.py`:

```python
# lra-demo/ch02/durable_loop.py
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

CHECKPOINT_FILE = Path(".lra/ch02_checkpoint.json")


@dataclass(frozen=True)
class Checkpoint:
    item_id: str
    status: str          # "pending" | "done" | "blocked"
    head_sha: str
    attempts: int


def load_checkpoint() -> Checkpoint | None:
    if not CHECKPOINT_FILE.exists():
        return None
    data = json.loads(CHECKPOINT_FILE.read_text())
    return Checkpoint(**data)


def save_checkpoint(cp: Checkpoint) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(asdict(cp), indent=2))


def verify(item_id: str) -> bool:
    """Deterministic verification: a real artifact must exist and be marked done."""
    artifact = Path(f"out/{item_id}.txt")
    return artifact.exists() and "done" in artifact.read_text()


def do_work(item_id: str) -> None:
    """Simulate the agent writing code. Flaky on purpose."""
    Path("out").mkdir(exist_ok=True)
    print(f"[{item_id}] writing artifact...")
    time.sleep(0.5)
    # 30% chance of a crash mid-write
    if random.random() < 0.3:
        raise RuntimeError(f"simulated crash while working on {item_id}")
    Path(f"out/{item_id}.txt").write_text(f"{item_id} done\n")


def run_item(item_id: str) -> Checkpoint:
    cp = load_checkpoint()
    attempts = 1
    if cp and cp.item_id == item_id:
        attempts = cp.attempts + 1

    status = "blocked"
    try:
        do_work(item_id)
        status = "done" if verify(item_id) else "blocked"
    except Exception as exc:
        print(f"[{item_id}] crashed: {exc}")

    # In the real system this would be `git rev-parse HEAD`
    head_sha = "fake-sha-for-demo"
    cp = Checkpoint(item_id=item_id, status=status, head_sha=head_sha, attempts=attempts)
    save_checkpoint(cp)
    return cp


def main(items: list[str]) -> None:
    for item in items:
        cp = load_checkpoint()
        if cp and cp.item_id == item and cp.status == "done":
            print(f"[{item}] already done — skipping")
            continue
        print(f"[{item}] starting (previous attempts: {cp.attempts if cp else 0})")
        cp = run_item(item)
        print(f"[{item}] outcome: {cp}")
    print("All items processed.")


if __name__ == "__main__":
    main(["parser", "server", "client"])
```

Run it, then kill it while it is working on `server`, then run it again:

```bash
cd lra-demo/ch02
python durable_loop.py
# while it is on "server", press Ctrl-C or `kill -9`
python durable_loop.py
```

You will see it skip `parser` if it finished, resume `server` from the saved checkpoint, and not lose the attempt count. That is durability in its simplest form.

## Wiring the Checkpoint into a Temporal Spine

The standalone loop above proves the idea, but a real long-running mission needs a scheduler that can:

- retry failed activities,
- sleep for hours without holding a process,
- replay exactly from the last successful activity,
- resume on a different worker after a crash.

LRA uses [Temporal](https://temporal.io) for this. The `AgentLoop` runs inside a Temporal activity, and the workflow orchestrator consumes the `CycleOutcome` it returns.

Here is the minimal shape. The full version is in `src/lra/durable/workflows.py`:

```python
# lra-demo/ch02/minimal_temporal_spine.py
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy


# In the real system this is imported from src/lra.agent.loop
class CycleOutcome:
    def __init__(self, item_id: str, verified: bool, is_complete: bool, head_sha: str):
        self.item_id = item_id
        self.verified = verified
        self.is_complete = is_complete
        self.head_sha = head_sha


@activity.defn
async def agent_cycle_activity(item_id: str) -> dict:
    """One durable unit of work. Temporal journals this call."""
    # Build the real AgentLoop from config (omitted for brevity)
    # loop = AgentLoop(model=..., dispatcher=..., verifier=..., anchor=...)
    # outcome = await loop.run_cycle(item_id)

    # Stubbed for this demo:
    outcome = CycleOutcome(
        item_id=item_id,
        verified=True,
        is_complete=item_id == "client",
        head_sha="abc123",
    )
    return outcome.__dict__


@workflow.defn
class MissionWorkflow:
    @workflow.run
    async def run(self, task: str, checklist: list[str]) -> dict:
        remaining = list(checklist)
        while remaining:
            item_id = remaining[0]

            outcome = await workflow.execute_activity(
                agent_cycle_activity,
                item_id,
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            if outcome["verified"]:
                remaining.pop(0)
            else:
                # Blocked items stay at the front; a human or later cycle can unblock them.
                await workflow.wait_condition(
                    lambda: False, timeout=timedelta(seconds=5)
                )

        return {"status": "DONE", "head_sha": outcome["head_sha"]}
```

Key points:

- The workflow never calls the model directly. It only schedules activities.
- The activity is the smallest unit of durable work. If it crashes, Temporal retries it.
- The workflow history records the *result* of each activity, so a restarted worker replays from the last outcome instead of re-running it.
- Idle time is spent in durable sleep, costing zero tokens and zero compute.

## Hands-On Exercise

1. Create `lra-demo/ch02/durable_loop.py` from the code above.
2. Run it once and let it finish. Inspect `.lra/ch02_checkpoint.json`.
3. Delete `out/server.txt` but keep the checkpoint. Run again. What happens? Why is the checkpoint not enough without the artifact?
4. Modify `do_work` so it writes the artifact **before** the simulated crash, and the checkpoint **after** verification. Kill the process mid-run and confirm it resumes without redoing finished work.
5. (Optional) If you have Docker running, start the Temporal dev server (`temporal server start-dev`) and run the minimal workflow spine. Watch the Temporal UI show each activity as a durable event.

## Key Takeaway

> Durability is an engineering property, not a model capability. The model thinks in short bursts; the system runs for weeks by keeping the real state outside the context window, journaling every step, and replaying from durable checkpoints.

## Next Chapter

In Chapter 03 we replace the JSON checkpoint with the real source of truth: a git repo. We will see how every commit becomes a checkpoint, how the mission anchor reads where things stand, and why "git as memory" makes a reboot on day 12 reconstruct situational awareness in seconds.