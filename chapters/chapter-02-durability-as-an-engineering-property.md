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

## From Chat-style Loop to Durable Loop

In Chapter 01 we saw that a ChatGPT-style agent is basically one long conversation. If the process dies, the context window fills, or an API call flakes, the run is over. The model itself is not the problem — it is still good at short bursts of reasoning. What is missing is the *system* around it: a way to remember where it is, retry what failed, and keep going after an interruption.

Durability is that system property. A durable agent does not trust its own memory. It treats every cycle as if it might be the last one before a crash. That means:

1. **Real state lives outside the model.** The context window is a cache, not a database.
2. **Every forward step is journaled.** If the process restarts, the agent reconstructs situational awareness by reading the journal, not by asking the model to remember.
3. **Work is verified by deterministic tests, not by the model's opinion.** A checkpoint is only a checkpoint if the verifier says it is green.

This chapter builds a minimal local version of that idea. The next chapters add git as the source of truth, Temporal as the durable spine, and the full agent organization.

## Assume Interruption

The single most important design rule in LRA is **Assume Interruption**. It changes how you write every loop:

- Instead of starting from "what did the model say last?", start from "what does the journal say is true right now?"
- Instead of holding state in Python variables, hold it in durable records that survive a reboot.
- Instead of trusting the model to report success, ask a deterministic verifier.

In `lra-demo/src/lra/agent/loop.py` this is encoded in the `AgentLoop` class. The loop is deliberately decoupled from Temporal so it can be unit-tested on its own. Temporal only provides the durable spine that schedules and replays it.

The contract for one cycle of work is captured by `CycleOutcome`:

```python
# src/lra/agent/loop.py
@dataclass
class CycleOutcome:
    item_id: str | None
    advanced: bool          # did the agent change the world (file, test, etc.)?
    verified: bool          # did the deterministic verifier pass?
    is_complete: bool       # can we mark the checklist item done?
    head_sha: str           # git commit the cycle landed on
    tool_calls: int
    turns: int
```

These six fields are the entire durable summary of one cycle. The model's full monologue can be discarded; the `CycleOutcome` is what matters for resuming.

## A Runnable Crash-Resume Loop

To make this concrete, here is a self-contained Python script that simulates a durable agent. It writes a JSONL journal of `CycleOutcome` records to disk. If you kill it mid-run and restart it, it reads the last checkpoint and continues from the next unverified item.

This version keeps its own local `CycleOutcome` so it runs without installing the full `lra` package, but the shape is identical to the real one in `src/lra/agent/loop.py`.

```python
# lra-demo/scripts/crash_resume_demo.py
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field

JOURNAL = ".lra/demo/journal.jsonl"


@dataclass
class CycleOutcome:
    """Minimal durable checkpoint matching src/lra/agent/loop.py."""
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


def load_last_checkpoint() -> CycleOutcome | None:
    if not os.path.exists(JOURNAL):
        return None
    with open(JOURNAL) as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        return None
    last = json.loads(lines[-1])
    return CycleOutcome(**last)


def save_checkpoint(outcome: CycleOutcome) -> None:
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(asdict(outcome), default=str) + "\n")


def deterministic_verify(item_id: str, attempt: int) -> bool:
    """
    Stand-in for the real verifier in src/lra/verify/.
    Odd-numbered items need two attempts to simulate a flaky test
    that the agent must retry.
    """
    if item_id.startswith("odd") and attempt < 2:
        return False
    return True


def run_one_cycle(item_id: str, attempt: int) -> CycleOutcome:
    print(f"[cycle] item={item_id} attempt={attempt}")
    verified = deterministic_verify(item_id, attempt)
    advanced = True
    is_complete = verified

    # Simulate a host crash 20% of the time *before* we save.
    # In a real system this would be a SIGKILL, OOM, or API timeout.
    if random.random() < 0.2:
        raise RuntimeError("simulated host crash before checkpoint")

    return CycleOutcome(
        item_id=item_id,
        advanced=advanced,
        verified=verified,
        is_complete=is_complete,
        head_sha=f"commit-{item_id}-{attempt}",
        tool_calls=attempt,
        turns=attempt,
    )


def main() -> None:
    items = ["even-1", "odd-2", "even-3", "odd-4", "even-5"]

    last = load_last_checkpoint()
    start_index = 0

    if last and last.item_id:
        print(f"[resume] last checkpoint: {last}")
        try:
            idx = items.index(last.item_id)
            # If the last item was completed, move to the next one.
            # If it was merely attempted, retry the same item.
            start_index = idx + (1 if last.is_complete else 0)
        except ValueError:
            start_index = 0
    else:
        print("[start] no checkpoint found, beginning mission")

    for i in range(start_index, len(items)):
        item = items[i]
        attempt = 1
        while True:
            outcome = run_one_cycle(item, attempt)
            save_checkpoint(outcome)
            print(f"  -> checkpoint saved: {asdict(outcome)}")

            if outcome.is_complete:
                break

            attempt += 1
            if attempt > 5:
                raise RuntimeError(f"item {item} blocked after 5 attempts")

    print("[done] mission complete")
    print(f"[journal] {JOURNAL}")
    with open(JOURNAL) as f:
        for line in f:
            print("  ", line.strip())


if __name__ == "__main__":
    main()
```

Run it a few times. Because of the simulated crash, it will often die before finishing. Restart it and notice that it never re-does a completed item:

```bash
cd lra-demo
mkdir -p .lra/demo
python scripts/crash_resume_demo.py
# (probably crashes)
python scripts/crash_resume_demo.py
# resumes from journal.jsonl
```

The journal is the durable source of truth for this local demo. In the real `lra` package, that role is split between:

- `src/lra/state/` — structured state files (checklist, decision log, events)
- `src/lra/durable/` — Temporal workflows and activities that journal every model/tool call
- `src/lra/state/mission_anchor.py` — `GitMissionAnchor`, which ties every checkpoint to an actual git commit

## Volatile Context vs. Durable State

A durable system keeps two categories separate:

| Volatile (can be lost) | Durable (must survive) |
|---|---|
| The model's current context window | The checklist and decision log |
| In-memory Python objects | `CycleOutcome` journal / git commits |
| API response text | Verified test results |
| Intermediate reasoning traces | Cost ledger entries |

The model's context window is a **lossy cache**. You can refill it from durable state when the process restarts. The durable state is the **ground truth**. This is why `AgentLoop` in `src/lra/agent/loop.py` is designed to be called fresh each cycle: it gathers state from the anchor, prompts the model, acts, verifies, and checkpoints. It does not rely on a long-lived conversation object.

## From Local Journal to Temporal Spine

The script above proves the idea on one machine. For a mission that runs for days or weeks, you need the same guarantees across process restarts, container reschedules, and host reboots. That is where Temporal comes in.

In `lra-demo/src/lra/durable/` the system defines:

- `MissionWorkflow` — the long-running orchestrator that owns the checklist
- `AgentActivity` — one unit of work (one gather → act → verify → checkpoint cycle)
- Replay-from-cache — Temporal re-executes the workflow from the event history, reusing cached activity results instead of re-calling the model

This means a crash on day 12 resumes exactly where it left off, without re-spending tokens on already-completed cycles. Idle time is spent in **durable sleep**, which costs nothing.

We will build the Temporal spine in Chapters 12–15. For now, the key insight is that the local journal and the Temporal event history are the same idea at different scales: **write down what is true before you trust it**.

## Hands-On Exercise

1. Run `python scripts/crash_resume_demo.py` until it completes. Inspect `.lra/demo/journal.jsonl`. How many `CycleOutcome` records were written? Did any items require more than one attempt?
2. Kill the script manually while it is processing an item (`Ctrl-C` or `kill -9`). Restart it. Confirm it resumes from the last saved checkpoint and does not redo a completed item.
3. Replace the stub `deterministic_verify` function with a real check: make the script create a small Python file and run `pytest` on it as the verifier. The `verified` flag should be `True` only when `pytest` exits with code 0. Hint: use `subprocess.run(["pytest", ...]).returncode`.
4. (Stretch) Add a `CostLedger` entry to each checkpoint. On resume, print the total "tokens spent so far" and prove that replay does not double-count completed work.

## Key Takeaway

> The model thinks in short bursts; the system runs for weeks by keeping the real state outside the window, journaling every step, and verifying progress with real tests. Durability is not something you ask the LLM to do — it is something you engineer around it.

## Next Chapter

In Chapter 03 we externalize the source of truth even further: we make **git** the mission's memory. Every checkpoint becomes a commit, every decision becomes a file, and the agent reconstructs its world from `git log` instead of from its own context window.