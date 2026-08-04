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

In Chapter 01 we saw that a ChatGPT-style agent is basically one long conversation. The model's context window holds the plan, the history, the failures, and the current file contents all at once. That works until something breaks: the process restarts, the API times out, or the context window fills up. Then the conversation is gone, and the agent has to start over.

Durability fixes this by moving the burden of memory from the model to the *system*. The model still thinks in short bursts, but the system keeps the mission alive across hours, days, or weeks. It does this by:

1. **Assuming interruption.** Every cycle is designed as if a crash could happen immediately after it.
2. **Writing real state to disk/git.** The context window is treated as a lossy cache, not the source of truth.
3. **Making progress verifiable.** A step is only "done" when an external check passes, not when the model says so.

This chapter focuses on the first two. We'll build a tiny crash-resume loop in plain Python, then connect it to the real `lra` code in `src/lra/agent/loop.py`.

---

## The "Assume Interruption" Design Rule

"Assume interruption" means: at any point in the agent's execution, the process might die. After it comes back, it must be able to figure out where it was and continue without redoing work or losing money on duplicate LLM calls.

This changes how you write every loop. Instead of:

```python
for item in checklist:
    result = model.act(item)      # expensive, volatile
    mark_done(item)               # lost if we crash here
```

you write:

```python
for item in load_remaining_checklist():
    if already_done(item):
        continue
    outcome = run_one_cycle(item)  # journaled, idempotent
    checkpoint(outcome)            # durable before moving on
```

The checkpoint is written *before* the next item starts. If the process dies right after the checkpoint, the next run reads the checkpoint and continues from the next item. If it dies *during* the cycle, the cycle is retried or resumed from its own internal journal.

This is exactly what the `AgentLoop` in `src/lra/agent/loop.py` does. It returns a `CycleOutcome`:

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

That small object is the durable contract between one cycle and the rest of the system. It tells the outer loop: which item was worked on, whether the repo advanced, whether verification passed, and which git commit (`head_sha`) now holds the result.

---

## Volatile Context vs. Durable State

| Volatile (can die) | Durable (must survive) |
|---|---|
| Model context window | Checklist on disk |
| In-memory variables | Git commits and tags |
| Open API connections | Event log and decision journal |
| Python object state | Checkpoint files |

The rule is simple: anything you would need to resume the mission must live outside the process. In `lra`, the durable state is managed by the `GitMissionAnchor` in `src/lra/state/mission_anchor.py`. It writes:

- `mission.json` — the checklist and current item
- `decisions.jsonl` — every decision the agents made
- `events.jsonl` — every cycle outcome and error
- git commits — the actual code produced

When the durable spine restarts, it does not ask the model "what were we doing?" It reads these files and replays the journal.

---

## A Runnable Crash-Resume Loop in Plain Python

To make this concrete, here is a minimal durable loop that does *not* use Temporal or any LLM. It simulates a long-running job that processes a checklist. It writes a checkpoint after every item, survives a simulated crash, and resumes without redoing finished work.

Create `lra-demo/ch02/crash_resume_demo.py`:

```python
"""Minimal crash-resume loop: durability as an engineering property.

This demo intentionally has no LLM and no Temporal. It shows the *shape*
of durability: write a checkpoint, read it back on restart, and continue.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CHECKPOINT_DIR = Path(".lra-demo/ch02/checkpoints")
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json"


@dataclass
class Checkpoint:
    version: int = 1
    next_index: int = 0
    completed: list[str] | None = None
    last_attempt_error: str | None = None

    def __post_init__(self):
        if self.completed is None:
            self.completed = []


def load_checkpoint() -> Checkpoint:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return Checkpoint(**data)
    return Checkpoint()


def save_checkpoint(cp: Checkpoint) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(asdict(cp), indent=2))


def flaky_work(item: str, attempt: int) -> str:
    """Simulates work that sometimes fails."""
    # 30% chance of failure on the first attempt, then stable.
    if attempt == 1 and random.random() < 0.3:
        raise RuntimeError(f"Simulated failure while processing {item!r}")
    time.sleep(0.1)
    return f"result-of-{item}"


def run_one_item(item: str, attempt: int) -> str:
    print(f"  → working on {item!r} (attempt {attempt})")
    result = flaky_work(item, attempt)
    print(f"  ✓ {item} -> {result}")
    return result


def main() -> None:
    checklist = [
        "parse-requirements",
        "scaffold-project",
        "write-first-test",
        "make-test-pass",
        "add-linting",
    ]

    cp = load_checkpoint()
    print(f"Loaded checkpoint: next_index={cp.next_index}, completed={cp.completed}")

    for i in range(cp.next_index, len(checklist)):
        item = checklist[i]
        attempt = 1 if item not in cp.completed else 2
        try:
            run_one_item(item, attempt)
        except RuntimeError as e:
            cp.last_attempt_error = str(e)
            save_checkpoint(cp)
            print(f"! Crash simulated at item {i}. Restart the process to resume.")
            sys.exit(1)

        cp.completed.append(item)
        cp.next_index = i + 1
        cp.last_attempt_error = None
        save_checkpoint(cp)
        print(f"Checkpoint saved: next_index={cp.next_index}\n")

    print("All items completed.")


if __name__ == "__main__":
    main()
```

Run it:

```bash
cd lra-demo
mkdir -p ch02
# paste the file above into ch02/crash_resume_demo.py
python ch02/crash_resume_demo.py
```

If the simulated crash fires, just run the same command again. The script reads `.lra-demo/ch02/checkpoints/checkpoint.json` and continues from `next_index`. Finished items are not reprocessed.

The checkpoint file looks like this after a partial run:

```json
{
  "version": 1,
  "next_index": 2,
  "completed": [
    "parse-requirements",
    "scaffold-project"
  ],
  "last_attempt_error": "Simulated failure while processing 'write-first-test'"
}
```

That is the whole idea in one file. The real `lra` system does the same thing, but the checkpoint is a git commit plus the `CycleOutcome` event log.

---

## Connecting the Demo to the Real `lra` Code

In `src/lra/agent/loop.py`, the `AgentLoop.run_cycle` method follows the same pattern:

1. **Gather** — read the current checklist, repo state, and recent events from the anchor.
2. **Act** — run the model/tool loop until the model signals it is done.
3. **Verify** — call the deterministic verifier.
4. **Checkpoint** — commit the result and emit a `CycleOutcome`.

The outer durable spine (a Temporal workflow, which we will cover in Chapters 12–15) calls `run_cycle` inside an *activity*. The activity is journaled and retried, so a crash during the cycle resumes from the last successful activity rather than restarting the whole mission.

The key insight: `AgentLoop` itself does not know about Temporal. It just knows how to do one durable cycle. That separation is what makes the system testable. You can unit-test the loop with a stub model and a fake verifier without spinning up a Temporal server.

---

## Hands-On Exercise

Make the demo truly idempotent across process death.

1. Copy `crash_resume_demo.py` to `crash_resume_idempotent.py`.
2. Add an `attempt_id` (a UUID) to the checkpoint before starting each item.
3. Make `run_one_item` write a side-effect file only if the `attempt_id` has not been seen before.
4. Simulate a crash by sending `SIGTERM` from another terminal while the script is sleeping:

   ```bash
   pkill -f crash_resume_idempotent.py
   ```

5. Restart the script and verify that:
   - No completed item is reprocessed.
   - The side-effect file for the interrupted item was written exactly once.

This is the same guarantee Temporal provides with its activity replay: work is either completed or retried, never duplicated.

---

## Key Takeaway

> Durability is not something you ask the model to do. It is a property of the system around the model: checkpoints on disk, idempotent activities, and an outer loop that assumes interruption. Get that right, and a cheap local model can run a mission for a week.

---

## Next Chapter

In **Chapter 03: Externalizing Truth — Git as Memory**, we will replace the JSON checkpoint with git commits. We'll see why the repo itself becomes the mission's source of truth, and how every `CycleOutcome` is anchored to a real `head_sha`.