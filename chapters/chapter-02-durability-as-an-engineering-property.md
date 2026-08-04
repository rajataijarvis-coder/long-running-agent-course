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

## From Chat-style to Durable

Chapter 01 showed that the ChatGPT-style agent is a single in-memory loop. It dies when the process restarts, the context window fills, or an API call fails. The instinctive fix is to ask for a "better" model — one with more reasoning, a longer context window, or cheaper tokens. That helps inside a single turn, but it does **nothing** for the loop itself. A smarter model still lives in the same process, still loses its memory on restart, and still repeats work after a crash.

Durability is not a model capability. It is an engineering property of the system around the model. A durable agent:

1. **Writes state to disk before it acts** so a restart can reconstruct where it was.
2. **Makes work idempotent** so repeating a step does not corrupt progress.
3. **Journals every step** so the next cycle, or a human, can read what happened.
4. **Assumes interruption** — it behaves as if the power could go out at any moment.

This chapter makes that concrete with a tiny local loop that survives a simulated crash. The same idea scales up to the full LRA system, where the loop runs inside a Temporal workflow and every model call is a journaled, replayed activity.

---

## Assume Interruption

The central design rule of LRA is **assume interruption**. Every cycle must be resumable from durable state alone. If the only record of progress lives in the LLM's context window, it is gone when the window is rebuilt. If it lives in a Python variable, it is gone when the process dies.

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

This object is small, but every field is durable-state glue:

- `item_id` — which checklist item was attempted.
- `advanced` — did the agent change the repo or state?
- `verified` — did the deterministic verifier pass?
- `is_complete` — is the item done?
- `head_sha` — the git commit SHA after the cycle, so the next cycle can re-read the exact state.
- `tool_calls` / `turns` — telemetry, useful for cost and replay.

The model never holds the "official" progress. The model only proposes the next action. The system writes the outcome to disk, commits it to git, and then — and only then — considers the cycle finished.

---

## Volatile Context vs. Durable State

A durable agent keeps two kinds of state:

| Volatile (lives in the LLM context) | Durable (lives on disk / in git) |
|---|---|
| Current task framing | Checklist of items |
| Recent observations | Decision log and event journal |
| This turn's reasoning | Git repo contents and commit graph |
| Tool results from the last few steps | Cost ledger and usage traces |
| Parsed action about to be executed | Checkpoint files and verifier results |

The context window is a **lossy cache**. It is rebuilt every cycle from durable files. That is why a reboot on day 12 reconstructs situational awareness in seconds: the agent does not remember the whole conversation; it re-reads the checklist, the git log, and the event journal.

---

## A Crash-Resume Loop in Plain Python

Before we add Temporal, git, or LLMs, we can prove the durability idea with a 60-line script. Save this as `lra-demo/ch02_crash_resume.py`:

```python
"""lra-demo/ch02_crash_resume.py

A minimal crash-resume loop. It writes a checkpoint after every step,
simulates a process crash, and resumes from the checkpoint without
re-doing work.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

CHECKPOINT = Path("lra-demo/.checkpoints/ch02.json")


@dataclass
class Checkpoint:
    item_id: str
    done: bool
    attempts: int
    output_file: str


def load_checkpoint() -> Checkpoint | None:
    if not CHECKPOINT.exists():
        return None
    data = json.loads(CHECKPOINT.read_text())
    return Checkpoint(**data)


def save_checkpoint(cp: Checkpoint) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(asdict(cp), indent=2))


def verify_step(cp: Checkpoint) -> bool:
    """Deterministic verifier: the step is done when the output file exists."""
    return Path(cp.output_file).exists()


def run_step(cp: Checkpoint) -> Checkpoint:
    """Do one unit of idempotent work."""
    print(f"[step] working on {cp.item_id} (attempt {cp.attempts + 1})")
    Path(cp.output_file).write_text(f"artifact for {cp.item_id}\n")
    return Checkpoint(
        item_id=cp.item_id,
        done=True,
        attempts=cp.attempts + 1,
        output_file=cp.output_file,
    )


def maybe_crash(step: int) -> None:
    """Simulate a process crash on step 2, but only once."""
    crash_marker = Path("lra-demo/.checkpoints/crashed_once")
    if step == 2 and not crash_marker.exists():
        crash_marker.write_text("yes")
        print("[FATAL] simulated process crash!")
        sys.exit(1)


def main() -> None:
    cp = load_checkpoint()
    if cp is None:
        cp = Checkpoint(
            item_id="ch02-hello-artifact",
            done=False,
            attempts=0,
            output_file="lra-demo/.artifacts/ch02_hello.txt",
        )

    # Resume semantics: if the verifier says we're done, skip the work.
    if verify_step(cp):
        cp = Checkpoint(
            item_id=cp.item_id,
            done=True,
            attempts=cp.attempts,
            output_file=cp.output_file,
        )
        print(f"[resume] already verified; nothing to redo.")
    else:
        cp = run_step(cp)

    save_checkpoint(cp)
    maybe_crash(2 if cp.done else 1)

    print(f"[done] checkpoint={CHECKPOINT}")
    print(f"[done] verified={verify_step(cp)}")


if __name__ == "__main__":
    main()
```

Run it twice to see the crash and resume:

```bash
cd lra-demo
python ch02_crash_resume.py
# [step] working on ch02-hello-artifact (attempt 1)
# [FATAL] simulated process crash!

python ch02_crash_resume.py
# [resume] already verified; nothing to redo.
# [done] checkpoint=.checkpoints/ch02.json
# [done] verified=True
```

The second run does not repeat the work because the checkpoint and the artifact already exist. The verifier is the ground truth, not the model's memory.

---

## Connecting to the Real LRA Loop

This tiny script is the seed of the full `AgentLoop` in `src/lra/agent/loop.py`. In the real system:

- `load_checkpoint()` becomes reading the git mission anchor and the event journal.
- `run_step()` becomes the model-driven tool loop.
- `verify_step()` becomes the deterministic verifier (tests, lint, build, typecheck).
- `save_checkpoint()` becomes a git commit plus a `CycleOutcome` written to the durable log.
- `maybe_crash()` becomes a host reboot, a pod eviction, or an API timeout.

Later chapters wrap this local loop inside a Temporal workflow. Temporal gives us **durable sleep**, **replay-from-cache**, and **continue-as-new** — but the underlying idea is the same: write state down, verify with real tests, and assume the process can die at any moment.

---

## Hands-on Exercise

Make the crash-resume loop idempotent across **three** steps instead of one:

1. Split the single `ch02-hello-artifact` item into three items:
   - `ch02-step-1-plan` — write a plan file.
   - `ch02-step-2-implement` — write an implementation file.
   - `ch02-step-3-test` — write a test file and run a real `python -m py_compile` check.
2. Store a list of completed item IDs in the checkpoint.
3. Simulate a crash at a random step.
4. Verify that after the crash the script resumes from the first not-yet-verified item and never re-runs a verified step.

Bonus: add a deterministic verifier that uses the **exit code** of `python -m py_compile` to decide whether `ch02-step-3-test` is done. This previews Chapter 06.

---

> **Key Takeaway:** A durable agent does not rely on the model to remember progress. It writes progress to disk, verifies it with real tests, and rebuilds context from that state every cycle. Durability is engineering, not intelligence.

---

## Next Chapter

In Chapter 03, we make the durable state **human-readable and auditable** by using git as the agent's memory. We will look at the mission anchor, the checklist file format, and why every checkpoint is a commit.