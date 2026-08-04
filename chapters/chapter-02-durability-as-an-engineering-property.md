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

## Why Smarter Models Do Not Help

In Chapter 01 we saw that chat-style agents collapse because they keep the plan, the progress, and the working memory inside one ephemeral context window. A bigger model, a better prompt, or a longer context window only delays the failure. The root problem is architectural: the agent treats memory as a cache instead of a store.

Durability is not about the model. It is about the system around the model. A durable agent assumes that at any moment the process may die, the host may reboot, the API may rate-limit, or the context window may fill. It builds every cycle so that a fresh process can pick up exactly where the old one left off.

This chapter makes that concrete with a small, runnable Python program. No Temporal, no database, no LLM API. Just a loop, a JSON file, and a deterministic rule for "done."

---

## The "Assume Interruption" Design Rule

The rule is simple: **every cycle must start by reading the truth from disk, and every cycle must end by writing the truth back to disk.** Nothing that matters can live only in memory.

This rule changes how you design every part of the agent:

| If you assume the process never dies | If you assume interruption |
|---|---|
| Keep the checklist in the prompt | Keep the checklist in a file and reload it |
| Mark an item done when the model says so | Mark an item done only after a verifier passes |
| Retry a failed API call in memory | Retry with a recorded attempt count and backoff |
| Sleep with `time.sleep` | Sleep with a durable timer that survives restarts |
| One long chat for the whole mission | Many short, journaled cycles |

The LRA system applies this rule at two scales. At the small scale, the inner agent loop in `src/lra/agent/loop.py` produces a `CycleOutcome` after every turn. At the large scale, the whole mission runs inside a Temporal workflow that journals every activity and replays from cache. This chapter focuses on the small scale; the Temporal spine comes in Chapters 12–15.

---

## Anatomy of a Checkpoint

Open `src/lra/agent/loop.py`. The `AgentLoop` returns one `CycleOutcome` per cycle:

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

Each field has a job:

- `item_id`: which checklist item this cycle worked on.
- `advanced`: did the mission move forward at all?
- `verified`: did the deterministic verifier say the work is good?
- `is_complete`: is the whole mission finished?
- `head_sha`: the git commit SHA that captured the state at the end of the cycle.
- `tool_calls` / `turns`: how much model work was consumed.

The important thing is that `CycleOutcome` is a **fact**, not an opinion. It is written to disk before the next cycle starts. If the process restarts, the next cycle reads the last outcome, reconstructs situational awareness, and continues.

---

## A Runnable Crash-Resume Loop

The full LRA system is large, but the durability idea is small enough to fit in one file. Create `lra-demo/ch02_crash_resume.py`:

```python
"""Crash-resume demo: a plain-Python durable loop.

Run:
    python lra-demo/ch02_crash_resume.py

Then press Ctrl-C while it is working. Run the same command again;
it resumes from the last checkpoint instead of starting over.
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

try:
    from lra.agent.loop import CycleOutcome
except Exception:  # pragma: no cover - demo runs without the full package
    @dataclass
    class CycleOutcome:
        item_id: str | None
        advanced: bool
        verified: bool
        is_complete: bool
        head_sha: str
        tool_calls: int
        turns: int


STATE_PATH = Path(".lra-demo/ch02_state.json")
CRASH_FLAG = Path(".lra-demo/ch02_crash.flag")

CHECKLIST = [
    "scaffold repo",
    "add hello.py",
    "add a test",
    "run the test suite",
]


@dataclass
class Checkpoint:
    head: int = 0                      # index of next item to do
    done: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)
    outcomes: list[dict] = field(default_factory=list)


def load_checkpoint() -> Checkpoint:
    if STATE_PATH.exists():
        raw = json.loads(STATE_PATH.read_text())
        return Checkpoint(
            head=raw.get("head", 0),
            done=raw.get("done", []),
            attempts=raw.get("attempts", {}),
            outcomes=raw.get("outcomes", []),
        )
    return Checkpoint()


def save_checkpoint(cp: Checkpoint) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(asdict(cp), indent=2))


def deterministic_verify(item: str, attempt: int) -> bool:
    """The verifier is dumb, deterministic, and never lies.

    In the real system this runs pytest / mypy / build. Here we just
    pretend 'add a test' fails on even attempts so we can see retry.
    """
    if item == "add a test":
        return attempt % 2 == 1
    return True


def do_one_cycle(cp: Checkpoint) -> CycleOutcome:
    if cp.head >= len(CHECKLIST):
        return CycleOutcome(
            item_id=None,
            advanced=False,
            verified=False,
            is_complete=True,
            head_sha=f"head={cp.head}",
            tool_calls=0,
            turns=0,
        )

    item = CHECKLIST[cp.head]
    cp.attempts[item] = cp.attempts.get(item, 0) + 1
    attempt = cp.attempts[item]

    print(f"[cycle] item={item!r} attempt={attempt}")

    # Simulate work. A crash can happen here and we must be able to resume.
    time.sleep(0.5)

    verified = deterministic_verify(item, attempt)

    outcome = CycleOutcome(
        item_id=item,
        advanced=verified,
        verified=verified,
        is_complete=False,
        head_sha=f"head={cp.head}",
        tool_calls=1,
        turns=1,
    )

    # Write the outcome BEFORE we advance the pointer. If we crash now,
    # the next run sees a blocked item and retries it — no duplicate work.
    cp.outcomes.append(asdict(outcome))
    save_checkpoint(cp)

    if verified:
        cp.done.append(item)
        cp.head += 1
        save_checkpoint(cp)
        print(f"[cycle] {item!r} -> DONE")
    else:
        print(f"[cycle] {item!r} -> BLOCKED, will retry")

    return outcome


def install_crash_handler() -> None:
    """If the user hits Ctrl-C, exit cleanly and leave the checkpoint intact."""

    def handler(signum, frame):
        print("\n[crash] interrupted — state is on disk; rerun to resume")
        sys.exit(130)

    signal.signal(signal.SIGINT, handler)


def main() -> None:
    install_crash_handler()

    cp = load_checkpoint()
    print(f"[resume] head={cp.head} done={cp.done} attempts={cp.attempts}")

    # Optional artificial crash flag for automated testing.
    if CRASH_FLAG.exists():
        CRASH_FLAG.unlink()
        raise RuntimeError("simulated crash mid-mission")

    while cp.head < len(CHECKLIST):
        do_one_cycle(cp)

    print("[mission] complete")
    print(json.dumps(asdict(cp), indent=2))


if __name__ == "__main__":
    main()
```

### Walkthrough

1. **State lives on disk.** `Checkpoint` is a plain dataclass serialized to `.lra-demo/ch02_state.json`. Every cycle begins with `load_checkpoint()` and ends with `save_checkpoint()`.
2. **The verifier is the only source of "done."** `deterministic_verify()` returns `True` or `False` based on a rule, not on a model guess. In the real LRA system this function runs `pytest`, `mypy`, or a build command.
3. **Crash safety is built into the write order.** The outcome is saved *before* the head pointer advances. If the process dies between the two writes, the next cycle sees the item as blocked and retries it.
4. **The signal handler makes interruption explicit.** When you press Ctrl-C, the program exits but leaves the JSON file consistent.

### Run It

```bash
# clean any previous run
rm -rf .lra-demo/ch02_state.json

# first run; interrupt it with Ctrl-C while "add a test" is running
python lra-demo/ch02_crash_resume.py

# run again — it resumes from the checkpoint
python lra-demo/ch02_crash_resume.py
```

Watch the state file while it runs:

```bash
cat .lra-demo/ch02_state.json
```

You will see `head` move forward only after an item is verified. The `attempts` counter prevents infinite retry loops and gives you a record of how many tries each item took.

---

## From Local Loop to Temporal Spine

This demo is intentionally tiny. The real LRA system does the same thing with more machinery:

- The `Checkpoint` becomes a git commit plus a structured mission log in `src/lra/state/`.
- The `deterministic_verify` function becomes the `Verifier` protocol in `src/lra/contracts/verify.py`.
- The `CycleOutcome` is returned by `AgentLoop.run()` in `src/lra/agent/loop.py`.
- The crash handler and durable sleep become Temporal workflows and activities in `src/lra/durable/`.

The mental model is identical: **read truth, do work, verify, write truth.** Temporal just guarantees that the same sequence of activities is replayed exactly after a process crash, without re-spending tokens or re-running tools.

---

## Hands-On Exercise

1. Run `lra-demo/ch02_crash_resume.py` and interrupt it with Ctrl-C during the "add a test" item. Inspect `.lra-demo/ch02_state.json` and confirm that `head` did not advance past the unverified item.
2. Resume the run. Confirm it retries the blocked item and eventually completes the mission.
3. Modify `deterministic_verify` so that "run the test suite" fails on the first attempt but succeeds on the second. Verify that the `attempts` counter and the checkpoint make this safe.
4. Add an `idempotency_token` field to `CycleOutcome` and write it into the checkpoint. Explain why an idempotency token matters when the underlying tool calls an external API.
5. (Stretch) Replace the JSON file with a tiny SQLite table and show that the same read-work-verify-write pattern still survives a crash.

---

> **Key Takeaway:** Durability is not a feature you bolt onto an agent; it is a property you design into every cycle. Assume interruption, keep the truth on disk, and never let the model declare its own work finished.

---

**Next Chapter:** Chapter 03 — *Externalizing Truth — Git as Memory*. We will replace the JSON checkpoint with a git repository and turn every verified cycle into a commit that a fresh process can inspect and resume from.