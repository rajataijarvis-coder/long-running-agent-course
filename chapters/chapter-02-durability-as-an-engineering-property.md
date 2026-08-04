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

In Chapter 01 we saw the ChatGPT-style agent: one process, one context window, one chain of API calls. It works until it doesn't — a reboot, a full context window, or a failed API call wipes out the run.

A long-running agent does the opposite. It treats interruption as the normal case and continuous execution as the special case. The LLM is still only asked to think in short bursts, but the *system* keeps the mission alive across days or weeks by:

1. **Writing real state to disk after every burst.**
2. **Re-reading that state at the start of every burst.**
3. **Only trusting deterministic verification to mark progress.**

This is the "Assume Interruption" rule. Every cycle must be able to resume from a cold start using only durable state.

## Volatile Context vs. Durable State

The context window is a lossy cache. It is fast to read, but it can be truncated, summarized incorrectly, or lost entirely. Durable state lives outside the model and is the source of truth.

In the `lra-demo/` project, durable state is stored in ordinary files that survive process restarts:

```text
lra-demo/
├── state/
│   ├── checklist.json      # what needs to be done
│   ├── events.jsonl        # what happened, append-only
│   └── decisions.jsonl     # why the agent chose what it chose
└── checkpoints/
    └── cycle_0003.json     # the last verified checkpoint
```

The agent does not "remember" where it is. It *reads* where it is from these files every cycle. A reboot on day 12 reconstructs situational awareness in seconds because the truth is on disk, not in the model's head.

## Anatomy of a Checkpoint

A checkpoint is the atomic unit of durable progress. It records: what item was attempted, whether the work advanced, whether it verified, whether the item is complete, the git commit that captured the file changes, and how many tool calls and turns were spent.

The real codebase uses this dataclass in `src/lra/agent/loop.py`:

```python
from dataclasses import dataclass

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

- `item_id`: which checklist item this cycle worked on.
- `advanced`: did the file state move forward (e.g., a file was written or modified).
- `verified`: did the deterministic verifier pass.
- `is_complete`: is the checklist item now done.
- `head_sha`: the git commit that captured the changes.
- `tool_calls` / `turns`: cost/effort telemetry.

Only when `verified=True` and `is_complete=True` does the mission advance. Everything else is recorded as an attempt so a later cycle, or a human, can pick it up.

## A Runnable Crash-Resume Loop in Plain Python

Before we add Temporal, Docker, or vector search, we can prove the core idea with a tiny Python script. The script below is a self-contained crash-resume loop. It maintains a checklist in `lra-demo/state/checklist.json`, appends events to `lra-demo/state/events.jsonl`, and writes checkpoints to `lra-demo/checkpoints/`.

Create `lra-demo/ch02_crash_resume.py`:

```python
"""lra-demo/ch02_crash_resume.py

A minimal crash-resume agent loop. It proves the "Assume Interruption" rule:
the process can be killed at any time and resumed without losing progress.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKLIST_PATH = STATE_DIR / "checklist.json"
EVENTS_PATH = STATE_DIR / "events.jsonl"


@dataclass
class CycleOutcome:
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


def ensure_state() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not CHECKLIST_PATH.exists():
        checklist = {
            "items": [
                {"id": "item_1", "title": "Create hello.py", "done": False},
                {"id": "item_2", "title": "Create test_hello.py", "done": False},
                {"id": "item_3", "title": "Run pytest and pass", "done": False},
            ]
        }
        CHECKLIST_PATH.write_text(json.dumps(checklist, indent=2))


def load_checklist() -> dict:
    return json.loads(CHECKLIST_PATH.read_text())


def save_checklist(checklist: dict) -> None:
    CHECKLIST_PATH.write_text(json.dumps(checklist, indent=2))


def log_event(record: dict) -> None:
    with EVENTS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def latest_checkpoint() -> CycleOutcome | None:
    files = sorted(CHECKPOINT_DIR.glob("cycle_*.json"))
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    return CycleOutcome(**data)


def save_checkpoint(outcome: CycleOutcome) -> None:
    idx = len(list(CHECKPOINT_DIR.glob("cycle_*.json")))
    path = CHECKPOINT_DIR / f"cycle_{idx:04d}.json"
    path.write_text(json.dumps(asdict(outcome), indent=2))


def pick_next_item(checklist: dict) -> dict | None:
    for item in checklist["items"]:
        if not item["done"]:
            return item
    return None


def act(item: dict) -> tuple[bool, int]:
    """Simulate the 'tool loop': write files, spend turns, maybe crash."""
    tool_calls = 0

    if item["id"] == "item_1":
        (ROOT / "hello.py").write_text('print("hello")\n')
        tool_calls += 1
    elif item["id"] == "item_2":
        (ROOT / "test_hello.py").write_text(
            "from hello import say_hello\n\ndef test_hello():\n    assert say_hello() == 'hello'\n"
        )
        # Oops — we forgot to add say_hello() to hello.py. The verifier will catch this.
        tool_calls += 1
    elif item["id"] == "item_3":
        # Simulate running pytest by checking the file exists.
        tool_calls += 1

    return True, tool_calls


def verify(item: dict) -> tuple[bool, str]:
    if item["id"] == "item_1":
        return (ROOT / "hello.py").exists(), "hello.py exists"
    if item["id"] == "item_2":
        # The verifier imports the module to prove it works.
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("hello", ROOT / "hello.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return hasattr(mod, "say_hello"), "hello.say_hello exists"
        except Exception as e:
            return False, f"import failed: {e}"
    if item["id"] == "item_3":
        return (ROOT / "test_hello.py").exists(), "test file exists"
    return False, "unknown item"


def run_cycle() -> CycleOutcome:
    ensure_state()
    checklist = load_checklist()
    item = pick_next_item(checklist)

    if item is None:
        print("All items complete.")
        return CycleOutcome(
            item_id=None,
            advanced=False,
            verified=True,
            is_complete=True,
            head_sha="n/a",
            tool_calls=0,
            turns=0,
        )

    print(f"Working on {item['id']}: {item['title']}")

    # Simulate a crash 30% of the time before the cycle finishes.
    # In a real system this would be a host reboot or API failure.
    if random.random() < 0.3:
        print("💥 Simulated crash before checkpoint!")
        sys.exit(1)

    advanced, tool_calls = act(item)
    verified, reason = verify(item)

    outcome = CycleOutcome(
        item_id=item["id"],
        advanced=advanced,
        verified=verified,
        is_complete=verified,
        head_sha="fake-sha-for-demo",
        tool_calls=tool_calls,
        turns=1,
    )

    if verified:
        item["done"] = True

    save_checklist(checklist)
    save_checkpoint(outcome)
    log_event({
        "item_id": item["id"],
        "advanced": advanced,
        "verified": verified,
        "reason": reason,
    })

    status = "✅ done" if verified else "🚧 blocked"
    print(f"  {status} — {reason}")
    return outcome


if __name__ == "__main__":
    # Keep cycling until everything is verified, surviving injected crashes.
    while True:
        outcome = run_cycle()
        if outcome.is_complete and outcome.item_id is None:
            print("Mission complete.")
            break
        time.sleep(0.5)
```

Run it:

```bash
cd lra-demo
python ch02_crash_resume.py
```

You will see it pick an item, sometimes crash, and always resume from the last checkpoint. Because `item_2` writes a test that imports `say_hello` before `hello.py` defines it, the verifier returns `False`. The item stays not-done, the checkpoint is still written, and a later cycle can retry.

This is the whole durability idea in one file: **the process is allowed to die, but the state is never allowed to lie.**

## How Temporal Extends This

The script above is intentionally not using Temporal. It proves the rule on your laptop with no servers. The real `lra` package wraps the same idea in a Temporal workflow so the loop can survive:

- Host reboots
- Worker process restarts
- API rate limits and retries
- Days of idle time via durable sleep

In `src/lra/durable/` (the Temporal control plane), every LLM call and tool call becomes a journaled activity. Temporal caches the result, so if a worker crashes and resumes, the model is not re-called and tokens are not re-spent. The local `CycleOutcome` you just ran is the same object passed back from a Temporal activity.

The design principle is: **keep the inner loop small and testable, then let the durable spine make it long-lived.**

## Hands-On Exercise

1. Run `python lra-demo/ch02_crash_resume.py` several times. Watch it survive the simulated crashes.
2. After it blocks on `item_2`, inspect:
   - `lra-demo/state/checklist.json`
   - `lra-demo/state/events.jsonl`
   - `lra-demo/checkpoints/cycle_*.json`
3. Fix the bug: edit `act()` so `item_1` writes a `say_hello()` function in `hello.py`. Re-run and confirm `item_2` verifies.
4. Add a new `item_4` that requires `pytest` to pass. Make the verifier actually run `pytest` using `subprocess.run`. Verify the mission completes only when the real test passes.

## Key Takeaway

> The model thinks in short bursts; the system runs for weeks by keeping the real state outside the model, journaling every step, and verifying progress with real tests. Durability is not something you ask the LLM to do — it is something you engineer into the loop.

## Next Chapter

In Chapter 03 we will make that durable state explicit and versioned. We replace fragile JSON files with **git as the source of truth** — every checkpoint becomes a commit, every decision becomes a note in the repo, and the agent can `git log` its own history to reconstruct context.