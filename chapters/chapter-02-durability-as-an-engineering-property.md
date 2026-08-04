# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why durability is a *system* concern, not a smarter-model concern
- The "Assume Interruption" design rule and how it changes every loop
- Volatile context vs. durable state: what lives in the window and what lives on disk/git
- The anatomy of a checkpoint using `CycleOutcome` from `src/lra/agent/loop.py`
- A runnable crash-resume loop in plain Python (`lra-demo/ch02_crash_resume.py`)
- How the durable spine (Temporal) extends this local idea to week-long missions

---

## From Chat Loop to Durable Loop

In Chapter 01 we saw that a ChatGPT-style agent is a single in-memory loop. It dies when the process restarts, the context window fills, or an API call fails. The fix is not to buy a bigger model or a bigger context window. The fix is to engineer the system around the model so that **interruption is the normal case, not the exception**.

This chapter introduces the central design rule of the whole course:

**Assume Interruption.** At any moment the process may be killed, the host may reboot, the API may rate-limit you, or the context window may be full. The next cycle must be able to reconstruct exactly where it left off by reading durable state, not by asking the model to remember.

The model still thinks in short bursts. The system, however, runs for weeks by doing three boring things very well:

1. **Keep the real state outside the model.**
2. **Journal every step.**
3. **Verify progress with real tests.**

That is what makes an agent *durable*.

---

## Volatile Context vs. Durable State

The context window is a lossy cache. It is fast to read from, but it disappears the moment the process dies. Anything that matters must live outside it.

| Volatile (in the window) | Durable (on disk / in git) |
|---|---|
| The model's current "understanding" | The checklist of remaining work |
| The last few tool outputs | The decision log and event history |
| The current plan in prose | The ownership map and ticket state |
| Cost estimates in the prompt | The cost ledger (`lra/governor/cost.py`) |
| File contents pasted into the prompt | The actual files in the workspace git repo |

When the system restarts, it does not ask the model, *"What were we doing?"* It reads the durable state and resumes. In `lra`, that durable state lives in:

- `src/lra/state/mission_anchor.py` — the git-backed anchor that owns the workspace
- `src/lra/contracts/state.py` — typed records for checkpoints, events, and the checklist
- `src/lra/agent/loop.py` — the `AgentLoop` that produces one `CycleOutcome` per step

---

## Anatomy of a Checkpoint

A checkpoint is a small, immutable record that says: *"After this cycle, here is exactly where things stand."* In `src/lra/agent/loop.py` the result of one cycle is captured by `CycleOutcome`:

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
- `advanced` — did the agent change anything in the workspace
- `verified` — did the deterministic verifier pass
- `is_complete` — is the whole mission finished
- `head_sha` — the git commit hash the workspace is pinned to
- `tool_calls` / `turns` — telemetry for cost and replay

The checkpoint is written *before* the model is asked to plan the next cycle. That ordering matters: if the process dies immediately after the checkpoint, the next run knows exactly what was accomplished and what is next.

---

## A Runnable Crash-Resume Loop

The full `lra` system uses Temporal for durable execution, but the idea is easier to see in plain Python first. Create `lra-demo/ch02_crash_resume.py`:

```python
"""Minimal crash-resume mission loop.

Run it once, kill it mid-way, then run it again. It continues from the
last checkpoint because the real state lives in a JSON file, not in memory.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# This demo keeps state in JSON. In the real lra package this is a git repo.
STATE_FILE = Path(".lra-demo/ch02_state.json")


@dataclass
class CycleOutcome:
    """Same shape as lra.contracts.state / lra.agent.loop.CycleOutcome."""
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


@dataclass
class MissionState:
    mission: str
    items: list[dict]
    ledger: dict = field(default_factory=lambda: {"usd": 0.0, "calls": 0})
    last_outcome: dict | None = None


def load_state() -> MissionState:
    if STATE_FILE.exists():
        raw = json.loads(STATE_FILE.read_text())
        return MissionState(**raw)

    return MissionState(
        mission="Create hello.py and a passing test",
        items=[
            {"id": "item-1", "title": "Write hello.py", "status": "todo", "attempts": 0},
            {"id": "item-2", "title": "Write test_hello.py", "status": "todo", "attempts": 0},
            {"id": "item-3", "title": "Run pytest", "status": "todo", "attempts": 0},
        ],
    )


def save_state(state: MissionState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(state), indent=2))


def verify(item_id: str) -> bool:
    if item_id == "item-1":
        return Path("hello.py").exists()
    if item_id == "item-2":
        return Path("test_hello.py").exists()
    if item_id == "item-3":
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_hello.py", "-q"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    return False


def act(item: dict) -> None:
    if item["id"] == "item-1":
        Path("hello.py").write_text(
            'def hello() -> str:\n    return "Hello, world!"\n\n'
            'if __name__ == "__main__":\n    print(hello())\n'
        )
    elif item["id"] == "item-2":
        Path("test_hello.py").write_text(
            'from hello import hello\n\n'
            'def test_hello() -> None:\n    assert hello() == "Hello, world!"\n'
        )
    elif item["id"] == "item-3":
        pass  # verification is the work for this item


def run_one_cycle(state: MissionState) -> CycleOutcome:
    todo = next((i for i in state.items if i["status"] != "done"), None)
    if todo is None:
        return CycleOutcome(
            item_id=None,
            advanced=False,
            verified=False,
            is_complete=True,
            head_sha="local",
            tool_calls=0,
            turns=0,
        )

    todo["attempts"] += 1
    print(f"[cycle] {todo['id']}: {todo['title']} (attempt {todo['attempts']})")

    act(todo)
    ok = verify(todo["id"])

    outcome = CycleOutcome(
        item_id=todo["id"],
        advanced=True,
        verified=ok,
        is_complete=ok and all(i["status"] == "done" for i in state.items),
        head_sha="local",
        tool_calls=1,
        turns=todo["attempts"],
    )

    state.ledger["calls"] += 1
    state.ledger["usd"] += 0.001

    if ok:
        todo["status"] = "done"
    else:
        todo["status"] = "blocked"

    state.last_outcome = asdict(outcome)
    save_state(state)
    return outcome


def simulate_crash(step: int) -> None:
    crash_at = os.environ.get("CRASH_AT")
    if crash_at and int(crash_at) == step:
        print(f"[crash] simulated failure at step {step}")
        raise SystemExit(1)


def main() -> int:
    state = load_state()
    print(
        f"[resume] mission={state.mission!r} "
        f"calls={state.ledger['calls']} usd=${state.ledger['usd']:.4f}"
    )

    for step in range(10):
        simulate_crash(step)
        outcome = run_one_cycle(state)
        print(f"[checkpoint] item={outcome.item_id} verified={outcome.verified} complete={outcome.is_complete}")

        if outcome.is_complete:
            print("[done] mission complete")
            return 0

        if not outcome.verified:
            print("[blocked] verification failed; will retry on next run")

    return 0


if __name__ == "__main__":
    import subprocess  # imported here so the module loads without it if desired
    raise SystemExit(main())
```

Run it:

```bash
cd lra-demo
rm -rf .lra-demo hello.py test_hello.py
python ch02_crash_resume.py
```

Then kill it after the first item is written, or simulate a deterministic crash:

```bash
CRASH_AT=1 python ch02_crash_resume.py
# [crash] simulated failure at step 1
```

Now run it again without the crash flag:

```bash
python ch02_crash_resume.py
```

The script reads `.lra-demo/ch02_state.json`, sees that `item-1` is already done, and continues with `item-2`. No work is repeated. The cost ledger is preserved. That is durability in its simplest form.

---

## From Local Checkpoint to Temporal Durable Spine

The demo above journals state to a JSON file. In `lra`, the same idea is pushed much further:

- The loop body (`AgentLoop.run_cycle` in `src/lra/agent/loop.py`) is called from inside a Temporal activity.
- Every LLM call, tool call, and verifier run is also an activity, so Temporal records it in the workflow history.
- If the worker crashes, a new worker picks up the workflow and **replays** from the cached activity results. No tokens are re-spent.
- Idle time is spent in **durable sleep**, which costs nothing and survives server restarts.

The local JSON demo and the Temporal workflow share the same mental model:

1. Read durable state.
2. Do one bounded unit of work.
3. Verify it deterministically.
4. Write a checkpoint.
5. Repeat.

Temporal just makes that model robust across process restarts, host reboots, and days of waiting.

---

## Hands-On Exercise

1. **Simulate a crash.** Run `CRASH_AT=1 python ch02_crash_resume.py`, then run it again without the flag. Inspect `.lra-demo/ch02_state.json` and confirm the checkpoint survived.

2. **Make the verifier stricter.** Change `verify("item-1")` to also check that `hello.py` contains the string `"Hello, world!"`, not just that the file exists. Delete `hello.py`, set the file to empty, and confirm the item becomes `blocked` instead of `done`.

3. **Add a retry budget.** Add a `max_attempts` field to each item. If an item is `blocked` and has exceeded its budget, mark it as `failed` and stop the mission. This is the seed of the governor and failure-trace ideas we will build in later chapters.

4. **(Preview)** Replace the JSON state file with a git commit after each verified item. This is exactly what `GitMissionAnchor` does in `src/lra/state/mission_anchor.py`, and it is the subject of Chapter 03.

---

> **Key Takeaway:** Durability is not something the model provides; it is something the system guarantees. The model thinks in short bursts, but the system runs for weeks by keeping the real state outside the context window, journaling every step, and resuming from the last verified checkpoint.

---

## Next Chapter

In **Chapter 03: Externalizing Truth — Git as Memory**, we will replace the JSON checkpoint file with a real git repository. You will see why git is the perfect source of truth for a long-running agent: every checkpoint is immutable, human-inspectable, and gives us a `head_sha` that the next cycle can trust.