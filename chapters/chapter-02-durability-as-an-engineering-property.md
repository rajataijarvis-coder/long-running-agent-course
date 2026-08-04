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

In Chapter 01 we saw the ChatGPT-style agent: one process, one context window, one API failure away from amnesia. The fix is not a smarter model. The fix is a system that *assumes interruption* and keeps enough state outside the model to resume anywhere.

Durability is an engineering property because it is built from boring, reliable primitives:

1. **Write the real state to disk/git before declaring progress.**
2. **Journal every step so the next process can reconstruct context.**
3. **Verify work with deterministic tests, not model self-assessment.**

The LLM still thinks in short bursts. The system, however, can run for days or weeks because each burst ends with a checkpoint.

### The "Assume Interruption" rule

Every cycle of the agent must be written as if the host will reboot immediately after it. That means:

- The model is not allowed to "remember" anything that is not also saved.
- The next cycle starts by *re-reading* the saved state, not by trusting the previous turn's context.
- A checkpoint is the only durable fact. Everything else is volatile.

This rule shows up directly in the LRA codebase. The inner agent loop in `src/lra/agent/loop.py` returns a `CycleOutcome` that is intentionally small and serializable:

```python
# src/lra/agent/loop.py
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

Those seven fields are the contract between the volatile inner loop and the durable spine. They answer the only questions the orchestrator needs to ask:

- `item_id`: what checklist item was attempted?
- `advanced`: did the state move forward at all?
- `verified`: did the deterministic verifier pass?
- `is_complete`: is the whole mission done?
- `head_sha`: which git commit now represents the saved state?
- `tool_calls` / `turns`: how much work was consumed?

The orchestrator does not need the model's reasoning trace. It needs the outcome. Everything else can be reconstructed from git and the mission log.

---

## Volatile Context vs. Durable State

| Volatile (lives in the window) | Durable (lives outside the window) |
|---|---|
| Model reasoning trace | Checklist of items and their status |
| Tool output summaries | Git commit graph and file tree |
| "I think we should..." | Decision log with IDs and rationale |
| Token budget in flight | Cost ledger written to disk |
| Current plan in prompt | `mission_state.json` / anchor |

The context window is a **lossy cache**. It is useful for one turn, but it is not the source of truth. The source of truth is the git repo plus the structured state files. When the process restarts, the agent re-reads those files and is back on the same page in seconds.

This is why LRA separates `AgentLoop` from the durable spine. `AgentLoop` is unit-testable on its own; it does not know whether it is running inside Temporal, a local script, or a test. The durable spine only cares about `CycleOutcome`.

---

## Code Walkthrough: A Crash-Resume Loop in Plain Python

Before we add Temporal, git anchors, or sandboxes, we can demonstrate the core idea with a tiny Python script. It keeps mission state in a JSON file, simulates a crash before saving 30% of the time, and resumes exactly where it left off.

Create `lra-demo/ch02/plain_durable_loop.py`:

```python
# lra-demo/ch02/plain_durable_loop.py
import json
import pathlib
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

STATE_PATH = pathlib.Path("mission_state.json")
LOG_PATH = pathlib.Path("mission_log.jsonl")


@dataclass
class CycleOutcome:
    """Same contract shape as src/lra/agent/loop.py, but standalone."""

    item_id: str
    advanced: bool
    verified: bool
    is_complete: bool
    cycle: int
    head_sha: str
    tool_calls: int
    turns: int


@dataclass
class MissionState:
    task: str
    items: dict[str, bool]
    current_cycle: int
    last_outcome: dict | None
    complete: bool


def initial_state() -> MissionState:
    return MissionState(
        task="Create hello.py and a test",
        items={
            "write_hello": False,
            "write_test": False,
            "run_test": False,
        },
        current_cycle=0,
        last_outcome=None,
        complete=False,
    )


def load_state() -> MissionState:
    if STATE_PATH.exists():
        raw = json.loads(STATE_PATH.read_text())
        return MissionState(**raw)
    return initial_state()


def save_state(state: MissionState, outcome: CycleOutcome) -> None:
    state.current_cycle = outcome.cycle
    state.last_outcome = asdict(outcome)

    if outcome.verified:
        state.items[outcome.item_id] = True

    state.complete = outcome.is_complete

    STATE_PATH.write_text(json.dumps(asdict(state), indent=2))

    with LOG_PATH.open("a") as log:
        log.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "cycle": outcome.cycle,
                    "outcome": asdict(outcome),
                }
            )
            + "\n"
        )


def pick_next_item(state: MissionState) -> str | None:
    for item_id, done in state.items.items():
        if not done:
            return item_id
    return None


def fake_git_commit(item_id: str, cycle: int) -> str:
    import hashlib

    payload = f"{item_id}:{cycle}:{time.time()}".encode()
    return hashlib.sha1(payload).hexdigest()[:7]


def run_one_cycle(state: MissionState) -> CycleOutcome:
    state.current_cycle += 1
    item = pick_next_item(state)

    if item is None:
        last_sha = (
            state.last_outcome.get("head_sha", "0000000")
            if state.last_outcome
            else "0000000"
        )
        return CycleOutcome(
            item_id="",
            advanced=False,
            verified=False,
            is_complete=True,
            cycle=state.current_cycle,
            head_sha=last_sha,
            tool_calls=0,
            turns=0,
        )

    # Simulate one tool call, one model turn, and a verifier that sometimes fails.
    verified = random.random() < 0.85
    advanced = True
    head = fake_git_commit(item, state.current_cycle)

    # Mission is complete only if this item verified and every earlier item is done.
    all_done = all(k == item or state.items[k] for k in state.items)
    is_complete = verified and all_done

    return CycleOutcome(
        item_id=item,
        advanced=advanced,
        verified=verified,
        is_complete=is_complete,
        cycle=state.current_cycle,
        head_sha=head,
        tool_calls=1,
        turns=1,
    )


def maybe_crash() -> None:
    """Simulate a process death BEFORE the checkpoint is written."""
    if random.random() < 0.3:
        print("💥 Simulated crash before checkpoint!")
        sys.exit(1)


def main() -> None:
    state = load_state()
    print(f"Resumed at cycle {state.current_cycle}, items={state.items}")

    while not state.complete:
        outcome = run_one_cycle(state)
        maybe_crash()
        save_state(state, outcome)

        icon = "✅" if outcome.verified else "🚫"
        print(
            f"Cycle {outcome.cycle}: {icon} {outcome.item_id} -> {outcome.head_sha}"
        )
        if not outcome.verified:
            print("  Item not verified; will retry on next cycle.")
        time.sleep(0.1)

    print(f"Mission complete in {state.current_cycle} cycles.")


if __name__ == "__main__":
    main()
```

Run it a few times. It will crash randomly, but each rerun picks up from the last saved `mission_state.json`:

```bash
cd lra-demo/ch02
python plain_durable_loop.py
# 💥 Simulated crash before checkpoint!
python plain_durable_loop.py
# Resumed at cycle 2, items={'write_hello': True, 'write_test': False, 'run_test': False}
# ...
```

The `maybe_crash()` call is deliberately placed **between** `run_one_cycle()` and `save_state()`. That is the dangerous gap. In a real agent, the dangerous gap is between "the model says it did the work" and "the verifier + git commit prove the work is done." A durable system never trusts progress until the checkpoint is on disk.

### Mapping this to the real LRA package

The standalone script mirrors the real design:

| Standalone script | LRA package |
|---|---|
| `MissionState` JSON file | `src/lra/state/mission_anchor.py` (`GitMissionAnchor`) |
| `CycleOutcome` dataclass | `src/lra/agent/loop.py` `CycleOutcome` |
| `save_state()` | Anchor commit + `TraceRecorder` in `src/lra/obs/events.py` |
| `maybe_crash()` | Host reboot, pod eviction, API timeout |
| `while not state.complete` | Temporal `MissionWorkflow` |

In LRA, the durable spine is Temporal. Every LLM call and tool call is wrapped in a Temporal **activity**. Temporal journals the activity inputs and outputs, retries transient failures, and replays from the journal after a crash. The agent does not re-spend tokens for work that already succeeded.

That is the difference between a chatbot and a long-running agent: the chatbot loses the conversation on restart; the agent loses only the work since the last checkpoint, and that work is measured in verified git commits.

---

## Hands-On Exercise: Build a Real Git-Backed Crash-Resume Loop

Replace the fake commit and fake verifier with real ones. The goal is a script that survives `Ctrl-C` or `kill -9` and finishes only when `pytest` passes.

1. Create a workspace:

```bash
mkdir -p lra-demo/ch02-real
cd lra-demo/ch02-real
git init
```

2. Write a `mission_state.json`:

```json
{
  "task": "Create hello.py and a passing test",
  "items": {
    "write_hello": false,
    "write_test": false,
    "run_pytest": false
  },
  "current_cycle": 0,
  "last_outcome": null,
  "complete": false
}
```

3. Write a Python script that:
   - Loads `mission_state.json`.
   - Picks the first unchecked item.
   - For `write_hello`, writes `hello.py` with a `hello()` function.
   - For `write_test`, writes `test_hello.py`.
   - For `run_pytest`, runs `pytest` with `subprocess.run(..., capture_output=True)` and treats the exit code as ground truth.
   - After each verified item, commits `mission_state.json` to git with a message like `checkpoint: run_pytest cycle 3`.
   - Saves the updated state before printing success.

4. Run the script, then randomly press `Ctrl-C` while it is running. Rerun it. Confirm:
   - It resumes from the last saved cycle.
   - It does not re-do already verified items.
   - It finishes only when `pytest` exits `0`.

5. Inspect the durable history:

```bash
git log --oneline
cat mission_state.json
cat mission_log.jsonl
```

This is the simplest version of the LRA durable contract: **verified work + saved state + git history = resumable mission**.

---

## Key Takeaway

> Durability is an engineering property, not a model capability. The model thinks in short bursts, and the system runs for weeks by doing three boring things very well: it keeps the real state outside the model, it journals every step, and it verifies progress with real tests.

---

## Next Chapter

**Chapter 03: Externalizing Truth — Git as Memory.** We will replace the JSON state file with a real git repo as the mission anchor, use commits as checkpoints, and see why the commit graph is the agent's true long-term memory.