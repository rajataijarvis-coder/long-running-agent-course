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

## From Chat Loop to Durable Machine

In Chapter 01 we saw that a ChatGPT-style agent is basically one long conversation. The moment the process dies, the context window fills, or an API call flakes, the run is over. That is fine for a quick answer, but it is catastrophic for a task that is supposed to take a week.

The fix is not to buy a smarter model. The fix is to build a system that *assumes interruption* and keeps going anyway. The LLM still thinks in short bursts — maybe a few hundred tokens at a time — but the system around it journals every burst, persists every decision, and verifies every step with real tests. If the host reboots on day 4, the agent picks up exactly where it left off. If the API rate-limits for an hour, the agent durable-sleeps and retries. If the context window fills, the agent compacts the window and re-reads the ground truth from disk.

That is what we mean by **durability as an engineering property**.

## Assume Interruption

The single most important design rule in LRA is:

> **Assume interruption.** Every cycle must be reconstructable from durable state alone.

This rule changes how you write every loop. You are not allowed to keep mission-critical state only in memory. You are not allowed to assume the previous API response is still in the context window. You are not allowed to mark work "done" just because the model said so.

Instead, the real state lives outside the model:

- The **checklist** of what needs to be built
- The **decision log** of why the agent chose what it chose
- The **event journal** of what actually happened
- The **git repo** that holds the real code
- The **checkpoint** that records whether the last cycle advanced, verified, or blocked

The model's context window is just a *lossy cache* — a scratchpad that gets rebuilt from the durable state at the start of every cycle.

## Volatile vs. Durable State

| Volatile (can die) | Durable (must survive) |
|---|---|
| Model context / prompt messages | `checklist.json` and `decisions.jsonl` |
| In-memory Python objects | Git commits in the workspace repo |
| Current API response | `events.jsonl` journal |
| Stack frames and local variables | `CycleOutcome` checkpoints |
| Open HTTP connections | Cost ledger on disk |

In `lra-demo/src/lra/agent/loop.py`, the `AgentLoop` class is deliberately decoupled from Temporal. It takes a model, a tool dispatcher, a verifier, and a `GitMissionAnchor`, and it returns a `CycleOutcome`. That outcome is the durable contract: it tells the rest of the system whether progress happened, whether the item is verified, and where the git head is.

Here is the checkpoint contract from `lra-demo/src/lra/agent/loop.py`:

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

- `item_id` — which checklist item this cycle worked on
- `advanced` — did the state move forward at all?
- `verified` — did the deterministic verifier pass?
- `is_complete` — is the whole mission done?
- `head_sha` — the git commit that captured the real state
- `tool_calls` / `turns` — telemetry for cost and loop detection

This small dataclass is the bridge between the "think/act" inner loop and the durable outer spine.

## A Runnable Crash-Resume Loop in Plain Python

Before we bring in Temporal, let's prove the idea with a tiny Python script that survives a simulated crash. Save this as `lra-demo/exercises/ch02_crash_resume.py`:

```python
"""A minimal durable loop that survives simulated process death.

This mirrors the LRA idea on a single machine with no servers:
- real state lives in a JSON file
- each cycle produces a CycleOutcome-like checkpoint
- a crash is simulated by raising SystemExit
- rerunning the script resumes exactly where it left off
"""

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# A tiny mission: build three pieces of a language server.
CHECKLIST = [
    {"id": "lsp-01", "title": "Initialize project skeleton"},
    {"id": "lsp-02", "title": "Parse JSON-RPC headers"},
    {"id": "lsp-03", "title": "Implement initialize handshake"},
]

STATE_FILE = Path(".lra/exercises/ch02_state.json")


@dataclass
class MissionState:
    current_index: int = 0
    attempts: int = 0
    verified_count: int = 0
    last_action: str = ""
    head_sha: str = "0000000"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


def load_state(path: Path) -> MissionState:
    if path.exists():
        return MissionState.from_dict(json.loads(path.read_text()))
    return MissionState()


def save_state(path: Path, state: MissionState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2))


def maybe_crash() -> None:
    """Simulate flaky infrastructure: 30% chance the process dies."""
    if random.random() < 0.3:
        raise SystemExit("Simulated process death (host reboot / API flake)")


def run_tool(item: dict, state: MissionState) -> str:
    """Pretend to do one unit of work. In reality this would be a tool call."""
    state.attempts += 1
    # Deterministic 'verification' for the demo: pass on the second attempt.
    if state.attempts >= 2:
        return f"verified:{item['id']}"
    return f"draft:{item['id']}"


def checkpoint(state: MissionState, item: dict, result: str) -> dict:
    """Record a CycleOutcome-like checkpoint and update durable state."""
    verified = result.startswith("verified:")
    advanced = verified or state.attempts > 0

    if verified:
        state.verified_count += 1
        state.current_index += 1
        state.attempts = 0
        state.head_sha = f"{item['id'][:7]}abc"

    state.last_action = result

    return {
        "item_id": item["id"],
        "advanced": advanced,
        "verified": verified,
        "is_complete": state.current_index >= len(CHECKLIST),
        "head_sha": state.head_sha,
        "tool_calls": state.attempts,
        "turns": state.attempts,
    }


def run_cycle(path: Path) -> MissionState:
    state = load_state(path)

    if state.current_index >= len(CHECKLIST):
        print("Mission already complete.")
        return state

    item = CHECKLIST[state.current_index]
    print(f"[cycle] working on {item['id']} — attempt {state.attempts + 1}")

    # The process may die here. If it does, the next rerun re-reads state.
    maybe_crash()

    result = run_tool(item, state)
    outcome = checkpoint(state, item, result)
    print(f"[checkpoint] {outcome}")

    save_state(path, state)
    return state


if __name__ == "__main__":
    # Optional reproducible seed: python ch02_crash_resume.py --seed 7
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        random.seed(int(sys.argv[idx + 1]))
    else:
        random.seed()

    while True:
        state = run_cycle(STATE_FILE)
        if state.current_index >= len(CHECKLIST):
            print(f"\nDone. verified={state.verified_count} head={state.head_sha}")
            break
        time.sleep(0.2)
```

Run it:

```bash
cd lra-demo
python exercises/ch02_crash_resume.py
```

You will probably see something like this:

```text
[cycle] working on lsp-01 — attempt 1
[cycle] working on lsp-01 — attempt 2
[checkpoint] {'item_id': 'lsp-01', 'advanced': True, 'verified': True, ...
[cycle] working on lsp-02 — attempt 1
```

Then it may crash. Rerun the exact same command. It resumes at `lsp-02` attempt 2. No work is lost, no tokens are re-spent (there are no tokens here, but the principle is the same), and the mission completes.

That is durability in a nutshell: **the loop is stateless; the state is on disk.**

## How Temporal Extends This Idea

The script above is durable against a single-process crash because it writes `STATE_FILE` before and after every cycle. But it is not durable against:

- a host reboot that kills the whole OS
- a network partition that leaves a tool call hanging
- a long idle period where you do not want to pay for a running Python process
- the need to retry an activity with exponential backoff
- replaying exactly the same sequence of events after months

That is why LRA wraps the inner loop in a [Temporal](https://temporal.io) workflow. In `lra-demo/src/lra/durable/`, the `MissionWorkflow` is a thin scheduler. Every LLM call, every tool call, and every verification is an **activity** that Temporal journals. If the worker crashes, a new worker resumes from the journal. If an activity fails, Temporal retries it with the policy you configured. If the workflow sleeps for six hours, it does so in **durable sleep** — the process exits, Temporal wakes it up later, and you pay nothing for the idle time.

The local JSON demo and the Temporal workflow share the same core insight: **keep the real state outside the model, journal every step, and make the loop reconstructable from that journal.**

## Hands-On Exercise

Make the crash-resume loop verify real files on disk instead of counting attempts.

1. In `run_tool`, for item `lsp-02`, create a file `exercises/ch02_workspace/parser.py` with the line `Content-Length: 43` if it does not already exist.
2. Add a `verify_parser() -> bool` function that returns `True` only when `parser.py` exists and contains the substring `Content-Length`.
3. Change `checkpoint` so that `verified` is set by `verify_parser()`, not by attempt count.
4. Run the script. Delete `parser.py` mid-run, or let the simulated crash happen, then rerun.

Expected behavior:

- The script never marks `lsp-02` done until `parser.py` contains the required string.
- If you delete the file and rerun, the script recreates it and eventually verifies it.
- The durable state file still tracks `current_index`, `attempts`, and `head_sha`.

This is the same contract LRA uses at scale: **verification is ground truth, and the durable journal records the attempt.**

## Key Takeaway

> Durability is an engineering property, not a model capability. The model thinks in short bursts; the system runs for weeks by keeping the real state outside the window, journaling every step, and making every cycle reconstructable from that journal.

---

## Next Chapter Teaser

In **Chapter 03: Externalizing Truth — Git as Memory**, we will replace the JSON state file with real git commits. Every checkpoint becomes a commit you can inspect, diff, and branch. We will see why git is the perfect durable memory for a long-running agent, and how the `GitMissionAnchor` in `lra-demo/src/lra/state/mission_anchor.py` turns the workspace repo into the single source of truth.