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

In Chapter 01 we saw that a ChatGPT-style agent is essentially one long conversation. The model's context window holds the plan, the partial results, the mistakes, and the recovery attempts. When the process restarts, the API times out, or the context window fills, that conversation is gone — and the agent loses the plot.

Durability fixes this by moving the plot *outside* the model. The LLM still thinks in short bursts, but the system around it:

1. **Writes every meaningful step to durable storage before moving on.**
2. **Re-reads that storage at the start of every burst.**
3. **Only calls an item "done" when a deterministic verifier says so.**

This is not a smarter model. It is stricter engineering. The model can still hallucinate, but the system never trusts a hallucination as progress because progress is measured by tests, builds, lint, or exit codes — not by the model's own claim.

In the `lra-demo` project this discipline is captured in `src/lra/agent/loop.py`. The `AgentLoop` class runs one cycle: gather state, let the model act, verify the result, and checkpoint. Its return type, `CycleOutcome`, is the contract between the inner agent and the durable spine:

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

Notice what is *not* in that dataclass: the model's reasoning, its feelings about the task, or a narrative summary. The durable spine only cares whether the item advanced, whether it verified, and what the new git head is. Everything else is ephemeral.

---

## The "Assume Interruption" Rule

The central design rule of a long-running agent is: **assume interruption.**

Every cycle must be written as if the process will die immediately after it. That means:

- No in-memory plan that has not been saved.
- No tool result that has not been journaled.
- No "done" status that has not been verified externally.
- On restart, the agent must reconstruct situational awareness by *reading*, not by *remembering*.

This rule changes how you write every loop. A chat agent asks, "What did I just say?" A durable agent asks, "What does the saved state say happened?" The difference is the difference between a conversation and a transaction.

In `lra-demo/src/lra/state/mission_anchor.py` the `GitMissionAnchor` implements this rule. It does not ask the LLM what the mission status is. It reads the git repo, the checklist file, and the decision log. A reboot on day 12 reconstructs awareness in seconds because the truth is on disk, not in a context window.

---

## Volatile Context vs. Durable State

It helps to be explicit about what belongs where.

| Volatile (context window / RAM) | Durable (disk / git / database) |
|---|---|
| Model's current reasoning trace | Checklist of work items |
| Parsed tool arguments for this turn | Decision log and event journal |
| Recent observations (capped) | Verified git commits and HEAD SHA |
| Prompt scratchpad | Cost ledger and attempt counts |
| Intermediate chain-of-thought | Human approvals and HITL gates |

The context window is a **lossy cache**. It can be truncated, summarized, or dropped. Durable state is the **source of truth**. It must be complete, atomic, and re-readable.

A practical consequence: the agent prompt should be rebuilt from durable state every cycle. Do not append to a growing message list and hope the model remembers. In `lra-demo/src/lra/agent/prompt.py` the prompt builder re-assembles the context from the anchor, the current checklist, and the latest events. This is slower in tokens but infinitely safer across days of runtime.

---

## A Local Crash-Resume Loop

Before we add Temporal, git, or an LLM backend, we can prove the durability idea with plain Python. The script below keeps mission state in a JSON file, re-reads it every cycle, injects a random crash, and resumes without repeating work.

Create `lra-demo/ch02_local_durability.py`:

```python
"""ch02_local_durability.py

A self-contained crash-resume loop that demonstrates durability as an
engineering property. It keeps the real mission state on disk, re-reads it
every cycle, and can resume exactly where it left off after a crash.

This is the same shape as lra-demo/src/lra/agent/loop.py, stripped down
to plain Python so you can run it without Temporal, git, or an LLM.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Checkpoint:
    """The durable record of one attempt on one work item."""
    item_id: str
    status: str  # "in_progress" | "done" | "blocked"
    attempt: int
    head_sha: str
    events: list[dict] = field(default_factory=list)


@dataclass
class CycleOutcome:
    """The same shape used by lra-demo/src/lra/agent/loop.py."""
    item_id: str
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


class MissionState:
    """External source of truth: a JSON file on disk, re-read every cycle."""
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.state_file = workdir / "state.json"
        self.checkpoints: list[Checkpoint] = []
        self.current_item_index = 0
        self.load()

    def load(self) -> None:
        if not self.state_file.exists():
            return
        data = json.loads(self.state_file.read_text())
        self.checkpoints = [
            Checkpoint(
                item_id=c["item_id"],
                status=c["status"],
                attempt=c["attempt"],
                head_sha=c["head_sha"],
                events=c.get("events", []),
            )
            for c in data["checkpoints"]
        ]
        self.current_item_index = data.get("current_item_index", 0)

    def save(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoints": [
                {
                    "item_id": c.item_id,
                    "status": c.status,
                    "attempt": c.attempt,
                    "head_sha": c.head_sha,
                    "events": c.events,
                }
                for c in self.checkpoints
            ],
            "current_item_index": self.current_item_index,
        }
        # Atomic-ish write: write to temp, then rename.
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.state_file)


class AgentLoop:
    """One cycle: gather -> act -> verify -> checkpoint."""
    def __init__(
        self,
        state: MissionState,
        verifier: Callable[[str], bool],
    ):
        self.state = state
        self.verifier = verifier

    def _checkpoint_for(self, item_id: str) -> Checkpoint:
        matches = [c for c in self.state.checkpoints if c.item_id == item_id]
        if matches:
            return matches[-1]
        new_cp = Checkpoint(item_id=item_id, status="in_progress", attempt=0, head_sha="")
        self.state.checkpoints.append(new_cp)
        return new_cp

    def run_cycle(self, item: str) -> CycleOutcome:
        cp = self._checkpoint_for(item)
        cp.attempt += 1
        cp.status = "in_progress"
        self.state.save()

        # Gather: in a real agent this would read the checklist, git state, etc.
        # Act: simulate tool calls.
        tool_calls = random.randint(1, 3)
        cp.events.append({
            "type": "act",
            "ts": time.time(),
            "tool_calls": tool_calls,
            "attempt": cp.attempt,
        })

        # Verify: the only ground truth we trust.
        verified = self.verifier(item)
        cp.status = "done" if verified else "blocked"
        cp.head_sha = f"sha{random.randint(10000, 99999)}"
        self.state.save()

        return CycleOutcome(
            item_id=item,
            advanced=True,
            verified=verified,
            is_complete=verified,
            head_sha=cp.head_sha,
            tool_calls=tool_calls,
            turns=1,
        )


def run_mission(
    items: list[str],
    workdir: Path,
    crash_prob: float = 0.0,
    max_attempts: int = 3,
) -> None:
    """Run until done or crash. State is always on disk, so a crash is safe."""
    state = MissionState(workdir)
    loop = AgentLoop(state, verifier=lambda item: "fail" not in item)

    while state.current_item_index < len(items):
        item = items[state.current_item_index]
        print(f"[cycle] item={item!r} index={state.current_item_index}")

        outcome = loop.run_cycle(item)
        print(f"  -> verified={outcome.verified}, sha={outcome.head_sha}")

        # Simulate a process crash.
        if random.random() < crash_prob:
            raise RuntimeError(f"Simulated crash after working on {item!r}")

        if outcome.verified:
            state.current_item_index += 1
            state.save()
        else:
            attempts = len([c for c in state.checkpoints if c.item_id == item])
            if attempts >= max_attempts:
                print(f"  -> max attempts reached, skipping {item!r}")
                state.current_item_index += 1
                state.save()

    print("Mission complete.")


def resume_mission(items: list[str], workdir: Path) -> None:
    """Resume from the on-disk state. No work is repeated, no tokens re-spent."""
    state = MissionState(workdir)
    last = state.checkpoints[-1] if state.checkpoints else None
    print(f"[resume] index={state.current_item_index}, "
          f"last_item={last.item_id if last else None}, "
          f"last_sha={last.head_sha if last else None}")
    run_mission(items, workdir, crash_prob=0.0)


if __name__ == "__main__":
    WORKDIR = Path(".lra-demo/ch02")
    ITEMS = [
        "Create hello.py",
        "Add a test for hello.py",
        "Add a README (this item will fail verification)",
        "Add pyproject.toml",
    ]

    try:
        run_mission(ITEMS, WORKDIR, crash_prob=0.25)
    except RuntimeError as exc:
        print(f"\nCRASH: {exc}\n")
        resume_mission(ITEMS, WORKDIR)
```

Run it:

```bash
cd lra-demo
python ch02_local_durability.py
```

You will see the mission start, process one or more items, crash, and then resume from the exact item and SHA it left behind. The third item contains the word "fail", so the deterministic verifier rejects it; after three attempts the loop skips it and continues. The final item always completes if the process survives that long.

### Code walkthrough

- **`MissionState`** is the external source of truth. It loads from `.lra-demo/ch02/state.json` on construction and writes to it after every meaningful change. The write is atomic (temp file + rename) so a crash during write does not corrupt the state.
- **`AgentLoop.run_cycle`** follows the same four steps as `lra-demo/src/lra/agent/loop.py`: gather the current checkpoint, act (simulated tool calls), verify with an external function, and checkpoint the result.
- **`CycleOutcome`** reports only durable facts: did the item advance, did it verify, what is the new head SHA. The model's internal monologue is intentionally not part of the contract.
- **`run_mission`** is the thin scheduler. It loops over items, saves progress, and can inject a crash. Because state is on disk, the crash is harmless.
- **`resume_mission`** reconstructs awareness by reading the same JSON file. It does not ask the user what was happening; it asks the saved state.

This is the local version of what Temporal provides at scale. Temporal adds retries, replay-from-cache, durable sleep, and distributed workers, but the *discipline* — write state before moving on — must exist in your loop first.

---

## Hands-On Exercise

1. **Run the script three times.** Count how many times it crashes and resumes. Notice that the resume always starts from the last saved `current_item_index`, never from the beginning.

2. **Inspect the durable state.** After a run, open `.lra-demo/ch02/state.json`. Identify:
   - The `current_item_index`.
   - The `head_sha` of the last completed item.
   - The `events` array for the failed README item, showing multiple attempts.

3. **Make the verifier real.** Replace the lambda verifier with one that checks whether a file exists on disk. For example, the item `"Create hello.py"` should only verify if `.lra-demo/ch02/hello.py` exists. Create the file manually between runs and watch the loop advance only after the external truth changes.

4. **Induce two crashes in one mission.** Raise `crash_prob` to `0.5` and wrap `resume_mission` in a retry loop so it keeps resuming until the mission is complete. Confirm that no item is processed twice after a successful verification.

5. **Map this to the real repo.** Open `lra-demo/src/lra/agent/loop.py` and compare its `CycleOutcome`, `AgentLoop.run_cycle`, and checkpoint calls to the simplified version above. Identify where it calls the real `GitMissionAnchor` and `Verifier`.

---

> **Key Takeaway:** Durability is not something the LLM does; it is something the loop guarantees by writing state to disk before trusting it, re-reading that state every cycle, and verifying progress with external tests. A durable agent can crash, reboot, or lose its context window and still resume exactly where it left off — because the truth lives outside the model.

---

## Next Chapter

In Chapter 03, **Externalizing Truth — Git as Memory**, we replace the JSON state file with a real git repository. You will learn why git commits are the perfect checkpoint format, how the `GitMissionAnchor` turns every verified cycle into immutable history, and how a human can read the agent's entire week of work with `git log`.