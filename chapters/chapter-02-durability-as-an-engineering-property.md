# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

-   Defining durability in the context of autonomous agents.
-   The "Assume Interruption" mindset shift.
-   Volatile memory (context) vs. Durable storage (Git/DB).
-   Implementing checkpointing with `CycleOutcome`.
-   Wrapping agent logic in a Temporal workflow for crash recovery.
-   Hands-on: Building a crash-resume loop.

---

## The Engineering Mindset

In Chapter 01, we identified the four failure modes that kill ChatGPT-style agents: process restarts, context window limits, API failures, and logical drift. The common thread across all four is **volatility**.

Standard agents keep their state in RAM or the LLM's context window. If the process dies, the state vanishes. If the context fills up, the state is truncated. If the API fails, the state is inconsistent.

**Durability** means the system survives these events without losing progress. To achieve this, we must treat the LLM as a stateless function. It takes input (state + task), produces output (action), and then shuts up. It does not "remember" anything between calls. The *system* remembers.

This requires three engineering guarantees:
1.  **Externalized State:** The source of truth lives on disk (Git) or in a database, not in RAM.
2.  **Atomic Checkpoints:** Every action is recorded before the next one begins.
3.  **Replayability:** If a crash occurs, the system can reload the last checkpoint and resume exactly where it left off.

## The "Assume Interruption" Principle

In traditional software, we assume continuity. In long-running agent systems, we **assume interruption**.

Every time the agent loop runs, it must act as if it is the first time it is running. It cannot rely on variables stored in memory from the previous iteration. It must reload its state from the durable store.

This sounds inefficient, but it is the only way to guarantee survival over days or weeks. A reboot on day 12 should reconstruct situational awareness in seconds by reading the log, not by hoping the process stayed alive.

## Code Walkthrough: The Checkpoint Cycle

In the `lra` codebase, durability is enforced at the boundary of the `AgentLoop`. The loop produces a `CycleOutcome`, which describes exactly what happened. This outcome must be persisted before the loop runs again.

Let's look at how we structure the data for durability. We use Pydantic models to ensure strict typing, which is critical when deserializing state after a crash.

### Defining the Checkpoint

First, we define what a checkpoint looks like. This lives in `src/lra/contracts/state.py`.

```python
# src/lra/contracts/state.py
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Checkpoint:
    """
    A durable snapshot of the mission state.
    This is written to disk/Git after every successful cycle.
    """
    mission_id: str
    cycle_count: int
    current_item_id: Optional[str]
    completed_items: List[str] = field(default_factory=list)
    head_sha: str  # The Git commit hash representing this state
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    cost_usd: float = 0.0
    status: str = "RUNNING"  # RUNNING, BLOCKED, DONE, FAILED
```

### The Agent Loop Outcome

The `AgentLoop` (from `src/lra/agent/loop.py`) returns a `CycleOutcome`. This is the transient result that must be converted into a permanent `Checkpoint`.

```python
# src/lra/agent/loop.py (excerpt)
from dataclasses import dataclass

@dataclass
class CycleOutcome:
    """The result of one agent cycle."""
    item_id: str | None
    advanced: bool          # Did we move to the next checklist item?
    verified: bool          # Did the deterministic verifier pass?
    is_complete: bool       # Is the whole mission done?
    head_sha: str           # New Git commit hash
    tool_calls: int         # Count for cost tracking
    turns: int              # LLM turns taken
```

### The Durable Workflow Wrapper

In production, we wrap this loop in a Temporal Workflow. Temporal ensures that if the Python process crashes between `run_cycle` and `save_checkpoint`, the workflow replays from the last known good state.

Here is a simplified version of the workflow logic found in `src/lra/durable/workflow.py`.

```python
# src/lra/durable/workflow.py
from temporalio import workflow
from lra.contracts.state import Checkpoint
from lra.agent.loop import AgentLoop, CycleOutcome

# Import activities (these are the actual Python functions)
from .activities import run_agent_cycle_activity, save_checkpoint_activity

@workflow.defn
class MissionWorkflow:
    @workflow.run
    async def run(self, mission_id: str, task: str) -> Checkpoint:
        # 1. Load existing state if replaying
        state = await workflow.execute_activity(
            load_state_activity,
            mission_id,
            start_to_close_timeout=timedelta(seconds=10)
        )

        while not state.is_complete:
            # 2. Run the agent cycle (LLM + Tools + Verify)
            # Temporal guarantees this activity retries on failure
            outcome: CycleOutcome = await workflow.execute_activity(
                run_agent_cycle_activity,
                mission_id,
                state.current_item_id,
                start_to_close_timeout=timedelta(minutes=30)
            )

            # 3. Update State
            state.cycle_count += 1
            state.head_sha = outcome.head_sha
            if outcome.advanced:
                state.completed_items.append(outcome.item_id)
            
            # 4. Checkpoint to Git/Disk
            # This is the durable anchor. If we crash after this, we resume fresh.
            await workflow.execute_activity(
                save_checkpoint_activity,
                state,
                start_to_close_timeout=timedelta(seconds=10)
            )

            if outcome.is_complete:
                state.status = "DONE"
                break

        return state
```

**Why this matters:** Notice that `state` is passed into every activity. The workflow itself holds no mutable state in memory that isn't serialized. If the worker node dies at 3:00 AM, Temporal restarts the workflow on a new node, loads the last `Checkpoint`, and continues the `while` loop. No tokens are wasted re-doing previous steps.

## Hands-on Exercise: Build a Crash-Resume Loop

In this exercise, you will simulate the durability pattern without setting up Temporal yet. You will build a script that saves its progress to a JSON file, simulates a crash, and resumes without losing data.

### Prerequisites
-   Python 3.12+
-   `pip install pydantic`

### Step 1: Create the State Model

Create a file named `durable_state.py`:

```python
from pydantic import BaseModel
from typing import List
from datetime import datetime

class MissionState(BaseModel):
    mission_id: str
    step: int
    history: List[str]
    last_saved: str

    def save(self, filename: str = "state.json"):
        with open(filename, "w") as f:
            f.write(self.model_dump_json(indent=2))
        print(f"[CHECKPOINT] Saved step {self.step} to {filename}")

    @classmethod
    def load(cls, filename: str = "state.json") -> "MissionState":
        try:
            with open(filename, "r") as f:
                return cls.model_validate_json(f.read())
        except FileNotFoundError:
            # Initial state if no checkpoint exists
            return cls(
                mission_id="demo-001",
                step=0,
                history=[],
                last_saved=datetime.utcnow().isoformat()
            )
```

### Step 2: Create the Durable Loop

Create a file named `runner.py`:

```python
import time
import os
import sys
from durable_state import MissionState

def work_step(state: MissionState) -> MissionState:
    """Simulates work being done."""
    state.step += 1
    state.history.append(f"Completed step {state.step}")
    state.last_saved = time.time()
    return state

def main():
    # 1. Load State (Assume Interruption)
    print("--- Starting Agent ---")
    state = MissionState.load()
    print(f"Resumed at step {state.step}")

    # 2. Work Loop
    while state.step < 5:
        # Simulate work
        state = work_step(state)
        print(f"Working... now at step {state.step}")

        # 3. Checkpoint (Durability Boundary)
        state.save()

        # 4. Simulate Crash on Step 3
        if state.step == 3:
            print("! CRASH SIMULATED !")
            # Delete the process memory (exit)
            # In a real crash, the OS kills us here.
            sys.exit(1)

        time.sleep(1)

    print("--- Mission Complete ---")

if __name__ == "__main__":
    main()
```

### Step 3: Run and Observe

1.  Run the script: `python runner.py`
2.  Observe it crash at step 3.
3.  Run the script again: `python runner.py`
4.  Observe it **resume** at step 3 and continue to 5 without re-doing steps 1 and 2.

**Expected Output (Second Run):**
```text
--- Starting Agent ---
[CHECKPOINT] Loaded step 3 from state.json
Resumed at step 3
Working... now at step 4
[CHECKPOINT] Saved step 4 to state.json
Working... now at step 5
[CHECKPOINT] Saved step 5 to state.json
--- Mission Complete ---
```

### Analysis
If you had not saved the state to `state.json`, the second run would have started at step 0. By externalizing the state, you made the work durable. In the `lra` system, `state.json` is replaced by a Git commit, making the history immutable and auditable.

## Key Takeaway

> "The model thinks in short bursts, and the system runs for weeks by doing three boring things very well. It keeps the real state outside the model, it journals every step, and it verifies progress with real tests."
> — Fareed Khan

Durability is not about making the AI smarter. It is about building a container around the AI that survives failure. By externalizing state and checkpointing every cycle, you transform a fragile chat script into a robust engineering system.

## Next Chapter

Now that we understand *why* state must be external, we need to decide *where* to put it. In Chapter 03, we will explore **Externalizing Truth — Git as Memory**. We will see why Git is the perfect database for agent state and how to structure commits so both humans and agents can read the history.