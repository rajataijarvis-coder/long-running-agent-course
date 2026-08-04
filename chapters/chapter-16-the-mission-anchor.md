# Chapter 16: The Mission Anchor

> The workflow is disposable. The git repo is the truth. But without a single root record, a resumed worker has no idea which truth belongs to which mission.
> — Fareed Khan

## What We'll Cover

- Why a week-long mission needs a single, durable root record that outlives any single workflow execution
- What fields belong in the **Mission Anchor** and why it is stored in git, not in worker memory
- How the anchor is loaded at the start of every cycle and updated after every verified checkpoint
- How a minimal anchor payload crosses **Continue-As-New** boundaries while the real state stays in the repo
- How the anchor lets a brand-new worker resume a crashed mission without any shared memory
- A runnable demo in `lra-demo/ch16_mission_anchor.py` that creates, persists, and resumes a mission anchor from a local git repo

---

## Why the Workflow Still Needs an Anchor

By now we have a durable control plane.

- Chapter 12 turned the agent loop into a Temporal `MissionWorkflow`.
- Chapter 13 moved every side effect into retried, replayable activities.
- Chapter 14 added durable sleep and `continue_as_new` so the workflow never outgrows its history.
- Chapter 15 showed that a killed worker can resume exactly where it left off.

But there is still a missing piece: **identity**.

A Temporal workflow has an ID, but the workflow is a process. It can be replaced by `continue_as_new`, restarted on a different host, or even lost if the server namespace is rebuilt. The git repo, meanwhile, is full of rich state — checklists, decision logs, code, tests — but none of it says *which mission this is* or *what the original task was*.

The **Mission Anchor** is the root record that ties it all together. It is a small JSON file committed into the repo at `.lra/mission_anchor.json`. It is created once when the mission starts and updated every time the system checkpoints. Every fresh worker, every resumed workflow, and every `continue_as_new` starts by reading it.

The anchor is not the *whole* state. It is the *entry point* to the state.

---

## The Anchor Schema

The anchor is intentionally small. It stores only what you need to reconstruct the mission: the task, the budget, the current cycle, the last known git HEAD, and pointers to the other saved-state files.

Here is the real implementation, which lives in `src/lra/state/anchor.py`:

```python
# src/lra/state/anchor.py
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class MissionAnchor(BaseModel):
    mission_id: str
    task: str
    workdir: Path
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    status: Literal["RUNNING", "PAUSED", "DONE", "FAILED"] = "RUNNING"
    cycle: int = 0
    head_commit: str | None = None

    budget_ceiling_usd: float = 400.0
    cum_usd: float = 0.0
    model_backend: str = "stub"

    # Pointers to the other saved-state files in the repo.
    checklist_path: str = "checklist.json"
    decision_log_path: str = "decision_log.jsonl"
    skills_dir: str = "skills"

    # Idempotency marker used by the durable workflow.
    last_event_id: str | None = None

    # Open-ended bucket for mission-level metadata.
    meta: dict = Field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.workdir / ".lra" / "mission_anchor.json"

    @classmethod
    def load(cls, workdir: Path) -> "MissionAnchor":
        path = workdir / ".lra" / "mission_anchor.json"
        if not path.exists():
            raise FileNotFoundError(f"No mission anchor at {path}")
        return cls.model_validate_json(path.read_text())

    def save(self, commit: bool = True) -> None:
        """Write the anchor to disk and commit it into git."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.model_dump_json(indent=2))

        if commit:
            self._git_commit(f"anchor: cycle={self.cycle} status={self.status}")

    def _git_commit(self, message: str) -> None:
        """Stage the anchor and commit, updating head_commit on success."""
        subprocess.run(
            ["git", "-C", str(self.workdir), "add", str(self.path)],
            check=True,
        )

        result = subprocess.run(
            ["git", "-C", str(self.workdir), "commit", "-m", message],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            self.head_commit = (
                subprocess.check_output(
                    ["git", "-C", str(self.workdir), "rev-parse", "HEAD"]
                )
                .decode()
                .strip()
            )
        elif "nothing to commit" not in result.stdout and "nothing to commit" not in result.stderr:
            # A real error, not just an empty commit.
            result.check_returncode()

    def to_continue_as_new_payload(self) -> dict:
        """The smallest possible state to pass across a workflow renewal."""
        return {
            "mission_id": self.mission_id,
            "task": self.task,
            "workdir": str(self.workdir),
            "model_backend": self.model_backend,
            "budget_ceiling_usd": self.budget_ceiling_usd,
        }

    @classmethod
    def from_continue_as_new_payload(cls, payload: dict) -> "MissionAnchor":
        """After renewal, rehydrate from git. The payload is only a locator."""
        return cls.load(Path(payload["workdir"]))
```

Notice the design:

- `workdir` is a `Path`, but the continue-as-new payload stores it as a string. The receiving workflow immediately calls `MissionAnchor.load()` to get the current truth from disk.
- `save()` both writes the JSON and commits it. The commit message is deterministic, so replaying the same activity produces the same git history.
- `head_commit` is updated only after a successful commit, so the anchor always knows which git revision it is anchored to.

---

## The Anchor Inside the Durable Loop

The workflow from Chapter 12 made decisions. The activities from Chapter 13 did the work. Now the workflow also owns the anchor: it loads it at the start of every iteration, passes a minimal payload to activities, and reloads it after each checkpoint.

Here is the shape of `src/lra/durable/mission_workflow.py` with the anchor wired in:

```python
# src/lra/durable/mission_workflow.py (simplified)
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from lra.durable.activities import (
        gather_activity,
        act_activity,
        verify_activity,
        checkpoint_activity,
    )
    from lra.state.anchor import MissionAnchor

MAX_CYCLES_BEFORE_RENEW = 100


@workflow.defn
class MissionWorkflow:
    @workflow.run
    async def run(self, payload: dict) -> str:
        # Every start — fresh worker, resumed worker, or continued workflow —
        # reconstructs situational awareness from the anchor in git.
        anchor = MissionAnchor.from_continue_as_new_payload(payload)

        while anchor.status == "RUNNING":
            # Durable sleep from Chapter 14. Zero-cost idle time.
            if anchor.cycle > 0:
                await workflow.sleep(timedelta(seconds=10))

            # 1) Gather
            plan = await workflow.execute_activity(
                gather_activity,
                args=(anchor.to_continue_as_new_payload(),),
                start_to_close_timeout=timedelta(minutes=2),
            )

            # 2) Act
            await workflow.execute_activity(
                act_activity,
                args=(anchor.to_continue_as_new_payload(), plan),
                start_to_close_timeout=timedelta(minutes=5),
            )

            # 3) Verify
            verified = await workflow.execute_activity(
                verify_activity,
                args=(anchor.to_continue_as_new_payload(),),
                start_to_close_timeout=timedelta(minutes=5),
            )

            if not verified:
                # The verifier is the only source of "done". No verification,
                # no checkpoint. Loop and try again.
                continue

            # 4) Checkpoint: update the anchor in git via an activity.
            anchor.cycle += 1
            event_id = f"{anchor.mission_id}-{anchor.cycle}"
            await workflow.execute_activity(
                checkpoint_activity,
                args=(anchor.to_continue_as_new_payload(), anchor.cycle, event_id),
                start_to_close_timeout=timedelta(minutes=1),
            )

            # Reload so the next iteration sees the committed truth.
            anchor = MissionAnchor.load(anchor.workdir)

            # Budget governor from Chapter 20.
            if anchor.cum_usd >= anchor.budget_ceiling_usid:
                anchor.status = "FAILED"
                anchor.save()
                break

            # Renew the workflow before the event history grows too large.
            if anchor.cycle >= MAX_CYCLES_BEFORE_RENEW:
                await workflow.continue_as_new(anchor.to_continue_as_new_payload())

        return anchor.status
```

The key pattern is **load → decide → act → reload**. The workflow never keeps stale state in memory. If the worker is killed after `checkpoint_activity` but before the reload, the next worker simply loads the latest anchor from git and continues.

---

## Idempotent Checkpointing

When Temporal replays a workflow after a crash, it re-executes the workflow code from the beginning. Without care, the `checkpoint_activity` could run twice and produce two git commits for the same cycle.

The anchor prevents this with `last_event_id`. The checkpoint activity checks the marker before writing:

```python
# src/lra/durable/activities.py
from lra.state.anchor import MissionAnchor


async def checkpoint_activity(payload: dict, cycle: int, event_id: str) -> dict:
    anchor = MissionAnchor.from_continue_as_new_payload(payload)

    # If this exact event already checkpointed, this is a replay. Do nothing.
    if anchor.last_event_id == event_id:
        return anchor.model_dump(mode="json")

    anchor.cycle = cycle
    anchor.last_event_id = event_id
    anchor.save(commit=True)

    return anchor.model_dump(mode="json")
```

Because the activity is journaled by Temporal, a replay sees the first execution in the event history and does not call the activity again. But the idempotency guard is still essential for manual retries, activity timeouts, or any future backend change.

---

## Runnable Demo: Create, Persist, and Resume an Anchor

The demo below is self-contained and does not need a Temporal server. It creates a temporary git repo, writes a `MissionAnchor`, simulates three cycles, **destroys the in-memory object** to mimic a worker crash, and then resumes from the file on disk.

Save this as `lra-demo/ch16_mission_anchor.py`:

```python
# lra-demo/ch16_mission_anchor.py
from __future__ import annotations

import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class MissionAnchor(BaseModel):
    mission_id: str
    task: str
    workdir: Path
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    status: Literal["RUNNING", "PAUSED", "DONE", "FAILED"] = "RUNNING"
    cycle: int = 0
    head_commit: str | None = None
    budget_ceiling_usd: float = 10.0
    cum_usd: float = 0.0
    model_backend: str = "stub"
    last_event_id: str | None = None

    @property
    def path(self) -> Path:
        return self.workdir / ".lra" / "mission_anchor.json"

    @classmethod
    def load(cls, workdir: Path) -> "MissionAnchor":
        path = workdir / ".lra" / "mission_anchor.json"
        if not path.exists():
            raise FileNotFoundError(f"No mission anchor at {path}")
        return cls.model_validate_json(path.read_text())

    def save(self, commit: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.model_dump_json(indent=2))
        if commit:
            self._git_commit(f"anchor: cycle={self.cycle} status={self.status}")

    def _git_commit(self, message: str) -> None:
        subprocess.run(
            ["git", "-C", str(self.workdir), "add", str(self.path)],
            check=True,
        )
        result = subprocess.run(
            ["git", "-C", str(self.workdir), "commit", "-m", message],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            self.head_commit = (
                subprocess.check_output(
                    ["git", "-C", str(self.workdir), "rev-parse", "HEAD"]
                )
                .decode()
                .strip()
            )
        elif "nothing to commit" not in result.stdout and "nothing to commit" not in result.stderr:
            result.check_returncode()

    def to_continue_as_new_payload(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "task": self.task,
            "workdir": str(self.workdir),
            "model_backend": self.model_backend,
            "budget_ceiling_usd": self.budget_ceiling_usd,
        }


def init_git_repo(workdir: Path) -> None:
    subprocess.run(["git", "-C", str(workdir), "init"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(workdir), "config", "user.email", "demo@lra.local"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(workdir), "config", "user.name", "LRA Demo"],
        check=True,
    )


def simulate_cycle(anchor: MissionAnchor, n: int) -> None:
    anchor.cycle = n
    anchor.cum_usd += 0.05
    anchor.last_event_id = f"evt-{n}"
    anchor.save(commit=True)
    print(f"  cycle={n} head={anchor.head_commit[:8]} cum_usd={anchor.cum_usd:.2f}")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="lra_anchor_demo_"))
    try:
        init_git_repo(workdir)

        anchor = MissionAnchor(
            mission_id=str(uuid4()),
            task="Create hello.py and a test for it",
            workdir=workdir,
        )
        anchor.save(commit=True)
        print(f"Created mission {anchor.mission_id}")

        for i in range(1, 4):
            simulate_cycle(anchor, i)

        # Simulate a worker crash: drop the in-memory object.
        print("\n--- worker dies ---\n")
        del anchor

        # A fresh worker arrives. It knows nothing except the workdir.
        resumed = MissionAnchor.load(workdir)
        resumed.cycle += 1
        resumed.cum_usd += 0.05
        resumed.last_event_id = f"evt-{resumed.cycle}"
        resumed.save(commit=True)
        print(f"Resumed mission {resumed.mission_id}")
        print(f"  cycle={resumed.cycle} head={resumed.head_commit[:8]} cum_usd={resumed.cum_usd:.2f}")

        print("\nGit log:")
        log = subprocess.check_output(
            ["git", "-C", str(workdir), "log", "--oneline"]
        ).decode()
        print(log)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
```

Run it:

```bash
cd lra-demo
python ch16_mission_anchor.py
```

You should see output like this:

```text
Created mission 0192a3f4-...
  cycle=1 head=a1b2c3d4 cum_usd=0.05
  cycle=2 head=e5f6a7b8 cum_usd=0.10
  cycle=3 head=9c0d1e2f cum_usd=0.15

--- worker dies ---

Resumed mission 0192a3f4-...
  cycle=4 head=3a4b5c6d cum_usd=0.20

Git log:
3a4b5c6d anchor: cycle=4 status=RUNNING
9c0d1e2f anchor: cycle=3 status=RUNNING
e5f6a7b8 anchor: cycle=2 status=RUNNING
a1b2c3d4 anchor: cycle=1 status=RUNNING
...
```

The mission identity survived the "crash" because the anchor was in git, not in the worker process.

---

## Hands-On Exercise

1. Run `lra-demo/ch16_mission_anchor.py` and inspect the printed git log.
2. Modify the demo so that after cycle 2 the anchor records a **scope cut**: set `anchor.meta["scope_cuts"] = 1` and append a decision to a `decisions.jsonl` file in the repo before saving.
3. Verify that a fresh `MissionAnchor.load(workdir)` still sees the scope cut in `meta`, and that `git log` shows the checkpoint commit.
4. Extra credit: add a `resume_from_head(workdir: Path)` helper that returns the anchor only if `head_commit` matches the current `git rev-parse HEAD`. This is the safety check a production worker would use before trusting a loaded anchor.

---

## Key Takeaway

> The Mission Anchor is the durable identity of the mission. Keep it small, keep it in git, and load it at the start of every cycle. A new worker should be able to walk up to a repo, read one file, and know exactly what mission it is continuing.

---

## Next Chapter

In **Chapter 17: Tiered Memory**, we move beyond the anchor and build a multi-layer memory system — working context, short-term event logs, and long-term skill embeddings — so the agent can recall what it learned yesterday without paying to re-read the entire git history every cycle.