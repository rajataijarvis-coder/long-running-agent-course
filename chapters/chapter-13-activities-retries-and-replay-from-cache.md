# Chapter 13: Activities, Retries, and Replay-from-Cache

> Every side effect that can fail must be a journaled, retried, replayable activity. The workflow only decides; it never does.
> — Fareed Khan

## What We'll Cover

- Why the durable workflow from Chapter 12 must not perform side effects directly
- How Temporal **activities** become the unit of work for gather, act, verify, and checkpoint
- Configuring **retry policies** so transient failures recover and permanent failures stop fast
- Building a **local idempotency cache** so retried activities do not re-spend tokens or duplicate external calls
- How **replay-from-cache** uses the workflow event history to resume after a crash without re-execution
- A runnable demo in `lra-demo/ch13_activities_retries_replay.py` that retries a flaky model call and survives a worker restart

---

## The Concept: Workflows Decide, Activities Execute

In Chapter 12 we turned the mission loop into a Temporal workflow. A workflow is a deterministic scheduler: given the same history, it must make the same decisions. That means a workflow cannot call an LLM, write a file, run a test, or sleep with `asyncio.sleep` — those are non-deterministic side effects.

The **activity** is the boundary where side effects are allowed. Temporal runs each activity, records its input and output in the workflow event history, and retries it when it fails. If the worker crashes after an activity completes, the new worker replays the workflow from the history; completed activities return their cached results instead of re-executing. This is **replay-from-cache**, and it is what makes a week-long mission cheap and safe.

The mapping to Chapter 04's cycle is direct:

| Cycle step | Activity | Retryable? | Idempotent? |
|---|---|---|---|
| Gather | `gather_context` | Yes | Yes |
| Act | `generate_patch` | Yes | Must be |
| Verify | `run_verification` | Yes | Yes |
| Checkpoint | `checkpoint_state` | Yes | Yes |

In the real `lra` package these live under `src/lra/durable/activities.py` and are called from `src/lra/durable/workflows.py`. The demo below keeps everything in one file so you can run it without the rest of the package.

---

## Demo: `lra-demo/ch13_activities_retries_replay.py`

This script defines four activities, wires them into a `MissionWorkflow`, starts a local worker, and runs a two-item mission. The `generate_patch` activity is intentionally flaky for the first two attempts so you can see retries in the logs.

```python
#!/usr/bin/env python3
"""lra-demo/ch13_activities_retries_replay.py

Demonstrates durable activities, retries, and replay-from-cache for the
gather -> act -> verify -> checkpoint cycle.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker


CYCLE_TASK_QUEUE = "mission-cycle-ch13"
WORKFLOW_ID = "ch13-mission-demo"
CACHE_FILE = ".lra-activity-cache.json"
CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PermanentError(Exception):
    """Raised when an activity should NOT be retried."""
    pass


class TransientError(Exception):
    """Raised when an activity MAY be retried."""
    pass


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MissionInput:
    workdir: str
    checklist: list[str]


@dataclasses.dataclass
class MissionStatus:
    completed: list[str]
    failed: list[str]
    cycles: int


@dataclasses.dataclass
class GatherInput:
    workdir: str
    item: str


@dataclasses.dataclass
class GatherResult:
    observations: list[str]


@dataclasses.dataclass
class ActInput:
    workdir: str
    item: str
    observations: list[str]


@dataclasses.dataclass
class ActResult:
    patch: str


@dataclasses.dataclass
class VerifyInput:
    workdir: str
    item: str
    patch: str


@dataclasses.dataclass
class VerifyResult:
    passed: bool
    stdout: str


@dataclasses.dataclass
class CheckpointInput:
    workdir: str
    item: str
    patch: str
    verify_stdout: str


# ---------------------------------------------------------------------------
# Local idempotency cache
# ---------------------------------------------------------------------------

def _cache_path(workdir: str) -> Path:
    return Path(workdir) / CACHE_FILE


def _load_cache(workdir: str) -> dict[str, Any]:
    path = _cache_path(workdir)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _save_cache(workdir: str, key: str, value: Any) -> None:
    path = _cache_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_LOCK:
        cache = _load_cache(workdir)
        cache[key] = value
        with path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)


def _idempotency_key(name: str, payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, default=str)
    return f"{name}:{hashlib.sha256(data.encode()).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

@activity.defn
async def gather_context(input: GatherInput) -> GatherResult:
    """Read-only gather step. Cached so re-runs do not re-spend tokens."""
    key = _idempotency_key("gather", dataclasses.asdict(input))
    cached = _load_cache(input.workdir).get(key)
    if cached:
        activity.logger.info("replay-from-local-cache", key=key)
        return GatherResult(**cached)

    activity.logger.info(
        "gather-execute",
        item=input.item,
        attempt=activity.info().attempt,
    )
    # Simulate reading the blackboard / codebase.
    await asyncio.sleep(0.2)
    result = GatherResult(observations=[f"item '{input.item}' needs implementation"])
    _save_cache(input.workdir, key, dataclasses.asdict(result))
    return result


@activity.defn
async def generate_patch(input: ActInput) -> ActResult:
    """The Lead Engineer's write step. First two attempts fail to show retries."""
    activity.logger.info(
        "act-execute",
        item=input.item,
        attempt=activity.info().attempt,
    )

    # Deterministic "flakiness" for the demo.
    if activity.info().attempt < 3:
        raise TransientError(
            f"model API timeout for {input.item} (attempt {activity.info().attempt})"
        )

    # Simulate writing code.
    await asyncio.sleep(0.1)
    patch = f"# implementation for {input.item}\nprint('done: {input.item}')\n"
    return ActResult(patch=patch)


@activity.defn
async def run_verification(input: VerifyInput) -> VerifyResult:
    """Deterministic verifier from Chapter 06."""
    activity.logger.info("verify-execute", item=input.item)
    passed = input.item.lower() in input.patch.lower()
    stdout = f"verify {'PASS' if passed else 'FAIL'} for {input.item}"
    return VerifyResult(passed=passed, stdout=stdout)


@activity.defn
async def checkpoint_state(input: CheckpointInput) -> None:
    """Write real state to disk and journal the cycle."""
    activity.logger.info("checkpoint", item=input.item)
    workdir = Path(input.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "checkpoints.jsonl"
    entry = {
        "item": input.item,
        "patch": input.patch,
        "verify_stdout": input.verify_stdout,
        "ts": time.time(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow.defn
class MissionWorkflow:
    @workflow.run
    async def run(self, input: MissionInput) -> MissionStatus:
        completed: list[str] = []
        failed: list[str] = []
        cycles = 0

        retry_policy = RetryPolicy(
            maximum_attempts=5,
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            non_retryable_error_types=["PermanentError"],
        )

        for item in input.checklist:
            cycles += 1
            workflow.logger.info("cycle-start", cycle=cycles, item=item)

            gather = await workflow.execute_activity(
                gather_context,
                GatherInput(workdir=input.workdir, item=item),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=retry_policy,
            )

            act = await workflow.execute_activity(
                generate_patch,
                ActInput(
                    workdir=input.workdir,
                    item=item,
                    observations=gather.observations,
                ),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=retry_policy,
            )

            verify = await workflow.execute_activity(
                run_verification,
                VerifyInput(workdir=input.workdir, item=item, patch=act.patch),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=retry_policy,
            )

            if verify.passed:
                await workflow.execute_activity(
                    checkpoint_state,
                    CheckpointInput(
                        workdir=input.workdir,
                        item=item,
                        patch=act.patch,
                        verify_stdout=verify.stdout,
                    ),
                    start_to_close_timeout=timedelta(seconds=10),
                    retry_policy=retry_policy,
                )
                completed.append(item)
                workflow.logger.info("cycle-done", item=item)
            else:
                failed.append(item)
                workflow.logger.warning("cycle-verify-failed", item=item)

        return MissionStatus(completed=completed, failed=failed, cycles=cycles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_worker() -> None:
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=CYCLE_TASK_QUEUE,
        workflows=[MissionWorkflow],
        activities=[
            gather_context,
            generate_patch,
            run_verification,
            checkpoint_state,
        ],
    )
    await worker.run()


async def run_starter() -> MissionStatus:
    client = await Client.connect("localhost:7233")
    workdir = ".lra/workspaces/ch13"
    handle = await client.start_workflow(
        MissionWorkflow.run,
        MissionInput(workdir=workdir, checklist=["hello.py", "math_utils.py"]),
        id=WORKFLOW_ID,
        task_queue=CYCLE_TASK_QUEUE,
    )
    print(f"started workflow {handle.id}")
    return await handle.result()


async def main() -> None:
    worker_task = asyncio.create_task(run_worker())
    await asyncio.sleep(1)  # let the worker connect
    try:
        result = await run_starter()
        print("MISSION RESULT:", result)
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Code Walkthrough

1. **Data contracts** — Plain `dataclasses` are used because Temporal's default payload converter serializes them cleanly. The real `lra` package uses Pydantic models in `src/lra/contracts/` and converts them at the activity boundary.

2. **`TransientError` vs `PermanentError`** — `generate_patch` raises `TransientError` for the first two attempts. Temporal sees that type is not in `non_retryable_error_types`, so it retries. A `PermanentError` would stop the retry chain immediately and surface to the workflow.

3. **Retry policy** — `maximum_attempts=5`, `initial_interval=1s`, `backoff_coefficient=2.0`. The flaky activity fails at 1s and 2s, then succeeds on the third attempt. In production you would tune these values per activity; model APIs usually deserve shorter initial intervals and fewer attempts than git writes.

4. **Local idempotency cache** — `_idempotency_key` builds a stable hash from the activity name and input. `_save_cache` writes the result to `.lra-activity-cache.json`. If `gather_context` is re-executed because of a worker crash or a retry, it returns the cached value instead of calling the model again. This is the second line of defense: Temporal history replay is the first.

5. **Workflow determinism** — `MissionWorkflow` contains no I/O, no `time.time`, no `random`, and no `asyncio.sleep`. It only calls `workflow.execute_activity`. That is what makes replay safe.

6. **Checkpoint activity** — The only activity that mutates durable state writes a JSONL log. In the full system this is where the git commit from Chapter 03 and the checklist update from Chapter 04 happen.

---

## Hands-On Exercise

1. Start a local Temporal server:

   ```bash
   temporal server start-dev
   ```

2. Run the demo:

   ```bash
   uv run python lra-demo/ch13_activities_retries_replay.py
   ```

3. Watch the logs. You should see `act-execute` for `hello.py` log `attempt=1`, `attempt=2`, then succeed on `attempt=3`. The gather step should execute once per item because the local cache hits on the second cycle.

4. Simulate a crash: while the mission is running, press `Ctrl-C` after the first checkpoint is written. Then run the script again. The workflow resumes from the same `WORKFLOW_ID`; already-completed activities are replayed from the workflow history, and the gather cache prevents a second "LLM call".

5. Inspect the durable output:

   ```bash
   cat .lra/workspaces/ch13/checkpoints.jsonl
   ```

6. Optional: change `generate_patch` to raise `PermanentError` on the first attempt and observe that the workflow stops immediately instead of retrying.

---

> **Key Takeaway:** Retries make transient failures recoverable; replay-from-cache makes completed work immortal. Together they turn a fragile LLM loop into a durable mission engine that survives crashes without re-spending tokens or corrupting state.

---

## Next Chapter

In **Chapter 14: Durable Sleep and Continue-As-New**, we will make the mission loop truly long-running. You will learn how `workflow.sleep` pauses execution for hours or days at zero cost, and how `continue_as_new` resets the workflow history so a mission can run for weeks without hitting Temporal's event limit.