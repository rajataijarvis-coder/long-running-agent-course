# Chapter 15: Surviving Crashes — Kill and Resume

> A durable workflow is not the one that never fails; it is the one that wakes up exactly where it left off.
> — Fareed Khan

## What We'll Cover

- Why a process kill, worker restart, or host reboot must not lose mission progress
- How Temporal's event history turns a crash into a **replay**, not a restart
- The split of responsibility: the server remembers the schedule, git remembers the truth, the worker is disposable
- How **idempotency markers** stop a resumed activity from duplicating side effects
- A runnable demo in `lra-demo/ch15_crash_resume.py` that starts a mission, kills the worker mid-activity, starts a fresh worker, and watches the mission finish
- How to verify the resume in the Temporal Web UI and in the git log

---

## The Crash Is Normal

By Chapter 12 you had a `MissionWorkflow`. By Chapter 13 every side effect was an activity with retries and a local replay cache. By Chapter 14 the workflow could sleep durably and renew itself with `continue_as_new`. This chapter closes the loop: the system must survive the worker process itself disappearing.

In a chat-style agent, killing the process means losing the context window and the mission. In LRA, three things make that impossible:

1. **The workflow state lives on the Temporal server**, not in the worker. The server stores the complete event history of every step, sleep, and activity result.
2. **The real state lives in git** (Chapter 03). The worker can be rebuilt from an empty container as long as it can read the repo.
3. **Activities are idempotent**. If an activity was half-done when the worker died, the retry does not duplicate the side effect because a marker on disk says it already happened.

When a new worker starts and connects to the same task queue, Temporal sends it the pending workflow execution along with its event history. The worker replays the history deterministically to reconstruct the workflow's internal state, then continues from the next uncompleted activity. No tokens are re-spent for already-completed steps because the results are in the history.

This is why the workflow code must be deterministic: no `datetime.now()`, no `random`, no direct I/O, no global mutable state. Anything non-deterministic must be expressed through Temporal primitives (`workflow.now()`, `workflow.random()`, `workflow.sleep()`) or pushed into activities.

---

## Demo: A Mission That Survives a Worker Kill

The file `lra-demo/ch15_crash_resume.py` contains a self-contained mission:

- It starts a local Temporal dev server.
- It starts a worker.
- It starts a `MissionWorkflow` that writes `hello.py`, writes a test, runs the test, and checkpoints each step to git.
- After a few seconds it **cancels the worker task** (simulating a crash).
- It starts a **second worker** and waits for the same workflow execution to finish.

The workflow, activities, and harness are shown below.

```python
# lra-demo/ch15_crash_resume.py
"""Demonstrate a Temporal mission workflow surviving a worker crash.

Run with:
    uv run python lra-demo/ch15_crash_resume.py

For manual testing against a local Temporal server:
    temporal server start-dev --ui-port 8233
    uv run python lra-demo/ch15_crash_resume.py --mode worker
    # In another terminal, start the workflow or use the Web UI.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import timedelta
from pathlib import Path

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ch15")


# --------------------------------------------------------------------------- #
# State helpers: git + JSON files are the real source of truth
# --------------------------------------------------------------------------- #
def state_path(workdir: str) -> Path:
    return Path(workdir) / ".lra" / "state.json"


def marker_path(workdir: str, key: str) -> Path:
    return Path(workdir) / ".lra" / "markers" / f"{key}.json"


def ensure_dirs(workdir: str) -> None:
    (Path(workdir) / ".lra" / "markers").mkdir(parents=True, exist_ok=True)


def git_commit(workdir: str, message: str) -> None:
    """Best-effort git checkpoint. Failure does not block the mission."""
    try:
        subprocess.run(["git", "add", "."], cwd=workdir, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message, "--quiet"],
            cwd=workdir,
            check=True,
            capture_output=True,
        )
    except Exception as exc:
        logger.warning("git commit skipped: %s", exc)


# --------------------------------------------------------------------------- #
# Activities: every side effect lives here, never in the workflow
# --------------------------------------------------------------------------- #
@activity.defn
async def load_state(mission_id: str, workdir: str) -> dict:
    """Return the current mission state, creating it if necessary."""
    ensure_dirs(workdir)
    path = state_path(workdir)
    if not path.exists():
        initial = {
            "mission_id": mission_id,
            "items": [
                {
                    "id": "1",
                    "title": "create hello.py",
                    "target": "hello.py",
                    "content": (
                        "def greet():\n"
                        "    return 'hello'\n\n"
                        "if __name__ == '__main__':\n"
                        "    print(greet())\n"
                    ),
                    "done": False,
                    "verify_cmd": ["python", "hello.py"],
                    "verify_contains": "hello",
                },
                {
                    "id": "2",
                    "title": "create test_hello.py",
                    "target": "test_hello.py",
                    "content": "from hello import greet\nassert greet() == 'hello'\n",
                    "done": False,
                    "verify_cmd": ["python", "test_hello.py"],
                    "verify_contains": None,
                },
            ],
        }
        path.write_text(json.dumps(initial, indent=2))
        git_commit(workdir, "init mission state")
        return initial
    return json.loads(path.read_text())


@activity.defn
async def execute_step(
    mission_id: str,
    workdir: str,
    item: dict,
    idempotency_token: str,
) -> dict:
    """Write the file for one checklist item. Idempotent via marker file."""
    ensure_dirs(workdir)
    key = f"{mission_id}-exec-{item['id']}-{idempotency_token}"
    marker = marker_path(workdir, key)

    if marker.exists():
        logger.info("execute_step replay/cache hit %s", key)
        return json.loads(marker.read_text())

    logger.info(
        "execute_step start %s (attempt %s)",
        key,
        activity.info().attempt,
    )
    # Simulate real work that can be interrupted mid-flight.
    await asyncio.sleep(2)

    target = Path(workdir) / item["target"]
    target.write_text(item["content"])

    result = {"written": str(target), "size": target.stat().st_size}
    marker.write_text(json.dumps(result))
    logger.info("execute_step done %s", key)
    return result


@activity.defn
async def verify_step(
    mission_id: str,
    workdir: str,
    item: dict,
    idempotency_token: str,
) -> bool:
    """Run deterministic verification for one item."""
    ensure_dirs(workdir)
    key = f"{mission_id}-verify-{item['id']}-{idempotency_token}"
    marker = marker_path(workdir, key)

    if marker.exists():
        logger.info("verify_step replay/cache hit %s", key)
        return json.loads(marker.read_text())["ok"]

    logger.info("verify_step start %s", key)
    proc = await asyncio.create_subprocess_exec(
        *item["verify_cmd"],
        cwd=workdir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    ok = proc.returncode == 0
    if item.get("verify_contains"):
        ok = ok and (item["verify_contains"] in stdout.decode())

    result = {
        "ok": ok,
        "rc": proc.returncode,
        "stdout": stdout.decode(errors="replace"),
    }
    marker.write_text(json.dumps(result))
    logger.info("verify_step done %s ok=%s", key, ok)
    return ok


@activity.defn
async def checkpoint_state(
    mission_id: str,
    workdir: str,
    item: dict,
    verified_ok: bool,
) -> dict:
    """Mark the item done and commit. Idempotent via marker + git state."""
    ensure_dirs(workdir)
    key = f"{mission_id}-checkpoint-{item['id']}"
    marker = marker_path(workdir, key)

    if marker.exists():
        logger.info("checkpoint replay/cache hit %s", key)
        return json.loads(marker.read_text())

    state = json.loads(state_path(workdir).read_text())
    for it in state["items"]:
        if it["id"] == item["id"]:
            it["done"] = verified_ok
            it["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state_path(workdir).write_text(json.dumps(state, indent=2))
    git_commit(
        workdir,
        f"checkpoint item {item['id']}: {item['title']} ok={verified_ok}",
    )

    result = {"item_id": item["id"], "ok": verified_ok}
    marker.write_text(json.dumps(result))
    logger.info("checkpoint done %s", key)
    return result


# --------------------------------------------------------------------------- #
# Workflow: deterministic scheduler; no side effects, no direct I/O
# --------------------------------------------------------------------------- #
@workflow.defn(name="MissionWorkflow")
class MissionWorkflow:
    @workflow.run
    async def run(self, mission_id: str, workdir: str) -> str:
        wf_logger = workflow.logger
        wf_logger.info("mission started mission_id=%s", mission_id)

        while True:
            # Re-read the real state every cycle (assume interruption).
            state = await workflow.execute_activity(
                load_state,
                args=(mission_id, workdir),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=5),
            )

            open_items = [it for it in state["items"] if not it["done"]]
            if not open_items:
                wf_logger.info("mission complete")
                break

            item = open_items[0]
            # A deterministic token that is stable across replay.
            token = workflow.now().isoformat()
            wf_logger.info("cycle item=%s token=%s", item["id"], token)

            await workflow.execute_activity(
                execute_step,
                args=(mission_id, workdir, item, token),
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=5),
                    maximum_attempts=10,
                ),
            )

            ok = await workflow.execute_activity(
                verify_step,
                args=(mission_id, workdir, item, token),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=5),
            )

            await workflow.execute_activity(
                checkpoint_state,
                args=(mission_id, workdir, item, ok),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=5),
            )

            if not ok:
                # The verifier is ground truth; back off and retry.
                await workflow.sleep(timedelta(seconds=2))
                continue

        return f"mission={mission_id} complete"


# --------------------------------------------------------------------------- #
# Demo harness: start worker, kill it, start another, watch resume
# --------------------------------------------------------------------------- #
async def run_auto_demo():
    logger.info("starting local Temporal dev server for crash/resume demo")
    async with await WorkflowEnvironment.start_local() as env:
        client = env.client
        mission_id = f"crash-demo-{uuid.uuid4().hex[:6]}"
        workdir = tempfile.mkdtemp(prefix="lra-ch15-")

        subprocess.run(["git", "init", "-q"], cwd=workdir, check=True)
        subprocess.run(["git", "config", "user.email", "lra@demo"], cwd=workdir, check=True)
        subprocess.run(["git", "config", "user.name", "LRA Demo"], cwd=workdir, check=True)

        logger.info("workdir=%s mission_id=%s", workdir, mission_id)

        # Worker 1: start and then kill it mid-mission.
        worker1 = Worker(
            client,
            task_queue="mission-queue",
            workflows=[MissionWorkflow],
            activities=[load_state, execute_step, verify_step, checkpoint_state],
        )
        worker1_task = asyncio.create_task(worker1.run())
        logger.info("worker 1 started")

        handle = await client.start_workflow(
            MissionWorkflow.run,
            mission_id,
            workdir,
            id=mission_id,
            task_queue="mission-queue",
        )
        logger.info("workflow started id=%s", mission_id)

        # Let the first activity start, then kill the worker.
        await asyncio.sleep(3)
        logger.info("KILLING worker 1 now (simulating crash/reboot)")
        worker1_task.cancel()
        try:
            await worker1_task
        except asyncio.CancelledError:
            pass

        # Worker 2: a fresh process picks up the same execution.
        worker2 = Worker(
            client,
            task_queue="mission-queue",
            workflows=[MissionWorkflow],
            activities=[load_state, execute_step, verify_step, checkpoint_state],
        )
        worker2_task = asyncio.create_task(worker2.run())
        logger.info("worker 2 started (resuming from Temporal server state)")

        try:
            result = await asyncio.wait_for(handle.result(), timeout=120)
            logger.info("workflow result: %s", result)
        except asyncio.TimeoutError:
            logger.error("workflow did not complete in time")
            raise
        finally:
            worker2_task.cancel()
            try:
                await worker2_task
            except asyncio.CancelledError:
                pass

        # Show that the real state survived on disk.
        log = subprocess.run(
            ["git", "-C", workdir, "log", "--oneline"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("git history:\n%s", log.stdout)
        logger.info("inspect workdir: %s", workdir)


def main():
    parser = argparse.ArgumentParser(description="Chapter 15 crash/resume demo")
    parser.add_argument(
        "--mode",
        choices=["auto", "worker"],
        default="auto",
        help="auto: full crash/resume simulation; worker: run a worker for manual testing",
    )
    args = parser.parse_args()

    if args.mode == "auto":
        asyncio.run(run_auto_demo())
    else:
        async def run_manual():
            client = await Client.connect("localhost:7233")
            worker = Worker(
                client,
                task_queue="mission-queue",
                workflows=[MissionWorkflow],
                activities=[load_state, execute_step, verify_step, checkpoint_state],
            )
            logger.info("manual worker running on localhost:7233; Ctrl-C to stop")
            await worker.run()

        asyncio.run(run_manual())


if __name__ == "__main__":
    main()
```

---

## Code Walkthrough

### Real state is on disk, not in memory

`load_state` reads `workdir/.lra/state.json` every cycle. If the file does not exist, it seeds the mission checklist and commits it to git. This is the "assume interruption" rule from Chapter 04: the workflow never trusts its own memory; it re-reads the ground truth each loop.

### Activities are idempotent

`execute_step`, `verify_step`, and `checkpoint_state` each compute a deterministic key and check a marker file before doing work:

```python
key = f"{mission_id}-exec-{item['id']}-{idempotency_token}"
marker = marker_path(workdir, key)

if marker.exists():
    return json.loads(marker.read_text())
```

The `idempotency_token` comes from `workflow.now().isoformat()`. Because the workflow is deterministic, the same token is produced during replay, so a retried activity finds the same marker and returns the cached result. This prevents duplicate file writes, duplicate test runs, and duplicate git commits after a crash.

### The workflow is pure scheduling

`MissionWorkflow` contains no `datetime.now()`, no `random`, no file I/O, and no `asyncio.sleep()`. It uses:

- `workflow.execute_activity` for side effects
- `workflow.sleep` for backoff
- `workflow.now()` for stable idempotency tokens
- `workflow.logger` for structured logging

This determinism is what allows a brand-new worker to replay the event history and land in exactly the same state as the dead worker.

### The crash is simulated by cancelling the worker task

The harness starts `worker1`, starts the workflow, waits three seconds, then cancels the worker task. The first `execute_step` is interrupted. Temporal server marks the activity as incomplete and schedules a retry. When `worker2` starts, it receives the same workflow execution, replays the history, sees that `execute_step` did not complete, and runs it again. The marker file ensures the retry does not write `hello.py` twice.

---

## Hands-On Exercise

1. **Run the automatic crash demo:**

   ```bash
   uv run python lra-demo/ch15_crash_resume.py
   ```

   Watch the logs. You should see `worker 1 started`, then `KILLING worker 1 now`, then `worker 2 started`, and finally `workflow result: mission=... complete`.

2. **Inspect the workdir.** The log prints a path like `/tmp/lra-ch15-...`. Run:

   ```bash
   git -C /tmp/lra-ch15-... log --oneline
   cat /tmp/lra-ch15-.../hello.py
   cat /tmp/lra-ch15-.../test_hello.py
   ```

   You should see commits for `init mission state`, `checkpoint item 1`, and `checkpoint item 2`.

3. **Inspect the marker cache:**

   ```bash
   ls /tmp/lra-ch15-.../.lra/markers
   ```

   Each activity that ran has a JSON marker. If the same activity was retried after the crash, the marker file is why the side effect did not repeat.

4. **Manual kill test with two terminals.** Start a local Temporal server:

   ```bash
   temporal server start-dev --ui-port 8233
   ```

   In terminal A, run the worker:

   ```bash
   uv run python lra-demo/ch15_crash_resume.py --mode worker
   ```

   In terminal B, start a workflow with a small Python starter or the Temporal CLI:

   ```bash
   temporal workflow start \
     --type MissionWorkflow \
     --task-queue mission-queue \
     --input '{"mission_id": "manual-crash", "workdir": "/tmp/lra-manual-crash"}'
   ```

   Wait until terminal A logs that it is executing a step, then press `Ctrl-C` or run `kill -9` on the worker process. Start the worker again in terminal A. The workflow should finish without restarting from scratch.

5. **Open the Temporal Web UI** at `http://localhost:8233`, find your workflow, and click through the event history. Look for the activity that was started by worker 1, did not complete, and was retried by worker 2.

---

> **Key Takeaway:** The worker is cattle, not a pet. The mission survives because the schedule lives on the Temporal server, the truth lives in git, and every activity can be replayed without duplicating side effects. Build your system so that `kill -9` is a recoverable event, not a catastrophe.

---

## Next Chapter Teaser

**Chapter 16: The Mission Anchor** — A durable workflow needs a single, stable identity that outlives every worker crash and every `continue_as_new` renewal. We will build the **Mission Anchor**: a small, versioned metadata file that binds a workflow ID to a git repository, a task queue, a budget ceiling, and a human approval policy, so the mission can always find its way home.