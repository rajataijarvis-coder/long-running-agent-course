# Chapter 14: Durable Sleep and Continue-As-New

> A week-long mission spends most of its time waiting. Make that wait durable, and make the workflow itself renewable before it grows too large to replay.
> — Fareed Khan

## What We'll Cover

- Why `asyncio.sleep` inside a Temporal workflow is a bug, and why `workflow.sleep` is the only safe way to pause
- How **durable sleep** moves the timer out of the worker process and onto the Temporal server, giving you zero-cost idle time
- Why a long-running workflow will eventually hit the **event history limit**, and how **Continue-As-New** resets it without losing the mission
- How to pass the smallest possible state across a Continue-As-New boundary because the real state lives in git
- A runnable demo in `lra-demo/ch14_durable_sleep_continue_as_new.py` that sleeps durably and renews itself automatically
- How to observe both behaviors in the Temporal Web UI

---

## The Problem This Solves

By Chapter 12 you had a `MissionWorkflow` running inside Temporal. By Chapter 13 you split every side effect into retried, replayable activities. That already gets you crash survival, but two long-horizon problems remain:

1. **Waiting is not free if the wait lives in your process.**  
   A mission that polls every hour for a week cannot hold a Python process open for a week doing `time.sleep(3600)`. The process costs money, dies on deploy, and loses the loop on restart.

2. **A workflow has a finite event history.**  
   Temporal keeps every event the workflow produced. A mission with thousands of cycles will eventually exceed the history limit and fail. You cannot keep appending cycles forever in the same run.

Temporal solves both with two primitives:

- **`workflow.sleep(delay)`** — schedules a durable timer. The worker process can exit, scale to zero, or reboot. When the timer fires, any worker can resume the workflow exactly where it left off.
- **`workflow.continue_as_new(*args)`** — ends the current run and starts a fresh run of the same workflow with the same workflow ID and a new `run_id`. The event history is reset, but the mission continues.

Because LRA keeps the real state in git (Chapter 03), the only thing that needs to cross the Continue-As-New boundary is the mission identity: `mission_id`, `workdir`, and maybe a generation counter. The new run reconstructs situational awareness by reading the checklist and decision log from disk.

---

## The Demo: `lra-demo/ch14_durable_sleep_continue_as_new.py`

This file is a self-contained Temporal workflow. It runs the gather → act → verify → checkpoint cycle from Chapter 04, sleeps durably between cycles, and renews itself with Continue-As-New after a configurable number of cycles.

```python
# lra-demo/ch14_durable_sleep_continue_as_new.py
import asyncio
import dataclasses
from dataclasses import dataclass
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker

TASK_QUEUE = "lra-ch14"
WORKFLOW_ID = "mission-ch14-weeklong"


@dataclass
class MissionInput:
    mission_id: str
    task: str
    workdir: str
    run_generation: int = 0
    # Renew the workflow run after this many cycles to keep history small.
    cycles_per_run: int = 5
    # Durable sleep between cycles.
    poll_interval_seconds: int = 30


@dataclass
class CycleResult:
    cycle: int
    generation: int
    observation: str
    commit: str
    done: bool


# --------------------------------------------------------------------------- #
# Activities: the only place side effects and non-determinism are allowed.
# --------------------------------------------------------------------------- #

@activity.defn
async def gather_context(task: str, workdir: str) -> str:
    """Read the current world state: git log, checklist, blackboard."""
    # In the real system this shells out to git and reads JSON state files.
    await asyncio.sleep(0.1)
    return f"gathered for '{task}' in {workdir}"


@activity.defn
async def act_and_verify(task: str, observation: str, cycle: int) -> tuple[str, bool]:
    """Lead engineer writes code; deterministic verifier decides if we are done."""
    await asyncio.sleep(0.1)
    commit = f"commit-{cycle}-{abs(hash(observation)) % 10000:04d}"
    # Deterministic demo completion: real LRA uses test exit codes (Chapter 06).
    done = cycle >= 4
    return commit, done


@activity.defn
async def checkpoint_to_git(workdir: str, result: CycleResult) -> str:
    """Persist the decision log and any code changes to git."""
    await asyncio.sleep(0.1)
    return f"{workdir}@gen{result.generation}-cycle{result.cycle}"


# --------------------------------------------------------------------------- #
# Workflow: scheduling only. No I/O, no sleep, no randomness here.
# --------------------------------------------------------------------------- #

@workflow.defn
class MissionWorkflow:
    @workflow.run
    async def run(self, inp: MissionInput) -> str:
        cycle = 0

        while True:
            # Renew before the event history grows unbounded.
            if cycle >= inp.cycles_per_run:
                next_inp = dataclasses.replace(
                    inp, run_generation=inp.run_generation + 1
                )
                # This ends the current run and starts a fresh one with the
                # same workflow ID. Code after this line never executes.
                await workflow.continue_as_new(next_inp)

            observation = await workflow.execute_activity(
                gather_context,
                args=(inp.task, inp.workdir),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            commit, done = await workflow.execute_activity(
                act_and_verify,
                args=(inp.task, observation, cycle),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=5),
            )

            result = CycleResult(
                cycle=cycle,
                generation=inp.run_generation,
                observation=observation,
                commit=commit,
                done=done,
            )

            await workflow.execute_activity(
                checkpoint_to_git,
                args=(inp.workdir, result),
                start_to_close_timeout=timedelta(seconds=10),
            )

            workflow.logger.info(
                "cycle complete",
                cycle=cycle,
                generation=inp.run_generation,
                commit=commit,
                done=done,
            )

            if done:
                return f"mission done at generation {inp.run_generation}, cycle {cycle}"

            cycle += 1

            # Durable sleep: the timer is journaled by Temporal, not this process.
            await workflow.sleep(inp.poll_interval_seconds)


# --------------------------------------------------------------------------- #
# Worker and starter
# --------------------------------------------------------------------------- #

async def run_worker() -> None:
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MissionWorkflow],
        activities=[gather_context, act_and_verify, checkpoint_to_git],
    )
    await worker.run()


async def start_workflow() -> None:
    client = await Client.connect("localhost:7233")
    handle = await client.start_workflow(
        MissionWorkflow.run,
        MissionInput(
            mission_id=WORKFLOW_ID,
            task="Create a hello.py that prints hello and a test for it",
            workdir=".lra/workspaces/ch14",
            cycles_per_run=2,          # force Continue-As-New quickly for the demo
            poll_interval_seconds=10,  # short sleep so you can see it resume
        ),
        id=WORKFLOW_ID,
        task_queue=TASK_QUEUE,
    )
    print(f"started workflow {handle.id} run_id={handle.result_run_id}")
    result = await handle.result()
    print(f"result: {result}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python ch14_durable_sleep_continue_as_new.py [worker|start]")

    if sys.argv[1] == "worker":
        asyncio.run(run_worker())
    elif sys.argv[1] == "start":
        asyncio.run(start_workflow())
    else:
        raise SystemExit("unknown command")
```

### Code Walkthrough

**`MissionInput`** carries only identity and tuning. `run_generation` increments each time the workflow Continue-As-News. `cycles_per_run` is the manual history-boundary knob. In production you can also react to `workflow.info().continue_as_new_suggested` if the SDK reports that the server is nearing its history limit.

**`gather_context`, `act_and_verify`, `checkpoint_to_git`** are activities. They are the only places allowed to sleep with `asyncio.sleep`, touch the filesystem, call APIs, or use non-deterministic logic. Their outputs are journaled, so replay does not re-execute them.

**The workflow loop** is pure scheduling. It calls activities, checks completion, and then sleeps with `workflow.sleep`. This sleep is deterministic: Temporal records the timer event in the workflow history. On replay, the SDK fast-forwards past the sleep instead of waiting again.

**Continue-As-New** happens when `cycle >= cycles_per_run`. We build `next_inp` with an incremented `run_generation` and pass it to `workflow.continue_as_new`. The current run ends cleanly; a new run starts with the same workflow ID and the new input. Because the real state is in git, we do not need to pass the checklist, blackboard, or memory — the next run can re-read them.

> **Determinism rule:** Never use `time.sleep`, `asyncio.sleep`, `random`, or any I/O inside a workflow method. Only `workflow.sleep`, `workflow.execute_activity`, `workflow.execute_child_workflow`, and deterministic computation are allowed.

---

## Hands-On Exercise

You need a Temporal server running. The fastest way is the dev server:

```bash
temporal server start-dev --ui-port 8080
```

### Part A — Durable Sleep Survives a Worker Kill

1. Start the worker in one terminal:

```bash
cd lra-demo
python ch14_durable_sleep_continue_as_new.py worker
```

2. Start the mission in a second terminal:

```bash
python ch14_durable_sleep_continue_as_new.py start
```

3. Watch the logs. After the first cycle completes, the workflow will print that it is sleeping for 10 seconds.

4. While it is sleeping, press `Ctrl-C` in the **worker** terminal to kill the process. Wait 15–20 seconds.

5. Restart the worker with the same command.

6. Observe that the workflow resumes at the pending timer. It does **not** re-run the cycle it already completed. The event history preserved the activity results; only the durable timer needed to fire.

### Part B — Continue-As-New Resets History

1. With `cycles_per_run=2` in the starter, the workflow will call `continue_as_new` after the second cycle.

2. Open the Temporal Web UI at `http://localhost:8080`, find workflow ID `mission-ch14-weeklong`.

3. Notice that the workflow ID stays the same, but the `run_id` changes. The new run has a tiny event history. The previous run is still visible under the “Previous Runs” tab.

4. Because the demo completes at `cycle >= 4`, and each run only does two cycles, the workflow will Continue-As-New repeatedly. Stop it with:

```bash
temporal workflow terminate --workflow-id mission-ch14-weeklong
```

### Part C — Complete in One Run

1. Edit the starter to set `cycles_per_run=10`.

2. Restart the worker, start a new workflow with a different ID, and watch it complete in a single run:

```bash
python ch14_durable_sleep_continue_as_new.py worker
# in another terminal
python - <<'PY'
import asyncio
from ch14_durable_sleep_continue_as_new import start_workflow
asyncio.run(start_workflow())
PY
```

3. Verify in the UI that the workflow returns `mission done at generation 0, cycle 4`.

---

## Key Takeaway

> Durable sleep turns idle time into zero-cost, crash-proof waiting. Continue-As-New turns a finite workflow run into an infinite mission. Together they are why the system can run for a week without holding a process open for a week.

---

## Next Chapter

**Chapter 15: Surviving Crashes — Kill and Resume**  
Now that sleeping and renewal are durable, we will deliberately kill workers, reboot the host, and even terminate the Temporal worker process mid-activity. You will see the mission resume exactly where it stopped, with no tokens re-spent and no work duplicated.