# Chapter 12: Temporal Workflows for Long-Running Agents

> A week-long mission cannot live inside a `while` loop. It needs an execution layer that treats interruption as the normal case.
> — Fareed Khan

## What We'll Cover

- Why a local Python loop dies on crashes, API failures, and reboots — and why that breaks the cycle from Chapter 04
- How Temporal provides a **durable control plane** that journals every step and resumes after failure
- The deterministic rules that separate **workflow code** (scheduling) from **activity code** (side effects)
- Mapping the **gather → act → verify → checkpoint** cycle into Temporal activities
- Wiring a `MissionWorkflow`, a Worker, and a starter with `temporalio`
- A runnable demo in `lra-demo/ch12_temporal_workflows.py` that survives a worker kill and resumes exactly where it left off

---

## From a Local Loop to a Durable Control Plane

In Chapters 04, 10, and 11 we built the agent's inner loop:

1. **Gather** context (blackboard, checklist, ownership map)
2. **Act** via the Lead Engineer or tool dispatcher
3. **Verify** with deterministic exit codes
4. **Checkpoint** to git

That loop works as long as the process stays alive. But a week-long mission will not. The host will reboot, the Python process will OOM, an API call will hang, or a context window will force a restart. When that happens, a local `while` loop loses everything that was not explicitly written to disk.

Temporal solves this by making the loop itself **durable**. A Temporal workflow is event-sourced: every activity result is written to a history. If the worker crashes, a new worker replays the history and continues from the last completed activity. Idle time is spent in **durable sleep**, which costs nothing.

In LRA, the `durable/` package is intentionally thin. It is the **scheduler**, not the brain. The brain stays in `agent/` and `agents/`. The workflow's job is to call activities in the right order and to enforce the cycle:

- `gather_activity` reads the blackboard and returns the next item
- `act_activity` runs the Lead Engineer / tool dispatcher and writes files
- `verify_activity` runs tests, lint, or build and returns an exit code
- `checkpoint_activity` commits to git if verification passed
- `review_activity` and `reflect_activity` are scheduled when the reviewer blocks progress or verification fails

This separation is the key design rule: **workflows are deterministic schedulers; activities are the only place side effects happen.**

### Determinism rules for workflows

Temporal replays workflow code against the history. If the workflow does something non-deterministic, the replay diverges and the workflow fails. Keep these rules:

| Don't do this in a workflow | Do this instead |
|---|---|
| `datetime.now()` | `workflow.now()` |
| `random.random()` | `workflow.random()` |
| `uuid.uuid4()` | `workflow.uuid4()` |
| File I/O, network calls, LLM calls | Put them in an activity |
| Mutate global state | Pass state through activity arguments and returns |

The activities themselves are normal Python functions. They can call LLMs, run `git`, execute tools in sandboxes, and talk to databases. Their results are journaled, so the workflow can replay without re-executing them.

---

## The Demo: `lra-demo/ch12_temporal_workflows.py`

The demo implements a minimal `MissionWorkflow` that writes `hello.py`, writes a test, runs the test, and commits. It uses `unittest` as the deterministic verifier so it runs without `pytest`.

```python
# lra-demo/ch12_temporal_workflows.py
import argparse
import asyncio
import dataclasses
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker


@dataclasses.dataclass
class MissionInput:
    task: str
    workdir: str
    max_cycles: int = 10


@dataclasses.dataclass
class CycleResult:
    cycle: int
    action: str
    verified: bool
    commit: str | None = None
    stdout: str = ""
    stderr: str = ""


# ------------------------------------------------------------------
# Activities: all side effects, I/O, and non-determinism live here.
# ------------------------------------------------------------------
@activity.defn
async def gather_activity(input: MissionInput, cycle: int) -> dict:
    """Read the tiny plan and decide what to do next."""
    activity.logger.info("gather", extra={"cycle": cycle})
    plan_path = Path(input.workdir) / ".lra" / "plan.txt"
    if not plan_path.exists():
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            "1. create hello.py\n"
            "2. create test_hello.py\n"
        )
    items = [
        line.strip()
        for line in plan_path.read_text().splitlines()
        if line.strip()
    ]
    idx = min(cycle - 1, len(items) - 1)
    return {"next": items[idx], "remaining": max(0, len(items) - cycle)}


@activity.defn
async def act_activity(input: MissionInput, cycle: int, next_item: str) -> str:
    """Write the file for this cycle. Idempotent: only writes if missing."""
    activity.logger.info("act", extra={"cycle": cycle, "item": next_item})
    workdir = Path(input.workdir)

    if "hello.py" in next_item and not (workdir / "hello.py").exists():
        (workdir / "hello.py").write_text(
            'def hello() -> str:\n'
            '    return "world"\n\n'
            'if __name__ == "__main__":\n'
            '    print(hello())\n'
        )
        return "wrote hello.py"

    if "test_hello.py" in next_item and not (workdir / "test_hello.py").exists():
        (workdir / "test_hello.py").write_text(
            'from hello import hello\n\n'
            'def test_hello() -> None:\n'
            '    assert hello() == "world"\n'
        )
        return "wrote test_hello.py"

    return "no-op"


@activity.defn
async def verify_activity(input: MissionInput, cycle: int) -> tuple[int, str, str]:
    """Deterministic ground truth: unittest exit code."""
    activity.logger.info("verify", extra={"cycle": cycle})
    workdir = Path(input.workdir)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(workdir),
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


@activity.defn
async def checkpoint_activity(input: MissionInput, cycle: int, verified: bool) -> str:
    """Commit to git only when verified. Idempotent via per-cycle tag."""
    activity.logger.info("checkpoint", extra={"cycle": cycle, "verified": verified})
    if not verified:
        return "no-checkpoint"

    workdir = Path(input.workdir)
    tag = f"lra-cycle-{cycle}"

    subprocess.run(["git", "init", "-q"], cwd=workdir, check=False)
    subprocess.run(["git", "-C", str(workdir), "add", "."], check=False)

    # If the tag already exists, this activity is a replay; return the old commit.
    existing = subprocess.run(
        ["git", "-C", str(workdir), "rev-parse", tag],
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        return existing.stdout.strip()

    subprocess.run(
        ["git", "-C", str(workdir), "commit", "-m", f"cycle {cycle} verified", "-q"],
        check=False,
    )
    subprocess.run(["git", "-C", str(workdir), "tag", tag], check=False)

    head = subprocess.run(
        ["git", "-C", str(workdir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return head.stdout.strip()


# ------------------------------------------------------------------
# Workflow: deterministic scheduler. No side effects here.
# ------------------------------------------------------------------
@workflow.defn(name="mission-workflow")
class MissionWorkflow:
    @workflow.run
    async def run(self, input: MissionInput) -> list[CycleResult]:
        results: list[CycleResult] = []

        for cycle in range(1, input.max_cycles + 1):
            workflow.logger.info("cycle start", extra={"cycle": cycle})

            # GATHER
            plan = await workflow.execute_activity(
                gather_activity,
                args=(input, cycle),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # ACT
            action = await workflow.execute_activity(
                act_activity,
                args=(input, cycle, plan["next"]),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # VERIFY
            exit_code, stdout, stderr = await workflow.execute_activity(
                verify_activity,
                args=(input, cycle),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            verified = exit_code == 0

            # CHECKPOINT
            commit = await workflow.execute_activity(
                checkpoint_activity,
                args=(input, cycle, verified),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            results.append(
                CycleResult(
                    cycle=cycle,
                    action=action,
                    verified=verified,
                    commit=commit,
                    stdout=stdout,
                    stderr=stderr,
                )
            )

            if verified and plan["remaining"] == 0:
                workflow.logger.info("mission complete")
                break

            if not verified:
                workflow.logger.warning("verification failed; retry next cycle")

        return results


# ------------------------------------------------------------------
# Runners: in-memory test, or real worker/starter against a server.
# ------------------------------------------------------------------
async def run_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input = MissionInput(
            task="Create hello.py and a test",
            workdir=tmpdir,
            max_cycles=5,
        )
        async with await WorkflowEnvironment.start_time_skipping() as env:
            worker = Worker(
                env.client,
                task_queue="mission-queue",
                workflows=[MissionWorkflow],
                activities=[
                    gather_activity,
                    act_activity,
                    verify_activity,
                    checkpoint_activity,
                ],
            )
            async with worker:
                result = await env.client.execute_workflow(
                    MissionWorkflow.run,
                    input,
                    id="demo-mission-1",
                    task_queue="mission-queue",
                )
                print("Mission result:")
                for r in result:
                    print(
                        f"  cycle={r.cycle} "
                        f"action={r.action!r} "
                        f"verified={r.verified} "
                        f"commit={r.commit}"
                    )


async def run_worker(server_url: str) -> None:
    client = await Client.connect(server_url)
    worker = Worker(
        client,
        task_queue="mission-queue",
        workflows=[MissionWorkflow],
        activities=[
            gather_activity,
            act_activity,
            verify_activity,
            checkpoint_activity,
        ],
    )
    print(f"Worker connected to {server_url}. Press Ctrl+C to stop.")
    await worker.run()


async def start_mission(server_url: str) -> None:
    client = await Client.connect(server_url)
    with tempfile.TemporaryDirectory() as tmpdir:
        input = MissionInput(
            task="Create hello.py and a test",
            workdir=tmpdir,
            max_cycles=5,
        )
        result = await client.execute_workflow(
            MissionWorkflow.run,
            input,
            id="demo-mission-2",
            task_queue="mission-queue",
        )
        print("Mission result:")
        for r in result:
            print(
                f"  cycle={r.cycle} "
                f"action={r.action!r} "
                f"verified={r.verified} "
                f"commit={r.commit}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 12 Temporal demo")
    parser.add_argument(
        "--server",
        help="Temporal server host:port (e.g. localhost:7233). "
             "If omitted, runs an in-memory test.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start a mission instead of running a worker.",
    )
    args = parser.parse_args()

    if args.server and args.start:
        asyncio.run(start_mission(args.server))
    elif args.server:
        asyncio.run(run_worker(args.server))
    else:
        asyncio.run(run_test())


if __name__ == "__main__":
    main()
```

### Code walkthrough

- **`MissionInput` / `CycleResult`** are plain dataclasses. Temporal serializes them to JSON, so they must contain only primitive-ish values — no `Path` objects, no open files.
- **Activities** are the only functions that touch the filesystem, run subprocesses, or log with `activity.logger`. Each activity is idempotent so retries are safe:
  - `act_activity` only writes a file if it is missing.
  - `checkpoint_activity` checks the per-cycle git tag before committing again.
- **`MissionWorkflow.run`** is the scheduler. It loops, calls activities, and decides whether to continue or stop. It never touches disk. It uses `workflow.execute_activity` with explicit timeouts and retry policies.
- **`run_test`** uses Temporal's in-memory time-skipping environment, so the demo runs without any server. This is the fastest way to iterate.
- **`run_worker` / `start_mission`** connect to a real Temporal server. This is what you use for the crash-resume exercise below.

### How this connects to earlier chapters

- Chapter 04's **gather → act → verify → checkpoint** cycle is now explicit in the workflow code.
- Chapter 05's **tool dispatcher and sandbox** run inside `act_activity` and `verify_activity`.
- Chapter 10's **Lead Engineer** is the activity that writes coupled code.
- Chapter 11's **Reviewer** and **Reflection** agents are additional activities scheduled when verification fails or the reviewer blocks.

The workflow does not replace those agents. It orchestrates them.

---

## Hands-On Exercise: Survive a Worker Kill

This exercise uses a real Temporal server so you can kill the worker mid-mission and watch the mission resume.

### 1. Install and start Temporal

```bash
# Install the Temporal CLI (macOS example; see https://docs.temporal.io/cli for others)
brew install temporal

# Start a local dev server with the Web UI
temporal server start-dev --ui-port 8080
```

### 2. Run the worker

In terminal A:

```bash
uv sync
python lra-demo/ch12_temporal_workflows.py --server localhost:7233
```

### 3. Start a mission

In terminal B:

```bash
python lra-demo/ch12_temporal_workflows.py --server localhost:7233 --start
```

Watch the logs. After you see `act cycle=1` or `verify cycle=1` complete, **kill terminal A with Ctrl+C**.

### 4. Restart the worker

In terminal A, run the same worker command again:

```bash
python lra-demo/ch12_temporal_workflows.py --server localhost:7233
```

Observe that:

- The workflow does **not** restart from cycle 1.
- Already-completed activities are replayed from history; their code is **not** re-executed.
- The workflow continues from the next pending activity.
- The final result still shows two verified cycles and two git commits.

Open http://localhost:8080 and search for workflow ID `demo-mission-2` to inspect the event history.

### 5. Optional: force a retry

Temporarily add a transient failure inside `act_activity`:

```python
if cycle == 1 and activity.info().attempt == 1:
    raise RuntimeError("transient failure")
```

Re-run the mission. You will see Temporal retry the activity according to the `RetryPolicy(maximum_attempts=3)` declared in the workflow. Remove the failure after observing the retry.

---

## Key Takeaway

> A long-running agent is not a long conversation with a model. It is a durable workflow that schedules short, deterministic, verifiable bursts of work — and remembers every burst even when the process dies.

---

## Next Chapter

**Chapter 13: Activities, Retries, and Replay-from-Cache** — we will make every LLM call, tool call, and verification a journaled activity, configure retry and caching policies, and prove that a crash mid-mission does not re-spend tokens on already-completed work.