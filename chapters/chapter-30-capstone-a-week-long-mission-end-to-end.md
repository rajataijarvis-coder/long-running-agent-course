# Chapter 30: Capstone — A Week-Long Mission End-to-End

## What We'll Cover

- Bringing up the full runtime stack from Chapter 27 (Temporal, Postgres, Ollama, Langfuse)
- Submitting a real, multi-cycle mission and watching it checkpoint into git
- Simulating a host-level crash and proving that the mission resumes without re-spending tokens
- Observing the asymmetric agent organization, loop detector, reviewer gate, and budget governor in one trace
- Inspecting the final artifact, cost report, and learned skills
- Exporting a captured skill to the offline eval harness from Chapter 26

---

## The Concept: A Week-Long Mission Is a System, Not a Prompt

By this point in the course you have built every layer of LRA:

- Chapters 1–6 argued that long-horizon autonomy is an engineering property, not a model capability, and that deterministic verification is the only acceptable definition of “done.”
- Chapters 7–11 gave you the local agent cycle and the asymmetric organization: one Lead Engineer, parallel Researchers, and an independent Reviewer.
- Chapters 12–15 moved that loop into Temporal so crashes, reboots, and API failures become resumable events.
- Chapters 16–19 externalized memory into git, Postgres, and pgvector.
- Chapters 20–23 added the governor, loop detection, HITL gates, and default-deny safety.
- Chapters 24–26 made failures inspectable and turned improvement into an offline loop.
- Chapters 27–29 gave you a reproducible runtime, replaceable workers, and observability.

This chapter wires all of those pieces together into one continuous run. The task is the same one from the source article: build a small Python LSP server from scratch. In a real classroom or production setting this takes days; in the demo harness below we scale the durable sleeps down so you can watch a representative mission complete in an hour or two, while still exercising every survival mechanism.

The artifact you run is `lra-demo/ch30_capstone.py`. It is not a toy. It starts real services, submits a real Temporal workflow, writes real git history, survives a real container kill, and prints a real cost report.

---

## Code Walkthrough: `lra-demo/ch30_capstone.py`

The script has three responsibilities: bootstrap the stack, drive the mission, and report the outcome. It is intentionally self-contained so you can run it from a fresh clone.

```python
#!/usr/bin/env python3
"""
lra-demo/ch30_capstone.py
End-to-end harness for a week-long LRA mission.
This script:
  1. Brings up the Docker Compose stack from Chapter 27.
  2. Waits for Temporal, Postgres, Ollama, and Langfuse to be healthy.
  3. Submits a MissionWorkflow for "build a minimal Python LSP server."
  4. Polls state, injects a worker crash, and verifies resume.
  5. Prints the git history, final checklist, and cost report.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import httpx
from temporalio.client import Client
from temporalio.common import RetryPolicy

# Allow running against the source tree without a package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lra.contracts.mission import MissionSpec
from lra.durable.workflows.mission import MissionWorkflow


# --------------------------------------------------------------------------- #
# 1. Stack bootstrap (Chapter 27)
# --------------------------------------------------------------------------- #

COMPOSE_FILE = REPO_ROOT / "lra-demo" / "compose.yaml"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def stack_up() -> None:
    run([
        "docker", "compose", "-f", str(COMPOSE_FILE),
        "--profile", "full", "up", "-d", "--build",
    ])


def stack_down() -> None:
    run(["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", "full", "down"])


async def wait_for_services(timeout: float = 300.0) -> None:
    """Health-check every dependency before submitting work."""
    checks = {
        "temporal": ("http://localhost:8233/health", lambda r: r.status_code == 200),
        "postgres": (None, lambda _: run(["pg_isready", "-h", "localhost", "-p", "5432"], check=False).returncode == 0),
        "ollama": ("http://localhost:11434/api/tags", lambda r: r.status_code == 200),
        "langfuse": ("http://localhost:3000/api/public/health", lambda r: r.status_code == 200),
    }

    deadline = time.time() + timeout
    async with httpx.AsyncClient() as client:
        for name, (url, ok) in checks.items():
            while time.time() < deadline:
                try:
                    if url is None:
                        healthy = ok(None)
                    else:
                        healthy = ok(await client.get(url, timeout=2.0))
                    if healthy:
                        print(f"[health] {name} ready")
                        break
                except Exception as exc:
                    print(f"[health] {name} not ready: {exc}")
                    await asyncio.sleep(2.0)
            else:
                raise RuntimeError(f"{name} did not become healthy in {timeout}s")


# --------------------------------------------------------------------------- #
# 2. Mission submission and polling
# --------------------------------------------------------------------------- #

async def start_mission(client: Client, task: str, workdir: Path) -> object:
    workdir.mkdir(parents=True, exist_ok=True)

    spec = MissionSpec(
        task=task,
        workdir=str(workdir.resolve()),
        budget_usd=15.0,
        model_backend="ollama/llama3.1:8b",
        max_cycles=250,
        durable_sleep_seconds=10,  # scaled down for the demo
        enable_reviewer=True,
        enable_loop_detector=True,
        enable_governor=True,
    )

    workflow_id = f"capstone-lsp-{int(time.time())}"
    handle = await client.start_workflow(
        MissionWorkflow.run,
        spec,
        id=workflow_id,
        task_queue="lra-missions",
        retry_policy=RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(minutes=1),
            maximum_attempts=3,
        ),
    )
    print(f"[mission] started {workflow_id}")
    return handle


async def query_state(handle):
    return await handle.query(MissionWorkflow.state)


async def wait_for_predicate(handle, predicate, timeout: float = 600.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = await query_state(handle)
        if predicate(state):
            return state
        await asyncio.sleep(5.0)
    raise TimeoutError("predicate never matched")


# --------------------------------------------------------------------------- #
# 3. Chaos injection (Chapters 14–15 and 28)
# --------------------------------------------------------------------------- #

def kill_active_worker() -> None:
    """
    Find whichever worker container is currently running and kill it.
    This simulates a host reboot or a blue/green swap without graceful drain.
    """
    result = run(["docker", "ps", "-q", "-f", "name=lra-worker"], check=False)
    ids = [line for line in result.stdout.splitlines() if line]
    if not ids:
        raise RuntimeError("no worker container found to kill")
    target = ids[0]
    print(f"[chaos] killing worker container {target[:12]}")
    run(["docker", "kill", target])


# --------------------------------------------------------------------------- #
# 4. Reporting
# --------------------------------------------------------------------------- #

def print_report(workdir: Path) -> None:
    print("\n=== git history (last 20 commits) ===")
    run(["git", "-C", str(workdir), "log", "--oneline", "-20"], check=False)

    state_path = workdir / ".lra" / "state.json"
    if state_path.exists():
        print("\n=== final mission state ===")
        print(json.dumps(json.loads(state_path.read_text()), indent=2))

    governor_path = workdir / ".lra" / "governor.json"
    if governor_path.exists():
        print("\n=== governor report ===")
        print(json.dumps(json.loads(governor_path.read_text()), indent=2))

    skills_path = workdir / ".lra" / "skills.jsonl"
    if skills_path.exists():
        print("\n=== learned skills ===")
        for line in skills_path.read_text().splitlines()[-5:]:
            print(line)


# --------------------------------------------------------------------------- #
# 5. Main orchestration
# --------------------------------------------------------------------------- #

async def main() -> int:
    task = (
        "Build a minimal Python LSP server from scratch. "
        "It must handle the 'initialize' and 'textDocument/didOpen' methods, "
        "include pytest tests, and pass a lint/typecheck gate. "
        "Keep the scope small; a working subset is enough."
    )
    workdir = REPO_ROOT / ".lra" / "workspaces" / "capstone-lsp"

    print("[capstone] starting stack")
    stack_up()
    await wait_for_services()

    print("[capstone] connecting to Temporal")
    client = await Client.connect("localhost:7233", namespace="default")

    print("[capstone] submitting mission")
    handle = await start_mission(client, task, workdir)

    print("[capstone] waiting for mission to make progress")
    await wait_for_predicate(handle, lambda s: s.get("cycles", 0) >= 10)

    print("[capstone] injecting worker crash")
    kill_active_worker()

    print("[capstone] waiting for resume")
    await wait_for_predicate(handle, lambda s: s.get("cycles", 0) >= 12)

    print("[capstone] waiting for completion or budget exhaustion")
    final = await wait_for_predicate(
        handle,
        lambda s: s.get("status") in {"DONE", "BUDGET_CUT", "FAILED"},
        timeout=7200.0,
    )

    print("\n=== mission result ===")
    print(json.dumps(final, indent=2))

    print_report(workdir)

    # Optional: export the newest learned skill to the eval harness (Chapter 26)
    eval_link = REPO_ROOT / "lra-demo" / "ch26_eval_harness.py"
    if eval_link.exists():
        print(f"\n[capstone] run {eval_link} to evaluate any captured skills offline.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n[capstone] interrupted; stack is still running. Run:")
        print(f"  docker compose -f {COMPOSE_FILE} --profile full down")
```

### What the script proves

1. **Durable execution.** The mission is a Temporal workflow. When `kill_active_worker()` fires, the in-flight activity is retried on a fresh worker and the workflow resumes from the last journal entry. The `cycles` counter does not reset.
2. **Git as memory.** Every verified checkpoint is a commit. After the crash, the new worker reconstructs state from `workdir/.lra/state.json` and the git log, not from a long context window.
3. **Deterministic verification.** Items move to `DONE` only when the verifier activity returns a zero exit code. The model is not allowed to declare victory.
4. **Governor and loop detector.** If the mission over-plans or starts oscillating, `governor.json` records scope cuts and `loop_trips`, and the run ends under budget.
5. **Observability.** Every LLM call, tool execution, and verification step is emitted through OpenTelemetry and visible in Langfuse (Chapter 29).

---

## The Mission Workflow Interface

The driver above assumes `MissionWorkflow` exposes a `run` class method and a `state` query. A minimal version of that workflow looks like this:

```python
# src/lra/durable/workflows/mission.py
from datetime import timedelta
from temporalio import workflow
from lra.contracts.mission import MissionSpec, MissionState

with workflow.unsafe.imports_passed_through():
    from lra.durable.activities import agent_cycle, verify, checkpoint


@workflow.defn
class MissionWorkflow:
    def __init__(self) -> None:
        self._state: MissionState = MissionState()

    @workflow.query
    def state(self) -> MissionState:
        return self._state

    @workflow.run
    async def run(self, spec: MissionSpec) -> MissionState:
        while not self._state.is_terminal():
            # Each cycle is a durable activity. If the worker dies here,
            # Temporal replays from cache on resume.
            self._state = await workflow.execute_activity(
                agent_cycle,
                (spec, self._state),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=5),
            )

            # Checkpoint to git only after a verified step.
            await workflow.execute_activity(
                checkpoint,
                (spec, self._state),
                start_to_close_timeout=timedelta(minutes=2),
            )

            # Long idle periods are durable sleep, not busy polling.
            if self._state.should_sleep():
                await workflow.sleep(timedelta(seconds=spec.durable_sleep_seconds))

            # Continue-as-new before the event history grows too large.
            if self._state.should_continue_as_new():
                return await workflow.execute_child_workflow(
                    MissionWorkflow.run, spec
                )

        return self._state
```

This is the same pattern from Chapters 12–15: the workflow is a thin scheduler, the activities do the real work, and the history is the source of truth.

---

## Hands-On Exercise

Run the capstone end-to-end and deliberately stress each survival layer.

1. **Start the stack and the mission:**

   ```bash
   uv sync --extra durable --extra embeddings --extra observability
   uv run python lra-demo/ch30_capstone.py
   ```

2. **Watch the first few cycles.** In another terminal, tail the mission state:

   ```bash
   watch -n 5 'cat .lra/workspaces/capstone-lsp/.lra/state.json | jq .cycles,.status'
   ```

3. **Inject a crash manually** while the script is running:

   ```bash
   docker ps -q -f name=lra-worker | head -1 | xargs docker kill
   ```

   Confirm in the Temporal UI at `http://localhost:8233` that the workflow is still `RUNNING` and that `cycles` resumed from where it left off.

4. **Inspect the git history:**

   ```bash
   git -C .lra/workspaces/capstone-lsp log --oneline -20
   ```

   Each checkpoint should have a message like `checkpoint: cycle=42 items=3/7`.

5. **Open Langfuse** at `http://localhost:3000`, find the trace for your workflow id, and identify:
   - one generation where the reviewer blocked a commit,
   - one verification activity with a non-zero exit code,
   - the total token cost at the end of the run.

6. **Export a learned skill.** If the mission captured a skill in `.lra/skills.jsonl`, copy it into `lra/evals/skills/` and run the eval harness from Chapter 26:

   ```bash
   uv run python lra-demo/ch26_eval_harness.py --skill .lra/workspaces/capstone-lsp/.lra/skills.jsonl
   ```

7. **(Stretch goal)** Modify the mission task to add a new LSP method, such as `textDocument/hover`, and run it again. Compare the final cost and cycle count to the first run.

---

> **Key Takeaway:** A week-long mission is not a long conversation with a model. It is a durable organization that keeps the real state in git, verifies progress with real tests, survives crashes through Temporal, and stays under budget through explicit governors. The model thinks in short bursts; the system around it runs for weeks.

---

## Next Chapter

**Afterword: Operating LRA in Production** — Turning the demo stack into a production deployment: multi-host Temporal namespaces, encrypted Postgres, model-provider fallbacks, retention policies, and the operator runbook for the day a mission has been running longer than you have.