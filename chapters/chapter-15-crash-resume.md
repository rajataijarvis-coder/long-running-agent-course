# Chapter 15: Surviving Crashes — Kill and Resume

> If you cannot demonstrate kill-and-resume, you do not have a durable agent.

## What We'll Cover

- Crash-resume as a first-class engineering property
- The exact kill-and-resume procedure for the LRA demo
- How to prove no work is lost
- How to prove no tokens are re-spent
- Writing an automated crash-resume test

---

## Durability Is Not a Claim; It Is a Demonstration

You can read about durable execution, but you do not really believe it until you see it. The most convincing test is:

1. Start a mission.
2. While it is running, `kill -9` the worker process.
3. Restart the worker.
4. Verify the mission resumes exactly where it stopped.

If this works, your architecture is durable. If it does not, you have a bug.

---

## What Should Survive a Crash

In the LRA demo, the following must survive a worker crash:

- The mission checklist in `.lra/checklist.json`
- The progress log in `.lra/progress.md`
- The event stream in `.lra/events.ndjson`
- The decisions log in `.lra/decisions.ndjson`
- Any files written by the Lead Engineer
- The Temporal workflow history on the server

What should **not** be lost:

- The current item being processed (it is in the workflow history)
- The result of any already-completed activity (it is in the workflow history)

What is allowed to restart:

- The in-memory worker process
- The Python interpreter state

---

## Manual Kill-and-Resume Procedure

### Step 1: prepare a mission directory

```bash
mkdir -p /tmp/lra-crash-test
cd /tmp/lra-crash-test
```

### Step 2: start the worker in one terminal

```bash
cd /Users/rajatjarvis/Downloads/projects/long-running-agent-course/lra-demo
uv run python run_worker.py
```

### Step 3: start a mission in another terminal

```python
import asyncio
from pathlib import Path
from temporalio.client import Client
from lra.temporal_workflow import MissionInput, MissionWorkflow
from lra.anchor import MissionAnchor

async def main():
    workdir = "/tmp/lra-crash-test"
    anchor = MissionAnchor(workdir)
    anchor.write_checklist({
        "items": [
            {"id": "1", "description": "Create hello.py", "status": "todo"},
            {"id": "2", "description": "Add test for hello.py", "status": "todo"},
        ]
    })

    client = await Client.connect("localhost:7233")
    await client.start_workflow(
        MissionWorkflow.run,
        MissionInput(workdir=workdir, items=[]),
        id="crash-test-1",
        task_queue="lra-mission",
    )

asyncio.run(main())
```

### Step 4: kill the worker while the first item is in progress

Find the worker PID:

```bash
pgrep -f run_worker.py
```

Then:

```bash
kill -9 <PID>
```

### Step 5: restart the worker

```bash
uv run python run_worker.py
```

The workflow will resume from the last completed activity. If the first activity (`load_anchor`) completed but `run_one_cycle` had not, the workflow will re-call `run_one_cycle`. Because that activity is idempotent (it checks the anchor status), no duplicate work is produced.

### Step 6: inspect the result

```bash
cat /tmp/lra-crash-test/.lra/checklist.json
```

Both items should eventually be `done`.

---

## Automated Crash-Resume Test

Manual tests are good; automated tests are better. Add the following to `lra-demo/tests/test_temporal.py` or a new file `tests/test_crash_resume.py`:

```python
import asyncio
import pytest
from pathlib import Path
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from lra.anchor import MissionAnchor
from lra.temporal_workflow import MissionInput, MissionWorkflow, load_anchor, reviewer_check, run_one_cycle

async def test_crash_resume_does_not_lose_state(env: WorkflowEnvironment, tmp_path: Path) -> None:
    workdir = str(tmp_path)
    anchor = MissionAnchor(workdir)
    anchor.write_checklist({
        "items": [
            {"id": "1", "description": "Create hello.py", "status": "todo"},
            {"id": "2", "description": "Add test for hello.py", "status": "todo"},
        ]
    })

    # First worker: run only the first activity then simulate a crash by shutting down the worker.
    async with Worker(env.client, task_queue="crash-test", workflows=[MissionWorkflow], activities=[load_anchor, run_one_cycle, reviewer_check]) as worker:
        handle = await env.client.start_workflow(
            MissionWorkflow.run,
            MissionInput(workdir=workdir, items=[]),
            id="crash-resume-1",
            task_queue="crash-test",
        )
        await handle.query(...)  # not shown; instead we use a short sleep
        await asyncio.sleep(0.5)
        # Worker exits context manager (simulates crash).

    # Second worker: resumes the same workflow.
    async with Worker(env.client, task_queue="crash-test", workflows=[MissionWorkflow], activities=[load_anchor, run_one_cycle, reviewer_check]) as worker:
        result = await handle.result()

    assert result["checklist"]["items"][0]["status"] == "done"
    assert result["checklist"]["items"][1]["status"] == "done"
```

Temporal's test server handles the journal for you, so the second worker resumes from the first worker's history automatically.

---

## Proving No Tokens Are Re-Spent

If you add an LLM call inside an activity, you can prove replay is free by:

1. Running the workflow once and recording the token count.
2. Killing and restarting the worker.
3. Checking that the token count in the LLM provider dashboard did not increase.

Temporal's history guarantees the activity result is returned from the cache. The LLM provider sees the call exactly once.

---

## Hands-On Exercise

1. Perform the manual kill-and-resume procedure above using `run_worker.py`.
2. Before killing, note the workflow ID and the current event count in the Temporal UI.
3. After resuming, confirm the event count increased but no earlier activities were duplicated.
4. Write a one-page test report in `.lra/crash_resume_report.md` documenting what survived and what did not.

---

## Key Takeaway

> Durability is only real when you can prove it. Kill the worker. Restart it. If the mission resumes exactly where it left off, you have built a long-running agent. If it does not, you have built a long chat.

---

## Next Chapter

**Chapter 16: The Mission Anchor** — we will do a deep dive into the four anchor files that hold all mission truth.
