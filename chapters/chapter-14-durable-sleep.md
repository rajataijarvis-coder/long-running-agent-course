# Chapter 14: Durable Sleep and Continue-As-New

> Durable sleep and Continue-As-New let missions run for months, not minutes.

## What We'll Cover

- Why `time.sleep` does not scale for long-running agents
- How `workflow.sleep` parks the workflow at zero cost
- Using durable timers for polling and human-in-the-loop waits
- `continue_as_new` to reset history size for unbounded missions
- Building a cadence workflow that runs forever

---

## Why Normal Sleep Fails

A naive agent might poll an API like this:

```python
while True:
    do_work()
    time.sleep(86400)  # wait a day
```

Problems:

1. The process must stay alive for the whole day.
2. If the machine reboots, the sleep is lost.
3. The workflow history grows forever.
4. You pay for a process that does nothing.

A durable agent cannot sleep like this.

---

## Durable Timers in Temporal

Temporal provides `workflow.sleep(timedelta)` and `workflow.execute_child_workflow(...)` for timed waits. When a workflow reaches `workflow.sleep(86400)`, the worker unloads it from memory. The server wakes it up after the timer fires. No process runs. No cost accrues.

If the server restarts during the sleep, the workflow still wakes up at the right time because the timer is in the durable history.

---

## A Cadence Workflow

The demo includes `CadenceWorkflow` in `src/lra/temporal_workflow.py`:

```python
@workflow.defn
class CadenceWorkflow:
    @workflow.run
    async def run(self, workdir: str, iteration: int = 0) -> None:
        if iteration == 0:
            anchor = MissionAnchor(workdir)
            anchor.write_checklist({
                "items": [
                    {"id": "1", "description": "Create hello.py", "status": "todo"},
                    {"id": "2", "description": "Add test for hello.py", "status": "todo"},
                ]
            })

        await workflow.execute_child_workflow(
            MissionWorkflow.run,
            MissionInput(workdir=workdir, items=[]),
            id=f"mission-iteration-{iteration}",
        )

        await workflow.sleep(timedelta(seconds=10))

        workflow.continue_as_new(workdir, iteration + 1)
```

This workflow:

1. Runs a child mission workflow.
2. Sleeps for ten seconds (use a day in production).
3. Calls `continue_as_new` to start a fresh workflow execution with the same ID.

`continue_as_new` resets the event history. Without it, a workflow that ran every day for a year would accumulate ~365 history pages and eventually hit limits. With it, each iteration starts with a clean history.

---

## Continue-As-New Semantics

`workflow.continue_as_new(...)` is **not** a recursive call. It tells the server:

> Throw away the current history, start a new workflow execution with these arguments, and continue from the first line of the new run.

The new run has the same workflow ID by default. The previous run is marked `Continued-As-New`. In the Temporal UI you will see a chain of executions.

Important: any local variables are lost. Only the arguments you pass survive. Therefore all durable state must live in the `MissionAnchor` (files on disk or database), not in memory.

---

## Real Use Cases

| Pattern | Durable primitive |
|---------|------------------|
| Daily report generation | `workflow.sleep(timedelta(days=1))` + `continue_as_new` |
| Wait for human approval | `workflow.sleep(timedelta(hours=48))` as a timeout |
| Poll an external API | `workflow.sleep(timedelta(minutes=5))` inside a bounded loop, then `continue_as_new` |
| Long training job | Execute child workflow, sleep, continue |

---

## Hands-On Exercise

1. Add `CadenceWorkflow` to your worker registration:
   ```python
   from lra.temporal_workflow import CadenceWorkflow, MissionWorkflow
   worker = Worker(
       client,
       task_queue="lra-mission",
       workflows=[MissionWorkflow, CadenceWorkflow],
       activities=[load_anchor, run_one_cycle, reviewer_check],
   )
   ```
2. Start a cadence workflow:
   ```python
   await client.execute_workflow(
       CadenceWorkflow.run,
       "/tmp/lra-cadence",
       id="cadence-1",
       task_queue="lra-mission",
   )
   ```
3. Watch the Temporal UI. You should see the parent workflow continue-as-new every ten seconds.
4. Stop the worker for a few seconds and restart it. The chain should resume without losing iteration count.

---

## Key Takeaway

> Durable timers make waiting free. Continue-As-New makes forever possible. Together they turn a short-lived script into a system that can run for months.

---

## Next Chapter

**Chapter 15: Surviving Crashes — Kill and Resume** — we will prove durability with a real crash test.
