# Chapter 02: Durability as an Engineering Property

What We'll Cover
- Defining durability in the context of long-running agents
- The limitations of model-based autonomy 
- Comparing agent approaches to long-horizon tasks
- Introducing Temporal and durable execution
- Hands-on Temporal demo with Python code

## Concept Explanation

Durability is an engineering property, not a model capability. Long-running agents need a system around the model to survive interruptions, crashes, context-window limits, and API call failures. The key is externalizing the real state (the source of truth) from inside the model's ephemeral context window.

Most "autonomous" agents are just single LLM loops that die when:
- The process restarts
- The context window fills up
- An API call fails

That's fine for quick chats, but it falls apart on long-horizon tasks that run for days or weeks. We need a system to keep the real state durable and consistent across interruptions.

The core idea is "assume interruption":
- The context window is a lossy cache
- The real state lives outside the model

## Python Code Walkthrough

Here's how you can get started with Temporal, the key durable execution engine:

```python
from temporalio import workflow

@workflow.defun
def my_durable_workflow():
    # This is a durable, long-running activity
    result = yield workflow.execute_activity(
        my_undurable_activity,
        start_to_end=True  # Make sure to wait for it to complete
    )
    return result

@workflow.defun
def my_undurable_activity():
    # This can be interrupted at any time and will resume from where it left off
    return "Hello, World!"

# Run the workflow
my_durable_workflow()
```

This code defines a durable `my_durable_workflow` that uses an undurable activity `my_undurable_activity`. The key is using Temporal's `execute_activity` to wait for the activity to complete and make sure it runs to end. If the process restarts, Temporal will resume the activity from its last known good state.

## Hands-on Exercise

1. Install Temporal:
```bash
pip install temporalio
```

2. Run this Python code in a script:

```python
from datetime import timedelta
import random
from typing import Any

import temporalio
from temporalio.activity import ActivityOptions
from temporalio.client import Client
from temporalio.types import WorkflowIdInfo


def my_undurable_activity(task: dict[str, Any]) -> str:
    print(f"Starting activity {task['activity_id']}")
    random_delay = random.randint(1, 5)
    # Simulate doing some work
    for i in range(random_delay):
        print(f"Activity {task['activity_id']} working...", end="")
        temporalio.time.sleep(timedelta(seconds=1))
    print(f"Finished activity {task['activity_id']}")
    return f"Result from activity {task['activity_id']}"

async def main():
    client = Client("localhost:7233")
    await client.start()

    workflow_id_info = WorkflowIdInfo(workflow="my_durable_workflow", run_id=client.new_run_id())
    
    # Schedule the activity
    activity_options = ActivityOptions(
        task_queue="my_task_queue",
        start_to_end=True,  # Wait for completion
    )
    result = await client.execute_activity(
        my_undurable_activity,
        activity_options=activity_options,
        workflow_id_info=workflow_id_info,
    )

    print(f"Workflow completed with result: {result}")

temporalio.run(main)
```

This code simulates a durable workflow that schedules an undurable activity. The `start_to_end=True` option makes sure the activity completes before moving on.

## Key Takeaway

> "Durability is not about making your model smarter. It's about engineering a system around it to survive interruptions."

Next Chapter
# Chapter 03: Externalizing Truth — Git as Memory

We'll dive into how git becomes the externalized memory for long-running agents, allowing them to checkpoint progress and recover state across interruptions.

The real state (codebase + checklist) lives in a git repo that the agent periodically checkpoints. This durable anchor allows the system to resume from where it left off after crashes or reboots.

By externalizing the truth into git, we decouple the model's context window from the actual state of the mission. The agent can survive interruptions and keep making verified progress over days or weeks.