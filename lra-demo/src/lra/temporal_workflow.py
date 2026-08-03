"""Temporal workflow and activity definitions for the durable LRA loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from lra.anchor import MissionAnchor
from lra.loop import AgentLoop
from lra.review import ReflectionResult, StaticReviewer
from lra.tools import ToolDispatcher
from lra.verify import DeterministicVerifier, VerificationResult


@dataclass
class MissionInput:
    workdir: str
    items: list[dict[str, str]]


@dataclass
class CycleInput:
    workdir: str
    item_id: str


@dataclass
class CycleOutput:
    item_id: str
    all_passed: bool
    reflection: ReflectionResult | None = None
    error: str | None = None


@activity.defn
async def load_anchor(workdir: str) -> dict[str, Any]:
    """Idempotent activity: reads the current anchor state."""
    anchor = MissionAnchor(workdir)
    return {
        "progress": anchor.read_progress(),
        "checklist": anchor.read_checklist(),
    }


@activity.defn
async def run_one_cycle(cycle_input: CycleInput) -> CycleOutput:
    """Runs gather → act → verify for one checklist item.

    This is the side-effecting core of the agent. Temporal journals the result.
    """
    anchor = MissionAnchor(cycle_input.workdir)
    loop = AgentLoop(
        anchor=anchor,
        tools=ToolDispatcher(),
        verifier=DeterministicVerifier.from_workdir(cycle_input.workdir),
    )
    item = next(
        (i for i in anchor.read_checklist()["items"] if i["id"] == cycle_input.item_id),
        None,
    )
    if item is None:
        return CycleOutput(item_id=cycle_input.item_id, all_passed=False, error="item not found")

    from lra.anchor import ChecklistItem
    checklist_item = ChecklistItem(**item)
    blocked = loop.cycle(checklist_item)
    result = VerificationResult(checks=[])
    reflection: ReflectionResult | None = None
    if blocked:
        _, reflect = __import__("lra.review", fromlist=["build_review_and_reflect"]).build_review_and_reflect(anchor)
        reflection = reflect.reflect(cycle_input.item_id)

    return CycleOutput(
        item_id=cycle_input.item_id,
        all_passed=not blocked,
        reflection=reflection,
    )


@activity.defn
async def reviewer_check(workdir: str, item_id: str) -> dict[str, Any]:
    """Independent reviewer activity: returns approval and findings."""
    anchor = MissionAnchor(workdir)
    reviewer = StaticReviewer(anchor)
    result = reviewer.review(item_id)
    return {
        "approved": result.approved,
        "findings": result.findings,
        "suggestions": result.suggestions,
    }


@workflow.defn
class MissionWorkflow:
    """Durable workflow that executes a mission one item at a time.

    Workflow code must be deterministic. All non-deterministic work (LLM, shell,
    filesystem, network) goes through activities.
    """

    @workflow.run
    async def run(self, mission_input: MissionInput) -> dict[str, Any]:
        # Use an idempotency key derived from the input so replay-from-cache works.
        anchor_state = await workflow.execute_activity(
            load_anchor,
            mission_input.workdir,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        items = anchor_state["checklist"].get("items", [])

        for item in items:
            if item.get("status") in ("done",):
                continue
            cycle_out: CycleOutput = await workflow.execute_activity(
                run_one_cycle,
                CycleInput(workdir=mission_input.workdir, item_id=item["id"]),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                    maximum_attempts=3,
                ),
            )
            if not cycle_out.all_passed:
                break

        # Return fresh anchor state for the client.
        return await workflow.execute_activity(
            load_anchor,
            mission_input.workdir,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )


@workflow.defn
class CadenceWorkflow:
    """Workflow that sleeps between mission iterations and then continues-as-new.

    This pattern lets a mission run indefinitely (daily polling, weekly report,
    etc.) without creating an infinitely long event history.
    """

    @workflow.run
    async def run(self, workdir: str, iteration: int = 0) -> None:
        if iteration == 0:
            # Bootstrap a tiny mission for the demo.
            anchor = MissionAnchor(workdir)
            anchor.write_checklist(
                {
                    "items": [
                        {"id": "1", "description": "Create hello.py", "status": "todo"},
                        {"id": "2", "description": "Add test for hello.py", "status": "todo"},
                    ]
                }
            )

        await workflow.execute_child_workflow(
            MissionWorkflow.run,
            MissionInput(workdir=workdir, items=[]),
            id=f"mission-iteration-{iteration}",
        )

        await workflow.sleep(timedelta(seconds=10))

        workflow.continue_as_new(workdir, iteration + 1)
