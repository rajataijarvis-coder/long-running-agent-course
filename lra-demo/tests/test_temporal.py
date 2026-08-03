"""Integration test for Temporal workflow replay.

This test runs the MissionWorkflow against the in-memory Temporal test server
so we can prove durable execution without Docker.
"""

from __future__ import annotations

import pytest
from pathlib import Path


try:
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"temporalio not installed: {exc}", allow_module_level=True)

from lra.anchor import MissionAnchor
from lra.temporal_workflow import MissionInput, MissionWorkflow, load_anchor, run_one_cycle, reviewer_check


@pytest.fixture(scope="module")
def env():
    with WorkflowEnvironment.start_time_skipping() as e:
        yield e


async def test_mission_workflow_runs_to_completion(env, tmp_path: Path) -> None:
    workdir = str(tmp_path)
    anchor = MissionAnchor(workdir)
    anchor.write_checklist(
        {
            "items": [
                {"id": "1", "description": "Create hello.py", "status": "todo"},
                {"id": "2", "description": "Add test for hello.py", "status": "todo"},
            ]
        }
    )

    async with Worker(
        env.client,
        task_queue="test-queue",
        workflows=[MissionWorkflow],
        activities=[load_anchor, run_one_cycle, reviewer_check],
    ):
        result = await env.client.execute_workflow(
            MissionWorkflow.run,
            MissionInput(workdir=workdir, items=[]),
            id="test-mission",
            task_queue="test-queue",
        )

    assert "checklist" in result
    assert result["checklist"]["items"][0]["status"] == "done"
