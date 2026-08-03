"""Tests for the agent loop."""

import pytest
from pathlib import Path

from lra.local_runner import LocalRunner


def test_local_runner_on_hello_task(tmp_path: Path) -> None:
    runner = LocalRunner(tmp_path)
    runner.set_task([
        {"id": "1", "description": "Create hello.py", "status": "todo"},
        {"id": "2", "description": "Add test for hello.py", "status": "todo"},
    ])
    summary = runner.run()
    assert summary["cycles"] >= 1
