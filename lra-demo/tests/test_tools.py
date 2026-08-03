"""Tests for tool dispatcher."""

from pathlib import Path

from lra.tools import ToolDispatcher


def test_write_and_read_file(tmp_path: Path) -> None:
    tool = ToolDispatcher()
    path = str(tmp_path / "demo.txt")
    write_result = tool.dispatch("write_file", {"path": path, "content": "hello"})
    assert write_result["ok"] is True

    read_result = tool.dispatch("read_file", {"path": path})
    assert read_result["content"] == "hello"


def test_dangerous_command_rejected(tmp_path: Path) -> None:
    tool = ToolDispatcher()
    result = tool.dispatch("run_command", {"cmd": "ls; rm -rf /"})
    assert result["ok"] is False
