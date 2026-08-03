"""Tool dispatch with default-deny allow-list."""

import hashlib
import subprocess
from pathlib import Path
from typing import Callable


def _read_file(path: str) -> dict[str, str]:
    target = Path(path)
    if not target.exists():
        return {"ok": False, "error": f"File not found: {path}"}
    return {"ok": True, "content": target.read_text()}


def _write_file(path: str, content: str) -> dict[str, str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"ok": True, "hash": hashlib.sha256(content.encode()).hexdigest()[:12]}


def _run_command(cmd: str, workdir: str | None = None) -> dict[str, str | int]:
    # Minimal allow-list validation: reject shell metacharacters
    dangerous = {";", "|", "&&", "||", "`", "$", ">", "<"}
    if any(ch in cmd for ch in dangerous):
        return {"ok": False, "error": "Command contains disallowed shell characters", "returncode": -1}
    result = subprocess.run(cmd, shell=True, cwd=workdir, capture_output=True, text=True)
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


ALLOWED_TOOLS: dict[str, Callable] = {
    "read_file": _read_file,
    "write_file": _write_file,
    "run_command": _run_command,
}


class ToolDispatcher:
    """Default-deny tool dispatcher."""

    def __init__(self, allowed: dict[str, Callable] | None = None):
        self.allowed = allowed or ALLOWED_TOOLS

    def dispatch(self, name: str, args: dict) -> dict[str, str | int]:
        if name not in self.allowed:
            return {"ok": False, "error": f"Tool {name} not in allow-list", "returncode": -1}
        return self.allowed[name](**args)
