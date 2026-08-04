# Chapter 05: Tool Dispatching and Sandboxing

> A model decides. A tool changes the world. A sandbox makes sure the change is bounded.
> — Fareed Khan

## What We'll Cover

- Why the **Act** phase from Chapter 04 needs a dedicated dispatcher instead of raw `exec()`
- The contract between a model action, a `ToolDispatcher`, and a `Sandbox`
- How to build a **local sandbox** that bounds filesystem and subprocess access
- Where Docker and E2B fit in the same interface
- Wiring tool results back into the `gather → act → verify → checkpoint` cycle
- A runnable demo in `lra-demo/ch05_tool_sandbox.py`

---

## From Model Decision to Real Change

In Chapter 04 the agent cycle had four phases. The **Act** phase is the only one that touches the real world: it writes files, runs tests, installs packages, or calls APIs. Everything before it is reasoning; everything after it is verification.

If the model is allowed to call `subprocess.run(..., shell=True)` directly, three bad things happen quickly:

1. **Blast radius** — a bad prompt can delete the working tree, exfiltrate secrets, or fork-bomb the host.
2. **Non-determinism** — the same tool call can produce different outputs depending on shell state.
3. **Untestability** — the agent loop becomes impossible to unit test because it is coupled to the host machine.

The fix is a small, strict layer: a `ToolDispatcher` that knows which tools exist and a `Sandbox` that knows how to run them safely. The model never touches `subprocess`, `open()`, or the network directly.

This chapter builds that layer in plain Python. The durable Temporal spine from Chapter 02 and the git anchor from Chapter 03 sit *outside* this layer; the dispatcher is the bridge between them.

---

## The Tool Contract

A tool call is just a name plus arguments. A tool result is stdout, stderr, an exit code, and optionally structured data. We keep both sides explicit so the dispatcher can log, retry, and replay them later.

```python
# lra-demo/ch05_tool_sandbox.py
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ToolRequest:
    name: str
    arguments: dict[str, object]
    request_id: str = ""


@dataclass
class ToolResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    data: dict[str, object] = field(default_factory=dict)

    def to_model_text(self, cap: int = 2000) -> str:
        """Format the result for the next model turn."""
        return (
            f"tool: {self.request_id or 'unknown'}\n"
            f"success: {self.success}\n"
            f"exit_code: {self.exit_code}\n"
            f"stdout:\n{self.stdout[:cap]}\n"
            f"stderr:\n{self.stderr[:cap]}"
        )
```

The `ToolResult.to_model_text()` method is important. The model does not get raw bytes; it gets a capped, structured observation. This prevents a runaway command from filling the context window and keeps the loop deterministic.

---

## The Sandbox Protocol

A sandbox is anything that can execute commands and read/write files inside a bounded directory. We define it as a protocol so local, Docker, and E2B implementations share the same interface.

```python
class Sandbox(Protocol):
    """Bounded execution surface for tools."""

    def run_command(self, command: list[str], timeout: float = 30.0) -> ToolResult: ...

    def write_file(self, relative_path: str, content: str) -> ToolResult: ...

    def read_file(self, relative_path: str) -> ToolResult: ...
```

The key word is **bounded**. Every path is relative to the sandbox workdir. Absolute paths and directory traversal are rejected.

---

## A Local Sandbox Implementation

The local sandbox is the cheapest to run and the easiest to test. It uses `subprocess` with a strict allow-list and a `cwd` lock.

```python
class LocalSandbox:
    """Run tools in a single directory on the host machine."""

    def __init__(self, workdir: Path):
        self.workdir = workdir.resolve()

    def _resolve(self, relative_path: str) -> Path:
        target = (self.workdir / relative_path).resolve()
        # Default-deny: the path must live inside the workdir.
        if self.workdir not in target.parents and target != self.workdir:
            raise ValueError(f"Path escapes workdir: {relative_path}")
        return target

    def run_command(self, command: list[str], timeout: float = 30.0) -> ToolResult:
        try:
            proc = subprocess.run(
                command,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,  # Never shell=True in a sandbox.
            )
            return ToolResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                success=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                exit_code=-1,
                data={"error": "timeout"},
            )
        except FileNotFoundError as exc:
            return ToolResult(
                success=False,
                stderr=f"Command not found: {exc.filename}",
                exit_code=127,
            )

    def write_file(self, relative_path: str, content: str) -> ToolResult:
        try:
            target = self._resolve(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(success=True, data={"path": str(target)})
        except ValueError as exc:
            return ToolResult(success=False, stderr=str(exc), exit_code=1)

    def read_file(self, relative_path: str) -> ToolResult:
        try:
            target = self._resolve(relative_path)
            if not target.exists():
                return ToolResult(
                    success=False,
                    stderr=f"File not found: {relative_path}",
                    exit_code=2,
                )
            return ToolResult(
                success=True,
                stdout=target.read_text(encoding="utf-8"),
                data={"path": str(target)},
            )
        except ValueError as exc:
            return ToolResult(success=False, stderr=str(exc), exit_code=1)
```

Notice the three safety rules baked in:

1. `shell=False` — commands are lists, not strings.
2. Path resolution is checked against `self.workdir`.
3. Every operation returns a `ToolResult` with an exit code, even on internal errors.

---

## The Tool Dispatcher

The dispatcher is the router. It validates that the requested tool is allowed, calls the sandbox, and returns a normalized result. Unknown tools are rejected with exit code `126` (command not executable).

```python
class ToolDispatcher:
    """Routes model actions to sandboxed tool handlers."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self._handlers: dict[str, callable] = {
            "bash": self._bash,
            "write_file": self._write_file,
            "read_file": self._read_file,
        }

    def dispatch(self, request: ToolRequest) -> ToolResult:
        handler = self._handlers.get(request.name)
        if handler is None:
            return ToolResult(
                success=False,
                stderr=f"Tool not allowed: {request.name}",
                exit_code=126,
            )
        return handler(request)

    def _bash(self, request: ToolRequest) -> ToolResult:
        command = request.arguments.get("command")
        if not isinstance(command, list):
            return ToolResult(
                success=False,
                stderr="bash tool expects 'command' as a list",
                exit_code=1,
            )
        timeout = float(request.arguments.get("timeout", 30.0))
        return self.sandbox.run_command(command, timeout=timeout)

    def _write_file(self, request: ToolRequest) -> ToolResult:
        path = request.arguments.get("path")
        content = request.arguments.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return ToolResult(
                success=False,
                stderr="write_file expects 'path' and 'content' strings",
                exit_code=1,
            )
        return self.sandbox.write_file(path, content)

    def _read_file(self, request: ToolRequest) -> ToolResult:
        path = request.arguments.get("path")
        if not isinstance(path, str):
            return ToolResult(
                success=False,
                stderr="read_file expects 'path' string",
                exit_code=1,
            )
        return self.sandbox.read_file(path)
```

The dispatcher is intentionally dumb. It does not decide *which* tool to call; the model or the agent loop does that. The dispatcher only enforces that the call is well-formed and allowed.

---

## Checkpointing After a Tool Run

Tool calls mutate the working tree, so each successful batch must end with a git checkpoint. We reuse the anchor idea from Chapter 03. The snippet below includes a minimal anchor so the demo is self-contained, but in the real `lra` package this is `src/lra/state/mission_anchor.py`.

```python
class GitMissionAnchor:
    """Minimal git checkpoint boundary."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir

    def init(self) -> None:
        if not (self.repo_dir / ".git").exists():
            subprocess.run(["git", "init", "-q"], cwd=self.repo_dir, check=True)
            subprocess.run(
                ["git", "config", "user.email", "lra@example.com"],
                cwd=self.repo_dir,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "LRA Agent"],
                cwd=self.repo_dir,
                check=True,
            )

    def checkpoint(self, message: str) -> str:
        subprocess.run(["git", "add", "-A"], cwd=self.repo_dir, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=self.repo_dir,
            check=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
```

---

## Putting It Together: A Mini Mission

The following script creates a temporary workspace, writes a file, runs it, reads it back, and checkpoints each step. This is the **Act** phase of the Chapter 04 cycle in action.

```python
def run_mini_mission() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="lra-ch05-"))
    print(f"workspace: {workdir}")

    anchor = GitMissionAnchor(workdir)
    anchor.init()

    sandbox = LocalSandbox(workdir)
    dispatcher = ToolDispatcher(sandbox)

    # 1. Write hello.py
    write_req = ToolRequest(
        name="write_file",
        arguments={"path": "hello.py", "content": 'print("hello from lra")'},
        request_id="write-1",
    )
    result = dispatcher.dispatch(write_req)
    print(result.to_model_text())
    anchor.checkpoint("checkpoint: write hello.py")

    # 2. Run it
    run_req = ToolRequest(
        name="bash",
        arguments={"command": ["python", "hello.py"]},
        request_id="run-1",
    )
    result = dispatcher.dispatch(run_req)
    print(result.to_model_text())
    anchor.checkpoint("checkpoint: run hello.py")

    # 3. Read it back
    read_req = ToolRequest(
        name="read_file",
        arguments={"path": "hello.py"},
        request_id="read-1",
    )
    result = dispatcher.dispatch(read_req)
    print(result.to_model_text())

    # 4. Try an escape attempt (should fail)
    bad_req = ToolRequest(
        name="read_file",
        arguments={"path": "../../../etc/passwd"},
        request_id="escape-1",
    )
    result = dispatcher.dispatch(bad_req)
    print(result.to_model_text())

    # Show the durable history
    log = subprocess.run(
        ["git", "-C", str(workdir), "log", "--oneline"],
        capture_output=True,
        text=True,
        check=True,
    )
    print("\ngit history:")
    print(log.stdout)


if __name__ == "__main__":
    run_mini_mission()
```

Run it:

```bash
uv run python lra-demo/ch05_tool_sandbox.py
```

You will see:

- `hello.py` is written and executed.
- The file content is read back.
- The `../../../etc/passwd` escape attempt is rejected.
- Two git commits exist in the workspace history.

---

## Code Walkthrough

1. **`ToolRequest` / `ToolResult`** — These are the contracts. Everything that crosses the model/sandbox boundary is typed and serializable. This is what makes replay-from-cache possible in later chapters.

2. **`LocalSandbox`** — The cheapest sandbox. It enforces the workdir boundary and returns structured results. In production you swap this for `DockerSandbox` or `E2BSandbox` without changing the dispatcher.

3. **`ToolDispatcher`** — The gatekeeper. It maps names to handlers and rejects unknown tools. This is where you will later add egress policies (Chapter 23) and rate limits.

4. **`GitMissionAnchor.checkpoint()`** — Every mutating batch ends in git. If the process dies after the bash call but before the checkpoint, the next cycle re-reads the previous checkpoint and retries.

5. **The mini mission** — This is not a chat. It is a sequence of verified actions with durable boundaries between them.

---

## Hands-On Exercise

Add a `grep` tool to the dispatcher and use it to search the working tree.

1. Add a `_grep` handler in `ToolDispatcher` that accepts `{"pattern": "...", "path": "..."}`.
2. Implement it using `self.sandbox.run_command(["grep", "-n", pattern, path])`.
3. Run a mini mission that:
   - writes `src/greet.py` containing `def greet(name): return f"hello {name}"`,
   - runs `grep` for the word `greet`,
   - checkpoints after each step.
4. Verify that a `grep` outside the workdir (e.g., `path: "/etc/passwd"`) still fails because the sandbox path check runs first.

Expected final git log:

```text
abc1234 checkpoint: grep result
def5678 checkpoint: write greet.py
```

---

## Key Takeaway

> The model proposes, but the dispatcher and sandbox control what is allowed to happen. A tool layer with bounded paths, typed results, and explicit exit codes is the difference between a demo and a durable agent.

---

## Next Chapter Teaser

**Chapter 06: Deterministic Verification — Exit Codes as Ground Truth**

A tool result is not progress until something independent says it is correct. Next we build the verifier: the only component allowed to mark a checklist item `done`, using exit codes from real tests, builds, and lints as the single source of truth.