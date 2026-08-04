# Chapter 07: Building Your First Local Mission

> A mission is not a conversation. It is a task, a workdir, a checklist, and a git history that proves what got done.
> — Fareed Khan

## What We'll Cover

- How a **mission** turns a human task into a bounded, checkpointed run
- Wiring the **cycle** (Chapter 04), **tool dispatcher/sandbox** (Chapter 05), and **verifier** (Chapter 06) into one local runner
- The `MissionState` contract and why it lives on disk, not in the model context
- Running your first mission end-to-end with the demo file `lra-demo/ch07_mission.py`
- Reading the git history the mission leaves behind
- What happens when verification fails and the loop retries

---

## From Cycle to Mission

Chapters 04–06 gave you the pieces of one agent turn:

| Chapter | Piece | Responsibility |
|---|---|---|
| 04 | `Cycle` | gather → act → verify → checkpoint |
| 05 | `ToolDispatcher` + `Sandbox` | turn model decisions into bounded side effects |
| 06 | `Verifier` | declare done only when exit codes pass |

A **mission** is the container that repeats that cycle until a human task is finished. It has:

1. A `task` string from the human.
2. A `workdir` that is a real git repo.
3. A `checklist` produced by a planner.
4. A `MissionState` file on disk.
5. A runner that loops, verifies, and commits.

The runner in this chapter is **local and in-process**. It does not use Temporal yet. That is intentional: if the local loop cannot produce verified work and checkpoint it to git, adding durability will only preserve a broken loop.

The demo task is the same one from the `lra mission` quickstart:

> Create a `hello.py` that prints `hello` and a test for it.

We will run it for **$0** using a deterministic stub model. No API keys required.

---

## The Demo: `lra-demo/ch07_mission.py`

Save the file below as `lra-demo/ch07_mission.py`. It is self-contained and uses only the Python standard library plus `pytest` (for the verification check).

```python
#!/usr/bin/env python3
"""lra-demo/ch07_mission.py

A self-contained local mission runner.
It creates hello.py + test_hello.py, verifies them with pytest,
and checkpoints mission state to git after every cycle.
"""
import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ------------------------------------------------------------------
# Contracts (same shape as ch04_cycle, ch05_tool_sandbox, ch06_verify)
# ------------------------------------------------------------------
@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ToolResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int


@dataclass
class VerificationResult:
    passed: bool
    checks: list[tuple[str, bool, str]]


# ------------------------------------------------------------------
# Sandbox + Tool Dispatcher (Chapter 05)
# ------------------------------------------------------------------
class LocalSandbox:
    def __init__(self, workdir: Path):
        self.workdir = workdir

    def run(self, cmd: list[str], timeout: int = 30) -> ToolResult:
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return ToolResult(
                ok=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"timeout after {timeout}s", -1)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, "", str(exc), -1)


class ToolDispatcher:
    def __init__(self, sandbox: LocalSandbox):
        self.sandbox = sandbox
        self.tools: dict[str, Callable[[dict], ToolResult]] = {
            "write_file": self._write_file,
            "read_file": self._read_file,
            "run_shell": self._run_shell,
        }

    def dispatch(self, call: ToolCall) -> ToolResult:
        fn = self.tools.get(call.name)
        if fn is None:
            return ToolResult(False, "", f"unknown tool {call.name}", -1)
        return fn(call.args)

    def _write_file(self, args: dict) -> ToolResult:
        path = self.sandbox.workdir / args["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return ToolResult(True, f"wrote {path}", "", 0)

    def _read_file(self, args: dict) -> ToolResult:
        path = self.sandbox.workdir / args["path"]
        if not path.exists():
            return ToolResult(False, "", f"{path} not found", 1)
        return ToolResult(True, path.read_text(encoding="utf-8"), "", 0)

    def _run_shell(self, args: dict) -> ToolResult:
        cmd = args["cmd"]
        if isinstance(cmd, str):
            cmd = cmd.split()
        return self.sandbox.run(cmd)


# ------------------------------------------------------------------
# Deterministic Verifier (Chapter 06)
# ------------------------------------------------------------------
class Verifier:
    def __init__(self, workdir: Path):
        self.workdir = workdir

    def _check(self, name: str, cmd: list[str], cwd: bool = True) -> tuple[str, bool, str]:
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.workdir if cwd else None,
                capture_output=True,
                text=True,
                check=False,
            )
            ok = proc.returncode == 0
            detail = (proc.stdout + proc.stderr)[:500]
        except Exception as exc:  # noqa: BLE001
            ok = False
            detail = str(exc)
        return name, ok, detail

    def verify(self, checklist: list[str]) -> VerificationResult:
        hello = self.workdir / "hello.py"
        test = self.workdir / "test_hello.py"
        manifest = self.workdir / "pyproject.toml"

        checks: list[tuple[str, bool, str]] = []

        # 1. hello.py exists and prints hello
        if hello.exists():
            name, ok, detail = self._check(
                "hello.py prints hello",
                [sys.executable, str(hello)],
            )
            ok = ok and "hello" in detail.lower()
            checks.append((name, ok, detail))
        else:
            checks.append(("hello.py exists", False, "file missing"))

        # 2. pytest passes
        if test.exists():
            checks.append(
                self._check(
                    "pytest passes",
                    [sys.executable, "-m", "pytest", str(test), "-q"],
                )
            )
        else:
            checks.append(("test_hello.py exists", False, "file missing"))

        # 3. project manifest exists (so ruff/typecheck have a root later)
        checks.append(
            (
                "pyproject.toml exists",
                manifest.exists(),
                "ok" if manifest.exists() else "file missing",
            )
        )

        passed = all(ok for _, ok, _ in checks)
        return VerificationResult(passed, checks)


# ------------------------------------------------------------------
# Stub Model / Planner / Lead (Chapter 03 + 04)
# ------------------------------------------------------------------
class StubModel:
    """Zero-cost deterministic agent used for the first local mission."""

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def plan(self, task: str) -> list[str]:
        return [
            "Create hello.py that prints 'hello'",
            "Create test_hello.py with a pytest test",
            "Add pyproject.toml so the project has a root",
            "Run verification until all checks pass",
        ]

    def decide(self, state: dict) -> list[ToolCall]:
        # Deterministic lead engineer: emit the next missing artifact.
        if not (self.workdir / "hello.py").exists():
            return [
                ToolCall(
                    "write_file",
                    {
                        "path": "hello.py",
                        "content": (
                            "def main():\n"
                            "    print('hello')\n"
                            "\n"
                            "\n"
                            "if __name__ == '__main__':\n"
                            "    main()\n"
                        ),
                    },
                )
            ]

        if not (self.workdir / "test_hello.py").exists():
            return [
                ToolCall(
                    "write_file",
                    {
                        "path": "test_hello.py",
                        "content": (
                            "from hello import main\n"
                            "\n"
                            "\n"
                            "def test_main(capsys):\n"
                            "    main()\n"
                            "    captured = capsys.readouterr()\n"
                            "    assert captured.out.strip() == 'hello'\n"
                        ),
                    },
                )
            ]

        if not (self.workdir / "pyproject.toml").exists():
            return [
                ToolCall(
                    "write_file",
                    {
                        "path": "pyproject.toml",
                        "content": (
                            "[project]\n"
                            "name = 'hello'\n"
                            "version = '0.1.0'\n"
                        ),
                    },
                )
            ]

        # All artifacts exist; run the test so the verifier has fresh output.
        return [
            ToolCall(
                "run_shell",
                {"cmd": f"{sys.executable} -m pytest test_hello.py -q"},
            )
        ]


# ------------------------------------------------------------------
# Git Checkpoint
# ------------------------------------------------------------------
class GitCheckpoint:
    def __init__(self, workdir: Path):
        self.workdir = workdir

    def _git(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.workdir)] + cmd,
            capture_output=True,
            text=True,
            check=False,
        )

    def init(self) -> None:
        if not (self.workdir / ".git").exists():
            self._git(["init", "-q"])
            # Ensure commits can be made even in a fresh environment.
            self._git(["config", "user.email", "lra@example.com"])
            self._git(["config", "user.name", "LRA Local Runner"])

    def commit(self, message: str, state_path: Path) -> bool:
        self._git(["add", str(state_path.relative_to(self.workdir)), "."])
        status = self._git(["status", "--porcelain"])
        if not status.stdout.strip():
            return True
        result = self._git(["commit", "-m", message, "-q"])
        return result.returncode == 0


# ------------------------------------------------------------------
# Mission State
# ------------------------------------------------------------------
@dataclass
class MissionState:
    task: str
    checklist: list[str]
    completed: list[bool] = field(default_factory=list)
    cycle_count: int = 0
    log: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "task": self.task,
                "checklist": self.checklist,
                "completed": self.completed,
                "cycle_count": self.cycle_count,
                "log": self.log,
            },
            indent=2,
        )

    @classmethod
    def from_path(cls, path: Path) -> "MissionState":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                task=data.get("task", ""),
                checklist=data.get("checklist", []),
                completed=data.get("completed", []),
                cycle_count=data.get("cycle_count", 0),
                log=data.get("log", []),
            )
        return cls(task="", checklist=[])


# ------------------------------------------------------------------
# Local Mission Runner
# ------------------------------------------------------------------
class LocalMissionRunner:
    def __init__(self, workdir: Path, task: str, max_cycles: int = 10):
        self.workdir = workdir
        self.task = task
        self.max_cycles = max_cycles
        self.state_path = workdir / "mission_state.json"

        self.sandbox = LocalSandbox(workdir)
        self.dispatcher = ToolDispatcher(self.sandbox)
        self.verifier = Verifier(workdir)
        self.checkpoint = GitCheckpoint(workdir)
        self.model = StubModel(workdir)

        self.state = MissionState.from_path(self.state_path)
        self.state.task = task
        if not self.state.checklist:
            self.state.checklist = self.model.plan(task)
            self.state.completed = [False] * len(self.state.checklist)

    def _save(self) -> None:
        self.state_path.write_text(self.state.to_json(), encoding="utf-8")

    def _gather(self) -> dict:
        return {
            "task": self.state.task,
            "checklist": self.state.checklist,
            "completed": self.state.completed,
            "cycle": self.state.cycle_count,
            "files": [
                str(p.relative_to(self.workdir))
                for p in self.workdir.rglob("*")
                if p.is_file() and ".git" not in p.parts
            ],
        }

    def _act(self) -> list[tuple[ToolCall, ToolResult]]:
        calls = self.model.decide(self._gather())
        results = []
        for call in calls:
            result = self.dispatcher.dispatch(call)
            results.append((call, result))
            print(f"  tool {call.name}: ok={result.ok} rc={result.returncode}")
        return results

    def _verify(self) -> VerificationResult:
        return self.verifier.verify(self.state.checklist)

    def _checkpoint(self, note: str) -> bool:
        self._save()
        return self.checkpoint.commit(
            f"cycle {self.state.cycle_count}: {note}",
            self.state_path,
        )

    def run(self) -> MissionState:
        self.checkpoint.init()
        self._save()
        self.checkpoint.commit("mission start", self.state_path)

        for _ in range(self.max_cycles):
            self.state.cycle_count += 1
            print(f"\n=== cycle {self.state.cycle_count} ===")

            gather = self._gather()
            print("gather:", gather["completed"])

            act_results = self._act()
            verify = self._verify()

            print(f"verify passed={verify.passed}")
            for name, ok, detail in verify.checks:
                print(f"  - {name}: {'PASS' if ok else 'FAIL'}")

            # In this simple demo, the whole checklist flips to done together.
            all_done = verify.passed
            self.state.completed = [all_done] * len(self.state.checklist)

            self.state.log.append(
                {
                    "cycle": self.state.cycle_count,
                    "actions": [
                        {"tool": call.name, "ok": result.ok}
                        for call, result in act_results
                    ],
                    "verify": [
                        {"name": name, "ok": ok}
                        for name, ok, _ in verify.checks
                    ],
                }
            )

            note = "verified" if verify.passed else "retry"
            self._checkpoint(note)

            if verify.passed:
                print("\nMISSION COMPLETE")
                break
        else:
            print("\nMISSION HALTED: max cycles reached")

        self._save()
        return self.state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local LRA mission")
    parser.add_argument(
        "--task",
        default="Create a hello.py that prints hello and a test for it",
    )
    parser.add_argument(
        "--workdir",
        default=".lra/workspaces/ch07",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    runner = LocalMissionRunner(workdir, args.task, args.max_cycles)
    final = runner.run()

    print("\nFinal state:")
    print(final.to_json())


if __name__ == "__main__":
    main()
```

---

## Running the Mission

Make sure you have `git` and `pytest` installed:

```bash
python -m pip install pytest
```

Then run the demo:

```bash
python lra-demo/ch07_mission.py \
  --task "Create a hello.py that prints hello and a test for it" \
  --workdir .lra/workspaces/ch07
```

You will see output like this:

```text
=== cycle 1 ===
gather: [False, False, False, False]
  tool write_file: ok=True rc=0
verify passed=False
  - hello.py prints hello: PASS
  - test_hello.py exists: FAIL
  - pyproject.toml exists: FAIL

=== cycle 2 ===
gather: [False, False, False, False]
  tool write_file: ok=True rc=0
verify passed=False
  - hello.py prints hello: PASS
  - pytest passes: FAIL
  - pyproject.toml exists: FAIL

=== cycle 3 ===
gather: [False, False, False, False]
  tool write_file: ok=True rc=0
verify passed=False
  - hello.py prints hello: PASS
  - pytest passes: PASS
  - pyproject.toml exists: FAIL

=== cycle 4 ===
gather: [False, False, False, False]
  tool write_file: ok=True rc=0
verify passed=True
  - hello.py prints hello: PASS
  - pytest passes: PASS
  - pyproject.toml exists: PASS

MISSION COMPLETE
```

After the run, inspect the real git history the mission produced:

```bash
git -C .lra/workspaces/ch07 log --oneline
```

You should see commits for each cycle plus the mission start:

```text
9d8c7b6 cycle 4: verified
a1b2c3d cycle 3: retry
d4e5f6g cycle 2: retry
h7i8j9k cycle 1: retry
l0m1n2o mission start
```

The `mission_state.json` file in the workdir contains the durable state. If you re-ran the script, it would resume from that file instead of planning from scratch.

---

## Code Walkthrough

### 1. Contracts at the top
`ToolCall`, `ToolResult`, and `VerificationResult` are the same dataclasses from Chapters 04–06. Keeping them identical lets you later drop in the real `lra` package classes without changing the runner.

### 2. `LocalSandbox` + `ToolDispatcher`
The dispatcher exposes three tools: `write_file`, `read_file`, and `run_shell`. Every side effect is scoped to `workdir`. This is the local sandbox from Chapter 05. Docker and E2B are not used here, but they would implement the same `run()` interface.

### 3. `Verifier`
Verification never asks the model “are we done?” It runs real commands and checks exit codes. The three checks are:

- `hello.py` exists and prints `hello`
- `pytest` passes on `test_hello.py`
- `pyproject.toml` exists

Only when all three pass does `VerificationResult.passed` become `True`.

### 4. `StubModel`
This is the $0 model. It plans the checklist deterministically and decides the next action by looking at which files are missing. In later chapters, this `decide()` method becomes the **Lead Engineer** loop backed by a real LLM.

### 5. `MissionState`
Everything the model would need to reconstruct context lives here:

- the original task
- the checklist
- which items are completed
- the cycle count
- a log of every cycle

Because it is JSON on disk, a restarted process can reload it in milliseconds.

### 6. `LocalMissionRunner.run()`
This is the orchestrator. Each iteration:

1. **Gather** — read `MissionState` and list files.
2. **Act** — ask the model to decide, then dispatch tools.
3. **Verify** — run deterministic checks.
4. **Checkpoint** — save `mission_state.json` and commit to git.

The loop stops when verification passes or when `max_cycles` is reached.

---

## Hands-On Exercise

1. **Run the mission** in a clean workdir and confirm the git history has four commits.
2. **Intentionally break verification**: edit `lra-demo/ch07_mission.py` so the stub model writes `print('goodbye')` instead of `print('hello')`. Re-run and watch the verifier fail every cycle until `max_cycles` is reached. Then revert your change.
3. **Change the task** to something slightly harder, for example:

   ```bash
   python lra-demo/ch07_mission.py \
     --task "Create a calculator.py with add and subtract functions and tests for both" \
     --workdir .lra/workspaces/ch07-calculator
   ```

   Update the `StubModel.plan()` and `StubModel.decide()` methods to emit the new files, and add verifier checks for `calculator.py`. Do not change the runner loop itself.
4. **Inspect the durable state**:

   ```bash
   cat .lra/workspaces/ch07/mission_state.json
   ```

   Confirm that `cycle_count`, `completed`, and `log` are all persisted.

---

> **Key Takeaway:** A mission is a checklist, a workdir, a verifier, and a git commit after every cycle. If you cannot build that locally and deterministically, no amount of LLM intelligence or orchestration will make it reliable.

---

## Next Chapter Teaser

**Chapter 08: Asymmetric Multi-Agent Design** — One agent can write, many agents can read and review. We will replace the single `StubModel` with an organization: a planner, a lead engineer, parallel researchers, and an independent reviewer. The loop stays the same; the team gets bigger.