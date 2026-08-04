# Chapter 24: Capturing Failure Traces

## What We'll Cover

- Why a week-long mission produces **too many failures to hold in a context window**
- What LRA considers a **failure trace** — a structured, durable, replayable record of a broken cycle
- The six failure kinds the system captures: **verification, tool crash, egress deny, HITL rejection, loop trip, and crash survival**
- How traces tie back to the **Mission Anchor** (Chapter 16), the **decision log**, and a git commit
- Why durable traces are the raw material for **offline evolution** (Chapter 25) and the **eval harness** (Chapter 26)
- A runnable demo in `lra-demo/ch24_failure_traces.py` that writes and inspects real traces

---

## The Concept: Failures Are Training Data

In a chat-style agent, a failed step is just another turn in the conversation. In a long-running mission, a failed step is a signal that has to survive for days, be understood by a different process, and ideally teach the system not to make the same mistake twice.

LRA treats failures as **first-class state**. Every time the deterministic verifier says no, a tool crashes, the loop detector trips, an egress policy blocks a request, a human rejects a gate, or the host reboots mid-cycle, the system writes a **failure trace** before it does anything else. That trace is JSON on disk, committed with the Mission Anchor, and re-read on the next cycle.

This works because the real state already lives outside the model (Chapter 3). The Mission Anchor (Chapter 16) gives every cycle a stable identity: a `mission_id`, a `cycle` number, and a git commit. The durable execution layer (Chapters 12–15) means a crash cannot erase a trace that was already flushed. The deterministic verifier (Chapter 6) gives the trace an objective ground-truth result: exit code, stdout, stderr. HITL gates (Chapter 22), egress policies (Chapter 23), and loop detection (Chapter 21) each add their own failure mode.

A trace without context is useless. A good trace contains:

| Field | Why it matters |
|---|---|
| `trace_id` / `mission_id` / `cycle` | Lets you find exactly which turn of which mission failed |
| `anchor_commit` | Lets you `git checkout` the repo state at the moment of failure |
| `checklist` | Shows what the agent believed was in progress |
| `action` | The tool, command, intent, and raw model output that produced the attempt |
| `verifier` | The deterministic result that declared the work not-done |
| `failure_kind` | A typed label: `VERIFICATION_FAILED`, `TOOL_CRASH`, `EGRESS_DENIED`, `HITL_REJECTED`, `LOOP_DETECTED`, `CRASH_SURVIVED` |
| `exception` | The actual error class/message when available |
| `resolution` | What finally happened: `fixed_by_repair`, `scope_cut`, `escalated_to_hitl`, `rejected`, `resumed`, `unresolved` |
| `linked_traces` | Pointers to earlier/later traces for the same root cause |

The trace is written **before** the cycle retries, escalates, or aborts. If the process dies immediately after the failure, the next process resumes from the durable activity journal and finds the trace already on disk.

---

## Code Walkthrough: `lra-demo/ch24_failure_traces.py`

The demo below is a self-contained script. It creates a small workspace, runs a few intentionally broken actions, and writes one trace per failure. It then loads the traces back and prints a summary.

```python
#!/usr/bin/env python3
"""lra-demo/ch24_failure_traces.py

Capture durable failure traces for a long-running agent mission.
Run with: python lra-demo/ch24_failure_traces.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_DIR = Path(".lra/traces/ch24_demo")
WORK_DIR = Path(".lra/workspaces/ch24_demo")


@dataclass
class ActionAttempt:
    tool: str
    command: str
    intent: str
    model_raw: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    egress_decision: str = "ALLOW"  # ALLOW / DENY / HITL (Chapter 23)


@dataclass
class VerifierResult:
    kind: str
    exit_code: int
    stdout: str
    stderr: str
    passed: bool


@dataclass
class FailureTrace:
    trace_id: str
    mission_id: str
    cycle: int
    timestamp: str
    anchor_commit: str
    checklist: list[str]
    action: ActionAttempt
    verifier: VerifierResult | None
    failure_kind: str
    exception: str | None
    resolution: str
    linked_traces: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_trace(trace: FailureTrace) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{trace.trace_id}.json"
    # Atomic-ish: write to a temp file, then rename, so a crash mid-write
    # leaves the previous trace intact (Chapter 15).
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(trace.to_json(), indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_traces(mission_id: str | None = None) -> list[FailureTrace]:
    traces: list[FailureTrace] = []
    if not TRACE_DIR.exists():
        return traces
    for p in sorted(TRACE_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        action = ActionAttempt(**data["action"])
        verifier = VerifierResult(**data["verifier"]) if data["verifier"] else None
        traces.append(FailureTrace(
            trace_id=data["trace_id"],
            mission_id=data["mission_id"],
            cycle=data["cycle"],
            timestamp=data["timestamp"],
            anchor_commit=data["anchor_commit"],
            checklist=data["checklist"],
            action=action,
            verifier=verifier,
            failure_kind=data["failure_kind"],
            exception=data["exception"],
            resolution=data["resolution"],
            linked_traces=data.get("linked_traces", []),
        ))
    if mission_id:
        traces = [t for t in traces if t.mission_id == mission_id]
    return traces


def run_tool(command: list[str], cwd: Path) -> ActionAttempt:
    """Run a local tool and return an ActionAttempt (Chapter 5)."""
    try:
        proc = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=10
        )
        return ActionAttempt(
            tool=command[0],
            command=" ".join(command),
            intent="execute tool for cycle",
            model_raw="",
            stdout=proc.stdout[-2000:] if proc.stdout else "",
            stderr=proc.stderr[-2000:] if proc.stderr else "",
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        return ActionAttempt(
            tool=command[0],
            command=" ".join(command),
            intent="execute tool for cycle",
            model_raw="",
            stdout=exc.stdout[-2000:] if exc.stdout else "",
            stderr="timeout",
            exit_code=None,
            exception="TimeoutExpired",
        )
    except Exception as exc:
        return ActionAttempt(
            tool=command[0],
            command=" ".join(command),
            intent="execute tool for cycle",
            model_raw="",
            stdout="",
            stderr=str(exc),
            exit_code=None,
            exception=type(exc).__name__,
        )


def verify_with_pytest(cwd: Path) -> VerifierResult:
    """Deterministic verifier from Chapter 6: exit code is ground truth."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return VerifierResult(
        kind="pytest",
        exit_code=proc.returncode,
        stdout=proc.stdout[-3000:],
        stderr=proc.stderr[-3000:],
        passed=proc.returncode == 0,
    )


def capture(
    mission_id: str,
    cycle: int,
    anchor_commit: str,
    checklist: list[str],
    action: ActionAttempt,
    verifier: VerifierResult | None,
    failure_kind: str,
    exception: str | None = None,
    resolution: str = "unresolved",
    linked: list[str] | None = None,
) -> FailureTrace:
    trace = FailureTrace(
        trace_id=str(uuid.uuid4()),
        mission_id=mission_id,
        cycle=cycle,
        timestamp=now(),
        anchor_commit=anchor_commit,
        checklist=list(checklist),
        action=action,
        verifier=verifier,
        failure_kind=failure_kind,
        exception=exception,
        resolution=resolution,
        linked_traces=linked or [],
    )
    write_trace(trace)
    return trace


def setup_workspace() -> Path:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)
    (WORK_DIR / "hello.py").write_text(
        'def hello():\n    return "hello"\n', encoding="utf-8"
    )
    (WORK_DIR / "test_hello.py").write_text(
        'from hello import hello\n\ndef test_hello():\n    assert hello() == "world"\n',
        encoding="utf-8",
    )
    return WORK_DIR


def summarize(mission_id: str) -> None:
    traces = load_traces(mission_id)
    print(f"\nMission {mission_id}: {len(traces)} failure trace(s)")
    for t in traces:
        status = "✓ resolved" if t.resolution != "unresolved" else "✗ unresolved"
        print(
            f"  cycle={t.cycle:02d} kind={t.failure_kind:20s} "
            f"{status} commit={t.anchor_commit}"
        )
        if t.verifier and not t.verifier.passed:
            print(f"           verifier={t.verifier.kind} exit={t.verifier.exit_code}")
        if t.linked_traces:
            print(f"           linked={t.linked_traces}")


def main() -> None:
    mission_id = "ch24-demo"
    anchor_commit = "9d8c7b6"  # pretend Mission Anchor commit (Chapter 16)
    checklist = ["implement hello", "add passing test"]

    if TRACE_DIR.exists():
        shutil.rmtree(TRACE_DIR)
    workspace = setup_workspace()

    # 1. Verification failure: tests fail (Chapter 6).
    action = run_tool([sys.executable, "-m", "pytest", "-q"], workspace)
    verifier = verify_with_pytest(workspace)
    verify_trace = capture(
        mission_id=mission_id,
        cycle=1,
        anchor_commit=anchor_commit,
        checklist=checklist,
        action=action,
        verifier=verifier,
        failure_kind="VERIFICATION_FAILED",
        resolution="unresolved",
    )

    # 2. Tool crash: command not found.
    action = run_tool(["not_a_real_tool", "--help"], workspace)
    capture(
        mission_id=mission_id,
        cycle=2,
        anchor_commit=anchor_commit,
        checklist=checklist,
        action=action,
        verifier=None,
        failure_kind="TOOL_CRASH",
        exception=action.exception,
        resolution="unresolved",
    )

    # 3. Egress deny (Chapter 23): network request blocked by default-deny policy.
    action = ActionAttempt(
        tool="curl",
        command="curl https://example.com",
        intent="download dependency",
        model_raw="",
        stdout="",
        stderr="blocked by default-deny egress policy",
        exit_code=None,
        egress_decision="DENY",
    )
    egress_trace = capture(
        mission_id=mission_id,
        cycle=3,
        anchor_commit=anchor_commit,
        checklist=checklist,
        action=action,
        verifier=None,
        failure_kind="EGRESS_DENIED",
        resolution="unresolved",
    )

    # 4. HITL rejection (Chapter 22): push to main rejected by human supervisor.
    action = ActionAttempt(
        tool="git",
        command="git push origin main",
        intent="publish completed work",
        model_raw="",
        stdout="",
        stderr="HITL gate rejected: push to main requires approval",
        exit_code=1,
    )
    capture(
        mission_id=mission_id,
        cycle=4,
        anchor_commit=anchor_commit,
        checklist=checklist,
        action=action,
        verifier=None,
        failure_kind="HITL_REJECTED",
        resolution="rejected",
    )

    # 5. Loop detected (Chapter 21): same egress-denied action attempted again.
    action = ActionAttempt(
        tool="curl",
        command="curl https://example.com",
        intent="download dependency (retry)",
        model_raw="",
        stdout="",
        stderr="blocked by default-deny egress policy",
        exit_code=None,
        egress_decision="DENY",
    )
    capture(
        mission_id=mission_id,
        cycle=5,
        anchor_commit=anchor_commit,
        checklist=checklist,
        action=action,
        verifier=None,
        failure_kind="LOOP_DETECTED",
        resolution="escalated_to_hitl",
        linked=[egress_trace.trace_id],
    )

    # 6. Crash survived (Chapter 15): a trace written after an interrupted cycle.
    capture(
        mission_id=mission_id,
        cycle=6,
        anchor_commit=anchor_commit,
        checklist=checklist,
        action=ActionAttempt(
            tool="unknown",
            command="interrupted activity",
            intent="resume after crash",
            model_raw="",
            stdout="",
            stderr="process was killed; resumed from durable activity journal",
            exit_code=None,
        ),
        verifier=None,
        failure_kind="CRASH_SURVIVED",
        resolution="resumed",
    )

    # Mark the verification failure as fixed after a hypothetical repair.
    verify_trace.resolution = "fixed_by_repair"
    write_trace(verify_trace)

    summarize(mission_id)


if __name__ == "__main__":
    main()
```

Run it:

```bash
python lra-demo/ch24_failure_traces.py
```

You will see six traces written to `.lra/traces/ch24_demo/`. Each file is a standalone JSON document that contains the full context of the failure, not just the error message.

A few things to notice in the code:

- **`write_trace` uses a temp-file rename.** This is the same durability idea as the Mission Anchor: never overwrite the canonical file in place, because a crash in the middle would corrupt the trace.
- **The verifier is separate from the action.** `ActionAttempt` records what the model tried; `VerifierResult` records what the test said. Keeping them separate lets you replay a failure even when the model output is ambiguous.
- **`linked_traces` connects related failures.** The loop trace points back to the original egress-deny trace, so a reviewer or evolver can see the full chain.
- **`anchor_commit` is a string in the demo, but in the real system it is the git SHA from the Mission Anchor.** That SHA is what lets you reproduce the failure later.

---

## Hands-On Exercise

1. Run `python lra-demo/ch24_failure_traces.py` and inspect one of the JSON files in `.lra/traces/ch24_demo/`. Confirm that `anchor_commit`, `checklist`, and `verifier` are present for the `VERIFICATION_FAILED` trace.

2. Add a new failure kind, `LINT_FAILED`, to the demo. Create a small Python file in `WORK_DIR` that violates a simple lint rule (for example, an unused import), run `ruff check .` or `python -m py_compile` as the verifier, and capture the trace.

3. Implement a `resolve(trace_id: str, resolution: str)` helper in `lra-demo/ch24_failure_traces.py` that loads an existing trace, updates its `resolution`, and writes it back. Use it to mark the `LINT_FAILED` trace as `fixed_by_repair` after you correct the code.

4. (Optional) Make the script commit the trace directory to a local git repo on every `write_trace`, so the failure history is part of the Mission Anchor. The commit message can include the `mission_id` and `cycle`.

---

> **Key Takeaway:** A failure you cannot replay is a failure you cannot learn from. In LRA, every failed cycle becomes a durable trace — tied to a git commit, a checklist state, and a verifier result — so the system, and the humans behind it, can diagnose, evolve, and improve.

---

**Next Chapter:** Chapter 25 — *Offline Evolution: Propose, Evaluate, Promote*. We will take the failure traces captured here, feed them into an evolver agent, and build a self-improving skill library that makes the next mission cheaper and more reliable.