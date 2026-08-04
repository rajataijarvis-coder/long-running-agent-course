# Chapter 11: Reviewer and Reflection Agents

> A reviewer with fresh context is the guardrail that keeps a single writer honest; a reflection agent is the recovery mechanism that turns a block into a plan.
> — Fareed Khan

## What We'll Cover

- Why the **Lead Engineer** from Chapter 10 still needs an independent, fresh-context **Reviewer**
- The **Reviewer** role: read-only adversarial audit that can block progress without writing code
- The **Reflection** role: diagnosing blocks and verification failures from the **decision log** and **blackboard**
- The contracts that connect them: `ReviewTask`, `ReviewReport`, `ReflectionTask`, `ReflectionReport`
- How reviewer/reflection fit into the **gather → act → verify → checkpoint** cycle from Chapter 04
- A runnable demo in `lra-demo/ch11_reviewer_reflection.py` that shows a reviewer blocking unsafe code and a reflection agent guiding recovery

---

## The Problem with a Single Writer Checking Its Own Work

Chapter 10 gave the **Lead Engineer** sole write access to coupled code. That prevents merge chaos, but it creates a new risk: the same agent that wrote the code is also interpreting test results and deciding what to do next. Its context window is contaminated by its own reasoning, so it can miss obvious mistakes, rationalize flaky tests, or silently weaken acceptance criteria.

The asymmetric organization from Chapter 08 solves this by adding two read-only agents:

1. **Researchers** — fan out to read the codebase and return facts (Chapter 09).
2. **Reviewer** — gets a fresh, minimal context and is asked one question: *should this change land?*
3. **Reflection agent** — only wakes up after a block or verification failure to answer: *why did we fail, and what is the smallest next step?*

Neither the reviewer nor the reflection agent writes code. They only write structured reports. That preserves the single-writer invariant while giving the Lead Engineer a clean signal to act on.

### Why fresh context matters

The context window is a lossy cache (Chapter 02). The Lead Engineer carries hours of reasoning, failed attempts, and partial hypotheses. A reviewer loaded with that same history will make the same blind spots. The production reviewer in `src/lra/agents/reviewer.py` is therefore invoked with:

- the current git diff,
- the checklist item being addressed,
- the acceptance criteria,
- the decision log,
- and nothing else.

It does not see the Lead Engineer's chain-of-thought. That adversarial distance is the point.

### Reflection is not just "try again"

When deterministic verification fails (Chapter 06) or the reviewer blocks, the worst thing the Lead Engineer can do is immediately rewrite the same file with the same mental model. The **Reflection agent** in `src/lra/agents/reflection.py` is a forced pause. It reads:

- the decision log,
- the blackboard,
- the verifier output,
- the review report,
- and the current file contents,

then emits a `ReflectionReport` with:

- `root_cause`,
- `hypothesis`,
- `next_step`,
- `risks`.

Only after the Lead Engineer consumes that report does the next act cycle begin.

---

## The Contracts

The production interfaces live in `src/lra/contracts/review.py`. The demo below mirrors them with plain dataclasses so you can run it without a Temporal server or an LLM backend.

```python
# lra-demo/ch11_reviewer_reflection.py
"""Reviewer + Reflection demo.

The Lead Engineer writes code, a fresh-context Reviewer audits it, and a
Reflection agent diagnoses the block. No agent except the Lead writes files.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


# ---------- Contracts (mirror src/lra/contracts/review.py) ----------
class Severity(str, Enum):
    BLOCKING = "blocking"
    ADVISORY = "advisory"


class Verdict(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"


@dataclass
class DecisionEvent:
    cycle: int
    agent: str
    action: str
    rationale: str


@dataclass
class ReviewTask:
    workspace: Path
    checklist_item: str
    files_changed: list[str]
    decision_log: list[DecisionEvent]
    acceptance_criteria: list[str]


@dataclass
class ReviewReport:
    verdict: Verdict
    severity: Severity
    findings: list[str]
    requested_changes: list[str] = field(default_factory=list)


@dataclass
class ReflectionTask:
    trigger: Literal["review_block", "verify_fail"]
    decision_log: list[DecisionEvent]
    review_report: ReviewReport | None
    test_output: str
    files: dict[str, str]


@dataclass
class ReflectionReport:
    root_cause: str
    hypothesis: str
    next_step: str
    risks: list[str]
```

### Deterministic verifier

The demo reuses the exit-code ground-truth pattern from Chapter 06. A checklist item is not done until `pytest` returns `0`.

```python
# ---------- Deterministic verifier (Chapter 06) ----------
def verify(workspace: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr
```

### Lead Engineer: first attempt

The Lead writes a happy-path `add()` function. It forgets input validation, which the acceptance criteria require.

```python
# ---------- Lead Engineer (Chapter 10) ----------
def lead_write_first_attempt(workspace: Path) -> list[str]:
    calc = workspace / "calculator.py"
    calc.write_text(textwrap.dedent('''\
        def add(a, b):
            return a + b
    '''))

    test = workspace / "test_calculator.py"
    test.write_text(textwrap.dedent('''\
        import pytest
        from calculator import add

        def test_add_numbers():
            assert add(2, 3) == 5

        def test_add_rejects_strings():
            with pytest.raises(TypeError):
                add("2", "3")
    '''))

    return ["calculator.py", "test_calculator.py"]
```

### Reviewer: fresh-context audit

The reviewer inspects the source, the tests, and the acceptance criteria. It does not see the Lead's reasoning. In production this would be an LLM prompt; here we use explicit rules so the demo is deterministic and reproducible.

```python
# ---------- Reviewer agent (src/lra/agents/reviewer.py) ----------
def reviewer_audit(task: ReviewTask) -> ReviewReport:
    files = {p: (task.workspace / p).read_text() for p in task.files_changed}
    findings: list[str] = []
    requested: list[str] = []

    # Check acceptance criteria against the implementation.
    for criterion in task.acceptance_criteria:
        if "numeric" in criterion.lower() or "TypeError" in criterion:
            src = files.get("calculator.py", "")
            if "TypeError" not in src or "isinstance" not in src:
                findings.append(
                    f"calculator.py does not validate inputs; "
                    f"acceptance criterion '{criterion}' is not met."
                )
                requested.append("Raise TypeError for non-numeric inputs in add().")

    # Check that tests exercise the criterion.
    test_src = files.get("test_calculator.py", "")
    if "TypeError" not in test_src:
        findings.append("No test exercises the TypeError acceptance criterion.")
        requested.append("Add a test that asserts TypeError for invalid inputs.")

    if findings:
        return ReviewReport(
            verdict=Verdict.REQUEST_CHANGES,
            severity=Severity.BLOCKING,
            findings=findings,
            requested_changes=requested,
        )

    return ReviewReport(
        verdict=Verdict.APPROVE,
        severity=Severity.ADVISORY,
        findings=[],
    )
```

### Reflection agent: diagnose the block

When the reviewer blocks, the Reflection agent produces a structured recovery plan. It reads the review report and the verifier output, not the Lead's internal monologue.

```python
# ---------- Reflection agent (src/lra/agents/reflection.py) ----------
def reflection_diagnose(task: ReflectionTask) -> ReflectionReport:
    # In production this is an LLM call. Here we emulate the structured output
    # it would produce by inspecting the failure signals.
    if task.trigger == "review_block" and task.review_report:
        causes = task.review_report.findings
    else:
        causes = [task.test_output]

    return ReflectionReport(
        root_cause="; ".join(causes),
        hypothesis="The lead implemented happy-path addition but forgot input validation.",
        next_step=(
            "Add an isinstance check at the top of add() and raise TypeError; "
            "the existing test already expects this behavior."
        ),
        risks=[
            "Over-validating could break legitimate numeric subclasses; "
            "consider numbers.Real or duck-typing in a real project."
        ],
    )
```

### Lead Engineer: repair from the reflection report

The Lead is the only agent that writes files. It consumes the reflection report and rewrites `calculator.py`.

```python
def lead_repair(workspace: Path, report: ReflectionReport) -> list[str]:
    calc = workspace / "calculator.py"
    calc.write_text(textwrap.dedent('''\
        def add(a, b):
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise TypeError("add() expects numeric arguments")
            return a + b
    '''))
    return ["calculator.py"]
```

### Decision log and checkpoint

The decision log is part of the coordination layer from Chapter 08. Every significant action is recorded before the next cycle begins. After approval, the state is checkpointed to git (Chapter 03).

```python
def log_event(log: list[DecisionEvent], cycle: int, agent: str, action: str, rationale: str):
    log.append(DecisionEvent(cycle, agent, action, rationale))


def checkpoint(workspace: Path, message: str):
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=workspace, check=True)
```

### Main loop

The loop below is a local simulation of the durable cycle we will move to Temporal in Chapter 12.

```python
def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)

        log: list[DecisionEvent] = []
        log_event(
            log, 1, "lead", "write_first_attempt",
            "Implement add() and a test for numeric input.",
        )

        files = lead_write_first_attempt(workspace)
        ok, output = verify(workspace)
        print("First verification:", "PASS" if ok else "FAIL")
        print(output)

        # Reviewer audits with fresh context.
        review_task = ReviewTask(
            workspace=workspace,
            checklist_item="Implement add() with type safety",
            files_changed=files,
            decision_log=log,
            acceptance_criteria=[
                "add() returns the sum of two numbers",
                "add() raises TypeError for non-numeric inputs",
            ],
        )
        review = reviewer_audit(review_task)
        print("\nReviewer verdict:", review.verdict.value, review.severity.value)
        for finding in review.findings:
            print(" -", finding)

        if review.verdict == Verdict.REQUEST_CHANGES:
            log_event(log, 2, "reviewer", "request_changes", "; ".join(review.findings))

            reflection_task = ReflectionTask(
                trigger="review_block",
                decision_log=log,
                review_report=review,
                test_output=output,
                files={p: (workspace / p).read_text() for p in files},
            )
            reflection = reflection_diagnose(reflection_task)
            print("\nReflection report:")
            print(json.dumps(reflection.__dict__, indent=2))

            log_event(log, 3, "reflection", "propose_fix", reflection.next_step)
            files = lead_repair(workspace, reflection)
            log_event(log, 4, "lead", "repair", reflection.next_step)

            ok, output = verify(workspace)
            print("\nAfter repair:", "PASS" if ok else "FAIL")
            print(output)

            if not ok:
                print("Repair failed; mission would escalate to human.")
                return 1

            # Re-review after repair.
            review_task = ReviewTask(
                workspace=workspace,
                checklist_item="Implement add() with type safety",
                files_changed=files,
                decision_log=log,
                acceptance_criteria=[
                    "add() returns the sum of two numbers",
                    "add() raises TypeError for non-numeric inputs",
                ],
            )
            review = reviewer_audit(review_task)
            print("\nRe-review verdict:", review.verdict.value)

        if review.verdict == Verdict.APPROVE:
            checkpoint(workspace, "ch11: verified add() with type safety")
            print("\nCheckpoint committed.")
            print("Final decision log:")
            for ev in log:
                print(f"  cycle={ev.cycle} agent={ev.agent} action={ev.action}")
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Run it from the repo root:

```bash
uv run python lra-demo/ch11_reviewer_reflection.py
```

You should see the first verification fail, the reviewer request changes, the reflection agent propose a fix, the Lead repair the code, the re-review approve, and a git checkpoint land.

---

## Hands-On Exercise

Extend `lra-demo/ch11_reviewer_reflection.py` with a second checklist item:

1. Add a `subtract(a, b)` function to `calculator.py`.
2. Add an acceptance criterion: *"subtract must not silently accept non-numeric inputs; it should raise TypeError."*
3. Introduce a bug that **passes pytest** but violates a design rule from the decision log — for example, make `subtract` catch `Exception` and return `0` instead of letting the TypeError propagate.
4. Update the reviewer to inspect the decision log and block the change because it contradicts the project's "fail fast" rule, even though tests are green.
5. Run the reflection agent and have it propose a fix that removes the broad exception handler.

This mirrors the production scenario where verification alone is not enough: the reviewer must also enforce design coherence recorded in the decision log.

---

> **Key Takeaway:** The Reviewer and Reflection agents are read-only by design. They do not write code, they write structured signals — a block, a finding, a recovery plan — that the single Lead Engineer consumes before its next act cycle. That separation is what stops long-horizon missions from drifting into silent, self-reinforcing failure.

---

## Next Chapter

In **Chapter 12: Temporal Workflows for Long-Running Agents**, we move the local `main()` loop into a durable Temporal workflow so a host reboot on day three resumes exactly where the Reviewer left off — no work lost, no tokens re-spent.