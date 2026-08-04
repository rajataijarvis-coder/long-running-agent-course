# Chapter 06: Deterministic Verification — Exit Codes as Ground Truth

> The model proposes. A deterministic process with an exit code closes.
> — Fareed Khan

## What We'll Cover

- Why the model must **never** be allowed to declare its own work done
- The `Verifier` contract: `Check` + `VerificationResult`
- Using real subprocess exit codes (`pytest`, `ruff`, `mypy`, `compileall`) as the single source of "done"
- Composite verification: a checklist item passes only when **every** registered check passes
- Trust bootstrap: proving your verifier is trustworthy by injecting intentional failures
- A runnable demo in `lra-demo/ch06_verify.py`

---

## The Problem: Models Hallucinate Success

In Chapter 04 we introduced the agent cycle:

```
gather → act → verify → checkpoint
```

The **Act** phase changes the world (writes code, runs commands). The **Verify** phase asks a separate, non-LLM question: *did the change actually work?*

This separation is non-negotiable. A model can say "the task is complete" while the tests are red, the module fails to import, or the file was never written. Over hundreds of cycles, that kind of self-graded homework compounds into a broken codebase and a wasted budget.

The fix is to make verification **deterministic** and **external**:

- A check is a command that runs in a real process.
- The command returns an exit code.
- Exit code `0` means pass. Anything else means fail.
- A checklist item is marked `done` only when **all** of its checks pass.

This is the "ground truth" that the article's final trace relies on:

```
2026-06-02T08:44:51.880Z INFO  mission.complete items=14/14 cycles=615 head=9d8c7b6 status=DONE
```

`items=14/14` means every checklist item survived deterministic verification. Not one of them was closed by a model saying "looks good".

---

## The Verifier Contract

In the real `lra` package the contract lives in `src/lra/contracts/verify.py` and the implementation lives in `src/lra/verify/deterministic.py`. The shape is intentionally tiny:

```python
@dataclass(frozen=True)
class Check:
    name: str
    command: list[str]
    cwd: Path | None = None
    timeout: float = 30.0
    exit_ok: tuple[int, ...] = (0,)

@dataclass(frozen=True)
class VerificationResult:
    check: str
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
```

A `Verifier` takes a list of `Check` objects and returns a list of `VerificationResult` objects plus an overall pass/fail boolean. That boolean is what the agent loop from Chapter 04 feeds into `CycleOutcome.verified`.

---

## Runnable Demo: `lra-demo/ch06_verify.py`

Create `lra-demo/ch06_verify.py`. It is self-contained: it builds a tiny project in a temporary directory, runs deterministic checks, injects a bug to show failure, and runs a trust-bootstrap mutation test.

```python
"""lra-demo/ch06_verify.py

Deterministic verification using exit codes as ground truth.
Run with: python lra-demo/ch06_verify.py
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class Check:
    """One deterministic command that must exit with an acceptable code."""

    name: str
    command: list[str]
    cwd: Path | None = None
    timeout: float = 30.0
    exit_ok: tuple[int, ...] = (0,)


@dataclasses.dataclass(frozen=True)
class VerificationResult:
    """The outcome of running one Check."""

    check: str
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int


class DeterministicVerifier:
    """Runs checks as real subprocesses. Exit codes are the only truth."""

    def run(self, checks: Iterable[Check]) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        for check in checks:
            cwd = check.cwd or Path.cwd()
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    check.command,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=check.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(
                    args=exc.cmd,
                    returncode=-1,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                )
            duration_ms = int((time.perf_counter() - start) * 1000)
            passed = proc.returncode in check.exit_ok
            results.append(
                VerificationResult(
                    check=check.name,
                    passed=passed,
                    exit_code=proc.returncode,
                    stdout=proc.stdout[-4000:],
                    stderr=proc.stderr[-4000:],
                    duration_ms=duration_ms,
                )
            )
        return results

    def verify(self, checks: Iterable[Check]) -> tuple[bool, list[VerificationResult]]:
        results = self.run(checks)
        return all(r.passed for r in results), results


class CompositeMissionVerifier:
    """Maps each checklist item to a list of checks that must all pass."""

    def __init__(
        self,
        registry: dict[str, list[Check]],
        base: DeterministicVerifier | None = None,
    ) -> None:
        self.registry = registry
        self.base = base or DeterministicVerifier()

    def verify_item(self, item_id: str, cwd: Path) -> tuple[bool, list[VerificationResult]]:
        checks = self.registry.get(item_id, [])
        # Resolve relative cwd against the mission workspace.
        checks = [dataclasses.replace(c, cwd=c.cwd or cwd) for c in checks]
        return self.base.verify(checks)


class TrustBootstrap:
    """Proves a verifier is trustworthy by checking that it fails on broken code."""

    def __init__(self, verifier: DeterministicVerifier) -> None:
        self.verifier = verifier

    def is_trusted(
        self,
        checks: list[Check],
        source_file: Path,
        sentinel: str = "# MUTATION",
    ) -> tuple[bool, str]:
        if not source_file.exists():
            return False, f"source file not found: {source_file}"
        original = source_file.read_text()
        mutant = original + f"\n{sentinel}\nraise RuntimeError('mutant injected by trust bootstrap')\n"
        try:
            source_file.write_text(mutant)
            passed, _ = self.verifier.verify(checks)
        finally:
            source_file.write_text(original)
        if passed:
            return False, "checks passed on intentionally broken code — not trustworthy"
        return True, "checks caught the mutant"


def setup_demo_project(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "hello.py").write_text(
        textwrap.dedent(
            """\
            def hello() -> str:
                return "Hello, world!"
            """
        )
    )
    (workspace / "test_hello.py").write_text(
        textwrap.dedent(
            """\
            from hello import hello

            def test_hello() -> None:
                assert hello() == "Hello, world!"
            """
        )
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="lra_ch06_") as tmp:
        workspace = Path(tmp)

        # 1. Build a tiny project.
        setup_demo_project(workspace)

        # 2. Register deterministic checks for the "implement hello" item.
        checks: list[Check] = [
            Check(
                name="py_compile",
                command=[sys.executable, "-m", "py_compile", "hello.py"],
                cwd=workspace,
            ),
            Check(
                name="import_assert",
                command=[
                    sys.executable,
                    "-c",
                    "from hello import hello; assert hello() == 'Hello, world!'",
                ],
                cwd=workspace,
            ),
            Check(
                name="pytest",
                command=[sys.executable, "-m", "pytest", "test_hello.py", "-q"],
                cwd=workspace,
            ),
        ]

        verifier = DeterministicVerifier()

        # 3. Run verification on good code.
        print("=== First run: good code ===")
        passed, results = verifier.verify(checks)
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.check} (exit={r.exit_code}, {r.duration_ms}ms)")
        print(f"overall: {'PASS' if passed else 'FAIL'}\n")

        # 4. Inject a bug and re-run to show the verifier catches it.
        print("=== Second run: broken code ===")
        hello_path = workspace / "hello.py"
        original = hello_path.read_text()
        hello_path.write_text(original.replace('"Hello, world!"', '"Goodbye, world!"'))
        passed, results = verifier.verify(checks)
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.check} (exit={r.exit_code})")
            if not r.passed and r.stderr:
                print(f"      stderr: {r.stderr.strip()[:200]}")
        print(f"overall: {'PASS' if passed else 'FAIL'}\n")
        hello_path.write_text(original)

        # 5. Trust bootstrap: prove the checks would fail on a mutant.
        print("=== Trust bootstrap ===")
        bootstrap = TrustBootstrap(verifier)
        trusted, message = bootstrap.is_trusted(checks, hello_path)
        print(f"trusted: {trusted} — {message}\n")

        # 6. CompositeMissionVerifier: map checklist items to checks.
        print("=== Composite mission verifier ===")
        registry = {"item-1-hello": checks}
        composite = CompositeMissionVerifier(registry)
        passed, results = composite.verify_item("item-1-hello", workspace)
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
        print(f"item-1-hello overall: {'PASS' if passed else 'FAIL'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### What the demo proves

1. **Good code passes.** `py_compile`, an import assertion, and `pytest` all return exit code `0`.
2. **Broken code fails.** Changing the return string makes the assertion check and `pytest` fail with non-zero exit codes.
3. **The verifier is trustworthy.** The `TrustBootstrap` injects a runtime error into `hello.py`, runs the same checks, and confirms they catch the mutation. If they had passed, the verifier would be untrustworthy and the mission could not safely continue.
4. **Composite verification scales.** `CompositeMissionVerifier` maps checklist item IDs to lists of checks, so the agent loop from Chapter 04 can ask: *is item X done?* and get a single boolean back.

---

## Wiring Verification into the Cycle

Recall the `AgentLoop` from `src/lra/agent/loop.py` in Chapter 04. After the **Act** phase runs one or more tools, the loop calls the verifier:

```python
# Simplified excerpt from the cycle
checks = build_checks_for_item(current_item)
passed, results = self.verifier.verify(checks)

if passed:
    self.anchor.mark_done(current_item.id, results)
else:
    self.anchor.record_attempt(current_item.id, results)
```

Only `anchor.mark_done` updates the mission checklist. The model is never asked "are we done yet?". It is asked "what tool should I run next?" while the verifier owns the done-state.

This is why the final mission trace can claim `status=DONE`. Every one of those 14 items passed real, deterministic checks.

---

## Hands-On Exercise

1. Open `lra-demo/ch06_verify.py`.
2. Add a new checklist item `item-2-greet` in the `registry` that requires:
   - a `greet(name: str) -> str` function in `hello.py`
   - a test in `test_hello.py` asserting `greet("Ada") == "Hello, Ada!"`
   - a `ruff check .` or `python -m compileall .` check
3. Introduce a bug in `greet` and run the script. Confirm the composite verifier returns `FAIL`.
4. Fix the bug and rerun. Confirm `item-2-greet` is `PASS`.
5. Run the trust bootstrap against `hello.py` and verify it reports `trusted: true`.

If `pytest` is not installed, replace the `pytest` check with another `python -c "..."` assertion. The point is not which tool you use; the point is that **every check must be a real command with a real exit code**.

---

> **Key Takeaway:** A checklist item is only done when a deterministic process with a zero exit code says it is done. The model proposes; the verifier closes. Anything else is just a chatbot grading its own homework.

---

## Next Chapter

**Chapter 07: Building Your First Local Mission** — We will wire the gather/act/verify/checkpoint cycle, the tool dispatcher and sandbox from Chapter 05, and the deterministic verifier from this chapter into a single `lra mission` command that plans a task, works the checklist, and survives a process restart.