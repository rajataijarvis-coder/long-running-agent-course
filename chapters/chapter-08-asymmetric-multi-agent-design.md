# Chapter 08: Asymmetric Multi-Agent Design

> One writer, many readers, one fresh reviewer. Parallelism is a privilege of read-only work.
> — Fareed Khan

## What We'll Cover

- Why "add more agents and let them all write" destroys coherence on long-horizon software missions
- The asymmetric organization from the article: **Lead Engineer**, **Researchers**, **Reviewer**, and **Integrator**
- How each role maps onto the `gather → act → verify → checkpoint` cycle from Chapter 04
- The coordination contracts that keep them from stepping on each other: the **blackboard**, **ownership map**, and **decision log**
- A runnable simulation in `lra-demo/ch08_asymmetric_agents.py` that shows a lead agent being blocked by a reviewer, then recovering
- Where the real implementation lives in `src/lra/agents/` and `src/lra/coordination/`

---

## The Problem with Symmetric Agents

By Chapter 07 you can run a single agent loop that writes files, runs tests, and checkpoints to git. That is enough for small tasks. But a week-long mission is not a small task. If you spin up five identical "coder" agents and let them all touch the same codebase, you get:

- **Merge conflicts** on coupled design decisions
- **Inconsistent abstractions** because no single agent owns the architecture
- **Wasted tokens** as agents redo each other's work
- **Blame diffusion** when something breaks

The article's answer is **asymmetric roles**. The model is not made smarter; the organization is made stricter:

| Role | Writes code? | Parallel? | Responsibility |
|---|---|---|---|
| **Planner** | No | No | Builds the checklist and ownership map |
| **Lead Engineer** | **Yes — sole writer** | No | Runs the inner `think → act → verify → checkpoint` loop |
| **Researchers** | No | **Yes** | Read the codebase, docs, and dependencies in parallel |
| **Reviewer** | No | No | Fresh-context adversarial check of every write |
| **Integrator** | No | No | Merges approved work to `main` and tags checkpoints |

This is why the package has `src/lra/agents/` for the actors and `src/lra/coordination/` for the shared contracts that keep them apart.

### Mapping to the cycle from Chapter 04

- **Gather**: Researchers fan out and return `ReadTicket`s. The Lead Engineer reads them before acting.
- **Act**: The Lead Engineer is the only agent that produces `WriteTicket`s and calls the tool dispatcher from Chapter 05.
- **Verify**: The deterministic verifier from Chapter 06 runs first; the Reviewer adds a second, model-based adversarial check.
- **Checkpoint**: The Integrator writes approved files and commits, exactly like the local mission runner in Chapter 07.

The real state still lives in git + the structured mission log. The agents are just different lenses on that state.

---

## The Contracts

In `src/lra/contracts/agent.py` the real system defines a small protocol family. The demo below uses a simplified version:

- `ReadTicket(query, files, summary)` — read-only research result
- `WriteTicket(item, plan, files)` — a proposed code change
- `ReviewResult(verdict, reason)` — `approve` or `block`
- `Blackboard` — the shared decision log and ownership map

The Lead Engineer owns every path it writes. Researchers own nothing. The Reviewer owns nothing. The Integrator only mutates the tree after approval. That separation is what makes the system inspectable and recoverable.

---

## Demo: `lra-demo/ch08_asymmetric_agents.py`

This file simulates a single checklist item — *Implement `calc.add(a, b)` and tests* — across four agents. The Lead Engineer deliberately writes a buggy first draft so you can see the Reviewer block it, the loop retry, and the Integrator commit only the fixed version.

```python
# lra-demo/ch08_asymmetric_agents.py
"""Asymmetric multi-agent simulation.

One Lead Engineer writes. Two Researchers read in parallel. One Reviewer
verifies with real subprocess exit codes. One Integrator commits approved work.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


class Verdict(enum.StrEnum):
    APPROVE = "approve"
    BLOCK = "block"


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class ReadTicket:
    query: str
    files: list[str]
    summary: str


@dataclasses.dataclass
class WriteTicket:
    item: str
    plan: str
    files: dict[str, str]  # relative path -> proposed content
    review: ReviewResult | None = None


@dataclasses.dataclass(frozen=True)
class ReviewResult:
    verdict: Verdict
    reason: str


# --------------------------------------------------------------------------- #
# Coordination surface (simplified from src/lra/coordination/blackboard.py)
# --------------------------------------------------------------------------- #

class Blackboard:
    """Shared state + decision log. In production this is backed by git state."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.reads: list[ReadTicket] = []
        self.writes: list[WriteTicket] = []
        self.ownership: dict[str, str] = {}  # path -> agent name
        self.log: list[dict] = []

    def post_read(self, ticket: ReadTicket) -> None:
        self.reads.append(ticket)
        self.log_event("researcher", "read", f"{ticket.query}: {ticket.summary[:80]}")

    def post_write(self, ticket: WriteTicket) -> None:
        self.writes.append(ticket)
        for path in ticket.files:
            self.ownership[path] = "lead"
        self.log_event("lead", "write", ticket.item)

    def log_event(self, agent: str, action: str, detail: str) -> None:
        self.log.append({"agent": agent, "action": action, "detail": detail})


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

class Researcher:
    """Read-only fan-out. Never mutates the working tree."""

    def __init__(self, name: str, board: Blackboard):
        self.name = name
        self.board = board

    async def research(self, query: str, paths: list[str]) -> ReadTicket:
        findings: list[str] = []
        for p in paths:
            full = self.board.workdir / p
            if full.exists():
                findings.append(f"{p}: {len(full.read_text().splitlines())} lines")
            else:
                findings.append(f"{p}: missing")
        summary = f"{self.name} saw " + "; ".join(findings)
        return ReadTicket(query=query, files=paths, summary=summary)


class LeadEngineer:
    """Sole writer for all coupled code. Owns design coherence."""

    def __init__(self, board: Blackboard, checklist: list[str]):
        self.board = board
        self.checklist = checklist
        self.completed: set[str] = set()
        self.attempts: dict[str, int] = {}

    async def cycle(self) -> WriteTicket | None:
        remaining = [i for i in self.checklist if i not in self.completed]
        if not remaining:
            return None
        item = remaining[0]
        self.attempts[item] = self.attempts.get(item, 0) + 1

        # Gather: use whatever the researchers have posted.
        reads = [r for r in self.board.reads if r.query == "current codebase"]
        context = "\n".join(r.summary for r in reads) or "no prior reads"

        # Act: produce a write ticket. First attempt is buggy to exercise the reviewer.
        if self.attempts[item] == 1:
            code = {
                "calc.py": "def add(a, b):\n    return a - b\n",
                "test_calc.py": "import calc\n\ndef test_add():\n    assert calc.add(2, 3) == 5\n",
            }
        else:
            code = {
                "calc.py": "def add(a, b):\n    return a + b\n",
                "test_calc.py": "import calc\n\ndef test_add():\n    assert calc.add(2, 3) == 5\n    assert calc.add(-1, 1) == 0\n",
            }

        ticket = WriteTicket(
            item=item,
            plan=f"Implement {item} based on reads:\n{context}",
            files=code,
        )
        self.board.post_write(ticket)
        return ticket


class Verifier:
    """Deterministic checks using real subprocess exit codes (Chapter 06)."""

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def run(self, ticket: WriteTicket) -> ReviewResult:
        scratch = self.workdir / ".lra_verify"
        if scratch.exists():
            for f in scratch.iterdir():
                f.unlink()
        else:
            scratch.mkdir()

        for path, content in ticket.files.items():
            (scratch / path).write_text(content)

        # 1. Syntax check
        for path in ticket.files:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(scratch / path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return ReviewResult(
                    Verdict.BLOCK,
                    f"syntax error in {path}: {result.stderr}",
                )

        # 2. Run tests via a generated unittest runner
        runner = scratch / "run_tests.py"
        runner.write_text(
            "import unittest, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "import calc\n\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_add(self): self.assertEqual(calc.add(2, 3), 5)\n"
            "    def test_add_neg(self): self.assertEqual(calc.add(-1, 1), 0)\n"
            "if __name__ == '__main__':\n"
            "    unittest.main(verbosity=2)\n"
        )
        result = subprocess.run(
            [sys.executable, str(runner)],
            cwd=scratch,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return ReviewResult(
                Verdict.BLOCK,
                f"tests failed:\n{result.stdout}\n{result.stderr}",
            )

        return ReviewResult(Verdict.APPROVE, "py_compile + tests passed")


class Reviewer:
    """Fresh-context adversarial reviewer. Only reads; verdict comes from Verifier."""

    def __init__(self, board: Blackboard):
        self.board = board
        self.verifier = Verifier(board.workdir)

    async def review(self, ticket: WriteTicket) -> ReviewResult:
        # The reviewer does NOT reuse the lead's plan. It re-derives the verdict
        # from deterministic checks and the proposed diff.
        result = self.verifier.run(ticket)
        self.board.log_event(
            "reviewer",
            result.verdict.value,
            result.reason[:120],
        )
        return result


class Integrator:
    """Merge approved writes to the working tree and checkpoint to git."""

    def __init__(self, board: Blackboard):
        self.board = board

    async def integrate(self, ticket: WriteTicket) -> None:
        for path, content in ticket.files.items():
            (self.board.workdir / path).write_text(content)
        self._git_checkpoint(ticket.item)

    def _git_checkpoint(self, message: str) -> None:
        wd = self.board.workdir
        env = os.environ.copy()
        env.update(
            {
                "GIT_AUTHOR_NAME": "lra-lead",
                "GIT_AUTHOR_EMAIL": "lead@lra.local",
                "GIT_COMMITTER_NAME": "lra-integrator",
                "GIT_COMMITTER_EMAIL": "integrator@lra.local",
            }
        )
        for cmd in (
            ["git", "init", "-q"],
            ["git", "add", "."],
            ["git", "commit", "-q", "-m", message],
        ):
            subprocess.run(cmd, cwd=wd, env=env, capture_output=True)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        board = Blackboard(workdir)

        lead = LeadEngineer(board, ["Implement calc.add(a, b) and tests"])
        researchers = [Researcher("r1", board), Researcher("r2", board)]
        reviewer = Reviewer(board)
        integrator = Integrator(board)

        for cycle in range(1, 5):
            print(f"\n--- cycle {cycle} ---")

            # Gather: researchers fan out in parallel.
            reads = await asyncio.gather(
                *(
                    r.research("current codebase", ["calc.py", "test_calc.py"])
                    for r in researchers
                )
            )
            for r in reads:
                board.post_read(r)

            # Act
            ticket = await lead.cycle()
            if ticket is None:
                print("all checklist items complete")
                break

            # Verify
            ticket.review = await reviewer.review(ticket)

            # Checkpoint only on approval
            if ticket.review.verdict == Verdict.APPROVE:
                await integrator.integrate(ticket)
                lead.completed.add(ticket.item)
                print(f"integrated: {ticket.item}")
            else:
                print(f"blocked: {ticket.review.reason}")

        print("\n--- decision log ---")
        print(json.dumps(board.log, indent=2))

        git_log = subprocess.run(
            ["git", "-C", str(workdir), "log", "--oneline"],
            capture_output=True,
            text=True,
        )
        if git_log.returncode == 0:
            print("\n--- git history ---")
            print(git_log.stdout)


if __name__ == "__main__":
    asyncio.run(main())
```

### How to run it

```bash
cd lra-demo
python ch08_asymmetric_agents.py
```

You should see:

- **Cycle 1**: Researchers report `calc.py` and `test_calc.py` are missing. The Lead writes a buggy `add` that subtracts. The Reviewer blocks it because `calc.add(2, 3)` returns `-1`.
- **Cycle 2**: Researchers now report the files exist. The Lead writes the correct version. The Reviewer approves. The Integrator commits.
- **Git history**: exactly one commit for the approved change, not the failed attempt.

The failed attempt is journaled in the blackboard log, but it never reaches `main`.

### Why this matters for the real package

In `src/lra/agents/` the same separation is enforced by types, not just convention:

- `LeadEngineer` is the only agent that emits `WriteTicket`.
- `Researcher` and `Reviewer` implement a read-only interface.
- `Integrator` checks the `ReviewResult` before calling `git commit`.

The `Blackboard` in `src/lra/coordination/blackboard.py` is backed by the mission state on disk, so a crash in cycle 2 resumes with the failed ticket already recorded and the correct next action known.

---

## Hands-On Exercise

1. Run `lra-demo/ch08_asymmetric_agents.py` and confirm the reviewer blocks the first cycle.
2. Add a second checklist item: `"Implement calc.mul(a, b) and tests"`. Make the Lead Engineer write it correctly on the first attempt. Observe how the Researchers' summaries change once `calc.py` exists.
3. **Break the design on purpose**: create a second `LeadEngineer` instance and let both produce `WriteTicket`s for the same file in the same cycle. Watch the ownership map disagree and the Integrator refuse to commit. This is why the article insists on a single writer.
4. Inspect the git history in the temporary directory (the script prints it). Confirm there is no commit for the blocked attempt.

---

> **Key Takeaway:** Long-horizon autonomy does not come from more agents writing in parallel. It comes from giving each agent a narrow, enforceable role and letting the system — not the model — decide what gets merged.

---

## Next Chapter

**Chapter 09: Research Fan-out (Parallel Reading)** — We will open up the **Gather** phase and show how to dispatch dozens of read-only research tasks in parallel, deduplicate overlapping queries, and cache their results so the Lead Engineer never pays for the same context twice.