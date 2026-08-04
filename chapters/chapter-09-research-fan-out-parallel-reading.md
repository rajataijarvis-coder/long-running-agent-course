# Chapter 09: Research Fan-out (Parallel Reading)

> Reading scales horizontally; writing does not. Fan-out is safe only when every parallel agent is read-only.
> — Fareed Khan

## What We'll Cover

- Why the **Gather** phase from Chapter 04 is the only part of the cycle that can safely run in parallel
- The read-only **Researcher** role in the asymmetric organization from Chapter 08
- The `ResearchTask` → `ResearchReport` contract and how reports land on the **blackboard**
- Dispatching many researchers concurrently with `asyncio` and the **tool dispatcher/sandbox** from Chapter 05
- A runnable demo in `lra-demo/ch09_research_fanout.py` that fans out three researchers over a small codebase
- How the **Lead Engineer** consumes the merged research before it writes anything

---

## The Concept

In Chapter 08 we established the asymmetric rule: **one writer, many readers, one fresh reviewer**. The Lead Engineer is the single writer because coupled code changes must stay coherent. But reading—exploring the codebase, scanning dependencies, grepping for patterns, fetching documentation—is almost entirely **embarrassingly parallel**. Different files do not interfere with each other when they are only being read.

Research fan-out is the system exploiting that parallelism during the **Gather** phase of the `gather → act → verify → checkpoint` cycle from Chapter 04. Before the Lead Engineer decides what to change, it asks a pool of Researcher agents to answer focused questions:

- Where is the parser currently implemented?
- Which tests already cover the LSP initialize request?
- What does the `pygls` documentation say about `workspace/symbol`?
- Are there any existing type stubs we can reuse?

Each Researcher receives a `ResearchTask`, runs read-only tools inside a sandbox (Chapter 05), and returns a `ResearchReport`. Reports are posted to the **blackboard** (Chapter 08), a shared read-only structure. The Lead Engineer reads the blackboard, not the researchers directly, and then enters the **Act** phase alone.

This design is deliberately restrictive:

1. **Researchers never write files.** They cannot create branches, edit code, or touch git.
2. **Researchers never call the verifier.** Verification belongs to the Lead Engineer's act-and-checkpoint cycle.
3. **Researchers share state only through the blackboard.** There is no direct agent-to-agent messaging that could create hidden coupling.
4. **Egress is default-deny.** A web-reading researcher is allowed only specific hosts (Chapter 23). A local researcher is allowed only specific paths inside the workdir.

The result is that the system can spin up ten researchers at once without risking the coherence of the mission. If one researcher crashes or hallucinates a file path, the others continue, and the Lead Engineer can request a retry.

---

## The Contracts

The real implementation lives in `src/lra/agents/researcher.py` and `src/lra/coordination/blackboard.py`. For this chapter we use a simplified but runnable version saved as `lra-demo/ch09_research_fanout.py`.

```python
# lra-demo/ch09_research_fanout.py
"""Research fan-out demo: read-only parallel exploration of a codebase.

Run with:
    uv run python lra-demo/ch09_research_fanout.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# --------------------------------------------------------------------------- #
# Contracts (simplified versions of src/lra/contracts/research.py)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    question: str
    # Read-only tools this researcher is allowed to invoke.
    allowed_tools: tuple[str, ...]
    # Where to look. In production this is constrained by the sandbox.
    search_paths: tuple[str, ...]


@dataclass(frozen=True)
class SourceCitation:
    path: str
    line: int
    snippet: str


@dataclass
class ResearchReport:
    task_id: str
    question: str
    answer: str
    citations: list[SourceCitation] = field(default_factory=list)
    # Read-only agents can still fail; the lead needs to know.
    error: str | None = None

    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------- #
# Sandbox-style read-only tool dispatcher
# --------------------------------------------------------------------------- #

class ReadOnlyToolBox:
    """A tiny version of src/lra/execution/tool_dispatcher.py.

    Every call is a read-only subprocess so a crash cannot corrupt the workdir.
    """

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    def read_file(self, rel_path: str) -> str:
        target = (self.workdir / rel_path).resolve()
        # Default-deny: refuse to leave the workdir.
        if not str(target).startswith(str(self.workdir.resolve())):
            raise PermissionError(f"Path outside workdir: {rel_path}")
        if not target.exists():
            raise FileNotFoundError(rel_path)
        return target.read_text()

    def grep(self, pattern: str, glob: str = "*.py") -> list[SourceCitation]:
        matches: list[SourceCitation] = []
        for path in self.workdir.rglob(glob):
            for i, line in enumerate(path.read_text().splitlines(), start=1):
                if pattern in line:
                    matches.append(
                        SourceCitation(
                            path=str(path.relative_to(self.workdir)),
                            line=i,
                            snippet=line.strip(),
                        )
                    )
        return matches

    def run_pytest_collect(self) -> list[str]:
        """Collect test names without running them. Read-only exploration."""
        result = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", "-q"],
            cwd=self.workdir,
            capture_output=True,
            text=True,
        )
        # We still treat a non-zero exit as information, not mission failure.
        return [
            line.strip()
            for line in (result.stdout + result.stderr).splitlines()
            if line.strip() and not line.startswith("=")
        ]


# --------------------------------------------------------------------------- #
# The Researcher agent
# --------------------------------------------------------------------------- #

class Researcher:
    """A read-only agent. It explores, reports, and stops. It never writes."""

    def __init__(self, agent_id: str, toolbox: ReadOnlyToolBox) -> None:
        self.agent_id = agent_id
        self.toolbox = toolbox

    async def run(self, task: ResearchTask) -> ResearchReport:
        # Simulate model "thinking" time and any network latency.
        await asyncio.sleep(0.1)

        try:
            if task.task_id == "find_parser":
                citations = self.toolbox.grep("class Parser", "*.py")
                return ResearchReport(
                    task_id=task.task_id,
                    question=task.question,
                    answer=f"Found {len(citations)} parser class(es).",
                    citations=citations[:5],
                )

            if task.task_id == "list_tests":
                tests = self.toolbox.run_pytest_collect()
                return ResearchReport(
                    task_id=task.task_id,
                    question=task.question,
                    answer=f"Collected {len(tests)} test item(s).",
                    citations=[
                        SourceCitation(path="pytest", line=0, snippet=t)
                        for t in tests[:5]
                    ],
                )

            if task.task_id == "doc_lookup":
                # In production this would be a sandboxed web fetch with an egress allowlist.
                text = self.toolbox.read_file("README.md")
                return ResearchReport(
                    task_id=task.task_id,
                    question=task.question,
                    answer="README contains the project overview.",
                    citations=[
                        SourceCitation(
                            path="README.md",
                            line=1,
                            snippet=text.splitlines()[0],
                        )
                    ],
                )

            return ResearchReport(
                task_id=task.task_id,
                question=task.question,
                answer="",
                error=f"Unknown task type: {task.task_id}",
            )

        except Exception as exc:
            return ResearchReport(
                task_id=task.task_id,
                question=task.question,
                answer="",
                error=f"{type(exc).__name__}: {exc}",
            )


# --------------------------------------------------------------------------- #
# The blackboard: shared, append-only research output
# --------------------------------------------------------------------------- #

class Blackboard:
    """Simplified version of src/lra/coordination/blackboard.py."""

    def __init__(self) -> None:
        self._reports: list[ResearchReport] = []

    def post(self, report: ResearchReport) -> None:
        self._reports.append(report)

    def by_task(self, task_id: str) -> ResearchReport | None:
        for r in self._reports:
            if r.task_id == task_id:
                return r
        return None

    def all_ok(self) -> bool:
        return all(r.ok() for r in self._reports)

    def summary(self) -> str:
        lines = [f"Blackboard has {len(self._reports)} report(s):"]
        for r in self._reports:
            status = "OK" if r.ok() else f"ERR: {r.error}"
            lines.append(f"  [{status}] {r.task_id}: {r.answer}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Fan-out orchestrator
# --------------------------------------------------------------------------- #

async def fan_out_research(
    workdir: Path,
    tasks: list[ResearchTask],
    make_researcher: Callable[[str], Researcher],
) -> Blackboard:
    toolbox = ReadOnlyToolBox(workdir)
    blackboard = Blackboard()

    async def execute(task: ResearchTask) -> None:
        researcher = make_researcher(f"researcher-{task.task_id}")
        report = await researcher.run(task)
        blackboard.post(report)

    # This is the fan-out: many read-only explorations in parallel.
    await asyncio.gather(*(execute(t) for t in tasks))
    return blackboard


# --------------------------------------------------------------------------- #
# Demo setup
# --------------------------------------------------------------------------- #

def create_demo_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# Tiny LSP\n\nA minimal language server prototype.\n"
    )
    (root / "parser.py").write_text(
        "class Parser:\n    def parse(self, src: str) -> list:\n        return src.split()\n"
    )
    (root / "server.py").write_text(
        "class LanguageServer:\n    def initialize(self):\n        return {'capabilities': {}}\n"
    )
    tests = root / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test_parser.py").write_text(
        "from parser import Parser\n\ndef test_parse():\n    assert Parser().parse('a b') == ['a', 'b']\n"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        create_demo_repo(workdir)

        tasks = [
            ResearchTask(
                task_id="find_parser",
                question="Where is the parser class defined?",
                allowed_tools=("grep",),
                search_paths=("*.py",),
            ),
            ResearchTask(
                task_id="list_tests",
                question="Which tests exist in the repo?",
                allowed_tools=("pytest_collect",),
                search_paths=("tests/",),
            ),
            ResearchTask(
                task_id="doc_lookup",
                question="What does the README say the project is?",
                allowed_tools=("read_file",),
                search_paths=("README.md",),
            ),
        ]

        def make_researcher(agent_id: str) -> Researcher:
            return Researcher(agent_id, ReadOnlyToolBox(workdir))

        print("Fanning out research...")
        blackboard = await fan_out_research(workdir, tasks, make_researcher)

        print(blackboard.summary())
        print(f"\nAll reports healthy: {blackboard.all_ok()}")

        # The Lead Engineer would now read the blackboard and enter Act.
        parser_report = blackboard.by_task("find_parser")
        if parser_report and parser_report.ok():
            print(
                "\nLead Engineer sees:",
                parser_report.answer,
                f"(citations: {len(parser_report.citations)})",
            )


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Code Walkthrough

### 1. Read-only contracts

`ResearchTask` and `ResearchReport` are frozen and dataclass-based stand-ins for the Pydantic contracts in `src/lra/contracts/research.py`. A task carries a question and the **allowed tools**. A report carries an answer plus `SourceCitation` objects so the Lead Engineer can verify claims against real files.

### 2. The toolbox is the sandbox boundary

`ReadOnlyToolBox` is a minimal version of `src/lra/execution/tool_dispatcher.py`. Every operation is either a local file read, a `grep`, or a subprocess call that is known to be read-only (`pytest --collect-only`). The `read_file` method resolves the path and refuses anything outside the workdir—an early preview of the default-deny egress policy we will build in Chapter 23.

### 3. The Researcher never writes

`Researcher.run` answers a question using only the toolbox. It has no `write_file`, no `git commit`, and no `verify` call. If a tool fails, it returns a report with `error` set. This is important: **a failed researcher must not crash the mission**. The Lead Engineer can decide to retry, skip, or reformulate the question.

### 4. Fan-out with `asyncio.gather`

`fan_out_research` creates one `Researcher` per task and runs them concurrently with `asyncio.gather`. Because every task is read-only, there is no locking, no merge conflict, and no race condition. The results are posted to a shared `Blackboard`.

### 5. The blackboard decouples producers from consumers

The blackboard is append-only. Researchers post to it; the Lead Engineer reads from it. This is the same coordination contract we introduced in Chapter 08. It prevents researchers from chatting directly with each other and creating implicit state.

### 6. The Lead Engineer consumes, then acts alone

After fan-out completes, the demo prints the blackboard summary and shows how the Lead Engineer would pull the parser report. In a real mission this is the hand-off from **Gather** to **Act**. The Lead Engineer, and only the Lead Engineer, will then edit files and run the verifier from Chapter 06.

---

## Hands-On Exercise

Save the file as `lra-demo/ch09_research_fanout.py` and run it:

```bash
uv run python lra-demo/ch09_research_fanout.py
```

Then make the following changes and observe the behavior:

1. **Add a fourth researcher.** Create a task `task_id="find_server"` that greps for `class LanguageServer`. Confirm the blackboard now has four reports and that the Lead Engineer can read all of them.

2. **Measure the concurrency.** Add `print` statements at the start and end of each `Researcher.run`. Run the script and verify that the end messages interleave, proving the researchers ran in parallel rather than sequentially.

3. **Introduce a path escape attempt.** Change the `doc_lookup` task to request `../README.md`. Confirm that `ReadOnlyToolBox.read_file` raises `PermissionError` and that the blackboard records the error without crashing the other researchers.

4. **Deduplicate citations.** Modify `ResearchReport` to include a `source_paths` property that returns the unique set of cited paths. Have the Lead Engineer print the union of all source paths discovered by the research pool.

5. **Retry a failed researcher.** After `fan_out_research`, collect reports with `error` set and re-run only those tasks once. Verify that transient failures can be recovered without restarting the whole mission.

---

## Key Takeaway

> Fan-out is a force multiplier for reading, not a shortcut for writing. Keep researchers read-only, post their findings to a shared blackboard, and let the single Lead Engineer decide what to change.

---

## Next Chapter Teaser

**Chapter 10: The Lead Engineer (Single Writer)** — Now that the research is gathered, we focus on the one agent allowed to touch the code. We will build the Lead Engineer's inner loop, its ownership of the `state/` directory, and how it turns a blackboard full of reports into verified commits.