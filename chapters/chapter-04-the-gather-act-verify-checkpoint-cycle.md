# Chapter 04: The Gather → Act → Verify → Checkpoint Cycle

> A model turn is a short burst. A cycle is a unit of verified progress. Never confuse the two.
> — Fareed Khan

## What We'll Cover

- How the four phases of one agent cycle replace the "chat → answer" pattern
- Why each phase has a single, testable responsibility
- The `CycleOutcome` contract from `src/lra/agent/loop.py`
- A runnable local implementation in `lra-demo/ch04_cycle.py`
- How a failed verification becomes data, not a dead end
- Why the cycle is deliberately decoupled from Temporal so it can be unit-tested on its own

---

## From Chat Turn to Verified Cycle

Chapters 1–3 established the foundation:

- **Chapter 01** showed why a single chat loop collapses on long-horizon work.
- **Chapter 02** reframed durability as an engineering property: the system must *assume interruption*.
- **Chapter 03** moved truth out of the context window and into git-backed files.

This chapter adds the engine that uses those pieces: **one cycle of work**. A cycle is not a model response. A cycle is a structured transaction that ends with either a verified checkpoint or a recorded failure.

The cycle has four phases:

1. **Gather** — Re-read durable state from the `GitMissionAnchor`. The model starts each turn fresh; the anchor is the source of truth.
2. **Act** — Run the model in a tool loop until it produces a concrete change (or gives up).
3. **Verify** — Ask the deterministic verifier whether the change satisfies the active checklist item.
4. **Checkpoint** — If verification passes, commit the work and update `checklist.json`, `events.jsonl`, and `decisions.jsonl`. If it fails, record the attempt and mark the item `blocked`.

This is the same shape you will see later inside Temporal activities. The local version in this chapter is intentionally identical in behavior so you can test it without a server.

---

## The `CycleOutcome` Contract

In `src/lra/agent/loop.py`, one cycle returns:

```python
@dataclass
class CycleOutcome:
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int
```

Each field answers one operational question:

| Field | Meaning |
|---|---|
| `item_id` | Which checklist item was attacked. |
| `advanced` | Did the model produce a concrete change? |
| `verified` | Did the deterministic verifier pass? |
| `is_complete` | Is the whole mission done? |
| `head_sha` | The git commit that now represents truth. |
| `tool_calls` / `turns` | Cost and loop-detection telemetry. |

The durable spine does not care *how* the model reasoned. It only cares about `CycleOutcome`. That boundary is what makes the system robust.

---

## A Minimal, Runnable Cycle

The file `lra-demo/ch04_cycle.py` implements the full four-phase cycle using only the standard library plus `git` on your PATH. It is a stripped-down version of `src/lra/agent/loop.py` so you can run it without Temporal, Pydantic, or any paid API.

Create the file:

```python
# lra-demo/ch04_cycle.py
"""Minimal Gather → Act → Verify → Checkpoint cycle.

This is a teaching implementation of the same logic in src/lra/agent/loop.py.
It runs one checklist item at a time, verifies with pytest, and checkpoints
every verified change to git.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


# --------------------------------------------------------------------------- #
# Contracts (tiny versions of the real ones in src/lra/contracts)
# --------------------------------------------------------------------------- #

class ModelProvider(Protocol):
    def complete(self, prompt: str, tools: list[dict]) -> tuple[str, list[dict]]:
        """Return (text, native_tool_calls)."""


class Verifier(Protocol):
    def check(self, workdir: Path, item: dict) -> tuple[bool, str]:
        """Return (passed, report)."""


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Action:
    done: bool = False
    tool: str | None = None
    arguments: dict = field(default_factory=dict)
    summary: str = ""


@dataclass
class CycleOutcome:
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


# --------------------------------------------------------------------------- #
# GitMissionAnchor (simplified from Chapter 03)
# --------------------------------------------------------------------------- #

class GitMissionAnchor:
    """Owns the durable files: checklist.json, events.jsonl, decisions.jsonl."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.checklist_path = workdir / "checklist.json"
        self.events_path = workdir / "events.jsonl"
        self.decisions_path = workdir / "decisions.jsonl"

    def init(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        if not (self.workdir / ".git").exists():
            self._run(["git", "init", "-q"], cwd=self.workdir)
            self._run(["git", "config", "user.email", "lra@example.com"], cwd=self.workdir)
            self._run(["git", "config", "user.name", "LRA"], cwd=self.workdir)

    def _run(self, cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)

    def read_checklist(self) -> dict:
        if not self.checklist_path.exists():
            return {"items": []}
        return json.loads(self.checklist_path.read_text())

    def write_checklist(self, checklist: dict) -> None:
        self.checklist_path.write_text(json.dumps(checklist, indent=2) + "\n")

    def append_event(self, record: dict) -> None:
        with self.events_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def head_sha(self) -> str:
        result = self._run(["git", "rev-parse", "HEAD"], cwd=self.workdir)
        return result.stdout.strip()

    def commit(self, message: str) -> str:
        self._run(["git", "add", "-A"], cwd=self.workdir)
        # Allow empty commits so tests can checkpoint even when only state files change.
        self._run(["git", "commit", "-q", "-m", message, "--allow-empty"], cwd=self.workdir)
        return self.head_sha()


# --------------------------------------------------------------------------- #
# Stub model that emits structured tool calls
# --------------------------------------------------------------------------- #

class StubModel:
    """A deterministic 'model' that knows how to finish one toy task.

    In production this is replaced by src/lra/model/openai_compat.py,
    claude.py, or ollama.py. The cycle does not care.
    """

    def __init__(self, task: str):
        self.task = task
        self.turn = 0

    def complete(self, prompt: str, tools: list[dict]) -> tuple[str, list[dict]]:
        self.turn += 1
        # Very small deterministic planner for the demo task.
        if "Create hello.py" in self.task:
            if self.turn == 1:
                return "", [ToolCall("write_file", {"path": "hello.py", "content": 'print("hello")'})]
            if self.turn == 2:
                return "", [ToolCall("write_file", {"path": "test_hello.py", "content": textwrap.dedent('''
                    import subprocess
                    import sys

                    def test_hello():
                        result = subprocess.run([sys.executable, "hello.py"], capture_output=True, text=True)
                        assert result.returncode == 0
                        assert "hello" in result.stdout.lower()
                ''')})]
        # Done signal.
        return '{"done": true, "summary": "no more actions"}', []


# --------------------------------------------------------------------------- #
# Tool dispatcher (local filesystem only)
# --------------------------------------------------------------------------- #

class LocalToolDispatcher:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.schema = [
            {
                "name": "write_file",
                "description": "Write a file relative to the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "read_file",
                "description": "Read a file relative to the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        ]

    def execute(self, tool: str, arguments: dict) -> str:
        if tool == "write_file":
            path = self.workdir / arguments["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments["content"])
            return f"wrote {arguments['path']}"
        if tool == "read_file":
            path = self.workdir / arguments["path"]
            if not path.exists():
                return "file not found"
            return path.read_text()
        raise ValueError(f"unknown tool: {tool}")


# --------------------------------------------------------------------------- #
# Deterministic verifier (pytest)
# --------------------------------------------------------------------------- #

class PytestVerifier:
    """The only authority that can call an item 'done'."""

    def check(self, workdir: Path, item: dict) -> tuple[bool, str]:
        # Install pytest if missing so the demo is self-contained.
        try:
            import pytest  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short"],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        passed = result.returncode == 0
        return passed, result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# Action parser (same logic as src/lra/agent/loop.py)
# --------------------------------------------------------------------------- #

def parse_action(text: str, native_tool_calls: list[ToolCall]) -> Action:
    if native_tool_calls:
        first = native_tool_calls[0]
        return Action(tool=first.name, arguments=dict(first.arguments))
    obj = _extract_json(text)
    if obj is None:
        return Action(done=True, summary=text[:200])
    if obj.get("done") is True:
        return Action(done=True, summary=str(obj.get("summary", "")))
    tool = obj.get("tool")
    if isinstance(tool, str):
        return Action(tool=tool, arguments=obj.get("arguments", {}))
    return Action(done=True, summary=text[:200])


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


# --------------------------------------------------------------------------- #
# The cycle
# --------------------------------------------------------------------------- #

class AgentLoop:
    """One gather → act → verify → checkpoint cycle."""

    def __init__(
        self,
        model: ModelProvider,
        dispatcher: LocalToolDispatcher,
        verifier: Verifier,
        anchor: GitMissionAnchor,
    ):
        self.model = model
        self.dispatcher = dispatcher
        self.verifier = verifier
        self.anchor = anchor

    def run_cycle(self) -> CycleOutcome:
        # -------------------- GATHER --------------------
        checklist = self.anchor.read_checklist()
        active_item = self._next_open_item(checklist)
        if active_item is None:
            return CycleOutcome(
                item_id=None,
                advanced=False,
                verified=False,
                is_complete=True,
                head_sha=self.anchor.head_sha(),
                tool_calls=0,
                turns=0,
            )

        # -------------------- ACT --------------------
        prompt = self._build_prompt(active_item, checklist)
        tool_calls = 0
        turns = 0
        advanced = False

        while turns < 10:
            turns += 1
            text, native = self.model.complete(prompt, self.dispatcher.schema)
            action = parse_action(text, [ToolCall(c["name"], c.get("arguments", {})) for c in native])

            if action.done:
                break

            tool_calls += 1
            result = self.dispatcher.execute(action.tool, action.arguments)
            advanced = True
            prompt += f"\n\nTool result: {result}\nWhat next? Reply with a tool call or {{\"done\": true}}."

        # -------------------- VERIFY --------------------
        verified, report = self.verifier.check(self.anchor.workdir, active_item)

        # -------------------- CHECKPOINT --------------------
        if verified:
            active_item["status"] = "done"
            self.anchor.write_checklist(checklist)
            sha = self.anchor.commit(f"verified {active_item['id']}: {active_item['title']}")
        else:
            active_item["status"] = "blocked"
            active_item.setdefault("attempts", []).append({"turns": turns, "report": report})
            self.anchor.write_checklist(checklist)
            sha = self.anchor.commit(f"blocked {active_item['id']}: {active_item['title']}")

        self.anchor.append_event({
            "cycle": {
                "item_id": active_item["id"],
                "advanced": advanced,
                "verified": verified,
                "tool_calls": tool_calls,
                "turns": turns,
                "head_sha": sha,
            }
        })

        return CycleOutcome(
            item_id=active_item["id"],
            advanced=advanced,
            verified=verified,
            is_complete=all(i["status"] == "done" for i in checklist["items"]),
            head_sha=sha,
            tool_calls=tool_calls,
            turns=turns,
        )

    def _next_open_item(self, checklist: dict) -> dict | None:
        for item in checklist["items"]:
            if item["status"] in ("open", "blocked"):
                return item
        return None

    def _build_prompt(self, item: dict, checklist: dict) -> str:
        done = [i["id"] for i in checklist["items"] if i["status"] == "done"]
        return textwrap.dedent(f"""
            You are working in a durable agent cycle.
            Mission: {self.model.task}
            Already done: {done}
            Active item: {item['id']} - {item['title']}

            Available tools: write_file, read_file.
            Reply with a JSON tool call or {{"done": true}}.
        """).strip()


# --------------------------------------------------------------------------- #
# CLI driver
# --------------------------------------------------------------------------- #

def main() -> int:
    workdir = Path(".lra/workspaces/ch04")
    task = "Create hello.py that prints hello and a test for it"

    anchor = GitMissionAnchor(workdir)
    anchor.init()

    # Seed the checklist once.
    if not anchor.checklist_path.exists():
        anchor.write_checklist({
            "items": [
                {"id": "py-1", "title": "Create hello.py that prints hello", "status": "open"},
                {"id": "py-2", "title": "Add a pytest test for hello.py", "status": "open"},
            ]
        })
        anchor.commit("seed checklist")

    model = StubModel(task)
    dispatcher = LocalToolDispatcher(workdir)
    verifier = PytestVerifier()
    loop = AgentLoop(model, dispatcher, verifier, anchor)

    for _ in range(5):
        outcome = loop.run_cycle()
        print(outcome)
        if outcome.is_complete:
            print("Mission complete.")
            return 0
    print("Mission did not complete within cycle budget.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

Run it:

```bash
cd lra-demo
python ch04_cycle.py
```

You will see two `CycleOutcome` lines: one for `py-1` and one for `py-2`. Then inspect the durable record:

```bash
git -C .lra/workspaces/ch04 log --oneline
cat .lra/workspaces/ch04/events.jsonl
cat .lra/workspaces/ch04/checklist.json
```

Every verified change is a git commit. Every cycle is a line in `events.jsonl`. The model's context was rebuilt from those files each time; no hidden chat history is required.

---

## Why Failure Is Just Another Event

Look at the verify/checkpoint branch:

```python
if verified:
    active_item["status"] = "done"
else:
    active_item["status"] = "blocked"
    active_item.setdefault("attempts", []).append({"turns": turns, "report": report})
```

A failed verification is not thrown away. It becomes structured data:

- The item stays `blocked`.
- The attempt is recorded with the verifier report.
- The git commit still happens, so the failure is reproducible.

This is how the system stops error compounding. A later cycle, a human, or a reviewer agent can read `attempts` and try a different approach. The model does not need to "remember" the failure because the anchor remembers it.

---

## Decoupling the Cycle from the Durable Spine

Notice that `AgentLoop` in this chapter has no Temporal imports. That is intentional and matches the real design in `src/lra/agent/loop.py`. The cycle is a pure function of:

- durable state (anchor),
- model (pluggable),
- tools (dispatcher),
- verifier.

Temporal only orchestrates *when* cycles run and *replays* them after crashes. The cycle logic itself is unit-testable with a stub model, fake tools, and a fake verifier. That separation is what lets you write fast tests for the brain without standing up a workflow server.

---

## Hands-On Exercise

1. Run `lra-demo/ch04_cycle.py` and verify that `hello.py` and `test_hello.py` are created and the tests pass.
2. Delete `hello.py`, then run the script again. Watch the first cycle mark `py-1` as `blocked` and the second cycle recover by rewriting the file.
3. Add a third checklist item that requires a new tool, e.g. `append_file`. Implement the tool in `LocalToolDispatcher` and make the stub model use it.
4. Inspect `events.jsonl` after a blocked cycle. Confirm that `verified: false` is recorded with the verifier report.

---

> **Key Takeaway:** A model produces text; a cycle produces a verified checkpoint. The four phases — gather, act, verify, checkpoint — turn a chatbot into a durable worker that can be interrupted, inspected, and resumed.

---

## Next Chapter

**Chapter 05: Tool Dispatching and Sandboxing** — The cycle is only as safe as the tools it can call. Next we replace the local filesystem dispatcher with a pluggable tool registry and run dangerous commands inside a sandbox.