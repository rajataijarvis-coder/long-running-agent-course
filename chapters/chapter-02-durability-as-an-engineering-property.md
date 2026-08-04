# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why durability belongs to the *system*, not the LLM
- The "Assume Interruption" design rule and how it changes every loop
- Volatile context vs. durable state: what lives in the window and what lives on disk/git
- The anatomy of a checkpoint using `CycleOutcome`
- A runnable crash-resume loop in plain Python
- How the durable spine (Temporal) extends this local idea to weeks-long missions

---

## From Chat Loop to Durable Loop

In Chapter 01 we saw the ChatGPT-style agent die when the process restarts, the context window fills, or an API call fails. The fix is not a smarter model. The fix is a system that *assumes interruption* and rebuilds itself from durable state every cycle.

The model still thinks in short bursts. The system, however, keeps the real state outside the model, journals every step, and verifies progress with real tests. When the process comes back, it does not remember what it was doing — it *re-reads* what it was doing.

This chapter builds the smallest version of that idea in plain Python. The full implementation lives in `lra-demo/src/lra/agent/loop.py`, but the pattern is the same: **gather → act → verify → checkpoint**.

---

## Assume Interruption

The single most important design rule in `lra-demo` is:

> Every cycle must be able to start from a cold boot.

That means:

1. The agent never trusts its own in-memory context.
2. At the start of every cycle it re-reads the checklist, the decision log, and the git state.
3. After every action it writes a checkpoint.
4. A crash is not an exception to handle — it is the normal case to optimize for.

This rule shows up concretely in `lra-demo/src/lra/agent/loop.py`. The `AgentLoop.run_one_cycle` method does not assume it knows the mission state. It receives a `GitMissionAnchor` and re-reads the anchor at the top of every cycle. The model is given only the *current* distilled context, not a growing chat history.

---

## Volatile Context vs. Durable State

| Volatile (can disappear) | Durable (must survive) |
|---|---|
| The model's context window | The git repo on disk |
| In-memory chat history | The structured checklist |
| The Python process heap | The event/decision log |
| API response cache | Checkpoint files |

The context window is a **lossy cache**. It is useful for one turn, but it is not the source of truth. The durable state lives in:

- `lra-demo/src/lra/state/mission_anchor.py` — writes the real state to git
- `lra-demo/src/lra/contracts/state.py` — typed contracts for `Checkpoint`, `EventRecord`, etc.
- `lra-demo/src/lra/agent/loop.py` — the `CycleOutcome` returned after every verified cycle

If you can kill the process at any point and resume correctly, you have durability. If you cannot, you have a chatbot.

---

## The Checkpoint Contract: `CycleOutcome`

In `lra-demo/src/lra/agent/loop.py`, one cycle produces exactly one `CycleOutcome`:

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

Each field has a job:

- `item_id` — which checklist item was attempted
- `advanced` — did the mission move to the next item?
- `verified` — did the deterministic verifier say the work is green?
- `is_complete` — is the whole mission done?
- `head_sha` — the git commit hash that captured the state (the anchor)
- `tool_calls` / `turns` — cost and effort counters for the governor

A cycle is only allowed to end after it has written a checkpoint. The checkpoint is the boundary that makes the work durable.

---

## A Runnable Crash-Resume Loop in Plain Python

The full `lra-demo` stack uses Temporal for durability, but the *idea* is testable without any server. Below is a minimal, self-contained crash-resume loop. It writes a JSONL checkpoint after every cycle, simulates a host reboot, and resumes from the last checkpoint.

Save this as `lra-demo/ch02_crash_resume.py` and run it:

```python
"""Minimal durable agent loop: gather -> act -> verify -> checkpoint.

This mirrors the CycleOutcome pattern in lra-demo/src/lra/agent/loop.py,
but uses a local JSONL file instead of git so it runs with zero setup.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class CycleOutcome:
    """Same contract as lra-demo/src/lra/agent/loop.py."""
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


@dataclass
class MissionState:
    """The durable slice of the mission. Everything else is volatile."""
    workdir: Path
    checklist: list[str]
    current_index: int = 0
    attempts: dict[str, int] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def checkpoint_path(self) -> Path:
        return self.workdir / ".lra" / "state.jsonl"

    def save(self) -> None:
        self.checkpoint_path().parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path(), "a") as f:
            f.write(json.dumps(asdict(self), default=str) + "\n")

    @classmethod
    def load_latest(cls, workdir: Path) -> MissionState | None:
        path = workdir / ".lra" / "state.jsonl"
        if not path.exists():
            return None
        last = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        return cls(**json.loads(last)) if last else None


class AgentCycle:
    """One model + tools + verifier + checkpoint cycle."""

    def __init__(
        self,
        state: MissionState,
        model: Callable[[dict, list[str]], dict],
        tools: dict[str, Callable],
        verifier: Callable[[str, Path], bool],
    ) -> None:
        self.state = state
        self.model = model
        self.tools = tools
        self.verifier = verifier
        self.tool_calls = 0
        self.turns = 0

    def current_item(self) -> str | None:
        if self.state.current_index >= len(self.state.checklist):
            return None
        return self.state.checklist[self.state.current_index]

    def run_once(self, crash_after: int | None = None) -> CycleOutcome:
        item = self.current_item()
        if item is None:
            return CycleOutcome(
                item_id="done",
                advanced=False,
                verified=True,
                is_complete=True,
                head_sha=f"v{len(self.state.events)}",
                tool_calls=self.tool_calls,
                turns=self.turns,
            )

        self.turns += 1

        # GATHER: re-read state every cycle (assume interruption)
        context = {
            "item": item,
            "attempts": self.state.attempts.get(item, 0),
            "recent_events": self.state.events[-3:],
        }
        action = self.model(context, list(self.tools.keys()))

        # Simulate a host reboot after N tool calls
        if crash_after is not None and self.tool_calls >= crash_after:
            raise RuntimeError("Simulated host reboot")

        # ACT
        if action["tool"] not in self.tools:
            raise ValueError(f"Unknown tool: {action['tool']}")
        result = self.tools[action["tool"]](**action.get("args", {}))
        self.tool_calls += 1

        self.state.events.append(
            {"item": item, "action": action, "result": str(result)[:200]}
        )

        # VERIFY: deterministic, never the model's opinion
        verified = self.verifier(item, self.state.workdir)

        advanced = False
        is_complete = False
        if verified:
            self.state.current_index += 1
            advanced = True
            is_complete = self.state.current_index >= len(self.state.checklist)
        else:
            self.state.attempts[item] = self.state.attempts.get(item, 0) + 1

        # CHECKPOINT
        self.state.save()

        return CycleOutcome(
            item_id=item,
            advanced=advanced,
            verified=verified,
            is_complete=is_complete,
            head_sha=f"v{len(self.state.events)}",
            tool_calls=self.tool_calls,
            turns=self.turns,
        )


def simple_model(context: dict, tool_names: list[str]) -> dict:
    """A tiny hard-coded 'model' that picks the next tool."""
    item = context["item"]
    if "Create hello.py" in item:
        return {
            "tool": "write_file",
            "args": {"path": "hello.py", "content": "print('hello')\n"},
        }
    if "Create a pytest test" in item:
        return {
            "tool": "write_file",
            "args": {
                "path": "test_hello.py",
                "content": (
                    "import subprocess, sys\n"
                    "result = subprocess.run([sys.executable, 'hello.py'], capture_output=True, text=True)\n"
                    "assert result.returncode == 0\n"
                    "assert 'hello' in result.stdout\n"
                ),
            },
        }
    if "Run the test" in item:
        return {"tool": "run_shell", "args": {"cmd": "python -m pytest test_hello.py -q"}}
    return {"tool": "run_shell", "args": {"cmd": "echo unknown item"}}


def deterministic_verifier(item: str, workdir: Path) -> bool:
    """Exit-code / file-existence truth. No model judgment."""
    if "Create hello.py" in item and "test" not in item:
        return (workdir / "hello.py").exists()

    if "Create a pytest test" in item:
        test_file = workdir / "test_hello.py"
        if not test_file.exists():
            return False
        proc = subprocess.run(
            ["python", "-m", "pytest", str(test_file), "-q"],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0

    if "Run the test" in item:
        proc = subprocess.run(
            ["python", "-m", "pytest", str(workdir / "test_hello.py"), "-q"],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0

    return False


def run_mission(workdir: Path, crash_after: int | None = None) -> None:
    state = MissionState.load_latest(workdir) or MissionState(
        workdir=workdir,
        checklist=[
            "Create hello.py that prints hello",
            "Create a pytest test for hello.py",
            "Run the test and make it pass",
        ],
    )

    cycle = AgentCycle(
        state=state,
        model=simple_model,
        tools={
            "write_file": lambda path, content: (workdir / path).write_text(content),
            "run_shell": lambda cmd: subprocess.run(
                cmd, shell=True, cwd=workdir, capture_output=True, text=True
            ).stdout,
        },
        verifier=deterministic_verifier,
    )

    try:
        while True:
            outcome = cycle.run_once(crash_after=crash_after)
            print(outcome)
            if outcome.is_complete:
                break
    except RuntimeError as exc:
        print(f"CRASH: {exc}")
        print(f"Checkpoint written to {state.checkpoint_path()}")
        print("Resuming now...")
        run_mission(workdir, crash_after=None)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        # First run: crash after 1 tool call, then resume
        run_mission(wd, crash_after=1)
```

When you run it, you will see:

1. The first cycle creates `hello.py` and checkpoints.
2. A simulated crash is raised.
3. The process reloads the latest `state.jsonl`.
4. It resumes from item 2, writes the test, runs it, and completes.

No work is lost because the durable state lives on disk, not in the model's context window.

---

## How Temporal Extends This Local Idea

The plain-Python loop above is intentionally decoupled from any orchestrator. In `lra-demo`, the same `AgentLoop` is called from inside a Temporal activity:

```python
# lra-demo/src/lra/durable/activities.py (conceptual)
from temporalio import activity

@activity.defn
async def agent_cycle_activity(ctx: CycleContext) -> CycleOutcome:
    anchor = GitMissionAnchor(ctx.workdir)
    loop = AgentLoop(
        model=ctx.model,
        dispatcher=ctx.dispatcher,
        verifier=ctx.verifier,
        anchor=anchor,
        ledger=ctx.ledger,
    )
    return await loop.run_one_cycle(ctx.item_id)
```

Temporal gives us three things the local loop cannot:

1. **Journaled replay** — every LLM/tool call is recorded; a crash resumes exactly where it left off without re-spending tokens.
2. **Durable sleep** — idle waiting is free because the workflow is suspended and resumed by the server.
3. **Continue-as-new** — long missions do not blow up Temporal's event history; the workflow periodically starts a fresh copy of itself with the same durable state.

The `temporalio` dependency is listed under `[project.optional-dependencies]` in `lra-demo/pyproject.toml` as `durable`. You can run the local loop with no servers at all; you only need Temporal when you want the same guarantee across process restarts, host reboots, and weeks of wall-clock time.

---

## Hands-On Exercise

1. Run `lra-demo/ch02_crash_resume.py` as shown above.
2. Inspect `.lra/state.jsonl` inside the temporary workdir. You will see one JSON line per checkpoint.
3. Change `crash_after` to `0`, `2`, and `4` and confirm the mission still completes every time.
4. Add a fourth checklist item: `"Add a README.md describing the project"`. Update `simple_model` and `deterministic_verifier` so the mission completes with a real `README.md` on disk.
5. (Stretch) Replace the JSONL file with a real git commit after every cycle. Use `subprocess.run(["git", "init"], cwd=workdir)` at startup and commit after `state.save()`. This prepares you for Chapter 03.

---

> **Key Takeaway:** Durability is not something the model provides; it is a property of the loop around the model. Re-read state every cycle, journal every action, verify with real tests, and checkpoint before you declare progress. If you can kill the process at any moment and resume correctly, you have built a long-running agent.

---

## Next Chapter Teaser

In Chapter 03 we externalize the durable state even further: we replace the JSONL file with a real git repo and treat git commits as the mission's memory. You will learn why `head_sha` is the single most important field in `CycleOutcome`, and how `GitMissionAnchor` makes every checkpoint auditable, diffable, and human-readable.