# Chapter 10: The Lead Engineer (Single Writer)

> One writer, many readers, one fresh reviewer. Parallelism is a privilege of read-only work.
> — Fareed Khan

## What We'll Cover

- Why the **Lead Engineer** is the only agent allowed to write coupled code
- How the Lead Engineer consumes the **blackboard** filled by read-only Researchers (Chapter 09)
- Mapping the Lead Engineer's work onto the **gather → act → verify → checkpoint** cycle (Chapter 04)
- Enforcing the **single writer** rule with an ownership map and a decision log
- A runnable demo in `lra-demo/ch10_lead_engineer.py` that completes a checklist by writing, testing, and committing
- What happens when verification fails and the Lead Engineer must retry before checkpointing

---

## The Concept

In Chapter 08 we introduced an asymmetric organization: many **Researchers** read in parallel, one **Reviewer** audits with a fresh context, and one **Lead Engineer** owns every coupled write. Chapter 09 proved that read-only fan-out is safe. This chapter focuses on the other side of that contract: **writes must stay single-threaded**.

The Lead Engineer is not "smarter" than the other agents. It is simply the **sole holder of write tokens** for files that affect the same design surface. If two agents could edit `hello.py` at the same time, the mission would accumulate merge conflicts, contradictory abstractions, and silent regressions over hundreds of cycles. By forcing all coupled writes through one agent, the system keeps the design coherent across days or weeks.

The Lead Engineer's loop is the same cycle from Chapter 04:

1. **Gather** — read the blackboard, the checklist, and the current files. No writes.
2. **Act** — decide the next code change and dispatch it through the tool/sandbox layer (Chapter 05).
3. **Verify** — run the deterministic command attached to the checklist item (Chapter 06).
4. **Checkpoint** — only if verification passes, update the checklist and commit to git (Chapter 07).

If verification fails, the Lead Engineer does **not** checkpoint. It retries the act phase with the failure output added to its context. This prevents error compounding, the main failure mode of long-horizon agents.

---

## Code Walkthrough

The demo file `lra-demo/ch10_lead_engineer.py` implements a minimal but fully runnable Lead Engineer. It uses hard-coded heuristics for the actual code generation so you can run it without an API key or model backend. In the real system, the `act_write` step would call the pluggable model layer in `src/lra/model/`.

```python
#!/usr/bin/env python3
"""lra-demo/ch10_lead_engineer.py

A minimal, runnable simulation of the Lead Engineer: the single writer in the
asymmetric organization. It consumes research from the blackboard, writes code
through the local tool dispatcher, and only checkpoints after deterministic
verification passes.
"""
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

WORKSPACE = Path(".lra/workspaces/ch10")
STATE_FILE = WORKSPACE / "mission_state.json"
GIT_USER = [
    "git",
    "-c", "user.email=lead@lra.local",
    "-c", "user.name=Lead Engineer",
]


@dataclass
class ChecklistItem:
    id: str
    title: str
    status: str = "open"          # open | in_progress | done
    owner: str = "lead"
    verification: str = ""


@dataclass
class MissionState:
    task: str
    checklist: list[ChecklistItem] = field(default_factory=list)
    blackboard: dict = field(default_factory=dict)
    ownership: dict = field(default_factory=dict)
    head_commit: Optional[str] = None


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Local tool dispatcher: run a command inside the mission workspace."""
    return subprocess.run(cmd, cwd=WORKSPACE, text=True, capture_output=True, **kwargs)


def init_workspace() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not (WORKSPACE / ".git").exists():
        run(["git", "init", "-q", str(WORKSPACE)], check=True)
        run(GIT_USER + ["commit", "-q", "--allow-empty", "-m", "mission start"], check=True)


def load_state() -> MissionState:
    if STATE_FILE.exists():
        raw = json.loads(STATE_FILE.read_text())
        return MissionState(
            task=raw["task"],
            checklist=[ChecklistItem(**i) for i in raw["checklist"]],
            blackboard=raw.get("blackboard", {}),
            ownership=raw.get("ownership", {}),
            head_commit=raw.get("head_commit"),
        )

    # Initial demo state.
    state = MissionState(
        task="Create hello.py with a greet(name) function and a pytest test.",
        checklist=[
            ChecklistItem(
                id="c1",
                title="Create hello.py with greet(name)",
                verification="python -m pytest tests/ -q",
            ),
            ChecklistItem(
                id="c2",
                title="Add pytest test for greet",
                verification="python -m pytest tests/ -q",
            ),
        ],
        blackboard={
            "research": [
                {
                    "topic": "project layout",
                    "finding": "Put source in hello.py and tests in tests/test_hello.py.",
                },
                {
                    "topic": "pytest convention",
                    "finding": "Functions named test_* are auto-discovered.",
                },
            ]
        },
        ownership={
            "hello.py": "lead",
            "tests/test_hello.py": "lead",
        },
    )
    save_state(state)
    return state


def save_state(state: MissionState) -> None:
    raw = {
        "task": state.task,
        "checklist": [
            {
                "id": i.id,
                "title": i.title,
                "status": i.status,
                "owner": i.owner,
                "verification": i.verification,
            }
            for i in state.checklist
        ],
        "blackboard": state.blackboard,
        "ownership": state.ownership,
        "head_commit": state.head_commit,
    }
    STATE_FILE.write_text(json.dumps(raw, indent=2) + "\n")


def gather_context(state: MissionState, item: ChecklistItem) -> str:
    """Gather phase: read the blackboard and current files. No writes."""
    lines = [
        f"Task: {state.task}",
        f"Current item: {item.title}",
        "",
        "Research on blackboard:",
    ]
    for report in state.blackboard.get("research", []):
        lines.append(f"- {report['topic']}: {report['finding']}")

    hello = WORKSPACE / "hello.py"
    test = WORKSPACE / "tests" / "test_hello.py"
    lines.extend(["", "Current files:"])
    lines.append(f"hello.py:\n{hello.read_text() if hello.exists() else '<missing>'}")
    lines.append(f"tests/test_hello.py:\n{test.read_text() if test.exists() else '<missing>'}")
    return "\n".join(lines)


def act_write(item: ChecklistItem, context: str) -> None:
    """Act phase: the Lead Engineer is the only agent allowed to write code.

    In the real system this prompt would go to the model layer. Here we use
    deterministic heuristics so the demo runs at $0.
    """
    if item.owner != "lead":
        raise PermissionError(f"Lead Engineer cannot write item owned by {item.owner}")

    if "hello.py" in item.title:
        file_owner = state.ownership.get("hello.py")
        if file_owner != "lead":
            raise PermissionError(f"hello.py is owned by {file_owner}, not lead")
        (WORKSPACE / "hello.py").write_text(
            'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n'
        )

    elif "test" in item.title:
        tests_dir = WORKSPACE / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "test_hello.py").write_text(
            'from hello import greet\n\n\ndef test_greet():\n    assert greet("world") == "Hello, world!"\n'
        )


def verify(item: ChecklistItem) -> tuple[bool, str]:
    """Verify phase: deterministic command, never the model's opinion."""
    result = run(item.verification.split())
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def checkpoint(state: MissionState, item: ChecklistItem, passed: bool) -> None:
    """Checkpoint phase: commit real state to git only when verification passes."""
    if passed:
        item.status = "done"
        run(GIT_USER + ["add", "-A"], check=True)
        run(GIT_USER + ["commit", "-q", "-m", f"{item.id}: {item.title}"], check=True)
        state.head_commit = run(GIT_USER + ["rev-parse", "HEAD"]).stdout.strip()
    else:
        item.status = "open"

    save_state(state)


def lead_cycle(state: MissionState) -> bool:
    """One Lead Engineer cycle: gather -> act -> verify -> checkpoint."""
    open_items = [i for i in state.checklist if i.status != "done"]
    if not open_items:
        return False

    item = open_items[0]
    item.status = "in_progress"
    save_state(state)

    print(f"\n[LEAD] working on {item.id}: {item.title}")
    context = gather_context(state, item)
    act_write(item, context)
    passed, output = verify(item)
    print(f"[LEAD] verification {'PASSED' if passed else 'FAILED'}\n{output}")
    checkpoint(state, item, passed)
    return True


def main() -> int:
    init_workspace()
    state = load_state()
    print(f"Mission: {state.task}")
    print(f"Workspace: {WORKSPACE.resolve()}")

    while lead_cycle(state):
        pass

    print("\nMission checklist:")
    for item in state.checklist:
        print(f"  [{item.status:12}] {item.id} {item.title}")
    print(f"Head commit: {state.head_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### How the pieces connect

- **Ownership enforcement.** Before writing, `act_write` checks that the checklist item and the target file are both owned by `lead`. This is the lightweight version of the ownership map in `src/lra/coordination/ownership.py`. In a multi-agent run, a Researcher or Reviewer that tries to write a file owned by the Lead Engineer would be rejected by the tool dispatcher.
- **Blackboard consumption.** The `gather_context` function reads the research reports that Researchers posted in Chapter 09. The Lead Engineer does not blindly trust them; it uses them as context for its own write, then lets verification decide if the result is correct.
- **Deterministic verification.** Each checklist item carries a `verification` command. The Lead Engineer is not allowed to mark an item done based on its own judgment. This is the same verifier contract from Chapter 06.
- **Git checkpointing.** Only after `pytest` passes does the Lead Engineer stage the files, commit, and update `head_commit`. If the mission is interrupted and resumed, `load_state` plus the git history reconstruct exactly where it left off.

Run it with:

```bash
uv run python lra-demo/ch10_lead_engineer.py
```

After it finishes, inspect the history the Lead Engineer produced:

```bash
git -C .lra/workspaces/ch10 log --oneline
```

You will see two commits, one per checklist item, and no commits for failed attempts.

---

## Hands-On Exercise

1. **Add a third checklist item** to `lra-demo/ch10_lead_engineer.py` that requires changing `hello.py` — for example, "Add a `farewell(name)` function to hello.py and a test for it." Give it the same verification command.
2. **Delete the workspace** `.lra/workspaces/ch10` and rerun the script. Watch the Lead Engineer rebuild the whole feature from the saved initial state.
3. **Introduce a deliberate failure:** change the `act_write` heuristic for the new item so it writes a function with the wrong return value. Run the script and confirm that:
   - the item stays `open`,
   - no checkpoint commit is created for that item,
   - the failure output appears in the console.
4. **Fix the bug** and rerun. Verify that the git log now contains a commit for the new item.
5. **Reflect:** why would allowing a Researcher to also write `hello.py` break this trace? Write a one-paragraph answer referencing the ownership map and deterministic verification.

---

## Key Takeaway

> The Lead Engineer is not the smartest agent; it is the only agent with write access to coupled code. Single-threaded writing plus deterministic verification is what keeps a week-long mission from turning into a pile of conflicting patches.

---

## Next Chapter

**Chapter 11: Reviewer and Reflection Agents.** The Lead Engineer writes, but it is not allowed to sign off on its own work. We will add an independent Reviewer that runs with a fresh context, can block a checkpoint, and forces the Lead Engineer to reflect and retry.