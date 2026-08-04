# Chapter 03: Externalizing Truth — Git as Memory

> The context window is a lossy cache. The real state lives outside it and is re-read every cycle.
> — Fareed Khan

## What We'll Cover

- Why the LLM context window is the wrong place for mission truth
- The three durable files that replace chat history: `checklist.json`, `decisions.jsonl`, `events.jsonl`
- How a git commit becomes an atomic checkpoint boundary
- Reconstructing "where things stand" after a crash or reboot
- A minimal, runnable `GitMissionAnchor` in plain Python (`lra-demo/ch03_git_memory.py`)
- Wiring that anchor into the crash-resume loop from Chapter 02

---

## From Volatile Context to Durable State

Chapter 01 showed that chat-style agents die because their entire understanding lives inside a single process and a single context window. Chapter 02 made the loop durable enough to survive a crash, but the *state* we carried was still in-memory or ad-hoc JSON. That is not enough for a week-long mission.

The next engineering property is **externalization**: the model thinks in short bursts, but the system keeps the real truth on disk — in git — and re-reads it at the start of every cycle. This is the "assume interruption" rule in practice. If the host reboots on day 4, the next cycle does not ask the model to remember anything. It reads the checklist, the decision log, and the event journal, then resumes.

Git is the right substrate because it is:

- **Atomic**: a commit groups many file changes into one checkpoint.
- **Versioned**: you can inspect exactly what changed and when.
- **Recoverable**: `git log` is an audit trail; `git checkout` is a time machine.
- **Already installed**: most software missions live in git anyway.

In the full `lra` repo this logic lives in `src/lra/state/mission_anchor.py`, backed by typed records in `src/lra/contracts/state.py`. For this chapter we will build a minimal version you can run without Temporal, without Docker, and without a paid model.

---

## The Three Files

A mission workdir contains a `mission/` directory with three structured files:

| File | Purpose | Replaces |
|---|---|---|
| `mission/checklist.json` | The plan: item IDs, descriptions, status, attempts, verification results | The model's memory of "what is left to do" |
| `mission/decisions.jsonl` | Design decisions and their rationale, timestamped | Long rationale threads in the context window |
| `mission/events.jsonl` | Every cycle, tool call, verification, and outcome | The chat transcript |

The model is never given the raw JSON as its full context. Instead, the anchor summarizes these files into a compact "where things stand" report before each cycle. The raw files are the **source of truth**; the model prompt is a **lossy cache** built from them.

---

## A Minimal Git Mission Anchor

Save this as `lra-demo/ch03_git_memory.py`. It requires only Python and a working `git` binary.

```python
#!/usr/bin/env python3
"""lra-demo/ch03_git_memory.py — externalize mission truth to git.

This is a minimal version of the full GitMissionAnchor found in
src/lra/state/mission_anchor.py. It demonstrates the three durable
files and atomic checkpoint commits without any external services.
"""

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Status = Literal["todo", "in_progress", "done", "blocked"]


@dataclass
class CheckItem:
    id: str
    description: str
    status: Status = "todo"
    attempts: int = 0
    verification: dict = field(default_factory=dict)


@dataclass
class EventRecord:
    cycle: int
    item_id: str
    action: str
    outcome: str
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class DecisionRecord:
    ts: str
    item_id: str
    decision: str
    rationale: str


class GitMissionAnchor:
    """Owns the mission workdir, the three durable files, and git checkpointing."""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.mission_dir = self.workdir / "mission"
        self.checklist_path = self.mission_dir / "checklist.json"
        self.events_path = self.mission_dir / "events.jsonl"
        self.decisions_path = self.mission_dir / "decisions.jsonl"

    # ------------------------------------------------------------------ git ops
    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.workdir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def init_repo(self) -> None:
        self.mission_dir.mkdir(parents=True, exist_ok=True)
        (self.workdir / ".gitignore").write_text("__pycache__/\n*.pyc\n")
        self._git("init", "-b", "main")
        self._git("add", ".")
        self._git("commit", "-m", "mission: init anchor")

    def commit_checkpoint(self, message: str) -> str:
        """Atomic boundary: everything written before this call is one checkpoint."""
        self._git("add", "-A")
        self._git("commit", "-m", message)
        return self._git("rev-parse", "HEAD").strip()

    # ------------------------------------------------------------------ state ops
    def write_checklist(self, items: list[CheckItem]) -> None:
        self.checklist_path.write_text(
            json.dumps([asdict(i) for i in items], indent=2) + "\n"
        )

    def read_checklist(self) -> list[CheckItem]:
        if not self.checklist_path.exists():
            return []
        data = json.loads(self.checklist_path.read_text())
        return [CheckItem(**row) for row in data]

    def append_event(self, event: EventRecord) -> None:
        with self.events_path.open("a") as f:
            f.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    def append_decision(self, decision: DecisionRecord) -> None:
        with self.decisions_path.open("a") as f:
            f.write(json.dumps(asdict(decision), sort_keys=True) + "\n")

    def where_things_stand(self) -> dict:
        """Build the situational-awareness summary the model sees each cycle."""
        items = self.read_checklist()
        return {
            "head": self._git("rev-parse", "--short", "HEAD").strip(),
            "total": len(items),
            "done": sum(1 for i in items if i.status == "done"),
            "in_progress": [i.id for i in items if i.status == "in_progress"],
            "blocked": [i.id for i in items if i.status == "blocked"],
            "todo": [i.id for i in items if i.status == "todo"],
        }


class SimpleVerifier:
    """Deterministic verifier from Chapter 02, now reading files from disk."""

    def verify(self, item: CheckItem, workdir: Path) -> bool:
        if item.id == "hello_file":
            target = workdir / "hello.py"
            return target.exists() and 'print("' in target.read_text()
        if item.id == "test_file":
            target = workdir / "test_hello.py"
            return target.exists() and "def test_" in target.read_text()
        return False


def run_one_cycle(anchor: GitMissionAnchor, verifier: SimpleVerifier) -> str:
    """One gather → act → verify → checkpoint cycle on the next open item."""
    state = anchor.where_things_stand()
    items = anchor.read_checklist()

    # Gather: pick the first item that is not done.
    item = next((i for i in items if i.status != "done"), None)
    if item is None:
        return "all_done"

    # Act: mark in-progress and do the work.
    item.attempts += 1
    item.status = "in_progress"
    anchor.write_checklist(items)
    anchor.append_event(
        EventRecord(
            cycle=state["done"] + 1,
            item_id=item.id,
            action="start",
            outcome="in_progress",
        )
    )

    # In a real agent these writes come from tool calls; here we hard-code them.
    if item.id == "hello_file":
        (anchor.workdir / "hello.py").write_text('print("hello from lra")\n')
    elif item.id == "test_file":
        (anchor.workdir / "test_hello.py").write_text(
            "def test_hello():\n    assert True\n"
        )

    # Verify: the model does not get to declare success; the verifier decides.
    ok = verifier.verify(item, anchor.workdir)
    item.status = "done" if ok else "blocked"
    item.verification = {"passed": ok, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    anchor.write_checklist(items)

    anchor.append_event(
        EventRecord(
            cycle=state["done"] + 1,
            item_id=item.id,
            action="verify",
            outcome="pass" if ok else "fail",
        )
    )

    # Checkpoint: commit the new state atomically.
    sha = anchor.commit_checkpoint(f"checkpoint: {item.id} -> {item.status}")
    return sha


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="lra-ch03-"))
    try:
        anchor = GitMissionAnchor(tmp)
        anchor.init_repo()

        # Seed the mission.
        anchor.write_checklist(
            [
                CheckItem(
                    id="hello_file",
                    description="Create hello.py that prints hello",
                ),
                CheckItem(
                    id="test_file",
                    description="Create test_hello.py with a real test function",
                ),
            ]
        )
        anchor.append_decision(
            DecisionRecord(
                ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                item_id="mission",
                decision="Use plain Python files and an in-process verifier",
                rationale="Keeps the demo runnable without Docker or paid APIs.",
            )
        )
        anchor.commit_checkpoint("mission: seed checklist and decisions")

        print("Initial state:", anchor.where_things_stand())

        while True:
            result = run_one_cycle(anchor, SimpleVerifier())
            print("Cycle result:", result, "state:", anchor.where_things_stand())
            if result == "all_done":
                break

        print("\nGit log:")
        print(anchor._git("log", "--oneline"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
```

Run it:

```bash
python lra-demo/ch03_git_memory.py
```

You will see each item move from `todo` → `in_progress` → `done`, followed by a git commit. The final `git log --oneline` shows a clear history: init, seed, checkpoint for `hello_file`, checkpoint for `test_file`.

---

## Code Walkthrough

### The anchor owns the workdir

`GitMissionAnchor` is not a global singleton. It is constructed for a specific `workdir`, which is the same `workdir` the CLI passes in:

```bash
uv run lra mission --task "..." --workdir .lra/workspaces/demo
```

This matters because every activity in the durable spine (Chapter 12) reconstructs the anchor from the workdir path. There is no shared in-memory object to lose in a crash.

### The three files are ordinary JSON/JSONL

- `checklist.json` is rewritten as a whole because the checklist is small and we want a clean diff.
- `events.jsonl` and `decisions.jsonl` are appended to because they are append-only journals.
- All three are text, so `git diff` shows exactly what changed.

### `commit_checkpoint` is the only trusted boundary

Everything before `commit_checkpoint` is tentative. If the process dies after `write_checklist` but before commit, the next cycle still reads from the last committed state. In the full `lra` implementation the anchor may also stage files and roll back uncommitted changes on resume; the principle is the same: **the commit is the checkpoint**.

### `where_things_stand` is the model's window

The model does not get the full git history. It gets a compact summary:

```python
{
    "head": "a1b2c3d",
    "total": 2,
    "done": 1,
    "in_progress": ["test_file"],
    "blocked": [],
    "todo": [],
}
```

This is how the system survives context-window limits. The raw files are unbounded; the summary is capped.

---

## Hands-On Exercise: Kill and Resume

The point of externalized truth is that a crash is harmless. Prove it.

### Step 1 — Make the workdir persistent

Edit `lra-demo/ch03_git_memory.py` so it uses a fixed directory and does not delete it:

```python
workdir = Path(".lra/ch03-resume")
workdir.mkdir(parents=True, exist_ok=True)
anchor = GitMissionAnchor(workdir)
# remove the tempfile and shutil.rmtree cleanup
```

### Step 2 — Simulate a crash after the first checkpoint

Insert this guard inside `main()` right after the seed commit:

```python
if anchor.where_things_stand()["done"] >= 1:
    raise KeyboardInterrupt("simulated crash after first checkpoint")
```

Run the script once. It crashes after `hello_file` is done.

Inspect the durable state:

```bash
git -C .lra/ch03-resume log --oneline
cat .lra/ch03-resume/mission/checklist.json
cat .lra/ch03-resume/mission/events.jsonl
```

### Step 3 — Write a resume script

Save this as `lra-demo/ch03_resume.py`. It does not re-initialize the repo. It just reconstructs the anchor and continues.

```python
#!/usr/bin/env python3
"""lra-demo/ch03_resume.py — resume a mission from its git anchor."""

from pathlib import Path
from ch03_git_memory import GitMissionAnchor, SimpleVerifier, run_one_cycle


def main() -> None:
    workdir = Path(".lra/ch03-resume")
    anchor = GitMissionAnchor(workdir)
    print("Resumed state:", anchor.where_things_stand())

    while True:
        result = run_one_cycle(anchor, SimpleVerifier())
        print("Cycle result:", result, "state:", anchor.where_things_stand())
        if result == "all_done":
            break

    print("\nFinal git log:")
    print(anchor._git("log", "--oneline"))


if __name__ == "__main__":
    main()
```

Run it:

```bash
python lra-demo/ch03_resume.py
```

Notice that `hello_file` is **not** rebuilt. The anchor read the committed checklist, saw it was `done`, and moved straight to `test_file`. The crash cost zero lost work and zero repeated tokens.

---

## Key Takeaway

> The model's context window is a lossy cache. The real state lives in git: a structured checklist, a decision log, and an event journal. Re-read them every cycle, commit every checkpoint, and a crash becomes a free resume.

---

## Next Chapter

In **Chapter 04: The Gather → Act → Verify → Checkpoint Cycle**, we wire the durable anchor from this chapter into the model and tool loop. We will replace the hard-coded file writes with real tool dispatching, and we will see why verification — not the model — is the only authority that can mark an item done.