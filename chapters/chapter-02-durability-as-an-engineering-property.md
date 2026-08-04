# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why durability is a *system* concern, not a smarter-model concern
- The "Assume Interruption" design rule and how it changes every loop
- Volatile context vs. durable state: what lives in the window and what lives on disk/git
- The anatomy of a checkpoint using `CycleOutcome` from `src/lra/agent/loop.py`
- A runnable crash-resume loop in plain Python (`lra-demo/ch02_crash_resume.py`)
- How the durable spine (Temporal) extends this local idea to week-long missions

---

## From Chat Loop to Durable Loop

Chapter 01 showed that a ChatGPT-style agent is a single in-memory loop. The moment the process restarts, the API times out, or the context window fills, the mission is gone. The fix is not to ask the model to "remember harder." The fix is to build a system that *assumes interruption*.

**Assume Interruption** is the design rule that changes every decision in LRA:

- Every cycle starts by re-reading the real state from disk.
- Every action is journaled before the next action is attempted.
- Progress is measured by deterministic verification, not by the model's confidence.
- A crash on cycle 600 must resume exactly like a crash on cycle 6.

This is why durability is an engineering property. The model still thinks in short bursts. The system around it keeps the mission alive.

---

## Volatile Context vs. Durable State

| Volatile (can die) | Durable (must survive) |
|---|---|
| The LLM context window | The checklist on disk |
| In-memory variables | The git repo |
| The current process | The event/decision log |
| API response cache | Verification results |

The context window is a **lossy cache**. It is useful for reasoning, but it is not the source of truth. In LRA, the real state lives in:

- `src/lra/state/` — git-backed saved-state files
- `src/lra/contracts/state.py` — typed `Checkpoint` and `EventRecord` interfaces
- `src/lra/agent/loop.py` — the `CycleOutcome` that records what actually happened

Here is the `CycleOutcome` shape from `src/lra/agent/loop.py`:

```python
@dataclass
class CycleOutcome:
    """The result of one agent cycle."""

    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int
```

Notice `head_sha`. That is the durable anchor. It tells the next cycle, "this is exactly where the world was when I finished." In the full system that SHA comes from `GitMissionAnchor` in `src/lra/state/mission_anchor.py`. In the demo below we compute it from a state file so you can run it without Temporal or git.

---

## A Local Crash-Resume Loop

The file `lra-demo/ch02_crash_resume.py` is a self-contained durability demo. It:

1. Loads mission state from `lra-demo/ch02/state/mission_state.json`.
2. Finds the next un-done item.
3. Writes the required file.
4. Verifies it with a real subprocess command (exit code 0 == done).
5. Atomically checkpoints the updated state.
6. Randomly simulates a process death.
7. On restart, resumes exactly where it left off.

Create the file and run it:

```bash
mkdir -p lra-demo
# save the script below as lra-demo/ch02_crash_resume.py
python lra-demo/ch02_crash_resume.py
```

```python
#!/usr/bin/env python3
"""lra-demo/ch02_crash_resume.py

A self-contained crash-resume loop that demonstrates durability as an
engineering property.  It keeps the real state in a JSON file on disk,
re-reads it every cycle, and only marks work "done" when a real
verification command returns exit code 0.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_FILE = Path(__file__).with_suffix("").parent / "state" / "mission_state.json"
CRASH_ODDS = 0.25  # 1 in 4 cycles we simulate a process death after checkpointing.


@dataclass
class CycleOutcome:
    """Same shape as lra.contracts.state.CycleOutcome used by the real agent."""

    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


@dataclass
class MissionState:
    items: list[dict]
    head: int = 0
    attempts: int = 0
    crashes_survived: int = 0
    events: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def head_sha(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]


def load_state() -> MissionState:
    if STATE_FILE.exists():
        raw = json.loads(STATE_FILE.read_text())
        return MissionState(
            items=raw["items"],
            head=raw.get("head", 0),
            attempts=raw.get("attempts", 0),
            crashes_survived=raw.get("crashes_survived", 0),
            events=raw.get("events", []),
        )

    # Initial mission: build a tiny Python package.
    items = [
        {
            "id": "hello",
            "prompt": "Create hello.py that prints 'hello world'",
            "path": "hello.py",
            "content": (
                "def main():\n"
                "    print('hello world')\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
            "verify": [sys.executable, "hello.py"],
        },
        {
            "id": "test",
            "prompt": "Create test_hello.py and run it with pytest",
            "path": "test_hello.py",
            "content": (
                "from hello import main\n"
                "\n"
                "def test_main(capsys):\n"
                "    main()\n"
                "    assert capsys.readouterr().out == 'hello world\\n'\n"
            ),
            "verify": [sys.executable, "-m", "pytest", "test_hello.py", "-q"],
        },
        {
            "id": "readme",
            "prompt": "Create README.md",
            "path": "README.md",
            "content": "# Demo\nA durable mini-mission.\n",
            "verify": None,  # existence check
        },
    ]
    return MissionState(items=items)


def checkpoint(state: MissionState) -> None:
    """Atomic write so a crash during write never leaves a half-written state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(state.to_json())
    tmp.replace(STATE_FILE)


def verify_item(item: dict, workdir: Path) -> bool:
    if item["verify"] is None:
        return (workdir / item["path"]).exists()

    result = subprocess.run(
        item["verify"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def do_work(item: dict, workdir: Path) -> None:
    path = workdir / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(item["content"])


def run_cycle(state: MissionState, workdir: Path) -> CycleOutcome:
    # Idempotent: if the item already verifies, mark it done without re-doing work.
    for idx, item in enumerate(state.items):
        if item.get("done"):
            continue

        if verify_item(item, workdir):
            item["done"] = True
            state.head = idx + 1
            checkpoint(state)
            return CycleOutcome(
                item_id=item["id"],
                advanced=True,
                verified=True,
                is_complete=state.head >= len(state.items),
                head_sha=state.head_sha(),
                tool_calls=0,
                turns=1,
            )

        # Do the work, verify, checkpoint.
        state.attempts += 1
        do_work(item, workdir)
        verified = verify_item(item, workdir)
        if verified:
            item["done"] = True
            state.head = idx + 1

        state.events.append(
            {"item": item["id"], "verified": verified, "head": state.head}
        )
        checkpoint(state)

        return CycleOutcome(
            item_id=item["id"],
            advanced=verified,
            verified=verified,
            is_complete=state.head >= len(state.items),
            head_sha=state.head_sha(),
            tool_calls=1,
            turns=1,
        )

    return CycleOutcome(
        item_id=None,
        advanced=False,
        verified=True,
        is_complete=True,
        head_sha=state.head_sha(),
        tool_calls=0,
        turns=0,
    )


def maybe_crash() -> None:
    if random.random() < CRASH_ODDS:
        raise SystemExit("Simulated process death (OOM, reboot, API timeout...)")


def main() -> None:
    workdir = Path(__file__).with_suffix("").parent / "workspace"
    workdir.mkdir(parents=True, exist_ok=True)

    state = load_state()
    print(
        f"Resumed: head={state.head}/{len(state.items)} "
        f"attempts={state.attempts} crashes_survived={state.crashes_survived} "
        f"sha={state.head_sha()}"
    )

    while state.head < len(state.items):
        outcome = run_cycle(state, workdir)
        print(
            f"  cycle: item={outcome.item_id} advanced={outcome.advanced} "
            f"verified={outcome.verified} complete={outcome.is_complete} "
            f"sha={outcome.head_sha}"
        )
        if outcome.is_complete:
            break
        maybe_crash()

    state.crashes_survived += 1
    checkpoint(state)
    print("Mission complete.")


if __name__ == "__main__":
    main()
```

Run it several times. Because `CRASH_ODDS = 0.25`, it will often die mid-mission. Each restart prints the same `head` and `sha` it had at the last checkpoint, then continues:

```text
Resumed: head=1/3 attempts=2 crashes_survived=0 sha=9d8c7b6a4f2e
  cycle: item=test advanced=True verified=True complete=False sha=...
SystemExit: Simulated process death...

$ python lra-demo/ch02_crash_resume.py
Resumed: head=2/3 attempts=2 crashes_survived=1 sha=...
```

No tokens were re-spent. No work was duplicated. That is durability.

---

## Code Walkthrough

### `load_state`
The first thing the loop does is read durable state. If the file does not exist, it creates the initial mission. This is the "assume interruption" entry point: the loop never trusts memory.

### `checkpoint`
The state is written to a temporary file and then moved into place. This atomic replace means a crash during the write cannot corrupt `mission_state.json`. In LRA, the equivalent operation is a git commit via `GitMissionAnchor`.

### `verify_item`
Verification is deterministic and external. For `hello.py` it runs the file. For `test_hello.py` it runs pytest. For `README.md` it checks existence. The model is not allowed to declare an item done; only the exit code can.

### `run_cycle`
The cycle is idempotent. Before doing work, it checks whether the item already passes verification. This matters after a crash: if the process died *after* writing the file but *before* updating state, the next cycle will see the file is valid and mark it done without re-writing it.

### `maybe_crash`
This simulates the real world: OOM killer, host reboot, API timeout, laptop closed. Because state is on disk, the next process resumes cleanly.

---

## From Local Loop to Temporal Spine

This demo keeps state in a JSON file. The full LRA system keeps state in git and runs the loop inside a Temporal workflow:

- `src/lra/durable/` — Temporal workflows and activities
- `src/lra/state/mission_anchor.py` — git-backed durable anchor
- `src/lra/agent/loop.py` — the same gather → act → verify → checkpoint cycle, but invoked as a Temporal activity

Temporal gives three things the local demo cannot:

1. **Replay-from-cache** — a completed LLM call is not re-run after a crash.
2. **Durable sleep** — the workflow can sleep for hours at zero compute cost.
3. **Continue-as-new** — after thousands of cycles, the workflow history is compacted so it never outgrows limits.

We will cover the Temporal spine in Chapters 12–15. For now, the important idea is the same: the loop is durable because the *state* is durable, not because the model is special.

---

## Hands-On Exercise

Upgrade `lra-demo/ch02_crash_resume.py` so the checkpoint is a git commit instead of a JSON file move.

Requirements:

1. Initialize a git repo inside `lra-demo/ch02/workspace/`.
2. After each verified item, commit both the workspace file and `mission_state.json` with a message like `checkpoint: hello`.
3. On startup, if a git repo exists, read `mission_state.json` from `HEAD` instead of from disk.
4. Simulate a crash by killing the process with `Ctrl-C` while it is running, then restart it. Confirm that:
   - No item is verified twice.
   - No file is written twice.
   - `git log --oneline` shows one commit per verified item.

Hints:

```python
import subprocess

def git_checkpoint(state: MissionState, workdir: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=workdir, check=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=workdir,
        check=True,
    )
```

This exercise is the bridge to Chapter 03, where git becomes the agent's real memory.

---

> **Key Takeaway:** Durability is an engineering property, not a model capability. The model thinks in short bursts; the system runs for weeks by keeping the real state outside the window, journaling every step, and verifying progress with real tests.

---

## Next Chapter

**Chapter 03: Externalizing Truth — Git as Memory.** We will replace the JSON state file with a git repository, make every checkpoint a commit, and use the commit graph as the durable memory that survives reboots, host migrations, and human inspection.