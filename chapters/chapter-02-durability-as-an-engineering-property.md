# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why durability belongs to the system, not the LLM
- The "Assume Interruption" design rule and how it changes every loop
- Volatile context vs. durable state: what lives in the window and what lives on disk/git
- The anatomy of a checkpoint using `CycleOutcome`
- A runnable crash-resume loop in plain Python
- How the durable spine (Temporal) extends this local idea to weeks-long missions

---

## From Chat Loop to Durable Loop

Chapter 01 showed that a ChatGPT-style agent is a single, fragile loop: one failed API call, one full context window, or one process restart and the run is dead. The fix is not a smarter model. It is a system that **assumes interruption** and makes every cycle re-entrant.

In `lra-demo/src/lra/agent/loop.py` the inner agent loop is deliberately small: it gathers state, acts through tools, verifies with real tests, and then checkpoints. The loop itself does not trust memory. It trusts the durable record it just wrote.

## Assume Interruption

Design every cycle as if the process could be killed at any moment. After restart, the agent must reconstruct situational awareness from durable artifacts, not from the model's context window.

This rule changes three things:

1. **Real state lives outside the model.** The checklist, decision log, and git repo are the truth; the context window is only a cache.
2. **Every cycle ends with a checkpoint.** A cycle is not done until its result is written somewhere durable.
3. **No in-flight work is trusted.** Only committed, verified state counts.

## Volatile Context vs. Durable State

| Volatile (can be lost) | Durable (must survive) |
|---|---|
| Model context window | Git repo in `lra-demo/.lra/workspaces/...` |
| In-memory variables | `Checkpoint` and `EventRecord` in `lra-demo/src/lra/contracts/state.py` |
| Current LLM response | `CycleOutcome` returned by `AgentLoop.run()` |
| Process stack | Temporal workflow history |

The model re-reads the durable state every cycle. A reboot on day 12 reconstructs awareness in seconds because the truth was never inside the process.

## Anatomy of a Checkpoint

In `lra-demo/src/lra/agent/loop.py` a single cycle returns this:

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

Each field is deliberate:

- `item_id` — which checklist item was attacked.
- `advanced` — did the repo move forward?
- `verified` — did the deterministic verifier pass?
- `is_complete` — is the whole mission done?
- `head_sha` — the git commit that pins the resulting state.
- `tool_calls` / `turns` — cost and effort telemetry.

The `head_sha` is the anchor. It lets a resumed process know exactly which commit to check out before doing anything else.

## A Runnable Crash-Resume Loop

The file `lra-demo/scripts/crash_resume_demo.py` below is a minimal, self-contained version of the same idea. It processes a checklist, runs a deterministic verifier, commits progress to git, and writes a JSON checkpoint. Kill it mid-run with `kill -9` and restart it: it resumes at the last verified item.

```python
#!/usr/bin/env python3
"""crash_resume_demo.py — a minimal re-entrant agent loop.

This is a stripped-down version of the idea in lra-demo/src/lra/agent/loop.py.
It keeps durable state in a JSON checkpoint and a git repo so it survives
SIGKILL and resumes exactly where it left off.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

WORK_DIR = Path(".lra/crash_resume_demo")
CHECKPOINT_FILE = WORK_DIR / "checkpoint.json"
REPO_DIR = WORK_DIR / "repo"


@dataclass
class Checkpoint:
    item_index: int
    attempts: dict[str, int]
    last_commit: str | None


def run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_DIR), *args],
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def init_repo() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    REPO_DIR.mkdir(exist_ok=True)
    if not (REPO_DIR / ".git").exists():
        subprocess.run(["git", "init", "-q", str(REPO_DIR)], check=True)
        run_git("config", "user.email", "agent@lra.local")
        run_git("config", "user.name", "LRA Agent")
        (REPO_DIR / "log.txt").write_text("mission log\n")
        run_git("add", ".")
        run_git("commit", "-q", "-m", "init")


def read_checkpoint() -> Checkpoint:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return Checkpoint(
            item_index=data["item_index"],
            attempts=data["attempts"],
            last_commit=data["last_commit"],
        )
    head = run_git("rev-parse", "HEAD")
    return Checkpoint(item_index=0, attempts={}, last_commit=head)


def save_checkpoint(cp: Checkpoint) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(asdict(cp), indent=2))


def verify(item: str, attempt: int) -> bool:
    """Deterministic verifier: 'fix flaky lint' only passes on attempt 2+."""
    if item == "fix flaky lint":
        return attempt >= 2
    return "fail" not in item.lower()


def act(item: str, attempt: int) -> None:
    """Simulate doing work. In the real system this is a tool/sandbox call."""
    print(f"  acting on {item!r} (attempt {attempt})...")
    time.sleep(1)
    with (REPO_DIR / "log.txt").open("a") as f:
        f.write(f"attempt {attempt}: {item}\n")


def checkpoint(cp: Checkpoint, item: str, passed: bool, attempt: int) -> str:
    """Commit the result and write the checkpoint."""
    msg = f"{'verify' if passed else 'attempt'}: {item} (#{attempt})"
    run_git("add", "log.txt")
    run_git("commit", "-q", "-m", msg)
    head = run_git("rev-parse", "HEAD")
    cp.last_commit = head
    if passed:
        cp.item_index += 1
    save_checkpoint(cp)
    return head


def main() -> None:
    init_repo()
    items = ["write README", "add tests", "fix flaky lint", "ship it"]

    cp = read_checkpoint()
    cp.last_commit = run_git("rev-parse", "HEAD")
    print(f"Resuming at item {cp.item_index}/{len(items)}, repo at {cp.last_commit[:7]}")

    while cp.item_index < len(items):
        item = items[cp.item_index]
        cp.attempts[item] = cp.attempts.get(item, 0) + 1

        act(item, cp.attempts[item])
        passed = verify(item, cp.attempts[item])
        head = checkpoint(cp, item, passed, cp.attempts[item])

        print(f"  {'verified' if passed else 'blocked'} -> checkpoint {head[:7]}")

    print("Mission complete.")


if __name__ == "__main__":
    main()
```

Run it:

```bash
cd lra-demo
python scripts/crash_resume_demo.py
```

While it is sleeping on `"fix flaky lint"`, open another terminal and kill the process:

```bash
pkill -9 -f crash_resume_demo.py
```

Then run it again. It reads `checkpoint.json`, checks out the recorded commit, and resumes from the next unverified item. No work is repeated.

## How Temporal Extends This Local Idea

The local checkpoint above is the same contract the full system uses, but Temporal provides the durable spine:

- Every LLM call, tool call, and verification is an **activity**.
- Temporal journals the activity inputs and outputs.
- On crash, the workflow **replays from cache**; no tokens are re-spent.
- Idle time is spent in **durable sleep**, costing nothing.

We will build that spine in Chapters 12–15. For now, the mental model is enough: a durable agent is a state machine whose transitions are committed, versioned, and replayable.

## Hands-on Exercise

1. Run `lra-demo/scripts/crash_resume_demo.py` and let it finish once. Inspect the git log with:

   ```bash
   git -C .lra/crash_resume_demo/repo log --oneline
   ```

2. Delete `checkpoint.json` but keep the repo. Run the script again. What happens? Why is the git history enough to reconstruct state?

3. Add a new checklist item that intentionally fails on the first attempt and passes on the second. Modify `verify()` to implement the rule.

4. Kill the process with `kill -9` during the `"fix flaky lint"` item, then restart it. Confirm it does not re-run already-verified items.

## Key Takeaway

> Durability is an engineering property, not a model capability. The model thinks in short bursts; the system runs for weeks by keeping real state outside the window, journaling every step, and making every cycle re-entrant.

## Next Chapter

**Chapter 03: Externalizing Truth — Git as Memory.** We will replace the ad-hoc JSON checkpoint with git as the single source of truth and introduce the Mission Anchor that every cycle consults before it acts.