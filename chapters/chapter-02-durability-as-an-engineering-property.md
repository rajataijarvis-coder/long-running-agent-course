# Chapter 02: Durability as an Engineering Property

> Durability is an engineering property, not a model capability.
> — Fareed Khan

## What We'll Cover

- Why a smarter model cannot fix a loop that dies on restart
- The "Assume Interruption" design rule and how it changes every decision
- Volatile context vs. durable state: what belongs in the LLM window and what belongs on disk
- The anatomy of a checkpoint using `CycleOutcome` from `src/lra/agent/loop.py`
- A runnable crash-resume loop in plain Python (`lra-demo/ch02_crash_resume.py`)
- How this local idea extends into the Temporal durable spine used in later chapters

---

## From a Chat Loop to a Durable Loop

In Chapter 01 we saw that a ChatGPT-style agent is essentially one long conversation. The model's context window holds the plan, the code, the test results, and the mistakes. If the process restarts, that conversation is gone. Calling a bigger model does not change that: the loop is still fragile.

Durability is the system's ability to keep a mission coherent across interruptions. An interruption can be a host reboot, an API rate-limit, a context-window overflow, a container eviction, or a developer redeploying the service. A durable agent treats every one of these as expected, not exceptional.

The central design rule is **Assume Interruption**. Every cycle must leave the real state on disk in a form that the next cycle can read cold. The model is a short-burst thinker; the system is a long-burst runner.

## Volatile Context vs. Durable State

The LLM context window is a **lossy cache**. It is fast to query and good for reasoning, but it is not the source of truth. Anything that must survive a restart belongs outside the window.

| Volatile (in the prompt) | Durable (on disk / in git) |
|---|---|
| Current item framing | The checklist and its done/blocked status |
| Recent tool output | The full tool log and decision journal |
| This cycle's reasoning | The committed code and test files |
| A summary of past failures | The failure trace and retry history |
| Cost estimate for this turn | The cumulative cost ledger |

In LRA this split is enforced by the `AgentLoop` in `src/lra/agent/loop.py`. The loop does not trust the model to remember where things stand. It re-reads the mission anchor, the checklist, and the event log at the start of every cycle. The result of a cycle is summarized in a small, serializable object:

```python
# From src/lra/agent/loop.py
@dataclass
class CycleOutcome:
    item_id: str | None
    advanced: bool          # did we move the mission forward?
    verified: bool          # did the deterministic verifier pass?
    is_complete: bool       # is the whole mission done?
    head_sha: str           # git commit that holds the real state
    tool_calls: int
    turns: int
```

`CycleOutcome` is the contract between the agent and the durable spine. It says: "Here is what happened this cycle, here is the commit that proves it, and here is whether the mission is finished." The model's opinion is not enough; the checkpoint is.

## A Runnable Crash-Resume Loop

The full LRA system uses Temporal for durable execution and git for durable state. Both are introduced in later chapters. Before we add those dependencies, we can prove the idea with a plain Python script that writes checkpoints to a JSONL file.

Create `lra-demo/ch02_crash_resume.py`:

```python
#!/usr/bin/env python3
"""lra-demo/ch02_crash_resume.py

A minimal crash-resume agent loop. It externalizes mission state to a JSONL
checkpoint file so a restarted process can reconstruct exactly where it was.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

WORK_DIR = Path(".lra/workspaces/ch02_demo")
CHECKPOINT_FILE = WORK_DIR / "checkpoints.jsonl"


@dataclass
class Item:
    id: str
    description: str
    done: bool = False
    attempts: int = 0
    last_error: str = ""


@dataclass
class MissionState:
    items: list[Item]
    current_index: int = 0
    cycles: int = 0
    head_sha: str = "local"


@dataclass
class Checkpoint:
    state: MissionState
    cycle: int
    timestamp: float
    note: str = ""


def load_state() -> MissionState:
    """Resume from the latest checkpoint, or start a fresh mission."""
    if CHECKPOINT_FILE.exists():
        with CHECKPOINT_FILE.open() as f:
            lines = [line.strip() for line in f if line.strip()]
        if lines:
            last = json.loads(lines[-1])
            items = [Item(**i) for i in last["state"]["items"]]
            state = MissionState(
                items=items,
                current_index=last["state"]["current_index"],
                cycles=last["state"]["cycles"],
                head_sha=last["state"]["head_sha"],
            )
            print(f"[resume] loaded checkpoint cycle={state.cycles} index={state.current_index}")
            return state

    return MissionState(
        items=[
            Item(id="hello", description="Write hello.py with a hello() function"),
            Item(id="verify", description="Run a Python one-liner that asserts hello() works"),
        ]
    )


def save_checkpoint(state: MissionState, note: str) -> None:
    """Append a checkpoint. We never overwrite the old one, so we can audit history."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cp = Checkpoint(state=state, cycle=state.cycles, timestamp=time.time(), note=note)
    with CHECKPOINT_FILE.open("a") as f:
        f.write(json.dumps(asdict(cp), default=str) + "\n")
    print(f"[checkpoint] cycle={cp.cycle} item={state.current_index} note={note}")


def write_hello(work_dir: Path) -> None:
    (work_dir / "hello.py").write_text('def hello():\n    return "hello"\n')


def verify_hello(work_dir: Path) -> tuple[bool, str]:
    """Deterministic verification: import the function and check its return value."""
    import subprocess

    cmd = [
        sys.executable,
        "-c",
        "from hello import hello; assert hello() == 'hello', hello()",
    ]
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=30)
    ok = result.returncode == 0
    detail = (result.stdout + result.stderr)[:300] if not ok else ""
    return ok, detail


def run_one_cycle(state: MissionState, work_dir: Path, crash_after: int | None) -> bool:
    """Run one cycle. Returns True if there is still work to do."""
    if state.current_index >= len(state.items):
        return False

    state.cycles += 1
    item = state.items[state.current_index]
    print(f"\n[cycle {state.cycles}] working on {item.id}: {item.description}")
    item.attempts += 1

    try:
        if item.id == "hello":
            write_hello(work_dir)
        elif item.id == "verify":
            pass  # the verification step does the work

        ok, detail = verify_hello(work_dir) if item.id == "verify" else (True, "")
        if item.id == "hello":
            ok, detail = verify_hello(work_dir)

        if ok:
            item.done = True
            item.last_error = ""
            state.current_index += 1
            save_checkpoint(state, note=f"verified {item.id}")
        else:
            item.last_error = detail
            save_checkpoint(state, note=f"blocked {item.id}: {detail[:80]}")
    except Exception as exc:
        save_checkpoint(state, note=f"crashed during {item.id}: {exc}")
        raise

    if crash_after is not None and state.cycles >= crash_after:
        print(f"[inject] simulated crash after cycle {state.cycles}")
        raise SystemExit(1)

    return state.current_index < len(state.items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crash-resume agent loop demo")
    parser.add_argument("--crash-after", type=int, default=None, help="simulate crash after N cycles")
    parser.add_argument("--workdir", type=Path, default=WORK_DIR)
    args = parser.parse_args()

    work_dir = args.workdir
    work_dir.mkdir(parents=True, exist_ok=True)

    state = load_state()
    done_count = sum(1 for i in state.items if i.done)
    print(f"[start] mission has {len(state.items)} items, {done_count} done")

    while run_one_cycle(state, work_dir, args.crash_after):
        time.sleep(0.5)

    if all(item.done for item in state.items):
        print(f"\n[done] mission complete in {state.cycles} cycles")
    else:
        print(f"\n[paused] item {state.current_index} is not done yet")


if __name__ == "__main__":
    main()
```

## Code Walkthrough

1. **`MissionState` and `Item`** are the durable model of progress. They are plain dataclasses that serialize cleanly to JSON.
2. **`load_state`** reads the last line of `checkpoints.jsonl`. If the file exists, the process resumes from the latest snapshot. If not, it starts fresh. This is the "assume interruption" entry point.
3. **`save_checkpoint`** appends a new checkpoint after every significant state change. We append rather than overwrite so we keep an audit trail. In the real LRA system this is replaced by a git commit plus a Temporal event, but the principle is identical.
4. **`verify_hello`** is a tiny deterministic verifier. It does not ask the model whether the code looks correct; it imports the function and asserts its output. This is the seed of the larger verifier system we build in Chapter 06.
5. **`run_one_cycle`** does one unit of work, verifies it, and checkpoints the result. The `--crash-after` flag lets you simulate an interruption at a precise point.

Run it once with a simulated crash:

```bash
python lra-demo/ch02_crash_resume.py --crash-after 1
```

You will see output like:

```text
[start] mission has 2 items, 0 done
[cycle 1] working on hello: Write hello.py with a hello() function
[checkpoint] cycle=1 item=1 note=verified hello
[inject] simulated crash after cycle 1
```

The process exits, but the checkpoint is already on disk. Now run it again without the crash flag:

```bash
python lra-demo/ch02_crash_resume.py
```

It resumes:

```text
[resume] loaded checkpoint cycle=1 index=1
[start] mission has 2 items, 1 done

[cycle 2] working on verify: Run a Python one-liner that asserts hello() works
[checkpoint] cycle=2 item=2 note=verified verify

[done] mission complete in 2 cycles
```

The second run did not re-write `hello.py` or re-spend any tokens. It read the durable state and continued. That is durability.

Inspect the checkpoint history:

```bash
cat .lra/workspaces/ch02_demo/checkpoints.jsonl
```

Each line is a complete snapshot of the mission at a point in time.

## Mapping to the Real LRA System

This script is intentionally minimal. In the full codebase the same pattern is implemented by three layers:

- **`src/lra/state/mission_anchor.py`** — writes the durable state to a git repo (checklist, decisions, code, tests) and returns a commit SHA.
- **`src/lra/agent/loop.py`** — the `AgentLoop` class runs the gather → act → verify → checkpoint cycle and produces a `CycleOutcome`.
- **`src/lra/durable/`** — the Temporal workflows and activities that call `AgentLoop`. Temporal journals every activity, replays from cache, and resumes after process crashes. We cover this in Chapters 12–15.

The JSONL checkpoint in this chapter is a stand-in for the git commit + Temporal journal combination. The design rule is the same: **never let the only copy of progress live inside the model's head.**

## Hands-On Exercise

1. Run the demo with a simulated crash after the first cycle and confirm it resumes correctly:

   ```bash
   rm -rf .lra/workspaces/ch02_demo
   python lra-demo/ch02_crash_resume.py --crash-after 1
   python lra-demo/ch02_crash_resume.py
   ```

2. Kill the process manually during the second cycle (`Ctrl-C` or `kill`) and rerun. Observe that it resumes from the last checkpoint rather than starting over.

3. Add a third item to the mission that writes a `README.md` file and verifies that it contains the word `hello`. Run the mission, inject a crash, and verify that the new item is resumed correctly.

4. (Stretch) Modify `load_state` to detect a corrupted final checkpoint line and fall back to the previous good line. This is a preview of the replay and recovery logic we will add with Temporal.

## Key Takeaway

> A durable agent does not remember its mission because the model has a good memory. It remembers because the system writes the mission down after every cycle, and the next cycle starts by reading it back. Durability is an engineering property of the loop, not a capability of the LLM.

## Next Chapter

In **Chapter 03: Externalizing Truth — Git as Memory**, we replace the JSONL checkpoint with a real git repository. We will see why git is the right source of truth for long-horizon software missions, how LRA structures its saved-state files, and how a reboot on day 12 reconstructs situational awareness in seconds.