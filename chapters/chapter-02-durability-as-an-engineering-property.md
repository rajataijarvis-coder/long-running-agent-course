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

In Chapter 01 we saw that a ChatGPT-style agent is basically one long conversation in one process. It dies when:

- the process restarts,
- the context window fills,
- an API call fails, or
- a tool returns something unexpected.

The fix is not to buy a bigger model or a bigger context window. The fix is to make the *system* durable. The LLM can still be a stateless, short-burst reasoner; the machinery around it must remember where it is, retry what failed, and resume exactly where it left off.

This chapter makes that idea concrete with a tiny local loop. The real LRA package wraps the same idea in Temporal workflows, git commits, and a team of agents, but the principle is identical: **write down the real state before you trust the next step.**

---

## Assume Interruption

The single most important design rule in LRA is **Assume Interruption**. Every cycle must be written as if the process could be killed at any moment and a *different* process could pick up the work later.

That changes how you write the loop:

- The model does not "remember" anything between cycles. It re-reads the state from disk.
- Every action that changes the world is journaled.
- Progress is declared only by a deterministic verifier, never by the model saying "I think I'm done."

In `src/lra/agent/loop.py` this is captured by `CycleOutcome`:

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

`head_sha` is the git commit that holds the state after the cycle. `verified` is the only field that can mark an item done. The model's opinion is not on the list.

---

## Volatile Context vs. Durable State

| Volatile (can be lost) | Durable (must survive) |
|---|---|
| The LLM context window | The git repo with code + tests |
| In-memory variables | `state/checklist.json` |
| The current process | `state/decisions.jsonl` |
| API response cache | `state/events.jsonl` |
| Cost so far in RAM | `governor/cost_ledger.json` |

The context window is a **lossy cache**. The real state lives outside it. In LRA the durable files are under `src/lra/state/` and are committed to git by the `GitMissionAnchor`. When the system resumes, it does not ask the model "what were we doing?" It reads the checklist, the latest commit SHA, and the event log.

This is why the README says:

> The model thinks in short bursts; the *system* runs for weeks by keeping the real state outside the model, journaling every step, and verifying progress with real tests.

---

## A Runnable Crash-Resume Loop

The file `lra-demo/ch02/crash_resume.py` demonstrates the core pattern without any LLM or Temporal. It writes a checkpoint after every work item, randomly kills itself mid-item, and resumes from the checkpoint.

Create the demo file:

```python
# lra-demo/ch02/crash_resume.py
"""A minimal crash-resume loop that demonstrates durability as an
engineering property.  It writes a JSON checkpoint after each item and
can resume from it even if the process is killed mid-item."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

STATE_DIR = Path(".lra-demo/ch02/state")
CHECKPOINT = STATE_DIR / "checkpoint.json"
ARTIFACTS = Path(".lra-demo/ch02/artifacts")

MISSION = [
    {
        "id": "hello",
        "task": "Create hello.py that prints 'hello'",
        "verify": f"python {ARTIFACTS / 'hello.py'}",
    },
    {
        "id": "test",
        "task": "Create test_hello.py and run pytest",
        "verify": f"python -m pytest {ARTIFACTS / 'test_hello.py'} -q",
    },
]


@dataclass
class Checkpoint:
    done: list[str]
    current: str | None
    attempts: int
    running: bool


def load_checkpoint() -> Checkpoint:
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text())
        return Checkpoint(**data)
    return Checkpoint(done=[], current=None, attempts=0, running=False)


def save_checkpoint(cp: Checkpoint) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(asdict(cp), indent=2))


def verify(item: dict) -> bool:
    """Deterministic verification: the command must exit 0."""
    result = subprocess.run(item["verify"], shell=True, capture_output=True, text=True)
    print(f"    verify command: {item['verify']}")
    print(f"    exit code: {result.returncode}")
    return result.returncode == 0


def do_work(item: dict) -> None:
    """In the real system this is the LLM + tool loop. Here we just write files."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if item["id"] == "hello":
        (ARTIFACTS / "hello.py").write_text('print("hello")\n')
    elif item["id"] == "test":
        (ARTIFACTS / "test_hello.py").write_text(
            "from hello import hello\n\ndef test_hello():\n    assert hello() == 'hello'\n"
        )
        # Make hello.py importable and testable.
        (ARTIFACTS / "hello.py").write_text(
            'def hello() -> str:\n    return "hello"\n\n\nif __name__ == "__main__":\n    print(hello())\n'
        )
    print(f"  [work] {item['task']}")
    time.sleep(0.3)


def maybe_crash() -> None:
    """Simulate a flaky environment by killing the process mid-item."""
    if random.random() < 0.4:
        print("  [crash] process terminated unexpectedly!")
        sys.exit(1)


def run_one_item(item: dict, cp: Checkpoint) -> bool:
    cp.current = item["id"]
    cp.running = True
    save_checkpoint(cp)

    do_work(item)
    maybe_crash()

    if verify(item):
        cp.done.append(item["id"])
        cp.attempts = 0
        cp.current = None
        cp.running = False
        save_checkpoint(cp)
        return True

    cp.attempts += 1
    save_checkpoint(cp)
    return False


def main() -> int:
    print("[resume] loading checkpoint...")
    cp = load_checkpoint()
    print(f"  done={cp.done} current={cp.current} attempts={cp.attempts}")

    for item in MISSION:
        if item["id"] in cp.done:
            print(f"[skip] {item['id']} already verified")
            continue

        while item["id"] not in cp.done and cp.attempts < 3:
            print(f"[cycle] {item['id']} attempt {cp.attempts + 1}")
            if run_one_item(item, cp):
                print(f"[done] {item['id']} verified")
                break
            print(f"[blocked] {item['id']} not verified, retrying")
        else:
            if item["id"] not in cp.done:
                print(f"[fail] {item['id']} exceeded retries")
                return 2

    print("[mission] complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run it repeatedly until it survives all the way through:

```bash
mkdir -p lra-demo/ch02
# create the file above, then:
while ! python lra-demo/ch02/crash_resume.py; do
  echo "--- resuming after crash ---"
done
```

You will see output like:

```text
[resume] loading checkpoint...
  done=['hello'] current='test' attempts=0
[cycle] test attempt 1
  [work] Create test_hello.py and run pytest
    verify command: python -m pytest .lra-demo/ch02/artifacts/test_hello.py -q
    exit code: 0
[done] test verified
[mission] complete
```

Even when the process dies, the checkpoint file `.lra-demo/ch02/state/checkpoint.json` tells the next process exactly where to continue.

---

## Connecting to the Real LRA

The demo above is intentionally tiny. In the real package the same pattern is implemented by `AgentLoop` in `src/lra/agent/loop.py`:

- `GitMissionAnchor` commits the repo after each cycle, so `head_sha` is the durable pointer.
- `CostLedger` tracks spend across restarts.
- `TraceRecorder` journals every model turn and tool call.
- `Verifier` runs real tests/lint/build/typecheck and returns a `VerificationResult`.

The durable spine is Temporal. `pyproject.toml` lists it as an optional dependency:

```toml
[project.optional-dependencies]
durable = ["temporalio>=1.7"]
```

Temporal turns the local checkpoint into a **workflow history**. Every LLM call and tool call becomes an *activity* that is journaled, retried, and replayed from cache. If the worker crashes, a new worker resumes from the exact history event where the old one stopped. Idle time is spent in *durable sleep*, which costs nothing.

So the local JSON checkpoint and the Temporal workflow are the same idea at different scales:

| Local demo | LRA production |
|---|---|
| `checkpoint.json` | Temporal workflow history |
| `save_checkpoint()` | Activity completion + cache |
| `load_checkpoint()` | Workflow replay |
| `maybe_crash()` | Host reboot, pod eviction, API outage |
| `verify()` | `src/lra/verify/` deterministic verifier |

---

## Hands-on Exercise

1. Run the crash-resume script in a loop until it completes:
   ```bash
   while ! python lra-demo/ch02/crash_resume.py; do echo "resuming..."; done
   ```
2. While it is running, press `Ctrl-C` manually. Inspect `.lra-demo/ch02/state/checkpoint.json` and confirm it records the last item attempted.
3. Change the `verify` command for the `hello` item to something that fails (for example, `python .lra-demo/ch02/artifacts/missing.py`). Run again and confirm the retry limit (3 attempts) is enforced.
4. Add a third mission item that creates `greet.py` and a test for it. Verify that the checkpoint correctly tracks all three items across multiple crashes.

---

> **Key Takeaway:** Durability is not something the model provides; it is something the system guarantees by keeping the real state outside the context window, journaling every step, and only declaring progress when a deterministic verifier passes. A reboot on day 12 should reconstruct situational awareness in seconds.

---

**Next Chapter:** Chapter 03: Externalizing Truth — Git as Memory. We will replace the JSON checkpoint with real git commits and introduce the **Mission Anchor**: the single SHA that tells a resumed process exactly where the truth lives.