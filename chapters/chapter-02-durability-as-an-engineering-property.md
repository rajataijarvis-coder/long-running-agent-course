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

## From Model Magic to System Property

In Chapter 01 we saw that a ChatGPT-style agent is basically one long conversation. If the process dies, the context window overflows, or an API call flakes, the whole run is gone. The model itself cannot fix that: it has no memory of yesterday's work unless yesterday's work is pasted back into the prompt.

Durability is the system's answer to that problem. It is not a smarter prompt or a bigger model. It is the property that the *agent organization* keeps its real state somewhere safe, journals every step, and can resume exactly where it left off after a crash, a reboot, or a weekend of idle time.

The design rule that makes this concrete is **Assume Interruption**. Every cycle of the agent must be written as if the process could be killed at the end of it. That changes how you write loops:

1. Read the current truth from durable storage at the start of the cycle.
2. Do one bounded unit of work.
3. Verify it with something deterministic.
4. Write the result back to durable storage.
5. Only then consider the cycle done.

If the process dies between steps 4 and 5, the next process reads the same truth and continues. Nothing is lost.

## Volatile Context vs. Durable State

The LLM context window is a **lossy cache**. It is fast to query, but it is limited, expensive, and disappears when the process restarts. The real state lives outside the window.

In `lra-demo/src/lra/state/mission_anchor.py` the real state is a git repo plus a structured log. In this chapter we will use a tiny JSON stand-in so you can run the demo without setting up git or Temporal. The semantics are the same:

- **Mission state**: the checklist, what is done, what is blocked, and the current "head" identifier.
- **Checkpoint**: a saved `CycleOutcome` that records what happened in one cycle.
- **Resume**: the next process loads the checkpoint and continues.

The `CycleOutcome` dataclass in `lra-demo/src/lra/agent/loop.py` is the contract:

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

Those fields are intentionally boring engineering facts, not model prose. Did we move forward? Was the work verified? What is the current head? How many turns did it cost? That is the durable record.

## A Runnable Crash-Resume Loop in Plain Python

The script below is a self-contained version of the same loop that `lra-demo/src/lra/agent/loop.py` runs inside an activity. It writes artifacts, runs real tests as a deterministic verifier, checkpoints to JSON, and can simulate a crash so you can watch it resume.

Save it as `durable_loop_demo.py` next to your `lra-demo/` checkout.

```python
# durable_loop_demo.py
# Mirrors the checkpoint semantics of lra-demo/src/lra/agent/loop.py
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

WORK_DIR = Path(".lra-demo-workdir")
STATE_FILE = WORK_DIR / "anchor.json"


@dataclass
class CycleOutcome:
    """Same fields as lra-demo/src/lra/agent/loop.py::CycleOutcome."""
    item_id: str | None
    advanced: bool
    verified: bool
    is_complete: bool
    head_sha: str
    tool_calls: int
    turns: int


class JsonMissionAnchor:
    """Stand-in for lra-demo/src/lra/state/mission_anchor.py::GitMissionAnchor.
    The real anchor commits to git; this one commits to JSON so the demo runs
    with no git setup required.
    """
    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"items": [], "cycle": 0, "done": [], "head": "0" * 7}

    def save(self, state: dict) -> str:
        text = json.dumps(state, indent=2, sort_keys=True)
        self.path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()[:7]


def verify_with_test(item: dict, workdir: Path) -> bool:
    """Deterministic verifier: a green exit code is the only truth."""
    script = workdir / item["verify"]
    if not script.exists():
        return False
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def run_one_cycle(anchor: JsonMissionAnchor, crash_after: int | None = None) -> CycleOutcome:
    state = anchor.load()
    todo = [i for i in state["items"] if i["id"] not in state["done"]]

    if not todo:
        return CycleOutcome(
            item_id=None,
            advanced=False,
            verified=False,
            is_complete=True,
            head_sha=state.get("head", "0" * 7),
            tool_calls=0,
            turns=0,
        )

    item = todo[0]
    print(f"[cycle {state['cycle']}] item={item['id']}")

    # Act: write the artifact. In the real system this is a tool call.
    artifact = anchor.path.parent / item["artifact"]
    artifact.write_text(item["content"])
    advanced = True

    # Verify: real test, real exit code.
    verified = verify_with_test(item, anchor.path.parent)

    if verified:
        state["done"].append(item["id"])
        print("  -> VERIFIED")
    else:
        print("  -> BLOCKED (will retry next cycle)")

    state["cycle"] += 1
    state["head"] = anchor.save(state)

    # Simulate a host crash/reboot right after checkpointing.
    if crash_after is not None and state["cycle"] >= crash_after:
        print("  -> SIMULATED CRASH (process exits)")
        raise SystemExit(1)

    return CycleOutcome(
        item_id=item["id"],
        advanced=advanced,
        verified=verified,
        is_complete=False,
        head_sha=state["head"],
        tool_calls=1,
        turns=1,
    )


def seed_mission(workdir: Path) -> None:
    anchor = JsonMissionAnchor(workdir / "anchor.json")
    anchor.save(
        {
            "items": [
                {
                    "id": "hello",
                    "artifact": "hello.py",
                    "content": "print('hello')",
                    "verify": "test_hello.py",
                },
                {
                    "id": "add",
                    "artifact": "math_ops.py",
                    "content": "def add(a, b):\n    return a + b\n",
                    "verify": "test_math.py",
                },
            ],
            "cycle": 0,
            "done": [],
            "head": "0" * 7,
        }
    )
    (workdir / "test_hello.py").write_text(
        "import subprocess, sys\n"
        "r = subprocess.run([sys.executable, 'hello.py'], capture_output=True, text=True)\n"
        "assert r.stdout.strip() == 'hello', r.stdout\n"
    )
    (workdir / "test_math.py").write_text(
        "import math_ops\n"
        "assert math_ops.add(2, 3) == 5\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--crash-after", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        shutil.rmtree(WORK_DIR, ignore_errors=True)

    if not STATE_FILE.exists():
        seed_mission(WORK_DIR)

    anchor = JsonMissionAnchor(STATE_FILE)
    while True:
        outcome = run_one_cycle(anchor, crash_after=args.crash_after)
        if outcome.is_complete:
            print(f"mission complete: head={outcome.head_sha}")
            break
```

### What the script demonstrates

1. **Assume Interruption**: `run_one_cycle` always starts by loading the anchor. It never assumes it knows the state from a previous prompt.
2. **Bounded work**: each cycle touches exactly one checklist item.
3. **Deterministic verification**: `verify_with_test` runs a real Python test and treats the exit code as ground truth. The model is not allowed to declare success.
4. **Checkpoint**: after verification, the anchor is saved. The `head_sha` is a content hash of the saved state, exactly the kind of identifier the real git anchor uses.
5. **Crash resume**: `--crash-after=1` kills the process after the first checkpoint. Rerunning the script loads the saved state and continues with the next item.

Run it:

```bash
python durable_loop_demo.py --reset
```

You should see both items verified and a final `mission complete` line.

Now simulate a crash:

```bash
python durable_loop_demo.py --reset
python durable_loop_demo.py --crash-after=1
python durable_loop_demo.py
```

The first full run finishes. The second run dies after verifying `hello`. The third run picks up at `add` and finishes. No item is repeated.

## How the Durable Spine Extends This

The plain-Python loop above is the local kernel of the idea. In `lra-demo/src/lra/durable/` the same pattern is lifted into a Temporal workflow:

- The outer `MissionWorkflow` is the durable scheduler.
- Each call to the model, each tool execution, and each verification is an **activity**.
- Temporal journals every activity completion. If the worker crashes, a new worker replays the journal from the last checkpoint instead of re-running the work.
- Idle time is spent in **durable sleep**, which costs nothing and survives process restarts.

That means the local `CycleOutcome` you just ran on your laptop is the same shape of fact that a week-long mission in Temporal uses. The engineering property is identical; only the runtime scale changes.

## Hands-On Exercise

1. Run the demo to completion and inspect `.lra-demo-workdir/anchor.json`. Confirm that `done` contains both item IDs and that `head` changed each cycle.
2. Run with `--crash-after=1`, then rerun without the flag. Verify that the second process resumes at the second item and that `hello` is not rebuilt.
3. Make one test fail on purpose (for example, change the assertion in `test_math.py` to `== 6`). Run the demo and observe that the item becomes `BLOCKED` but the checkpoint is still written. Rerun and confirm it retries the same item every cycle until you fix the test.
4. Extra credit: convert the JSON anchor to a real git anchor. Make `save()` run `git add . && git commit -m "checkpoint cycle N"` and use the commit SHA as `head_sha`. The rest of the loop should not need to change.

## Key Takeaway

> Durability is an engineering property, not a model capability. The model thinks in short bursts; the system runs for weeks by keeping the real state outside the window, journaling every step, and verifying progress with real tests.

## Next Chapter

In Chapter 03 we replace the JSON stand-in with the real source of truth: **git**. You will learn why git is the perfect memory for a long-running agent, how the mission anchor structures commits, and how to read the entire history of a mission with plain `git log`.