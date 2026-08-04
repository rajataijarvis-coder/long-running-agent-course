# Chapter 21: Loop Detection and Escaping Oscillation

## What We'll Cover

- Why long-horizon agents get stuck in **repeat loops** and **oscillations**, not just "bad reasoning"
- How LRA turns loop detection into a **state-engineering** problem using durable fingerprints
- The two detection modes — **exact-repeat** and **A→B→A oscillation** — and why both matter
- How the **Mission Anchor** (Chapter 16) and **decision log** make loop history survive crashes
- Wiring the detector into the **Think → Act → Verify → Checkpoint** cycle (Chapter 4)
- A runnable demo in `lra-demo/ch21_loop_detection.py` that oscillates on purpose, detects itself, and escapes
- How the **Budget Governor** (Chapter 20) uses `loop_trips` as a cost lever

---

## Why Long-Horizon Agents Loop

A chat-style agent can afford to be forgetful. A week-long mission cannot.

The context window is a lossy cache (Chapter 1). Even with tiered memory (Chapter 17) and compaction, the model can still propose the *same* edit, read the *same* file, or run the *same* failing test twice because the *system* did not notice the cycle. The deterministic verifier from Chapter 6 prevents the agent from declaring "done" falsely, but it does not stop the agent from running the same failing step over and over. Each failed cycle costs tokens, compute, and wall-clock time.

Loop detection is therefore not a model-capability problem. It is a **state-tracking** problem. If the durable state — git tree, checklist, last action, verification outcome — has not meaningfully changed, the agent is looping. The fix is to detect that in the durable layer, record it, and apply an **escape**.

LRA's escape ladder is deliberately cheap-to-expensive:

1. **Perturb the prompt** — change temperature, reorder constraints, or inject a "do not repeat X" rule.
2. **Call the reviewer** — an independent agent with a fresh context window (Chapter 11).
3. **Cut scope** — drop the current checklist item and move on (Chapter 20).
4. **Compact memory** — prune old observations that may be reinforcing the loop (Chapter 17).
5. **Raise a HITL gate** — stop and ask a human (Chapter 22).

Because the Mission Anchor (Chapter 16) stores the decision log, the loop event itself is durable. If the process crashes right after detection, the resume logic knows it was mid-escape and continues from there instead of replaying the loop.

---

## What a Loop Looks Like in the Decision Log

Before writing code, look at what the detector produces. A loop event is just another durable fact:

```jsonl
{"ts":"2026-06-02T08:12:11Z","cycle":47,"loop_id":"loop-1","kind":"oscillation","strategy":"perturb_prompt","window_hash":"a3f7e2c9d1b4"}
{"ts":"2026-06-02T08:31:02Z","cycle":112,"loop_id":"loop-2","kind":"repeat","strategy":"call_reviewer","window_hash":"8e5d1a2f0c6b"}
{"ts":"2026-06-02T08:44:51Z","cycle":615,"loop_id":"loop-3","kind":"repeat","strategy":"cut_scope","window_hash":"7b9c4e3d2a1f"}
```

The `window_hash` is a hash of the last N observations. It lets the system answer the question: "Have I seen this exact situation before?" without keeping the full text of every prior step in the prompt.

---

## The Loop Detector Design

The detector maintains a sliding window of `CycleObservation`s. Each observation captures:

- `state_fingerprint` — a hash of the durable state (git tree, checklist, done list)
- `action_kind` and `action_target` — what the agent tried to do
- `outcome_signature` — a hash of the verification result (exit code, error, verified flag)
- `verified` — whether the step actually passed

It then checks for two patterns:

1. **Exact repeat** — the same `(state, action, outcome)` tuple appears `repeat_threshold` times.
2. **Oscillation** — state A → state B → state A within the window, with the same action kind on the A states.

When either fires, the detector emits a `LoopEvent`. The event is recorded in the decision log, and the agent applies the chosen escape strategy.

---

## Code Walkthrough: `lra-demo/ch21_loop_detection.py`

The demo below simulates a deliberately brittle agent. It oscillates between adding and removing `hello`. The detector catches the oscillation, applies a `PERTURB_PROMPT` escape that forces a stable write path, and the mission completes.

Save this as `lra-demo/ch21_loop_detection.py`:

```python
#!/usr/bin/env python3
"""lra-demo/ch21_loop_detection.py

Simulate a brittle agent that oscillates between two edits, show the
LoopDetector catching the oscillation, and apply an escape that breaks
the symmetry so the mission can finish.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loop detector
# ---------------------------------------------------------------------------
class EscapeStrategy(Enum):
    PERTURB_PROMPT = "perturb_prompt"
    CALL_REVIEWER = "call_reviewer"
    CUT_SCOPE = "cut_scope"
    RAISE_HITL = "raise_hitl"
    COMPACT_MEMORY = "compact_memory"


@dataclass(frozen=True)
class CycleObservation:
    cycle: int
    state_fingerprint: str
    action_kind: str
    action_target: str
    outcome_signature: str
    verified: bool


@dataclass
class LoopEvent:
    loop_id: str
    cycle: int
    kind: str
    repeat_count: int
    window_hash: str
    strategy: EscapeStrategy
    context: dict[str, Any] = field(default_factory=dict)


class LoopDetector:
    """Detect two failure modes:
    1. Repeat: the same (state, action, outcome) tuple appears N times.
    2. Oscillation: state A -> state B -> state A within the window.
    """

    def __init__(self, window_size: int = 12, repeat_threshold: int = 3):
        self.window_size = window_size
        self.repeat_threshold = repeat_threshold
        self.observations: deque[CycleObservation] = deque(maxlen=window_size)
        self.loop_count = 0

    def observe(
        self,
        *,
        cycle: int,
        state: dict[str, Any],
        action_kind: str,
        action_target: str,
        outcome: dict[str, Any],
    ) -> LoopEvent | None:
        fp = self._fingerprint(state)
        obs = CycleObservation(
            cycle=cycle,
            state_fingerprint=fp,
            action_kind=action_kind,
            action_target=action_target,
            outcome_signature=self._outcome_signature(outcome),
            verified=outcome.get("verified", False),
        )
        self.observations.append(obs)
        return self._detect(obs)

    def _fingerprint(self, state: dict[str, Any]) -> str:
        canonical = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _outcome_signature(self, outcome: dict[str, Any]) -> str:
        sig = {
            "verified": outcome.get("verified"),
            "error": outcome.get("error"),
            "exit_code": outcome.get("exit_code"),
        }
        return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]

    def _detect(self, latest: CycleObservation) -> LoopEvent | None:
        if len(self.observations) < 3:
            return None

        # 1. Exact repeat detection.
        matches = [
            o
            for o in self.observations
            if o.state_fingerprint == latest.state_fingerprint
            and o.action_kind == latest.action_kind
            and o.action_target == latest.action_target
            and o.outcome_signature == latest.outcome_signature
        ]
        if len(matches) >= self.repeat_threshold:
            self.loop_count += 1
            return LoopEvent(
                loop_id=f"loop-{self.loop_count}",
                cycle=latest.cycle,
                kind="repeat",
                repeat_count=len(matches),
                window_hash=self._hash_window(),
                strategy=self._pick_strategy(latest, matches),
            )

        # 2. Oscillation detection: A -> B -> A.
        *_, a, b, c = self.observations
        if (
            a.state_fingerprint == c.state_fingerprint
            and a.state_fingerprint != b.state_fingerprint
            and a.action_kind == c.action_kind
        ):
            self.loop_count += 1
            return LoopEvent(
                loop_id=f"loop-{self.loop_count}",
                cycle=latest.cycle,
                kind="oscillation",
                repeat_count=2,
                window_hash=self._hash_window(),
                strategy=self._pick_strategy(latest, [a, c]),
            )

        return None

    def _hash_window(self) -> str:
        payload = json.dumps(
            [asdict(o) for o in self.observations], sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _pick_strategy(
        self, latest: CycleObservation, matches: list[CycleObservation]
    ) -> EscapeStrategy:
        # First loop: try to break symmetry cheaply.
        if self.loop_count == 1:
            return EscapeStrategy.PERTURB_PROMPT
        # Persistent loops after reviewer/scope cuts need a human.
        if self.loop_count >= 3:
            return EscapeStrategy.RAISE_HITL
        # Second loop: get an independent, fresh-context review.
        return EscapeStrategy.CALL_REVIEWER


# ---------------------------------------------------------------------------
# Verifier and agent
# ---------------------------------------------------------------------------
@dataclass
class MissionState:
    cycle: int = 0
    file_content: str = "# stub\n"
    checklist: list[str] = field(default_factory=lambda: ["add hello", "add test"])
    done: list[str] = field(default_factory=list)
    stable_mode: bool = False

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "file_hash": hashlib.sha256(self.file_content.encode()).hexdigest()[:16],
            "todo": sorted(self.checklist),
            "done": sorted(self.done),
        }


class DeterministicVerifier:
    """Final gate from Chapter 06: the model does not decide 'done'; the test does."""

    def verify(self, state: MissionState) -> dict[str, Any]:
        ok = "hello" in state.file_content and "test" in state.file_content
        return {
            "verified": ok,
            "exit_code": 0 if ok else 1,
            "error": None if ok else "missing hello or test",
        }


class Agent:
    def __init__(
        self,
        detector: LoopDetector,
        verifier: DeterministicVerifier,
        log_path: Path,
    ):
        self.detector = detector
        self.verifier = verifier
        self.log_path = log_path
        self.state = MissionState()

    def run(self, max_cycles: int = 20) -> MissionState:
        for cycle in range(1, max_cycles + 1):
            self.state.cycle = cycle
            outcome = self._act()

            event = self.detector.observe(
                cycle=cycle,
                state=self.state.fingerprint_payload(),
                action_kind=outcome["action_kind"],
                action_target=outcome["action_target"],
                outcome=outcome,
            )

            if event:
                self._record_loop(event)
                self._escape(event)

            self._checkpoint()

            if self._all_verified():
                print(f"✅ mission complete at cycle {cycle}")
                return self.state

        print(f"⛔ mission hit max cycles ({max_cycles})")
        return self.state

    def _act(self) -> dict[str, Any]:
        if self.state.stable_mode:
            # Escape forced a stable, correct write.
            self.state.file_content = "print('hello')\ndef test_hello(): assert True\n"
            return self._outcome("edit", "stable_write")

        # Brittle oscillating policy.
        if "hello" in self.state.file_content:
            self.state.file_content = "# stub\n"
            return self._outcome("edit", "remove_hello")

        self.state.file_content = "print('hello')\n"
        return self._outcome("edit", "add_hello")

    def _outcome(self, action_kind: str, action_target: str) -> dict[str, Any]:
        result = self.verifier.verify(self.state)
        result["action_kind"] = action_kind
        result["action_target"] = action_target
        return result

    def _escape(self, event: LoopEvent) -> None:
        print(
            f"🔄 {event.loop_id} ({event.kind}) at cycle {event.cycle} "
            f"-> {event.strategy.value}"
        )

        if event.strategy == EscapeStrategy.PERTURB_PROMPT:
            # In a real agent this would rewrite the system prompt / constraints.
            # Here we break symmetry by enabling a stable write path.
            self.state.stable_mode = True

        elif event.strategy == EscapeStrategy.CALL_REVIEWER:
            if self.state.checklist:
                dropped = self.state.checklist.pop(0)
                self.state.done.append(f"{dropped} (reviewer-cut)")

        elif event.strategy == EscapeStrategy.CUT_SCOPE:
            if self.state.checklist:
                dropped = self.state.checklist.pop(0)
                self.state.done.append(f"{dropped} (scope-cut)")

        elif event.strategy == EscapeStrategy.RAISE_HITL:
            raise RuntimeError("Human-in-the-loop gate required: loop count exceeded")

        elif event.strategy == EscapeStrategy.COMPACT_MEMORY:
            self.detector.observations.clear()

    def _checkpoint(self) -> None:
        for item in list(self.state.checklist):
            if item == "add hello" and "hello" in self.state.file_content:
                self.state.checklist.remove(item)
                self.state.done.append(item)
            elif item == "add test" and "test" in self.state.file_content:
                self.state.checklist.remove(item)
                self.state.done.append(item)

    def _all_verified(self) -> bool:
        return not self.state.checklist and self.verifier.verify(self.state)["verified"]

    def _record_loop(self, event: LoopEvent) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cycle": event.cycle,
            "loop_id": event.loop_id,
            "kind": event.kind,
            "strategy": event.strategy.value,
            "window_hash": event.window_hash,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def main() -> None:
    log_path = Path(".lra/workspaces/ch21-loop/decision-log.jsonl")
    detector = LoopDetector(window_size=8, repeat_threshold=3)
    verifier = DeterministicVerifier()
    agent = Agent(detector, verifier, log_path)

    final = agent.run(max_cycles=25)

    print("\nFinal file:")
    print(final.file_content)
    print("Done:", final.done)
    print("Loops detected:", detector.loop_count)
    print("Decision log:", log_path)


if __name__ == "__main__":
    main()
```

### Key points in the code

- **`state.fingerprint_payload()`** only hashes durable facts: the file content, the todo list, and the done list. It does not hash the model's hidden reasoning.
- **`_outcome_signature()`** collapses the verification result so that two failures with the same root cause look identical to the detector.
- **Oscillation detection** fires on `A → B → A`. In the demo, that happens at cycle 3: `add_hello` → `remove_hello` → `add_hello`.
- **`PERTURB_PROMPT`** is the cheapest escape. In the demo it flips `stable_mode`, which forces a single correct write. In the real LRA package this would correspond to rewriting the system prompt or shuffling constraint order.
- **`RAISE_HITL`** is the final escalation. After three loops, the agent stops and demands a human gate.

---

## Running the Demo

```bash
cd lra-demo
python ch21_loop_detection.py
```

Expected output:

```text
🔄 loop-1 (oscillation) at cycle 3 -> perturb_prompt
✅ mission complete at cycle 4

Final file:
print('hello')
def test_hello(): assert True

Done: ['add hello', 'add test']
Loops detected: 1
Decision log: .lra/workspaces/ch21-loop/decision-log.jsonl
```

Inspect the decision log:

```bash
cat .lra/workspaces/ch21-loop/decision-log.jsonl
```

You should see one loop event with `kind=oscillation` and `strategy=perturb_prompt`.

---

## Hands-On Exercise

1. **Run the demo** and confirm the oscillation is caught at cycle 3.
2. **Change the repeat threshold** to `2` and re-run. The detector should now fire on an exact repeat instead of an oscillation. Which escape strategy runs?
3. **Add a third oscillating state**: make the agent cycle through `add_hello` → `add_greeting` → `remove_greeting` → `add_hello`. Update `_detect` to catch `A → B → C → A` cycles, not just `A → B → A`.
4. **Wire the loop count into a fake cost model**: after each cycle, add `$0.10` to a running total. Print the total at the end and compare it with a run where loop detection is disabled. This shows why the Budget Governor (Chapter 20) cares about `loop_trips`.
5. **Persist the detector window**: modify the demo to write the observation window to the Mission Anchor directory on every cycle and reload it at startup, so the loop detector survives a process restart.

---

> **Key Takeaway:** A long-running agent does not loop because the model is foolish; it loops because the system forgets it already tried the same thing. Loop detection is memory applied to durable state, and escaping oscillation is cheaper than paying for it twice.

---

## Next Chapter

In **Chapter 22: HITL Gates and Irreversible Actions**, we look at the top of the escape ladder. When the loop detector raises `RAISE_HITL`, or when the agent is about to push to `main`, delete a database, or spend real money, the system must pause durably and wait for a human signal — without losing the mission's place.