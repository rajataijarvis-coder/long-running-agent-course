# Chapter 20: Budget Governor and Cost Caps

## What We'll Cover

- Why a week-long mission needs a **budget governor** in the control plane, not just an invoice at the end
- How LRA treats the **Mission Anchor** from Chapter 16 as the durable ledger for cumulative spend
- The three economic levers the governor can pull: **loop detection**, **reviewer blocking**, and **scope cuts**
- How every cycle records a `CostEvent` from the model layer and the tool dispatcher
- Wiring the governor into the **Think → Act → Verify → Checkpoint** cycle so the mission pauses before the ceiling is breached
- A runnable demo in `lra-demo/ch20_budget_governor.py` that simulates a mission, survives a crash, and enforces a hard USD ceiling

---

## The Economic Layer of Durability

Durability is usually discussed in terms of crashes and reboots, but money is also a finite resource. A long-horizon mission can easily issue thousands of LLM calls, run hundreds of tests, and spawn dozens of sandboxes. Without a governor, the first sign of trouble is a surprise API bill.

LRA solves this by making the budget a **first-class engineering property**:

1. **The Mission Anchor** (Chapter 16) stores `cum_usd`, `ceiling_usd`, `loop_trips`, `reviewer_blocking`, `scope_cuts`, `crashes_survived`, and `skills_learned`. Because the anchor is committed to git, the governor's state survives any crash.
2. **Every activity is a `CostEvent`**. The model layer (Chapter model), tool dispatcher (Chapter 5), and verifier (Chapter 6) all emit token counts and estimated USD.
3. **The governor is in the loop**. Before each new cycle it checks the projected spend against three thresholds: warn, scope-cut, and hard pause.
4. **If the mission is over-planned, it cuts scope**. Dropped work is compacted into reusable skills (Chapters 17–19), so the system learns instead of forgetting.

The final log line from the article is the governor's report card:

```
governor.final cum_usd=178.60 ceiling=400.00 crashes_survived=1
                 loop_trips=1 reviewer_blocking=2 scope_cuts=1 skills_learned=2
```

That line is not an afterthought. It is produced every time the anchor is persisted.

---

## Code Walkthrough: `lra-demo/ch20_budget_governor.py`

The demo simulates a mission with a hard budget ceiling. It records every cycle's cost, survives a synthetic crash, and can cut scope or pause when spend crosses a threshold. In the real `src/lra/governor/` package, the same logic is invoked by the Temporal `MissionWorkflow` (Chapter 12) and writes into the git-backed Mission Anchor (Chapter 16).

```python
#!/usr/bin/env python3
"""lra-demo/ch20_budget_governor.py

A self-contained simulation of the LRA budget governor. It shows how a
week-long mission records every cycle's cost, enforces a hard ceiling, and
can cut scope or pause rather than overspend.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# Token prices per 1k tokens (input, output). "stub" and local models cost $0.
PRICE_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "stub": (0.0, 0.0),
    "ollama/llama3": (0.0, 0.0),
    "openai/gpt-4o-mini": (0.00015, 0.0006),
    "anthropic/claude-3-5-sonnet": (0.003, 0.015),
}


class CostEvent(BaseModel):
    """One billable unit of work."""

    cycle: int
    activity: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    estimated_usd: float = 0.0
    ts: float = Field(default_factory=time.time)

    @classmethod
    def from_openai_compat(
        cls, cycle: int, activity: str, response: dict[str, Any]
    ) -> "CostEvent":
        """Convert a real OpenAI-compatible response into a CostEvent."""
        usage = response.get("usage", {})
        model = response.get("model", "unknown")
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        inp_price, out_price = PRICE_PER_1K_TOKENS.get(model, (0.0, 0.0))
        estimated = (inp / 1000.0) * inp_price + (out / 1000.0) * out_price
        return cls(
            cycle=cycle,
            activity=activity,
            model=model,
            input_tokens=inp,
            output_tokens=out,
            estimated_usd=round(estimated, 6),
        )


class BudgetGovernor:
    """Hard ceiling + soft levers for a long-horizon mission."""

    def __init__(
        self,
        ceiling_usd: float,
        warn_ratio: float = 0.5,
        scope_cut_ratio: float = 0.75,
        hard_ratio: float = 0.95,
        anchor_path: Path | None = None,
    ):
        self.ceiling_usd = ceiling_usd
        self.warn_ratio = warn_ratio
        self.scope_cut_ratio = scope_cut_ratio
        self.hard_ratio = hard_ratio
        self.anchor_path = anchor_path or Path(".lra/budget_anchor.json")

        self.cum_usd = 0.0
        self.events: list[CostEvent] = []
        self.status = "RUNNING"
        self.loop_trips = 0
        self.reviewer_blocks = 0
        self.scope_cuts = 0
        self.crashes_survived = 0
        self.skills_learned = 0

    @property
    def warn_usd(self) -> float:
        return self.ceiling_usd * self.warn_ratio

    @property
    def scope_cut_usd(self) -> float:
        return self.ceiling_usd * self.scope_cut_ratio

    @property
    def hard_usd(self) -> float:
        return self.ceiling_usd * self.hard_ratio

    def record(self, event: CostEvent) -> None:
        """Add a cost event to the running total."""
        self.events.append(event)
        self.cum_usd += event.estimated_usd

    def persist_state(self, checklist: list[str], completed: list[str]) -> None:
        """Write the full budget + progress state to the Mission Anchor."""
        self.anchor_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ceiling_usd": self.ceiling_usd,
            "cum_usd": round(self.cum_usd, 6),
            "status": self.status,
            "loop_trips": self.loop_trips,
            "reviewer_blocks": self.reviewer_blocks,
            "scope_cuts": self.scope_cuts,
            "crashes_survived": self.crashes_survived,
            "skills_learned": self.skills_learned,
            "event_count": len(self.events),
            "checklist": checklist,
            "completed": completed,
        }
        self.anchor_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, anchor_path: Path | None = None, **overrides: Any) -> "BudgetGovernor":
        """Resume a mission from its persisted anchor."""
        path = anchor_path or Path(".lra/budget_anchor.json")
        if not path.exists():
            raise FileNotFoundError(f"no anchor at {path}; run without --resume")
        data = json.loads(path.read_text())
        inst = cls(
            ceiling_usd=data["ceiling_usd"],
            anchor_path=path,
            **overrides,
        )
        inst.cum_usd = data["cum_usd"]
        inst.status = data["status"]
        inst.loop_trips = data["loop_trips"]
        inst.reviewer_blocks = data["reviewer_blocks"]
        inst.scope_cuts = data["scope_cuts"]
        inst.crashes_survived = data["crashes_survived"]
        inst.skills_learned = data["skills_learned"]
        return inst

    def check(self, planned_cost: float = 0.0) -> dict[str, Any]:
        """Return control signals based on projected spend."""
        projected = self.cum_usd + planned_cost
        signals: dict[str, Any] = {
            "pause": False,
            "scope_cut": False,
            "hitl": False,
            "reason": None,
        }
        if projected >= self.hard_usd:
            self.status = "PAUSED_BUDGET"
            signals["pause"] = True
            signals["reason"] = (
                f"projected ${projected:.2f} hits hard cap ${self.hard_usd:.2f}"
            )
        elif projected >= self.scope_cut_usd:
            signals["scope_cut"] = True
            signals["reason"] = (
                f"projected ${projected:.2f} crossed scope-cut line ${self.scope_cut_usd:.2f}"
            )
        elif projected >= self.warn_usd:
            signals["hitl"] = True
            signals["reason"] = (
                f"projected ${projected:.2f} crossed warn line ${self.warn_usd:.2f}"
            )
        return signals

    def cut_scope(self, checklist: list[str]) -> list[str]:
        """Drop the second half of the checklist and count it as a learned skill."""
        if len(checklist) <= 1:
            return checklist
        keep = max(1, len(checklist) // 2)
        self.scope_cuts += 1
        self.skills_learned += 1
        return checklist[:keep]

    def register_loop_trip(self) -> None:
        self.loop_trips += 1

    def register_reviewer_block(self) -> None:
        self.reviewer_blocks += 1

    def register_crash_survived(self) -> None:
        self.crashes_survived += 1

    def summary(self) -> dict[str, Any]:
        return {
            "cum_usd": round(self.cum_usd, 2),
            "ceiling_usd": self.ceiling_usd,
            "remaining_usd": round(max(0.0, self.ceiling_usd - self.cum_usd), 2),
            "status": self.status,
            "crashes_survived": self.crashes_survived,
            "loop_trips": self.loop_trips,
            "reviewer_blocking": self.reviewer_blocks,
            "scope_cuts": self.scope_cuts,
            "skills_learned": self.skills_learned,
        }


def sample_cost(cycle: int, activity: str, model: str) -> CostEvent:
    """Reproducible, synthetic cost for the demo."""
    rng = random.Random(cycle)
    if activity == "lead_write":
        inp = rng.randint(2000, 8000)
        out = rng.randint(500, 2500)
    elif activity == "review":
        inp = rng.randint(1000, 4000)
        out = rng.randint(200, 800)
    else:
        inp = rng.randint(500, 2000)
        out = rng.randint(100, 500)
    inp_price, out_price = PRICE_PER_1K_TOKENS.get(model, (0.0, 0.0))
    usd = (inp / 1000.0) * inp_price + (out / 1000.0) * out_price
    # Add a small sandbox/tool cost so verification is never free.
    usd += rng.random() * 0.02
    return CostEvent(
        cycle=cycle,
        activity=activity,
        model=model,
        input_tokens=inp,
        output_tokens=out,
        estimated_usd=round(usd, 6),
    )


def deterministic_verify(cycle: int) -> bool:
    """Stand-in for the real verifier from Chapter 6."""
    return random.Random(cycle).random() > 0.20


def run_mission(
    ceiling: float,
    items: int = 10,
    model: str = "openai/gpt-4o-mini",
    crash_cycle: int | None = 3,
    resume: bool = False,
) -> None:
    anchor_path = Path(".lra/budget_anchor.json")

    if resume:
        gov = BudgetGovernor.load(anchor_path=anchor_path)
        if gov.status == "PAUSED_BUDGET":
            print(
                "anchor says PAUSED_BUDGET; raise ceiling or delete anchor to restart"
            )
            return
        data = json.loads(anchor_path.read_text())
        checklist: list[str] = data.get("checklist", [f"item_{i}" for i in range(items)])
        completed: list[str] = data.get("completed", [])
    else:
        if anchor_path.exists():
            anchor_path.unlink()
        gov = BudgetGovernor(ceiling_usd=ceiling, anchor_path=anchor_path)
        checklist = [f"item_{i}" for i in range(items)]
        completed = []

    cycle = len(gov.events)
    while checklist and gov.status == "RUNNING":
        cycle += 1
        current = checklist[0]

        # Simulate a crash/resume once, early in the mission.
        if cycle == crash_cycle and not resume:
            gov.register_crash_survived()
            print(
                f"cycle={cycle} CRASH_RESUME survived={gov.crashes_survived} "
                f"cum_usd={gov.cum_usd:.2f}"
            )

        # 1. Plan / think
        gov.record(sample_cost(cycle, "plan", model))

        # 2. Lead engineer writes (Chapter 10)
        gov.record(sample_cost(cycle, "lead_write", model))

        # 3. Deterministic verification (Chapter 6)
        if deterministic_verify(cycle):
            completed.append(current)
            checklist.pop(0)
            print(
                f"cycle={cycle} VERIFY_OK item={current} "
                f"cum_usd={gov.cum_usd:.2f}"
            )
        else:
            print(
                f"cycle={cycle} VERIFY_FAIL item={current} "
                f"cum_usd={gov.cum_usd:.2f}"
            )
            # Every fourth failure is treated as a loop trip (Chapter 21).
            if cycle % 4 == 0:
                gov.register_loop_trip()
                checklist = gov.cut_scope(checklist)
                print(
                    f"cycle={cycle} LOOP_TRIP scope_cuts={gov.scope_cuts} "
                    f"remaining={len(checklist)}"
                )
            gov.persist_state(checklist, completed)
            continue

        # 4. Independent review with probability (Chapter 11)
        if random.Random(cycle).random() > 0.85:
            gov.record(sample_cost(cycle, "review", model))
            gov.register_reviewer_block()
            print(
                f"cycle={cycle} REVIEWER_BLOCK blocks={gov.reviewer_blocks} "
                f"cum_usd={gov.cum_usd:.2f}"
            )

        # 5. Governor check before next cycle
        signals = gov.check(planned_cost=0.05)
        if signals["pause"]:
            print(f"cycle={cycle} PAUSE reason={signals['reason']}")
            gov.persist_state(checklist, completed)
            break
        elif signals["scope_cut"]:
            checklist = gov.cut_scope(checklist)
            print(
                f"cycle={cycle} SCOPE_CUT reason={signals['reason']} "
                f"remaining={len(checklist)}"
            )
        elif signals["hitl"]:
            print(f"cycle={cycle} HITL_WARN reason={signals['reason']}")

        gov.persist_state(checklist, completed)

    if not checklist and gov.status == "RUNNING":
        gov.status = "DONE"

    gov.persist_state(checklist, completed)

    head = "9d8c7b6"  # In LRA this is the git short SHA from Chapter 3/16.
    print(
        f"mission.complete items={len(completed)}/{items} "
        f"cycles={cycle} head={head} status={gov.status}"
    )
    summary = gov.summary()
    print(
        f"governor.final cum_usd={summary['cum_usd']:.2f} "
        f"ceiling={summary['ceiling_usd']:.2f} "
        f"crashes_survived={summary['crashes_survived']} "
        f"loop_trips={summary['loop_trips']} "
        f"reviewer_blocking={summary['reviewer_blocking']} "
        f"scope_cuts={summary['scope_cuts']} "
        f"skills_learned={summary['skills_learned']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LRA budget governor demo")
    parser.add_argument("--ceiling", type=float, default=2.0, help="USD ceiling")
    parser.add_argument("--items", type=int, default=10, help="checklist size")
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        choices=list(PRICE_PER_1K_TOKENS.keys()),
    )
    parser.add_argument(
        "--crash-cycle",
        type=int,
        default=3,
        help="cycle on which to simulate a crash/resume",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the persisted budget anchor",
    )
    args = parser.parse_args()

    run_mission(
        ceiling=args.ceiling,
        items=args.items,
        model=args.model,
        crash_cycle=args.crash_cycle,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
```

### How the pieces fit together

- **`CostEvent`** is the universal currency. The real system produces these from the model layer, the tool dispatcher, and the verifier. The demo uses `sample_cost` so it runs without an API key.
- **`BudgetGovernor`** owns the ceiling and the counters. It never asks the model whether the mission can continue; it compares `cum_usd` against `warn_usd`, `scope_cut_usd`, and `hard_usd`.
- **`persist_state`** writes the anchor. Because the anchor is in git, a crash on cycle 3 leaves a committed snapshot of `cum_usd`, `checklist`, and `completed`.
- **`load`** reconstructs the governor from that snapshot. This is exactly what the Temporal workflow does when it resumes after a worker restart (Chapter 15).
- **`cut_scope`** is the governor's most aggressive lever. When the mission is over-planned, it keeps the highest-priority half of the checklist and records a `skills_learned` increment. The dropped work can later be compacted into the skill library (Chapters 17–19).

---

## Hands-On Exercise

1. **Run the demo with the default ceiling and the stub model** (zero cost):

   ```bash
   uv run python lra-demo/ch20_budget_governor.py --model stub --ceiling 2.0
   ```

   Inspect the final `governor.final` line and the anchor at `.lra/budget_anchor.json`.

2. **Trigger a scope cut** by lowering the ceiling or switching to a more expensive model:

   ```bash
   uv run python lra-demo/ch20_budget_governor.py --model anthropic/claude-3-5-sonnet --ceiling 0.30
   ```

   Watch for `SCOPE_CUT` or `PAUSE` events. How many items finish before the ceiling stops the mission?

3. **Simulate a real crash and resume**. Start a run, press `Ctrl-C` after you see `cycle=2 VERIFY_OK`, then resume:

   ```bash
   uv run python lra-demo/ch20_budget_governor.py --model stub --ceiling 2.0
   # Ctrl-C after cycle 2
   uv run python lra-demo/ch20_budget_governor.py --resume
   ```

   Verify that `cum_usd` did not reset and that `crashes_survived` is at least 1.

4. **Add a cost spike**. Edit `sample_cost` to double the cost on cycle 5. Re-run and confirm the governor hits `PAUSED_BUDGET` before the raw ceiling is exceeded.

5. **Wire a real API response**. Replace one `sample_cost` call with `CostEvent.from_openai_compat(cycle, "lead_write", openai_response_dict)` and confirm the estimated USD matches the pricing table.

---

> **Key Takeaway:** A week-long agent is not finished when the tests pass; it is finished when the tests pass under budget. The governor turns the mission from an open-ended API bill into a bounded, auditable project.
> — Fareed Khan

---

## Next Chapter Teaser

**Chapter 21: Loop Detection and Escaping Oscillation.** The governor can cut scope, but only if it recognizes when the agent is going in circles. Next we build the loop detector that watches the episodic journal and triggers a clean escape before time and money are wasted.