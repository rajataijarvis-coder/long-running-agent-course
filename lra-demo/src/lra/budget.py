"""Budget governor and cost cap safety system.

The BudgetGovernor is a durable, in-memory safe-guard that tracks cumulative
LLM spend and refuses new calls once a cap is reached. It is intended to be
used inside an activity so Temporal journals the refusal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BudgetConfig:
    usd_cap: float = 10.0
    # Cost per 1k tokens by model. Estimates for free/low-cost models.
    token_rates: dict[str, dict[str, float]] | None = None

    def __post_init__(self) -> None:
        if self.token_rates is None:
            self.token_rates = {
                "ollama": {"input": 0.0, "output": 0.0},
                "gpt-4o-mini": {"input": 0.15, "output": 0.60},
                "gpt-4o": {"input": 5.0, "output": 15.0},
            }


class BudgetGovernor:
    """Tracks spend in a JSON ledger and raises when the cap is reached.

    The ledger is stored next to the mission anchor so it survives crashes.
    Because this is an activity-side helper, the workflow can decide whether
    to park, continue, or escalate when the cap is hit.
    """

    def __init__(self, workdir: Path | str, config: BudgetConfig | None = None):
        self.workdir = Path(workdir)
        self.lra_dir = self.workdir / ".lra"
        self.lra_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or BudgetConfig()
        self.ledger_path = self.lra_dir / "budget.json"
        self._load()

    def _load(self) -> None:
        if self.ledger_path.exists():
            self.ledger: dict[str, Any] = json.loads(self.ledger_path.read_text())
        else:
            self.ledger = {"spent_usd": 0.0, "calls": 0, "history": []}

    def _save(self) -> None:
        self.ledger_path.write_text(json.dumps(self.ledger, indent=2) + "\n")

    def record_call(self, model: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
        rates = (self.config.token_rates or {}).get(model, {"input": 0.0, "output": 0.0})
        cost_usd = (input_tokens / 1000.0) * rates.get("input", 0.0) + (
            output_tokens / 1000.0
        ) * rates.get("output", 0.0)

        self.ledger["spent_usd"] += cost_usd
        self.ledger["calls"] += 1
        self.ledger["history"].append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost_usd, 6),
            }
        )
        self._save()
        return {
            "ok": self.ledger["spent_usd"] < self.config.usd_cap,
            "spent_usd": round(self.ledger["spent_usd"], 4),
            "cap_usd": self.config.usd_cap,
            "remaining_usd": round(self.config.usd_cap - self.ledger["spent_usd"], 4),
        }

    def can_spend(self, estimated_usd: float = 0.0) -> bool:
        return (self.ledger["spent_usd"] + estimated_usd) < self.config.usd_cap

    def assert_room(self, estimated_usd: float) -> None:
        if not self.can_spend(estimated_usd):
            raise BudgetExceededError(
                f"Budget cap {self.config.usd_cap} USD would be exceeded "
                f"(already spent {self.ledger['spent_usd']:.4f} USD, estimated {estimated_usd:.4f} more)."
            )

    def summary(self) -> dict[str, Any]:
        return {
            "cap_usd": self.config.usd_cap,
            "spent_usd": round(self.ledger["spent_usd"], 4),
            "calls": self.ledger["calls"],
            "remaining_usd": round(self.config.usd_cap - self.ledger["spent_usd"], 4),
            "at_risk": self.ledger["spent_usd"] >= 0.8 * self.config.usd_cap,
        }


class BudgetExceededError(Exception):
    """Raised when a planned call would exceed the cost cap."""
