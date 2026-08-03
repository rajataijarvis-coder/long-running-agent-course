"""Reviewer and reflection agents.

A reviewer gives a second opinion on the Lead Engineer's work without seeing its
reasoning chain. A reflection agent examines a failure trace and proposes new
hypotheses so the next cycle does not repeat the same mistake.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReviewResult:
    approved: bool
    findings: list[str]
    suggestions: list[str]


@dataclass
class ReflectionResult:
    root_cause: str
    hypotheses: list[str]
    next_action: str


class StaticReviewer:
    """Deterministic reviewer that checks surface invariants.

    This is intentionally simple. In production the reviewer would be another
    LLM call with a fresh prompt and no access to the Lead's chain-of-thought.
    """

    def __init__(self, anchor: Any, max_findings: int = 5):
        self.anchor = anchor
        self.max_findings = max_findings

    def review(self, item_id: str) -> ReviewResult:
        checklist = self.anchor.read_checklist()
        item = next((i for i in checklist["items"] if i["id"] == item_id), None)
        if item is None:
            return ReviewResult(False, [f"Item {item_id} not found"], [])

        findings: list[str] = []
        suggestions: list[str] = []

        # Rule 1: done items must have at least one verification record.
        if item["status"] == "done" and not item.get("verified_by"):
            findings.append(f"Item {item_id} is marked done but has no verification record.")
            suggestions.append("Run deterministic verification and record the check names.")

        # Rule 1b: a verified done item with a claimed file should have that file.
        # This rule is applied only when there is a verification record; otherwise
        # the missing record is the primary issue.
        if item["status"] == "done" and item.get("verified_by"):
            claimed_file = self._guess_output_file(item["description"])
            full_path = self.anchor.workdir / claimed_file if claimed_file else None
            if full_path and not full_path.exists():
                finding = f"Claimed output file {claimed_file} does not exist."
                findings.append(finding)
                suggestions.append(f"Write {claimed_file.name} or update the item description.")

        # Rule 2: blocked items must explain why in the progress log.
        progress = self.anchor.read_progress()
        if item["status"] == "blocked" and f"Item {item_id}" not in progress:
            findings.append(f"Item {item_id} is blocked but progress.md contains no entry.")
            suggestions.append("Append a failure note to progress.md before retrying.")

        # Rule 3: non-done items that create files are checked only when not blocked.
        if item["status"] not in ("done", "blocked"):
            if "create" in item["description"].lower() or "write" in item["description"].lower():
                claimed_file = self._guess_output_file(item["description"])
                full_path = self.anchor.workdir / claimed_file if claimed_file else None
                if full_path and not full_path.exists():
                    finding = f"Claimed output file {claimed_file} does not exist."
                    findings.append(finding)
                    suggestions.append(f"Write {claimed_file.name} or update the item description.")

        approved = len(findings) == 0
        return ReviewResult(
            approved=approved,
            findings=findings[: self.max_findings],
            suggestions=suggestions[: self.max_findings],
        )

    @staticmethod
    def _guess_output_file(description: str) -> Path | None:
        words = description.split()
        for word in words:
            if word.endswith((".py", ".md", ".json", ".yml", ".yaml")):
                return Path(word)
        return None


class ReflectionAgent:
    """Produces structured failure reflections from events and verification output."""

    def __init__(self, anchor: Any):
        self.anchor = anchor

    def reflect(self, item_id: str) -> ReflectionResult:
        events = list(self._read_events())
        failures = [
            e for e in events if e.get("item_id") == item_id and e.get("action") == "verify" and not e.get("passed")
        ]
        last = failures[-1] if failures else {}
        checks = last.get("checks", [])
        failed_names = [c["name"] for c in checks if not c.get("passed")]

        # Deterministic hypothesis generation.
        hypotheses: list[str] = []
        if "pytest" in failed_names:
            hypotheses.append("The generated code does not satisfy the test assertion.")
        if "ruff" in failed_names:
            hypotheses.append("The generated code has syntax or lint errors.")
        if "mypy" in failed_names:
            hypotheses.append("Type annotations are missing or inconsistent.")
        if not failed_names:
            hypotheses.append("Verification was skipped or the item has no matching implementation.")

        if len(hypotheses) < 3:
            hypotheses.append("The act step selected the wrong tool or arguments for this item.")
        if len(hypotheses) < 3:
            hypotheses.append("The anchor state (progress.md or checklist.json) is stale or inconsistent.")

        root_cause = (
            f"Verification failed for item {item_id}"
            + (f": {', '.join(failed_names)}" if failed_names else ".")
        )
        next_action = (
            f"Pick the most likely hypothesis ({hypotheses[0]}) and retry with a corrected act step."
        )
        return ReflectionResult(root_cause=root_cause, hypotheses=hypotheses[:3], next_action=next_action)

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.anchor.events_path.exists():
            return []
        return [json.loads(line) for line in self.anchor.events_path.read_text().strip().split("\n") if line.strip()]


def build_review_and_reflect(anchor: Any) -> tuple[StaticReviewer, ReflectionAgent]:
    """Factory for the two agents used in the durable loop."""
    return StaticReviewer(anchor), ReflectionAgent(anchor)
