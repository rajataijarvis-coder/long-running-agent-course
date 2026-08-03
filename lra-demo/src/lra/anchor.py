"""Mission anchor: durable state store for an agent mission."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


@dataclass
class ChecklistItem:
    id: str
    description: str
    status: str = "todo"  # todo | in_progress | done | blocked
    verified_by: list[str] = field(default_factory=list)


class MissionAnchor:
    """Reads/writes the four anchor files that hold mission truth."""

    def __init__(self, workdir: Path | str):
        self.workdir = Path(workdir)
        self.lra_dir = self.workdir / ".lra"
        self.lra_dir.mkdir(parents=True, exist_ok=True)

    @property
    def progress_path(self) -> Path:
        return self.lra_dir / "progress.md"

    @property
    def checklist_path(self) -> Path:
        return self.lra_dir / "checklist.json"

    @property
    def decisions_path(self) -> Path:
        return self.lra_dir / "decisions.ndjson"

    @property
    def events_path(self) -> Path:
        return self.lra_dir / "events.ndjson"

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------
    def read_progress(self) -> str:
        if not self.progress_path.exists():
            return "# Mission Progress\n\nNo progress recorded yet.\n"
        return self.progress_path.read_text()

    def write_progress(self, text: str) -> None:
        self.progress_path.write_text(text)

    # ------------------------------------------------------------------
    # Checklist
    # ------------------------------------------------------------------
    def read_checklist(self) -> dict[str, Any]:
        if not self.checklist_path.exists():
            return {"items": []}
        return json.loads(self.checklist_path.read_text())

    def write_checklist(self, checklist: dict[str, Any]) -> None:
        self.checklist_path.write_text(json.dumps(checklist, indent=2) + "\n")

    def next_item(self) -> ChecklistItem | None:
        data = self.read_checklist()
        for raw in data["items"]:
            if raw["status"] == "todo":
                return ChecklistItem(**raw)
        return None

    def update_item(self, item_id: str, status: str, verified_by: list[str] | None = None) -> None:
        data = self.read_checklist()
        for raw in data["items"]:
            if raw["id"] == item_id:
                raw["status"] = status
                if verified_by is not None:
                    raw["verified_by"] = verified_by
        self.write_checklist(data)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def log_event(self, **fields: Any) -> None:
        fields["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.events_path, "a") as f:
            f.write(json.dumps(fields) + "\n")

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------
    def record_decision(self, decision: str, reason: str) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "reason": reason,
        }
        with open(self.decisions_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
