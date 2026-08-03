"""Tests for MissionAnchor."""

import json
from pathlib import Path

from lra.anchor import MissionAnchor, ChecklistItem


def test_anchor_round_trip(tmp_path: Path) -> None:
    anchor = MissionAnchor(tmp_path)
    anchor.write_checklist({
        "items": [
            {"id": "1", "description": "First task", "status": "todo"},
        ]
    })
    item = anchor.next_item()
    assert isinstance(item, ChecklistItem)
    assert item.id == "1"


def test_event_logging(tmp_path: Path) -> None:
    anchor = MissionAnchor(tmp_path)
    anchor.log_event(action="test", value=42)
    lines = (tmp_path / ".lra" / "events.ndjson").read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["action"] == "test"
