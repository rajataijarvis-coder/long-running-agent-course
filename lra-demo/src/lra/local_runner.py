"""Local runner for missions without Temporal."""

from pathlib import Path

from lra.anchor import MissionAnchor
from lra.loop import AgentLoop
from lra.tools import ToolDispatcher
from lra.verify import DeterministicVerifier, build_default_checks


class LocalRunner:
    """Runs a mission end-to-end in a local workdir."""

    def __init__(self, workdir: Path | str):
        self.anchor = MissionAnchor(workdir)
        self.loop = AgentLoop(
            anchor=self.anchor,
            tools=ToolDispatcher(),
            verifier=DeterministicVerifier(build_default_checks()),
        )

    def set_task(self, items: list[dict[str, str]]) -> None:
        self.anchor.write_checklist({"items": items})
        self.anchor.write_progress("# Mission Progress\n\n")
        self.anchor.record_decision(
            "Use local runner", f"Running mission with {len(items)} checklist items"
        )

    def run(self) -> dict:
        cycles = 0
        while True:
            item = self.anchor.next_item()
            if item is None:
                break
            cycles += 1
            blocked = self.loop.cycle(item)
            if blocked:
                break
        return {
            "cycles": cycles,
            "progress": self.anchor.read_progress(),
            "checklist": self.anchor.read_checklist(),
        }
