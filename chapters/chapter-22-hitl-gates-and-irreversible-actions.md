# Chapter 22: HITL Gates and Irreversible Actions

## What We'll Cover

- Why a week-long mission cannot be fully unattended — some actions need a human *before* they run
- What LRA treats as **irreversible**: writes to `main`, deployments, destructive migrations, egress, and large spends
- How a HITL gate is a **durable checkpoint**, not a `input()` prompt that vanishes on reboot
- The gate lifecycle: **propose → pending → approve / steer / reject**
- Wiring gates into the **Think → Act → Verify → Checkpoint** cycle from Chapter 4
- A runnable demo in `lra-demo/ch22_hitl_gates.py` that pauses, survives a simulated crash, and resumes after human approval

---

## The Concept: Asynchronous Supervision

By Chapter 21 the agent can plan, spend, loop, and recover. But long-horizon autonomy is not the same as unsupervised autonomy. Some actions are too risky to let the model commit on its own:

| Action | Why it needs a gate |
|---|---|
| Push to `main` | Rewrites shared history; teammates may base work on it |
| Deploy to production | Changes live state for users |
| Run a destructive DB migration | Cannot be undone without backups |
| Open an egress connection | Could leak data or call an unexpected paid API |
| Spend above a micro-budget | Even small per-step costs compound |

LRA solves this with **HITL gates**. A gate is a durable signal stored in the same place as the mission truth: the **Mission Anchor** from Chapter 16 and the git-backed state. When the agent wants to do something irreversible, it does not ask the model "is this okay?" It writes a proposal to disk, marks the action as pending, and the whole mission enters a **durable sleep**. A human can approve, steer, or reject hours or days later. After a reboot, the workflow re-reads the anchor, sees the gate is still pending, and goes back to sleep — no tokens re-spent, no work lost.

This is the same durability idea we applied to activities in Chapter 13 and to crashes in Chapter 15. The only difference is that the external event we are waiting for is a human, not an API.

### Gate statuses

- `pending` — waiting for human input; the cycle must not proceed
- `approved` — human said yes; execute and checkpoint
- `rejected` — human said no; the agent must choose a fallback or cut scope
- `steered` — human did not answer yes/no but gave a new direction; the agent updates its plan

The **Budget Governor** from Chapter 20 and the **Loop Detector** from Chapter 21 already know how to cut scope or pause. A rejected gate plugs into the same fallback path: if the risky action was essential, the mission may need to re-plan; if it was optional, it is skipped.

---

## Code Walkthrough

### `src/lra/hitl/gate.py`

The HITL module is small and stateless. All durable state lives in the anchor directory so that the gate survives process restarts.

```python
# src/lra/hitl/gate.py
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GateStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    STEERED = "steered"


class GateKind(str, Enum):
    APPROVE = "approve"   # yes/no gate
    STEER = "steer"       # human gives new direction
    INFO = "info"         # notify only, no block


class Irreversibility(str, Enum):
    NONE = "none"                 # safe, no log needed
    REVERSIBLE = "reversible"     # can be undone, log for audit
    IRREVERSIBLE = "irreversible" # must block for human


class ActionProposal(BaseModel):
    action_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent: str
    description: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    irreversibility: Irreversibility = Irreversibility.NONE
    affected_paths: list[str] = Field(default_factory=list)
    estimated_cost_usd: float | None = None
    reason: str = ""


class GateResponse(BaseModel):
    action_id: str
    status: GateStatus
    human_note: str = ""
    steer_payload: dict[str, Any] | None = None
    resolved_at: datetime | None = None


class GatePendingError(Exception):
    """Raised when the agent cycle must pause for human input."""
    def __init__(self, action_id: str, proposal: ActionProposal):
        self.action_id = action_id
        self.proposal = proposal
        super().__init__(f"Gate pending for action {action_id}: {proposal.description}")


class HITLManager:
    def __init__(self, anchor_dir: Path):
        self.anchor_dir = Path(anchor_dir)
        self.gates_dir = self.anchor_dir / "hitl" / "gates"
        self.gates_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, action_id: str) -> Path:
        return self.gates_dir / f"{action_id}.json"

    def propose(self, proposal: ActionProposal) -> GateResponse:
        """Idempotent propose. Auto-approves reversible actions; blocks irreversible ones."""
        if proposal.irreversibility in (Irreversibility.NONE, Irreversibility.REVERSIBLE):
            return self._resolve(
                proposal.action_id,
                GateStatus.APPROVED,
                human_note="auto-approved: reversible action",
            )

        path = self._path(proposal.action_id)
        if path.exists():
            return GateResponse.model_validate_json(path.read_text())

        gate = GateResponse(action_id=proposal.action_id, status=GateStatus.PENDING)
        path.write_text(gate.model_dump_json(indent=2))

        # Human-readable inbox for operators.
        inbox = self.gates_dir / "inbox.txt"
        with inbox.open("a") as f:
            f.write(
                f"[{datetime.now(timezone.utc).isoformat()}] {proposal.action_id}\n"
                f"  agent: {proposal.agent}\n"
                f"  tool:  {proposal.tool}\n"
                f"  desc:  {proposal.description}\n"
                f"  paths: {', '.join(proposal.affected_paths)}\n"
                f"  cost:  {proposal.estimated_cost_usd}\n"
                f"  file:  {path}\n\n"
            )
        raise GatePendingError(proposal.action_id, proposal)

    def resolve(
        self,
        action_id: str,
        status: GateStatus,
        human_note: str = "",
        steer_payload: dict[str, Any] | None = None,
    ) -> GateResponse:
        path = self._path(action_id)
        if not path.exists():
            raise FileNotFoundError(f"No gate found for {action_id}")

        gate = GateResponse.model_validate_json(path.read_text())
        gate.status = status
        gate.human_note = human_note
        gate.steer_payload = steer_payload
        gate.resolved_at = datetime.now(timezone.utc)
        path.write_text(gate.model_dump_json(indent=2))

        # Durable decision log.
        log_path = self.gates_dir / "decisions.jsonl"
        with log_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": gate.resolved_at.isoformat(),
                        "action_id": action_id,
                        "status": status.value,
                        "note": human_note,
                        "steer": steer_payload,
                    },
                    default=str,
                )
                + "\n"
            )
        return gate

    def status(self, action_id: str) -> GateResponse:
        path = self._path(action_id)
        if not path.exists():
            raise FileNotFoundError(action_id)
        return GateResponse.model_validate_json(path.read_text())

    def check(self, proposal: ActionProposal) -> GateResponse:
        """Idempotent check. Raises GatePendingError if the gate is still open."""
        path = self._path(proposal.action_id)
        if path.exists():
            gate = GateResponse.model_validate_json(path.read_text())
            if gate.status == GateStatus.PENDING:
                raise GatePendingError(proposal.action_id, proposal)
            return gate
        return self.propose(proposal)
```

Key design points:

1. **Idempotency.** Calling `check()` twice with the same `action_id` never creates duplicate gates. This matters because the Temporal workflow from Chapter 12 may replay the activity that proposed the gate.
2. **Durable storage.** The gate file is the source of truth. The in-memory `HITLManager` is just a reader/writer.
3. **Human inbox.** `inbox.txt` is plain text so an operator can scan pending gates without running Python.
4. **Decision log.** Every resolution is appended to `decisions.jsonl` for audit and offline evolution in Chapter 25.

### Classifying irreversibility

The agent does not decide per-action risk by itself. A small rule table maps tool + target to an `Irreversibility` level. This lives next to the tool dispatcher from Chapter 5.

```python
# src/lra/hitl/classify.py
from lra.hitl.gate import Irreversibility

IRREVERSIBLE_TOOLS = {
    "git_push",
    "deploy",
    "db_migrate_destructive",
    "egress_open",
}

REVERSIBLE_TOOLS = {
    "write_file",
    "shell_test",
    "git_commit",
}


def classify_irreversibility(tool_name: str, affected_paths: list[str]) -> Irreversibility:
    if tool_name in IRREVERSIBLE_TOOLS:
        return Irreversibility.IRREVERSIBLE
    if tool_name in REVERSIBLE_TOOLS:
        return Irreversibility.REVERSIBLE
    # Default-deny: if we do not know the tool, treat it as irreversible.
    return Irreversibility.IRREVERSIBLE
```

Notice the default-deny pattern. We will expand it into full egress and tool policies in Chapter 23.

### Wiring the gate into the agent cycle

In Chapter 4 the cycle was Think → Act → Verify → Checkpoint. The gate fits between Think and Act: before any tool runs, the agent checks whether the planned action needs human approval.

```python
# src/lra/agent/cycle.py (excerpt)
from lra.hitl.gate import ActionProposal, GatePendingError, GateStatus, HITLManager
from lra.hitl.classify import classify_irreversibility


async def act_with_gate(cycle, tool_call, hitl: HITLManager):
    proposal = ActionProposal(
        action_id=tool_call.id,
        agent=cycle.agent_name,
        description=tool_call.description,
        tool=tool_call.name,
        args=tool_call.args,
        irreversibility=classify_irreversibility(tool_call.name, tool_call.affected_paths),
        affected_paths=tool_call.affected_paths,
        estimated_cost_usd=tool_call.estimated_cost_usd,
        reason=tool_call.reason,
    )

    try:
        gate = hitl.check(proposal)
    except GatePendingError as exc:
        # Handed up to the Temporal workflow, which durable-sleeps.
        cycle.log.info("hitl.pending", action_id=exc.action_id)
        raise

    if gate.status == GateStatus.REJECTED:
        cycle.log.warning("hitl.rejected", action_id=gate.action_id, note=gate.human_note)
        raise ActionRejectedError(gate)

    if gate.status == GateStatus.STEERED and gate.steer_payload:
        cycle.log.info("hitl.steer", payload=gate.steer_payload)
        cycle.apply_steer(gate.steer_payload)

    return cycle.execute_tool(tool_call)
```

In a real Temporal workflow from Chapter 12, the `GatePendingError` is caught at the workflow boundary and the workflow waits on a condition:

```python
# src/lra/durable/workflows.py (excerpt)
from temporalio import workflow

@workflow.signal
async def human_gate_response(response: GateResponse) -> None:
    workflow.logger.info("hitl.signal_received", action_id=response.action_id)
    # The signal handler writes the resolved gate file via the HITLManager.
    hitl = workflow.get_hithl_manager()
    hitl.resolve(response.action_id, response.status, response.human_note, response.steer_payload)


async def await_gate_resolution(hitl: HITLManager, action_id: str) -> None:
    def _resolved() -> bool:
        return hitl.status(action_id).status != GateStatus.PENDING

    await workflow.wait_condition(_resolved)
```

Because the wait is durable, the workflow sleeps at zero compute and zero tokens. A crash mid-wait resumes at the same `await`.

---

## Demo: `lra-demo/ch22_hitl_gates.py`

This script runs without Temporal or a real LLM. It simulates the gate lifecycle on disk.

```python
#!/usr/bin/env python3
# lra-demo/ch22_hitl_gates.py
"""Chapter 22 demo: propose an irreversible write, pause, approve, resume."""
from pathlib import Path

from lra.hitl.gate import (
    ActionProposal,
    GatePendingError,
    GateStatus,
    HITLManager,
    Irreversibility,
)

ANCHOR = Path(".lra-demo/ch22_anchor")
ANCHOR.mkdir(parents=True, exist_ok=True)
hitl = HITLManager(ANCHOR)


def simulate_crash():
    """Wipe the in-memory manager to prove state lives on disk."""
    print("\n--- simulated crash: rebuilding HITLManager from disk ---")
    return HITLManager(ANCHOR)


# 1. Propose an irreversible write to main.
proposal = ActionProposal(
    agent="lead-engineer",
    description="Overwrite hello.py on main with the final LSP server entrypoint",
    tool="write_file",
    args={"path": "hello.py", "content": "# final entrypoint\n"},
    irreversibility=Irreversibility.IRREVERSIBLE,
    affected_paths=["hello.py"],
    estimated_cost_usd=0.02,
    reason="This replaces the demo file; main branch history is permanent.",
)

print("--- cycle 1: proposing gate ---")
try:
    hitl.check(proposal)
except GatePendingError as exc:
    print(f"PAUSED: {exc}")
    print(f"Human inbox: {hitl.gates_dir / 'inbox.txt'}")

# 2. Simulate a crash while the gate is open.
hitl = simulate_crash()

# 3. Human approves by resolving the gate file.
print("\n--- human approves via resolve() ---")
hitl.resolve(proposal.action_id, GateStatus.APPROVED, human_note="LGTM, ship it.")

# 4. Re-check resumes without re-proposing.
print("\n--- cycle 2: re-check after crash + approval ---")
gate = hitl.check(proposal)
print(f"RESUMED: {gate.status.value} — {gate.human_note}")

# 5. A second action is proposed and rejected.
deploy = ActionProposal(
    action_id="deploy-001",
    agent="integrator",
    description="Deploy the LSP server to production",
    tool="deploy",
    args={"target": "production"},
    irreversibility=Irreversibility.IRREVERSIBLE,
    affected_paths=[],
    estimated_cost_usd=5.00,
)

print("\n--- cycle 3: deploy gate is rejected ---")
try:
    hitl.check(deploy)
except GatePendingError:
    hitl.resolve(deploy.action_id, GateStatus.REJECTED, human_note="Wait for reviewer pass.")
    gate = hitl.check(deploy)
    print(f"REJECTED: {gate.status.value} — {gate.human_note}")

# 6. Show the durable decision log.
print("\n--- decision log ---")
log = hitl.gates_dir / "decisions.jsonl"
for line in log.read_text().splitlines():
    print(line)
```

Run it:

```bash
uv run python lra-demo/ch22_hitl_gates.py
```

Expected output:

```text
--- cycle 1: proposing gate ---
PAUSED: Gate pending for action 9f8e7d6c2a1b: Overwrite hello.py on main with the final LSP server entrypoint
Human inbox: .lra-demo/ch22_anchor/hitl/gates/inbox.txt

--- simulated crash: rebuilding HITLManager from disk ---

--- human approves via resolve() ---

--- cycle 2: re-check after crash + approval ---
RESUMED: approved — LGTM, ship it.

--- cycle 3: deploy gate is rejected ---
REJECTED: rejected — Wait for reviewer pass.

--- decision log ---
{"ts": "...", "action_id": "9f8e7d6c2a1b", "status": "approved", "note": "LGTM, ship it.", "steer": null}
{"ts": "...", "action_id": "deploy-001", "status": "rejected", "note": "Wait for reviewer pass.", "steer": null}
```

The demo proves three things:

1. The gate pauses the cycle.
2. The gate survives a process restart because the state is on disk.
3. A rejected gate gives the agent a clear fallback signal instead of an ambiguous model refusal.

---

## Hands-On Exercise

1. **Run the demo** and inspect `.lra-demo/ch22_anchor/hitl/gates/inbox.txt`. Confirm it contains the human-readable summary of the pending gate.

2. **Add a steer gate.** Create a third proposal in `lra-demo/ch22_hitl_gates.py` for a destructive migration, e.g.:

   ```python
   migration = ActionProposal(
       action_id="migrate-001",
       agent="integrator",
       description="Drop the legacy_skills table",
       tool="db_migrate_destructive",
       irreversibility=Irreversibility.IRREVERSIBLE,
   )
   ```

   Instead of approving or rejecting, resolve it with `GateStatus.STEERED` and a payload:

   ```python
   hitl.resolve(
       migration.action_id,
       GateStatus.STEERED,
       human_note="Do not drop; archive instead.",
       steer_payload={"alternative": "rename_table", "new_name": "legacy_skills_archive"},
   )
   ```

   Update the script to print the steer payload and have the agent print `Switching to alternative: rename_table`.

3. **Simulate a real crash.** After a gate is raised, delete the `.lra-demo/ch22_anchor` process in your terminal, restart the script, and verify that `hitl.check(proposal)` returns `pending` again without creating a duplicate gate file.

4. **Wire a gate into your local mission from Chapter 7.** Add a `write_file` call that targets `main` and force `Irreversibility.IRREVERSIBLE`. Run `uv run lra mission` and confirm the mission pauses. Resolve the gate and resume.

---

> **Key Takeaway:** A long-running agent is not unsupervised; it is asynchronously supervised. HITL gates turn irreversible actions into durable checkpoints, so a human can approve, steer, or reject at any speed while the mission sleeps at zero cost and zero token spend.

---

**Next Chapter:** In Chapter 23 we add **Egress Policies and Default-Deny Security**. Even an approved action should not be allowed to phone home to an unknown endpoint, so we will layer a default-deny network and tool-allow list underneath the HITL gate.