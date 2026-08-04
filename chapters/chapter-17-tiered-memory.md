# Chapter 17: Tiered Memory

> A week-long mission generates far more history than any prompt can hold. The trick is not a bigger context window; it is a hierarchy of memory where each tier is read and written by the right rule.
> — Fareed Khan

## What We'll Cover

- Why a single context window is the wrong place to store a week of work
- The three tiers of memory in LRA: **working** (git state), **episodic** (event journal), and **semantic** (skills/embeddings)
- How the **Mission Anchor** from Chapter 16 becomes the root pointer that binds the tiers together
- How the agent loads only a curated slice of memory into the prompt each cycle
- How **compaction** turns an overflowing episodic journal into reusable semantic skills without losing ground truth
- A runnable demo in `lra-demo/ch17_tiered_memory.py` that writes, reads, compacts, and recalls across all three tiers

---

## The Context Window Is Not Memory

By now we have beaten this idea into the ground, but it is worth repeating because everything in this chapter rests on it.

The context window is a **lossy cache**, not a database. In Chapter 1 we saw why chat-style agents die when the window fills. In Chapter 3 we moved the real state into git. In Chapter 16 we added the Mission Anchor so a brand-new worker could find that git state after a crash. Those chapters solved **durability** and **resumability**.

They did not solve **recall**.

A mission that runs for a week produces thousands of events: plans, edits, test failures, reviewer objections, scope cuts, loop trips, skills learned. You cannot dump all of that into the prompt. Even if a model had a million-token window, stuffing it full of noise would make the agent slower, more expensive, and worse at reasoning.

The fix is a tiered memory system:

| Tier | Name | Authority | Lifetime | Storage | Used For |
|---|---|---|---|---|---|
| L1 | Working memory | The current truth | Hours to days | Git repo + anchor | Checklist, ownership map, decision log, current code |
| L2 | Episodic memory | A faithful history | Days to weeks | Append-only journal | What happened, what failed, what the reviewer said |
| L3 | Semantic memory | Distilled lessons | Weeks to months | Skills + embeddings | Reusable patterns, heuristics, failure signatures |

Only a small, curated subset of L1, L2, and L3 is promoted into the prompt context (L0) each cycle. The rest stays on disk, in git, or in a vector store.

This is the same principle as the human brain: you do not load every memory you have ever formed when you decide what to type next. You load the task at hand, a few recent events, and a few relevant skills.

## Where Each Tier Lives in the Repo

In the full LRA package these responsibilities are separated under `src/lra/memory/` and `src/lra/state/`:

```
src/lra/
├── state/
│   ├── anchor.py          # MissionAnchor (L1 root pointer)
│   ├── checklist.py       # current checklist (L1)
│   ├── ownership.py       # who owns what file (L1)
│   └── decision_log.py    # durable decisions (L1)
└── memory/
    ├── episodic.py        # journal of cycles/events (L2)
    ├── semantic.py        # skills + embeddings (L3)
    └── compaction.py      # summarizing L2 into L3
```

The demo in this chapter collapses that into one file so you can see the rules without navigating the whole package.

## L1 — Working Memory

Working memory is everything the agent needs to know **right now** to pick the next action. It is authoritative, small, and re-read at the start of every cycle.

It includes:

- The Mission Anchor (`src/lra/state/anchor.py`)
- The checklist of remaining work
- The ownership map (which agent owns which files)
- The decision log (irreversible choices already made)

Because these files live in git, they survive crashes, Continue-As-New, and worker replacement. We covered that in Chapters 3 and 16.

The rule for L1 is: **write it to git, read it every cycle, never trust a model's memory of it.**

## L2 — Episodic Memory

Episodic memory is an append-only journal of what happened. It is not authoritative for the current state — the checklist is — but it is authoritative for the history.

A typical episode record looks like this:

```json
{
  "cycle": 42,
  "timestamp": "2026-06-02T08:44:51.880Z",
  "agent": "lead",
  "action": "write_file",
  "target": "src/server.py",
  "observation": "pytest passed 3/3",
  "tags": ["write", "test", "server"]
}
```

The journal grows monotonically. That is fine for a while, but after hundreds or thousands of cycles the agent cannot read the whole thing each turn. That is where compaction comes in.

## L3 — Semantic Memory

Semantic memory holds **skills**: compact, reusable lessons distilled from episodes.

A skill might capture:

- "When `pytest` fails with `ModuleNotFoundError`, check `pyproject.toml` for missing deps before editing code."
- "This codebase uses `structlog`; prefer `logger.bind(...)` over string formatting."
- "Reviewer often blocks PRs missing type hints; add them on the first pass."

Skills are stored with embeddings so the agent can recall the few that are relevant to the current task. In this chapter we use a deterministic stub embedder. In Chapter 18 we will replace it with real vector search.

## The Compaction Pipeline

Compaction is the bridge from L2 to L3. When the episodic journal grows past a configured threshold, the agent:

1. Reads old episodes (e.g., everything before the last 50 cycles).
2. Summarizes them into one or more skills.
3. Stores those skills in L3 with embeddings.
4. Archives the old episodes to a separate file (still in git, still auditable).
5. Updates the anchor to point at the new skill set.

The archived episodes remain the ground truth. The skills are a compressed retrieval cache. If a skill ever seems wrong, you can always go back to the raw journal.

## The Demo: `lra-demo/ch17_tiered_memory.py`

This script implements a minimal but complete tiered memory store. It uses git for L1 durability, a JSONL journal for L2, and a JSONL skill file with a stub embedder for L3.

```python
# lra-demo/ch17_tiered_memory.py
"""Minimal tiered memory demo for LRA Chapter 17.

L1: working memory  -> Mission Anchor in .lra/anchor.json (git)
L2: episodic memory -> append-only journal in .lra/journal.jsonl (git)
L3: semantic memory -> skills in .lra/skills.jsonl with stub embeddings
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tiered_memory")


class GitStore:
    """Tiny helper: treat a directory as a git repo and commit LRA state."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, cwd=self.repo, check=True, capture_output=True, text=True)

    def commit(self, message: str) -> str | None:
        self.run(["git", "add", "-A"])
        try:
            self.run(["git", "commit", "-m", message])
        except subprocess.CalledProcessError as e:
            if "nothing to commit" in (e.stdout or ""):
                return None
            raise
        return self.run(["git", "rev-parse", "HEAD"]).stdout.strip()

    def head(self) -> str:
        return self.run(["git", "rev-parse", "HEAD"]).stdout.strip()


@dataclass
class MissionAnchor:
    """L1 root pointer. Everything else is found through this file."""
    mission_id: str
    task: str
    status: str
    created_at: str
    head_commit: str | None = None
    current_cycle: int = 0
    checklist: list[dict[str, Any]] = field(default_factory=list)
    journal_path: str = ".lra/journal.jsonl"
    skills_path: str = ".lra/skills.jsonl"
    archive_path: str = ".lra/journal.archive.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionAnchor:
        return cls(**data)


class StubEmbedder:
    """Deterministic, dependency-free embedder for the demo.

    In production this is replaced by sentence-transformers or an API.
    """
    DIM = 32

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [((b / 255.0) - 0.5) * 2.0 for b in h[: self.DIM]]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class TieredMemory:
    """One object exposes all three memory tiers."""

    def __init__(self, workdir: str | Path, mission_id: str | None = None, task: str = "") -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        self.lra = self.workdir / ".lra"
        self.anchor_path = self.lra / "anchor.json"
        self.journal_path = self.lra / "journal.jsonl"
        self.skills_path = self.lra / "skills.jsonl"
        self.archive_path = self.lra / "journal.archive.jsonl"
        self.git = GitStore(self.workdir)
        self.embedder = StubEmbedder()

        self._ensure_repo()
        self.anchor = self._load_or_create_anchor(mission_id, task)

    # ------------------------------------------------------------------ #
    # L1: working memory (anchor + checklist)
    # ------------------------------------------------------------------ #

    def _ensure_repo(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        if not (self.workdir / ".git").exists():
            self.git.run(["git", "init", "-q"])
            (self.workdir / ".gitignore").write_text(".lra/*.archive.jsonl\n__pycache__\n")
            self.git.commit("chore: init mission repo")
        self.lra.mkdir(parents=True, exist_ok=True)

    def _load_or_create_anchor(self, mission_id: str | None, task: str) -> MissionAnchor:
        if self.anchor_path.exists():
            data = json.loads(self.anchor_path.read_text())
            return MissionAnchor.from_dict(data)

        anchor = MissionAnchor(
            mission_id=mission_id or f"mission-{datetime.now(timezone.utc).isoformat()}",
            task=task or "untitled mission",
            status="RUNNING",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save_anchor(anchor, "init: mission anchor")
        return anchor

    def _save_anchor(self, anchor: MissionAnchor, message: str) -> None:
        anchor.head_commit = self.git.head()
        self.anchor_path.write_text(json.dumps(anchor.to_dict(), indent=2) + "\n")
        new_head = self.git.commit(message)
        if new_head:
            anchor.head_commit = new_head

    def update_checklist(self, checklist: list[dict[str, Any]]) -> None:
        self.anchor.checklist = checklist
        self._save_anchor(self.anchor, "update: checklist")

    # ------------------------------------------------------------------ #
    # L2: episodic memory (append-only journal)
    # ------------------------------------------------------------------ #

    def log_episode(
        self,
        cycle: int,
        agent: str,
        action: str,
        observation: str,
        target: str = "",
        tags: list[str] | None = None,
    ) -> None:
        record = {
            "cycle": cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "action": action,
            "target": target,
            "observation": observation,
            "tags": tags or [],
        }
        with self.journal_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.anchor.current_cycle = max(self.anchor.current_cycle, cycle)
        # Anchor is cheap; keep its cycle counter in git.
        self._save_anchor(self.anchor, f"episode: cycle {cycle} {agent}/{action}")

    def read_journal(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        return [json.loads(line) for line in self.journal_path.read_text().splitlines() if line.strip()]

    def search_episodes(
        self,
        tags: list[str] | None = None,
        since_cycle: int = 0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        episodes = self.read_journal()
        filtered = [
            e
            for e in episodes
            if e["cycle"] >= since_cycle and (not tags or any(t in e["tags"] for t in tags))
        ]
        return filtered[-limit:]

    # ------------------------------------------------------------------ #
    # L3: semantic memory (skills + stub embeddings)
    # ------------------------------------------------------------------ #

    def remember_skill(
        self,
        name: str,
        description: str,
        trigger_tags: list[str] | None = None,
    ) -> None:
        skill = {
            "name": name,
            "description": description,
            "trigger_tags": trigger_tags or [],
            "embedding": self.embedder.embed(description),
        }
        with self.skills_path.open("a") as f:
            f.write(json.dumps(skill) + "\n")
        self._save_anchor(self.anchor, f"skill: {name}")

    def read_skills(self) -> list[dict[str, Any]]:
        if not self.skills_path.exists():
            return []
        return [json.loads(line) for line in self.skills_path.read_text().splitlines() if line.strip()]

    def recall_skills(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = self.embedder.embed(query)
        skills = self.read_skills()
        scored = [
            {**s, "_score": self.embedder.similarity(query_vec, s["embedding"])}
            for s in skills
        ]
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------ #
    # Compaction: L2 -> L3
    # ------------------------------------------------------------------ #

    def compact(self, before_cycle: int) -> None:
        episodes = self.read_journal()
        keep = [e for e in episodes if e["cycle"] >= before_cycle]
        archive = [e for e in episodes if e["cycle"] < before_cycle]

        if not archive:
            logger.info("No episodes to compact before cycle %d", before_cycle)
            return

        # Write archive (ground truth stays in git).
        with self.archive_path.open("a") as f:
            for e in archive:
                f.write(json.dumps(e) + "\n")

        # Stub summarizer: build a skill from the most common tags/actions.
        tags: set[str] = set()
        actions: set[str] = set()
        for e in archive:
            tags.update(e.get("tags", []))
            actions.add(e["action"])

        summary = (
            f"Archived {len(archive)} episodes before cycle {before_cycle}. "
            f"Common actions: {', '.join(sorted(actions))}. "
            f"Common tags: {', '.join(sorted(tags))}."
        )
        self.remember_skill(
            name=f"archive-cycle-{before_cycle}",
            description=summary,
            trigger_tags=sorted(tags),
        )

        # Replace active journal with kept episodes.
        self.journal_path.write_text("".join(json.dumps(e) + "\n" for e in keep))
        self._save_anchor(self.anchor, f"compact: archived {len(archive)} episodes")

        logger.info(
            "Compacted %d episodes into skill 'archive-cycle-%d'; %d episodes remain active",
            len(archive),
            before_cycle,
            len(keep),
        )

    # ------------------------------------------------------------------ #
    # Building the prompt context (L0)
    # ------------------------------------------------------------------ #

    def build_context(
        self,
        cycle: int,
        query: str | None = None,
        recent_limit: int = 10,
        skill_top_k: int = 3,
    ) -> dict[str, Any]:
        """Return the curated subset that actually goes into the model prompt."""
        recent = self.search_episodes(since_cycle=max(0, cycle - recent_limit), limit=recent_limit)
        skills = self.recall_skills(query or self.anchor.task, top_k=skill_top_k)
        return {
            "mission_id": self.anchor.mission_id,
            "task": self.anchor.task,
            "status": self.anchor.status,
            "checklist": self.anchor.checklist,
            "recent_episodes": recent,
            "relevant_skills": [
                {"name": s["name"], "description": s["description"]} for s in skills
            ],
        }


if __name__ == "__main__":
    mem = TieredMemory(
        workdir=".lra/workspaces/ch17",
        mission_id="ch17-tiered-memory",
        task="Build a small LSP server in Python with tests",
    )

    # L1: set the current working state.
    mem.update_checklist([
        {"id": "1", "title": "Scaffold project layout", "status": "done"},
        {"id": "2", "title": "Implement initialize handler", "status": "in_progress"},
        {"id": "3", "title": "Add pytest coverage", "status": "pending"},
    ])

    # L2: log some episodes.
    mem.log_episode(1, "planner", "plan", "Created 3 checklist items", tags=["plan"])
    mem.log_episode(2, "lead", "write", "Added pyproject.toml and src/lsp/", tags=["write", "scaffold"])
    mem.log_episode(3, "lead", "write", "Implemented initialize handler", tags=["write", "handler"])
    mem.log_episode(4, "reviewer", "review", "Missing type hints in initialize.py", tags=["review"])
    mem.log_episode(5, "lead", "fix", "Added type hints; pytest passes", tags=["fix", "test"])

    # L3: add a reusable skill manually.
    mem.remember_skill(
        name="type-hints-first-pass",
        description="Reviewer frequently blocks code missing type hints. Add annotations before first review.",
        trigger_tags=["review", "python"],
    )

    # Build the prompt context for cycle 6.
    context = mem.build_context(cycle=6, query="type hints review")
    print(json.dumps(context, indent=2))

    # Compact old episodes and show what changed.
    mem.compact(before_cycle=4)
    print("\nRecalled skills after compaction:")
    for s in mem.recall_skills("review", top_k=3):
        print(f"  - {s['name']}: {s['description'][:80]}...")
```

## Code Walkthrough

### `MissionAnchor` and L1

The anchor is the root pointer. It tells a resumed worker which mission this repo belongs to, what the current task is, and where to find the journal and skills. We keep it tiny so it can be passed across a Continue-As-New boundary (Chapter 14) and re-read after a crash (Chapter 15).

### `log_episode` and L2

Every cycle appends one line to `.lra/journal.jsonl`. The file is append-only, which makes it durable and auditable. Because we commit the anchor after each episode, the latest cycle counter is in git, but the journal itself is also on disk and can be committed in batches if you prefer fewer git objects.

### `remember_skill` / `recall_skills` and L3

Skills are stored with embeddings. The `StubEmbedder` produces a deterministic vector from a SHA-256 hash so the demo has no external dependencies. In production this interface is backed by `sentence-transformers` or a hosted embedding API, and the skill store is backed by `pgvector` or a vector database.

### `compact`

Compaction is the most important method in this chapter. It splits the journal into active and archived episodes, writes the archive to a separate file, and creates a skill that summarizes the old batch. The raw episodes are not deleted; they move to `.lra/journal.archive.jsonl`, which is still in git and still the ground truth.

### `build_context`

This is the gatekeeper. It decides what actually enters the prompt. It loads:

- The full L1 state (anchor + checklist) — this is small.
- The last N episodes from L2.
- The top-K relevant skills from L3.

The result is bounded in size even when the total history is not.

## Hands-On Exercise

1. Run the demo:

   ```bash
   python lra-demo/ch17_tiered_memory.py
   ```

2. Inspect the git history:

   ```bash
   git -C .lra/workspaces/ch17 log --oneline
   ```

3. Look at the files the demo created:

   ```bash
   cat .lra/workspaces/ch17/.lra/anchor.json
   cat .lra/workspaces/ch17/.lra/journal.jsonl
   cat .lra/workspaces/ch17/.lra/skills.jsonl
   ```

4. Add synthetic load. Create a small script that calls `log_episode` 200 times with random cycles, agents, and tags. Then run `build_context(cycle=200)` and confirm the returned context stays small regardless of how many episodes you added.

5. Run compaction at cycle 150 and verify:
   - `.lra/journal.archive.jsonl` now contains the old episodes.
   - `.lra/journal.jsonl` only contains cycles 150–200.
   - A new skill was created from the archived batch.

6. Simulate a crash: delete the in-memory `TieredMemory` object, create a new one pointing at the same `workdir`, and confirm it reconstructs the same anchor, checklist, and recent episodes.

## Key Takeaway

> Memory in a long-running agent is not how much history you can stuff into a prompt. It is a tiered storage system where git holds the truth, a journal holds the history, and skills hold the lessons — with only a curated slice reaching the model each cycle.

## Next Chapter

**Chapter 18: Vector Search and Skill Libraries** — the stub embedder gets replaced with real embeddings, and the skill library becomes searchable across missions, so lessons learned on one task can accelerate the next.