"""Tiered memory: episodic, semantic, and procedural stores.

Episodic memory = events.ndjson (already in MissionAnchor).
Semantic memory = facts and observations (key/value + optional vector).
Procedural memory = skill library (retrievable instructions keyed by trigger).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Fact:
    id: str
    topic: str
    content: str
    source: str = ""
    embedding_model: str = "none"
    embedding: list[float] = field(default_factory=list)


@dataclass
class Skill:
    id: str
    trigger: str
    instructions: list[str]
    tools: list[str] = field(default_factory=list)
    embedding_model: str = "none"
    embedding: list[float] = field(default_factory=list)


class SemanticMemory:
    """Key/value semantic store with embedding metadata.

    For the demo we use a JSON file on disk. In production this maps to a
    pgvector-backed table with a separate column per embedding model version.
    """

    def __init__(self, workdir: Path | str):
        self.workdir = Path(workdir)
        self.lra_dir = self.workdir / ".lra"
        self.lra_dir.mkdir(parents=True, exist_ok=True)
        self.facts_path = self.lra_dir / "semantic_memory.json"

    def read(self) -> list[Fact]:
        if not self.facts_path.exists():
            return []
        return [Fact(**entry) for entry in json.loads(self.facts_path.read_text())]

    def write(self, facts: list[Fact]) -> None:
        self.facts_path.write_text(
            json.dumps(
                [
                    {
                        "id": f.id,
                        "topic": f.topic,
                        "content": f.content,
                        "source": f.source,
                        "embedding_model": f.embedding_model,
                        "embedding": f.embedding,
                    }
                    for f in facts
                ],
                indent=2,
            )
            + "\n"
        )

    def add(self, fact: Fact) -> None:
        facts = {f.id: f for f in self.read()}
        facts[fact.id] = fact
        self.write(list(facts.values()))

    def find(self, topic: str | None = None, source: str | None = None) -> list[Fact]:
        facts = self.read()
        if topic:
            facts = [f for f in facts if topic.lower() in f.topic.lower()]
        if source:
            facts = [f for f in facts if source.lower() in f.source.lower()]
        return facts


class SkillLibrary:
    """Procedural memory: a library of reusable skills keyed by trigger."""

    def __init__(self, workdir: Path | str):
        self.workdir = Path(workdir)
        self.lra_dir = self.workdir / ".lra"
        self.lra_dir.mkdir(parents=True, exist_ok=True)
        self.skills_path = self.lra_dir / "skills.json"

    def read(self) -> list[Skill]:
        if not self.skills_path.exists():
            return []
        return [Skill(**entry) for entry in json.loads(self.skills_path.read_text())]

    def write(self, skills: list[Skill]) -> None:
        self.skills_path.write_text(
            json.dumps(
                [
                    {
                        "id": s.id,
                        "trigger": s.trigger,
                        "instructions": s.instructions,
                        "tools": s.tools,
                        "embedding_model": s.embedding_model,
                        "embedding": s.embedding,
                    }
                    for s in skills
                ],
                indent=2,
            )
            + "\n"
        )

    def add(self, skill: Skill) -> None:
        skills = {s.id: s for s in self.read()}
        skills[skill.id] = skill
        self.write(list(skills.values()))

    def retrieve(self, query: str, k: int = 3) -> list[Skill]:
        """Naive keyword retrieval. Replaced with vector search in Chapter 18."""
        words = set(query.lower().split())
        scored: list[tuple[float, Skill]] = []
        for skill in self.read():
            score = sum(1 for w in words if w in skill.trigger.lower() or any(w in i.lower() for i in skill.instructions))
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:k]]


class TieredMemory:
    """Unified facade over episodic, semantic, and procedural memory."""

    def __init__(self, anchor: Any):
        self.anchor = anchor
        self.semantic = SemanticMemory(anchor.workdir)
        self.skills = SkillLibrary(anchor.workdir)

    def record_event(self, **fields: Any) -> None:
        self.anchor.log_event(**fields)

    def remember(self, fact: Fact) -> None:
        self.semantic.add(fact)

    def recall(self, topic: str) -> list[Fact]:
        return self.semantic.find(topic=topic)

    def learn_skill(self, skill: Skill) -> None:
        self.skills.add(skill)

    def use_skill(self, query: str, k: int = 3) -> list[Skill]:
        return self.skills.retrieve(query, k=k)


@dataclass
class EmbeddingVersion:
    """Schema helper for Chapter 19. Every embedding must carry its model version."""

    model: str
    version: str
    dims: int
    created_at: str

    def key(self) -> str:
        return f"{self.model}-{self.version}"
