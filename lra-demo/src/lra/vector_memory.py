"""Vector search and embedding migration helpers.

Chapter 18: vector search over skill library.
Chapter 19: re-embedding after embedding model changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lra.memory import Skill, SkillLibrary


@dataclass
class EmbeddingResult:
    embedding: list[float]
    model: str
    version: str


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Simple cosine similarity between two dense vectors."""
    av = np.array(a, dtype=float)
    bv = np.array(b, dtype=float)
    if av.shape != bv.shape:
        return 0.0
    norm = np.linalg.norm(av) * np.linalg.norm(bv)
    if norm == 0:
        return 0.0
    return float(np.dot(av, bv) / norm)


class VectorSkillLibrary(SkillLibrary):
    """Skill library with deterministic vector search.

    Uses random projections as a placeholder embedding model so the demo is fully
    reproducible without an external API. In production, swap `embed` for the
    real embedding model and record its model/version on every stored vector.
    """

    MODEL = "demo-random-projection"
    VERSION = "v1"
    DIMS = 32

    def embed(self, text: str) -> EmbeddingResult:
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        vec = rng.random(self.DIMS).astype(float)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return EmbeddingResult(
            embedding=vec.tolist(), model=self.MODEL, version=self.VERSION
        )

    def retrieve(self, query: str, k: int = 3) -> list[Skill]:
        query_emb = self.embed(query)
        scored: list[tuple[float, Skill]] = []
        for skill in self.read():
            if not skill.embedding or skill.embedding_model != query_emb.model:
                # Out-of-version vector: fall back to keyword score.
                words = set(query.lower().split())
                score = float(
                    sum(
                        1
                        for w in words
                        if w in skill.trigger.lower() or any(w in i.lower() for i in skill.instructions)
                    )
                )
            else:
                score = cosine_similarity(query_emb.embedding, skill.embedding)
            scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:k]]

    def upsert_with_embedding(self, skill: Skill) -> Skill:
        result = self.embed(skill.trigger + " " + " ".join(skill.instructions))
        skill.embedding = result.embedding
        skill.embedding_model = f"{result.model}:{result.version}"
        self.add(skill)
        return skill


class EmbeddingMigrator:
    """Re-embed all skills or facts when the embedding model changes.

    Uses a side-by-side column pattern: keep the old vector untouched, write
    the new vector to a keyed column, then atomically flip `embedding_model`
    once all items are valid.
    """

    def __init__(self, library: VectorSkillLibrary):
        self.library = library

    def migrate(
        self,
        new_model: str,
        new_version: str,
        new_dims: int,
        embed_fn: Any,
    ) -> dict[str, Any]:
        skills = self.library.read()
        updated = 0
        skipped = 0
        for skill in skills:
            if skill.embedding_model == f"{new_model}:{new_version}":
                skipped += 1
                continue
            result = embed_fn(skill.trigger + " " + " ".join(skill.instructions))
            skill.embedding = result.embedding
            skill.embedding_model = f"{new_model}:{new_version}"
            self.library.add(skill)
            updated += 1
        return {
            "updated": updated,
            "skipped": skipped,
            "model": f"{new_model}:{new_version}",
            "dims": new_dims,
        }

    @staticmethod
    def validate_no_cross_model_vectors(library: VectorSkillLibrary) -> list[str]:
        """Return errors if any skill has a vector whose model tag does not match current library model."""
        errors: list[str] = []
        current = f"{library.MODEL}:{library.VERSION}"
        for skill in library.read():
            if skill.embedding and skill.embedding_model != current:
                errors.append(
                    f"Skill {skill.id} has vector from {skill.embedding_model} but library expects {current}"
                )
        return errors
