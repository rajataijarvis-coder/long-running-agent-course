# Chapter 19: Re-embedding After Model Changes

## What We'll Cover

- Why swapping an embedding model invalidates every cached vector, not just the query vector
- How the **Mission Anchor** from Chapter 16 stores an embedder fingerprint that acts as a version gate
- Detecting a stale index by comparing the anchor fingerprint with the index manifest
- Rebuilding the vector index from git-stored skills while keeping the index itself a disposable cache
- Making re-embedding idempotent, atomic, and safe for long-running missions
- A runnable demo in `lra-demo/ch19_reembedding.py` that changes embedders and rebuilds the index automatically

---

## The Problem: Vectors Are Model-Specific

In Chapter 18 we turned compacted episodic memory into a **skill library** and indexed it with vector search. The important detail is that a vector is not a universal description of a sentence. It is a coordinate in one particular model's latent space. If you change the model, the old coordinates and the new coordinates are no longer comparable. Running a cosine similarity between a new query vector and old skill vectors is like measuring distance on two different maps.

This means the vector index is a **rebuildable cache**, not durable truth. The durable truth stays where it has been since Chapter 16 and Chapter 17: the skill text and metadata in git. The index only needs to be fast and correct for the current embedder.

The **Mission Anchor** is the natural place to record which embedder is authoritative. It stores a fingerprint — name, version, dimension, and any other canonical signature. On every startup or resume the worker compares the anchor fingerprint with the index manifest. If they differ, the worker rebuilds the index from the git skill store before any agent cycle runs.

This also covers less obvious model changes: dimension changes, quantization, fine-tuning, and provider-side silent updates. Any of them shift the latent space, so the fingerprint must capture enough to detect them.

---

## The Re-embedding Contract

A minimal re-embedding system has four rules:

1. **Skills live in git.** The text, tags, and source metadata are the durable record.
2. **The index is a cache.** It lives in a directory that can be deleted and regenerated.
3. **The anchor owns the model fingerprint.** Whoever changes the embedder must update the anchor.
4. **Rebuild on mismatch.** If the anchor fingerprint does not match the index manifest, re-embed every skill before serving search results.

The rebuild should be atomic: write the new vectors and manifest to a temporary location, then rename or swap them. That way a crash in the middle does not leave a half-written index that looks fresh.

---

## Code Walkthrough: `lra-demo/ch19_reembedding.py`

The demo below uses a deterministic stub embedder so it runs at `$0` and shows the same behavior every time. It stores skills in a JSONL file, writes the Mission Anchor to JSON, and keeps the vector index in a `.vector_cache` directory.

The key flow is:

1. Load the anchor and construct the embedder it specifies.
2. Try to load the existing index.
3. If the index is missing, the dimension differs, or the fingerprint mismatches, rebuild from `skills.jsonl`.
4. After a simulated model upgrade, the anchor fingerprint changes and the next run rebuilds automatically.

```python
# lra-demo/ch19_reembedding.py
"""Re-embedding after an embedding model change.

Skills are the durable truth in git. The vector index is a model-specific
cache that rebuilds itself whenever the Mission Anchor's embedder fingerprint
no longer matches the index manifest.
"""
from __future__ import annotations

import json
import math
import hashlib
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

DEMO_ROOT = Path(__file__).with_suffix("").parent / "ch19_store"
SKILLS_FILE = DEMO_ROOT / "skills.jsonl"
INDEX_DIR = DEMO_ROOT / ".vector_cache"
ANCHOR_FILE = DEMO_ROOT / "mission_anchor.json"


# ---------------------------------------------------------------------------
# Embedder spec + deterministic stub embedder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbedderSpec:
    name: str
    version: str
    dimension: int

    @property
    def fingerprint(self) -> str:
        payload = f"{self.name}:{self.version}:{self.dimension}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]


class Embedder(Protocol):
    spec: EmbedderSpec

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class StubEmbedder:
    """Deterministic, version-sensitive embedder. Free to run, easy to test."""

    def __init__(self, name: str = "stub", version: str = "1.0.0", dimension: int = 8):
        self.spec = EmbedderSpec(name, version, dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            # The fingerprint is part of the seed, so changing the model changes
            # every vector. This is exactly what happens with a real embedder.
            seed = f"{self.spec.fingerprint}:{text}".encode()
            digest = hashlib.sha256(seed).digest()
            vec = [(b / 255.0) * 2 - 1 for b in digest[: self.spec.dimension]]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


# ---------------------------------------------------------------------------
# Mission Anchor: the durable record of which embedder is authoritative
# ---------------------------------------------------------------------------
@dataclass
class MissionAnchor:
    mission_id: str
    embedder_spec: EmbedderSpec

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "embedder": {
                "name": self.embedder_spec.name,
                "version": self.embedder_spec.version,
                "dimension": self.embedder_spec.dimension,
                "fingerprint": self.embedder_spec.fingerprint,
            },
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_path(cls, path: Path) -> MissionAnchor:
        data = json.loads(path.read_text())
        e = data["embedder"]
        return cls(
            mission_id=data["mission_id"],
            embedder_spec=EmbedderSpec(e["name"], e["version"], e["dimension"]),
        )


# ---------------------------------------------------------------------------
# Skill library: the durable source of truth, stored as JSONL in git
# ---------------------------------------------------------------------------
@dataclass
class SkillRecord:
    skill_id: str
    name: str
    text: str
    tags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "skill_id": self.skill_id,
                "name": self.name,
                "text": self.text,
                "tags": self.tags,
            }
        )

    @classmethod
    def from_json(cls, line: str) -> SkillRecord:
        d = json.loads(line)
        return cls(skill_id=d["skill_id"], name=d["name"], text=d["text"], tags=d["tags"])


class SkillLibrary:
    def __init__(self, path: Path):
        self.path = path
        self.skills: list[SkillRecord] = []
        self._load()

    def _load(self) -> None:
        self.skills = []
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                self.skills.append(SkillRecord.from_json(line))

    def add(self, skill: SkillRecord) -> None:
        if any(s.skill_id == skill.skill_id for s in self.skills):
            raise ValueError(f"duplicate skill_id: {skill.skill_id}")
        self.skills.append(skill)
        with self.path.open("a") as f:
            f.write(skill.to_json() + "\n")

    def save(self) -> None:
        self.path.write_text("\n".join(s.to_json() for s in self.skills) + "\n")


# ---------------------------------------------------------------------------
# Vector index: a rebuildable cache keyed by embedder fingerprint
# ---------------------------------------------------------------------------
@dataclass
class VectorIndex:
    embedder: Embedder
    index_dir: Path
    ids: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def vectors_path(self) -> Path:
        return self.index_dir / "vectors.jsonl"

    def is_fresh_for(self, spec: EmbedderSpec) -> bool:
        if not self.manifest_path.exists() or not self.vectors_path.exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return (
            manifest.get("model_fingerprint") == spec.fingerprint
            and manifest.get("dimension") == spec.dimension
        )

    def load(self) -> bool:
        if not self.is_fresh_for(self.embedder.spec):
            return False
        self.ids = []
        self.vectors = []
        for line in self.vectors_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            sid, vec = json.loads(line)
            self.ids.append(sid)
            self.vectors.append(vec)
        return True

    def build(self, skills: list[SkillRecord]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        texts = [s.text for s in skills]
        self.vectors = self.embedder.embed(texts)
        self.ids = [s.skill_id for s in skills]

        # Atomic-ish write: write to a temp directory, then swap in.
        tmp_dir = self.index_dir.with_suffix(".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_vectors = tmp_dir / "vectors.jsonl"
        tmp_manifest = tmp_dir / "manifest.json"

        tmp_vectors.write_text(
            "\n".join(json.dumps([sid, vec]) for sid, vec in zip(self.ids, self.vectors)) + "\n"
        )
        manifest = {
            "model_fingerprint": self.embedder.spec.fingerprint,
            "dimension": self.embedder.spec.dimension,
            "indexed_count": len(self.ids),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

        # Replace old cache atomically.
        tmp_vectors.replace(self.vectors_path)
        tmp_manifest.replace(self.manifest_path)
        # Clean up temp dir if anything was left behind.
        if tmp_dir.exists():
            tmp_dir.rmdir()

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q_vec = self.embedder.embed([query])[0]
        scored = []
        for sid, vec in zip(self.ids, self.vectors):
            dot = sum(a * b for a, b in zip(q_vec, vec))
            scored.append((sid, dot))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def ensure_index(embedder: Embedder, library: SkillLibrary) -> VectorIndex:
    """Load a fresh index or rebuild it when the model or skill set changed."""
    index = VectorIndex(embedder, INDEX_DIR)
    loaded = index.load()

    if loaded and len(index.ids) == len(library.skills):
        print(f"Index fresh for {embedder.spec.fingerprint} ({len(index.ids)} skills).")
        return index

    reason = "missing or stale" if not loaded else f"count mismatch ({len(index.ids)} vs {len(library.skills)})"
    print(f"Rebuilding index: {reason}")
    index.build(library.skills)
    return index


# ---------------------------------------------------------------------------
# Git checkpoint helper: skills and anchor are real state, index is ignored
# ---------------------------------------------------------------------------
def git_checkpoint(repo: Path, message: str) -> None:
    """Commit durable files. The vector cache should be in .gitignore."""
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repo, check=False
    )
    if result.returncode == 0:
        return
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def seed_library(library: SkillLibrary) -> None:
    if library.skills:
        return
    library.add(
        SkillRecord(
            skill_id="skill-1",
            name="Pytest exception assertions",
            text="Use pytest.raises(ContextManager) to assert that a callable raises a specific exception. Prefer matching the exception message with a regex when the message carries semantic meaning.",
            tags=["testing", "pytest"],
        )
    )
    library.add(
        SkillRecord(
            skill_id="skill-2",
            name="LSP initialize handshake",
            text="Respond to the LSP initialize request with a capabilities object. Do not send other messages until initialize has been acknowledged.",
            tags=["lsp", "protocol"],
        )
    )
    library.add(
        SkillRecord(
            skill_id="skill-3",
            name="Checkpoint after verification",
            text="Only commit a checkpoint after deterministic verification passes. The commit message should reference the checklist item that was completed.",
            tags=["durability", "git"],
        )
    )
    library.save()


def main() -> None:
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)

    # Load or create the anchor. In production this comes from the durable
    # Mission Anchor stored in the mission's git repo (Chapter 16).
    if not ANCHOR_FILE.exists():
        anchor = MissionAnchor("demo-ch19", EmbedderSpec("stub", "1.0.0", 8))
        anchor.write(ANCHOR_FILE)
    else:
        anchor = MissionAnchor.from_path(ANCHOR_FILE)

    library = SkillLibrary(SKILLS_FILE)
    seed_library(library)

    # Build or load the index for the current anchor.
    embedder = StubEmbedder(
        anchor.embedder_spec.name,
        anchor.embedder_spec.version,
        anchor.embedder_spec.dimension,
    )
    index = ensure_index(embedder, library)
    git_checkpoint(DEMO_ROOT, "ch19: seed skills and anchor")

    query = "how do I assert an exception in a test"
    print(f"\nQuery: {query!r}")
    print("Top skills:", index.search(query, k=2))

    # Simulate an operator upgrading the embedder model. In a real system this
    # would be a config change that updates the Mission Anchor.
    print("\n--- Simulating embedder upgrade ---")
    new_spec = EmbedderSpec(anchor.embedder_spec.name, "2.0.0", anchor.embedder_spec.dimension)
    upgraded = MissionAnchor(anchor.mission_id, new_spec)
    upgraded.write(ANCHOR_FILE)

    embedder_v2 = StubEmbedder(new_spec.name, new_spec.version, new_spec.dimension)
    index_v2 = ensure_index(embedder_v2, library)
    git_checkpoint(DEMO_ROOT, "ch19: re-embed after model upgrade")

    print(f"\nQuery: {query!r}")
    print("Top skills after re-embedding:", index_v2.search(query, k=2))


if __name__ == "__main__":
    main()
```

A few things to note in the code:

- `EmbedderSpec.fingerprint` is a short hash of name, version, and dimension. The manifest stores this fingerprint.
- `StubEmbedder` includes the fingerprint in its seed, so changing the version changes every vector. This mirrors real embedder behavior.
- `VectorIndex.build` writes to a temporary directory first and then replaces the old files. A crash during the rebuild leaves the old manifest and vectors intact.
- `ensure_index` rebuilds not only on fingerprint mismatch but also when the skill count in the library differs from the indexed count. That handles incremental additions.
- `git_checkpoint` commits the durable files. In a real LRA mission the `.vector_cache` directory is in `.gitignore`; only the skills, anchor, and `.gitignore` itself are tracked.

---

## Hands-On Exercise

1. Run the demo:

   ```bash
   uv run python lra-demo/ch19_reembedding.py
   ```

   The first run creates `lra-demo/ch19_store/`, seeds three skills, builds an index for `stub:1.0.0`, and prints the top search results.

2. Inspect the manifest:

   ```bash
   cat lra-demo/ch19_store/.vector_cache/manifest.json
   ```

   Note the `model_fingerprint` and `indexed_count`.

3. Run the script a second time. The console should say the index is fresh and skip the rebuild.

4. Simulate a model change by editing `lra-demo/ch19_store/mission_anchor.json` and bumping the embedder version from `1.0.0` to `2.0.0`. Then rerun the script. You should see:

   ```text
   Rebuilding index: missing or stale
   ```

   and the search results may change because the latent space changed.

5. Add a new skill to `lra-demo/ch19_store/skills.jsonl` without changing the model. Rerun. The script should detect the count mismatch and rebuild incrementally while keeping the same embedder fingerprint.

6. (Optional) Add `lra-demo/ch19_store/.vector_cache/` to a `.gitignore` file in that directory, run `git -C lra-demo/ch19_store log --oneline`, and confirm that the durable skill records are committed while the cache is not.

---

> A vector index is a model-specific projection. Change the projector, and every point must be re-projected. The only durable truth is the skill text in git; the index is a cache that rebuilds itself the moment the Mission Anchor says the model changed.

---

## Next Chapter

**Chapter 20: Budget Governor and Cost Caps.** Long-running missions can spend tokens for days. We will add a durable ledger that tracks cumulative cost, enforces a ceiling, and can cut scope or pause the mission before the budget is exhausted.