# Chapter 18: Vector Search and Skill Libraries

> A week of episodic memory is a firehose. A skill library is the reservoir: small, searchable, and reusable.
> — Fareed Khan

## What We'll Cover

- Why semantic memory needs **vector search**, not just keyword grep
- How **skills** are distilled from compacted episodic memory (Chapter 17) into reusable, embeddable units
- Building a skill library with a **pluggable embedder** and a **hybrid keyword + vector retriever**
- Binding the library to a mission through the **Mission Anchor** (Chapter 16)
- A runnable demo in `lra-demo/ch18_vector_search_skills.py` that embeds, indexes, searches, and injects skills into an agent prompt
- Why the skill library is stored in git and the vector index is treated as a **rebuildable cache**

---

## The Concept

Chapter 17 split memory into three tiers:

- **Working memory** = the git repo (source of truth)
- **Episodic memory** = the event journal (what happened, when)
- **Semantic memory** = skills and embeddings (what the agent learned)

The episodic journal grows linearly. By day five it can contain thousands of cycles. You cannot dump all of that into the prompt. Compaction turns a cluster of related episodes into a **skill**: a short, reusable procedure with a trigger, context, action steps, and a verification rule.

But skills are useless if the agent cannot find them. Keyword search fails when the agent describes a problem with different words than the skill uses. That is where vector search comes in: it maps the query and each skill into the same embedding space, so semantically similar phrases match even when the vocabulary differs.

LRA's skill library therefore has three parts:

1. **Storage** — `skills.jsonl` in the repo, versioned like any other truth.
2. **Embedding** — a pluggable embedder. The default is a deterministic stub for $0 runs; production can swap in a local sentence-transformers model or an API.
3. **Retrieval** — a hybrid scorer that combines cosine similarity with keyword overlap, because exact names like `pytest` or `LSP` still matter.

The Mission Anchor (Chapter 16) records which skill library revision belongs to the mission. If a worker dies and resumes (Chapter 15), the new worker reloads the anchor, reads `skills.jsonl` from git, rebuilds the index, and continues.

The vector index itself is **not** precious. It is a cache. If the embedder changes, you re-embed every skill. We will cover that migration in Chapter 19.

---

## Code Walkthrough

The full runnable demo lives at `lra-demo/ch18_vector_search_skills.py`. It implements a miniature version of the same interface used by `src/lra/memory/skill_library.py` in the real repo.

### 1. The skill model

A skill is not a raw log line. It is a distilled procedure:

```python
# lra-demo/ch18_vector_search_skills.py
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class Skill:
    skill_id: str
    name: str
    trigger_phrases: list[str] = field(default_factory=list)
    context_template: str = ""
    action_steps: list[str] = field(default_factory=list)
    verification: str = ""
    source_episode_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None

    def to_text(self) -> str:
        """Single string used for embedding and keyword search."""
        parts = [
            self.name,
            " ".join(self.trigger_phrases),
            self.context_template,
            " ".join(self.action_steps),
            self.verification,
        ]
        return " ".join(parts)
```

### 2. Pluggable embedders

The demo ships with two embedders. `StubEmbedder` is deterministic and dependency-free, so the demo runs at $0. `SentenceTransformerEmbedder` is used when the optional `embeddings` extra is installed.

```python
import hashlib


class StubEmbedder:
    """Deterministic, zero-cost embedder for tests and local demos."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode()).digest()
        vec = [(seed[i % len(seed)] / 255.0) - 0.5 for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """Local embedding model. Requires: uv pip install 'lra[embeddings]'"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()
```

### 3. The skill library

`SkillLibrary` loads skills from `skills.jsonl`, embeds them, and answers queries with a hybrid score.

```python
class SkillLibrary:
    def __init__(self, repo_dir: Path, embedder: Embedder) -> None:
        self.repo_dir = repo_dir
        self.skills_dir = repo_dir / ".lra" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.skills: dict[str, Skill] = {}
        self._index: list[tuple[str, list[float]]] = []

    def load(self) -> "SkillLibrary":
        path = self.skills_dir / "skills.jsonl"
        if path.exists():
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.skills[data["skill_id"]] = Skill(**data)
        self._rebuild_index()
        return self

    def save(self) -> None:
        path = self.skills_dir / "skills.jsonl"
        with path.open("w") as f:
            for skill in self.skills.values():
                f.write(
                    json.dumps(
                        {
                            "skill_id": skill.skill_id,
                            "name": skill.name,
                            "trigger_phrases": skill.trigger_phrases,
                            "context_template": skill.context_template,
                            "action_steps": skill.action_steps,
                            "verification": skill.verification,
                            "source_episode_ids": skill.source_episode_ids,
                        }
                    )
                    + "\n"
                )
        self._rebuild_index()

    def add(self, skill: Skill) -> None:
        skill.embedding = None
        self.skills[skill.skill_id] = skill
        self.save()

    def _rebuild_index(self) -> None:
        pairs = [(sid, s.to_text()) for sid, s in self.skills.items()]
        if not pairs:
            self._index = []
            return
        ids, texts = zip(*pairs)
        vectors = self.embedder.embed(list(texts))
        self._index = list(zip(ids, vectors))

    def search(
        self, query: str, top_k: int = 3, keyword_boost: float = 0.15
    ) -> list[tuple[Skill, float]]:
        if not self._index:
            return []

        q_vec = self.embedder.embed([query])[0]
        q_norm = math.sqrt(sum(v * v for v in q_vec)) or 1.0
        q_vec = [v / q_norm for v in q_vec]

        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[Skill, float]] = []

        for sid, vec in self._index:
            dot = sum(a * b for a, b in zip(q_vec, vec))

            skill = self.skills[sid]
            haystack = " ".join(
                skill.trigger_phrases + [skill.name, skill.context_template]
            ).lower()
            kw_hits = sum(1 for tok in query_tokens if tok in haystack)
            kw_score = kw_hits / max(len(query_tokens), 1)

            score = dot + keyword_boost * kw_score
            scored.append((skill, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

### 4. From episode to skill

This function shows how Chapter 17's compaction step feeds the library. A compacted episode becomes a skill with concrete steps and a verification rule.

```python
def compact_episode_to_skill(
    name: str,
    trigger_phrases: list[str],
    context: str,
    steps: list[str],
    verification: str,
    episode_ids: list[str],
) -> Skill:
    skill_id = name.lower().replace(" ", "-").replace("/", "-")
    return Skill(
        skill_id=skill_id,
        name=name,
        trigger_phrases=trigger_phrases,
        context_template=context,
        action_steps=steps,
        verification=verification,
        source_episode_ids=episode_ids,
    )
```

### 5. Injecting skills into the prompt

Retrieval is only useful if the agent actually sees the result. This renderer turns the top skills into a compact system-prompt section.

```python
def render_skills_prompt(skills: list[Skill]) -> str:
    if not skills:
        return ""

    blocks = []
    for s in skills:
        steps = "\n".join(
            f"  {i + 1}. {step}" for i, step in enumerate(s.action_steps)
        )
        blocks.append(
            f"Skill: {s.name}\n"
            f"When: {s.context_template}\n"
            f"Do:\n{steps}\n"
            f"Verify: {s.verification}"
        )

    return (
        "## Reusable skills from previous missions\n\n"
        + "\n\n---\n\n".join(blocks)
    )
```

### 6. The demo mission

The `main()` function creates a workspace, loads or creates a library, adds skills distilled from earlier LSP-building episodes, and runs a few semantic queries.

```python
def main() -> None:
    workdir = Path(".lra/workspaces/ch18")
    workdir.mkdir(parents=True, exist_ok=True)

    embedder: Embedder
    try:
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        print("Using local sentence-transformers embedder.\n")
    except Exception as exc:
        print(f"Local embedder unavailable ({exc}); using deterministic stub.\n")
        embedder = StubEmbedder(dim=64)

    lib = SkillLibrary(workdir, embedder).load()

    # Skills distilled from compacted episodic memory (Chapter 17)
    lib.add(
        compact_episode_to_skill(
            name="Add a new LSP method handler",
            trigger_phrases=["lsp method", "new method", "handler", "register"],
            context="The language server needs to support a new LSP method.",
            steps=[
                "Register the method in the server capability list.",
                "Add a dispatch case in handle_request().",
                "Write a unit test that sends the request JSON and asserts the response.",
                "Run pytest and update the method until tests pass.",
            ],
            verification="pytest tests/ passes and the new method responds correctly.",
            episode_ids=["ep-1042", "ep-1089"],
        )
    )

    lib.add(
        compact_episode_to_skill(
            name="Debug a failing pytest",
            trigger_phrases=["pytest", "test failure", "red test", "assertion"],
            context="A test fails after a code change.",
            steps=[
                "Read the failing test and the traceback.",
                "Identify the smallest input that reproduces the failure.",
                "Fix the code, not the test, unless the test is wrong.",
                "Run only the failing test, then the full suite.",
            ],
            verification="The previously failing test passes and no new tests fail.",
            episode_ids=["ep-1103"],
        )
    )

    lib.add(
        compact_episode_to_skill(
            name="Wire up a JSON-RPC notification",
            trigger_phrases=["notification", "json-rpc", "didChange", "event"],
            context="The server must handle an asynchronous LSP notification.",
            steps=[
                "Add the notification method to the dispatcher.",
                "Update server state in the handler; do not write a response.",
                "Add a test that sends the notification and inspects state.",
                "Run the test suite to confirm no regressions.",
            ],
            verification="Notifications are handled without crashing and state is updated.",
            episode_ids=["ep-1156", "ep-1157"],
        )
    )

    queries = [
        "How do I support another LSP method?",
        "A test is failing after my change. What do I do?",
        "The client sends a didChange notification. How should I handle it?",
    ]

    for q in queries:
        print(f"Query: {q}")
        for skill, score in lib.search(q, top_k=2):
            print(f"  {score:.3f}  {skill.name}")
        print()

    # Show what the lead engineer prompt would see
    top_skills = [s for s, _ in lib.search("add a new LSP method", top_k=2)]
    print(render_skills_prompt(top_skills))


if __name__ == "__main__":
    main()
```

Run it with:

```bash
uv run python lra-demo/ch18_vector_search_skills.py
```

If `sentence-transformers` is not installed, the stub embedder still produces deterministic rankings. With the local model, the semantic matches become much sharper.

### 7. Binding to the Mission Anchor

The skill library is part of the mission's ground truth, so the anchor records its revision. In the real repo, `src/lra/state/anchor.py` does this; the demo below shows the same contract.

```python
@dataclass
class MissionAnchor:
    mission_id: str
    repo_dir: Path

    def path(self) -> Path:
        return self.repo_dir / ".lra" / "anchor.json"

    def load(self) -> dict:
        if self.path().exists():
            return json.loads(self.path().read_text())
        return {}

    def save(self, payload: dict) -> None:
        self.path().write_text(json.dumps(payload, indent=2))


def attach_library_to_anchor(anchor: MissionAnchor, library: SkillLibrary) -> None:
    state = anchor.load()
    state["mission_id"] = anchor.mission_id
    state["skills_dir"] = str(library.skills_dir.relative_to(anchor.repo_dir))
    state["skill_count"] = len(library.skills)
    state["skill_ids"] = sorted(library.skills.keys())
    anchor.save(state)
```

When a new worker resumes the mission, it reads the anchor, knows where `skills.jsonl` lives, and rebuilds the index. No shared memory, no lost context.

---

## Hands-On Exercise

1. **Run the demo** and inspect the generated files:
   ```bash
   uv run python lra-demo/ch18_vector_search_skills.py
   cat .lra/workspaces/ch18/.lra/skills/skills.jsonl
   ```

2. **Add a new skill** from a fake compacted episode. Distill this episode:
   > *Episode ep-1201: The agent tried to add a `textDocument/formatting` handler, forgot to register the capability, and spent three cycles debugging. The fix was adding the method to the capability list before dispatch.*

   Turn it into a skill named `Register an LSP capability`, add it to the library, and search with the query `"Why is my new handler not being called?"`. Verify it ranks at the top.

3. **Compare embedders**. If you have the `embeddings` extra installed, run the demo once with the stub embedder and once with `all-MiniLM-L6-v2`. Notice how the local model handles paraphrases like `"My handler is ignored"` better than the stub.

4. **Simulate a resume**. Delete the in-memory `SkillLibrary` object, create a new one pointing at the same `repo_dir`, call `.load()`, and confirm the same search results come back. This is what a resumed Temporal worker does after a crash.

5. **Commit the library**. From the workspace repo, run:
   ```bash
   git -C .lra/workspaces/ch18 add .lra/skills
   git -C .lra/workspaces/ch18 commit -m "ch18: seed skill library"
   ```
   This is the real state. The vector index is not committed; it can be rebuilt from `skills.jsonl` at any time.

---

> **Key Takeaway:** Skills are not memories; they are reusable procedures. Vector search is the retrieval layer that lets the agent recall the right procedure without being told its exact name. Store the skills in git, treat the index as a cache, and the mission stays portable across crashes, embedder swaps, and worker restarts.
> — Fareed Khan

---

## Next Chapter Teaser

**Chapter 19: Re-embedding After Model Changes** — Swapping from a stub embedder to a local model, or from one local model to another, changes every vector in the library. We will cover how to detect the drift, version the embedding model in the Mission Anchor, and rebuild the index safely without losing the structured skill records that live in git.