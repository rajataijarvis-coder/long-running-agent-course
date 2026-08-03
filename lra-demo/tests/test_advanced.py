"""Tests for reviewer, reflection, budget, and memory modules."""

import json
from pathlib import Path

import pytest

from lra.anchor import MissionAnchor
from lra.budget import BudgetConfig, BudgetGovernor, BudgetExceededError
from lra.memory import Fact, SemanticMemory, Skill, SkillLibrary, TieredMemory
from lra.review import ReflectionAgent, StaticReviewer
from lra.vector_memory import VectorSkillLibrary


def test_reviewer_flags_done_without_verification(tmp_path: Path) -> None:
    anchor = MissionAnchor(tmp_path)
    anchor.write_checklist({"items": [{"id": "1", "description": "Create app.py", "status": "done"}]})
    reviewer = StaticReviewer(anchor)
    result = reviewer.review("1")
    assert result.approved is False
    assert "no verification record" in result.findings[0]


def test_reviewer_approves_verified_done(tmp_path: Path) -> None:
    anchor = MissionAnchor(tmp_path)
    anchor.write_checklist(
        {"items": [{"id": "1", "description": "Create app.py", "status": "done", "verified_by": ["pytest"]}]}
    )
    reviewer = StaticReviewer(anchor)
    result = reviewer.review("1")
    assert result.approved is True


def test_reflection_hypotheses_on_pytest_failure(tmp_path: Path) -> None:
    anchor = MissionAnchor(tmp_path)
    anchor.log_event(action="verify", item_id="1", passed=False, checks=[{"name": "pytest", "passed": False}])
    reflect = ReflectionAgent(anchor)
    result = reflect.reflect("1")
    assert "pytest" in result.root_cause
    assert len(result.hypotheses) == 3


def test_semantic_memory_round_trip(tmp_path: Path) -> None:
    sm = SemanticMemory(tmp_path)
    sm.add(Fact(id="f1", topic="testing", content="Always write tests before code."))
    found = sm.find(topic="testing")
    assert len(found) == 1
    assert found[0].content == "Always write tests before code."


def test_skill_library_keyword_retrieval(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path)
    lib.add(
        Skill(
            id="s1",
            trigger="write fastapi endpoint",
            instructions=["Create router", "Add pydantic model"],
        )
    )
    hits = lib.retrieve("fastapi endpoint")
    assert len(hits) == 1
    assert hits[0].id == "s1"


def test_vector_skill_library_retrieval(tmp_path: Path) -> None:
    lib = VectorSkillLibrary(tmp_path)
    lib.upsert_with_embedding(
        Skill(
            id="s1",
            trigger="write fastapi endpoint",
            instructions=["Create router", "Add pydantic model"],
        )
    )
    hits = lib.retrieve("fastapi endpoint")
    assert len(hits) == 1
    assert hits[0].id == "s1"


def test_budget_governor_blocks_at_cap(tmp_path: Path) -> None:
    gov = BudgetGovernor(tmp_path, config=BudgetConfig(usd_cap=1.0))
    summary = gov.record_call("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
    assert summary["ok"] is False
    assert summary["spent_usd"] >= summary["cap_usd"]


def test_budget_governor_assert_room_raises(tmp_path: Path) -> None:
    gov = BudgetGovernor(tmp_path, config=BudgetConfig(usd_cap=0.01))
    gov.record_call("gpt-4o", input_tokens=1000, output_tokens=1000)
    with pytest.raises(BudgetExceededError):
        gov.assert_room(0.1)


def test_embedding_migrator_updates_model_version(tmp_path: Path) -> None:
    lib = VectorSkillLibrary(tmp_path)
    lib.upsert_with_embedding(
        Skill(id="s1", trigger="deploy to docker", instructions=["Build image", "Run container"])
    )
    from lra.vector_memory import EmbeddingMigrator

    def new_embed(text: str):
        class R:
            embedding = [0.0] * 8
            model = "new-model"
            version = "v2"

        return R()

    migrator = EmbeddingMigrator(lib)
    report = migrator.migrate("new-model", "v2", 8, new_embed)
    assert report["updated"] == 1
    updated = lib.read()[0]
    assert updated.embedding_model == "new-model:v2"
    assert len(updated.embedding) == 8
