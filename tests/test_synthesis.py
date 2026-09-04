from datetime import date

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import Base, DailyRun, SourceItem, TopicBrief
from quant_research_radar.synthesis import (
    DEPTH_ABSTRACT,
    DEPTH_FULL_TEXT,
    DEPTH_METADATA_ONLY,
    evidence_depth,
    synthesize_daily_topics,
)


def _session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def _daily() -> DailyRun:
    return DailyRun(
        logical_date=date(2026, 1, 2),
        status="SUCCESS",
        code_sha="abc",
        market_status="SUCCESS",
        academic_status="SUCCESS",
        practitioner_status="SUCCESS",
        analysis_status="SUCCESS",
        knowledge_status="SUCCESS",
        audit_status="SUCCESS",
        failure_reasons=[],
        source_health={},
        llm_summary={},
    )


def test_evidence_depth_never_infers_method_from_metadata() -> None:
    metadata = SourceItem(
        source_type="ACADEMIC",
        source_name="arxiv",
        external_id="m",
        title="Metadata title",
        raw_text="",
        raw_metadata={},
        content_sha256="x",
    )
    abstract = SourceItem(
        source_type="ACADEMIC",
        source_name="arxiv",
        external_id="a",
        title="Abstract title",
        raw_text="",
        raw_metadata={"abstract": "A bounded abstract."},
        content_sha256="x",
    )
    full = SourceItem(
        source_type="ACADEMIC",
        source_name="arxiv",
        external_id="f",
        title="Full title",
        raw_text="Methods and results are explicitly archived.",
        raw_metadata={},
        content_sha256="x",
    )
    assert evidence_depth(metadata) == DEPTH_METADATA_ONLY
    assert evidence_depth(abstract) == DEPTH_ABSTRACT
    assert evidence_depth(full) == DEPTH_FULL_TEXT


def test_topic_brief_depth_matches_persisted_sources(monkeypatch) -> None:
    s = _session()
    daily = _daily()
    s.add(daily)
    s.commit()
    monkeypatch.setattr(
        "quant_research_radar.synthesis.build_daily_topic_packet",
        lambda session, record: {
            "logical_date": record.logical_date.isoformat(),
            "run_id": str(record.id),
            "status": record.status,
            "source_items": [
                {
                    "source_item_id": "m",
                    "url": "https://example.test",
                    "evidence_depth": DEPTH_METADATA_ONLY,
                }
            ],
            "findings": [{"title": "Metadata finding", "question": "What changed?"}],
            "critic_reasons": {},
            "research": {},
            "knowledge": {},
            "market": {},
            "inputs": {},
            "provenance": {"code_sha": record.code_sha},
        },
    )
    topic = synthesize_daily_topics(s, daily.id)[0]
    assert topic.brief["evidence_depth"] == DEPTH_METADATA_ONLY
    assert topic.brief["sources"][0]["source_item_id"] == "m"

    s = _session()
    daily = _daily()
    s.add(daily)
    s.commit()
    before = len(s.scalars(select(SourceItem)).all())
    first = synthesize_daily_topics(s, daily.id)
    second = synthesize_daily_topics(s, daily.id)
    assert len(first) == len(second)
    assert len(s.scalars(select(SourceItem)).all()) == before
    assert len(s.scalars(select(TopicBrief)).all()) == len(first)
    if first:
        assert first[0].model_metadata["role"] == "DETERMINISTIC_TOPIC_SYNTHESIS"
        assert first[0].input_packet_hash == second[0].input_packet_hash
