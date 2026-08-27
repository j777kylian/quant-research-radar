from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import Base, Claim, Hypothesis, SourceItem
from quant_research_radar.llm import FakeLLMClient, TriageOutput
from quant_research_radar.pipeline import analyze, daily_report, ingest
from quant_research_radar.sources import ArxivSource, HyperliquidSource


def session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_fake_schema_and_validation() -> None:
    result = FakeLLMClient().triage("title", "text")
    assert result.retain
    assert TriageOutput.model_validate(result.model_dump()).relevance_score == 75


def test_ingestion_is_idempotent() -> None:
    db = session()
    adapter = ArxivSource()
    assert ingest(db, adapter, 1, offline=True) == 1
    assert ingest(db, adapter, 1, offline=True) == 0
    assert len(db.scalars(select(SourceItem)).all()) == 1


def test_pipeline_persists_separated_objects_and_report_has_labels(
    tmp_path: Path,
) -> None:
    db = session()
    ingest(db, ArxivSource(), 1, offline=True)
    ingest(db, HyperliquidSource(), 3, offline=True)
    assert analyze(db, FakeLLMClient(), 10) == 4
    assert db.scalars(select(Claim)).first() is not None
    assert db.scalars(select(Hypothesis)).first() is not None
    report = daily_report(db, str(tmp_path))
    text = report.read_text()
    assert "FACT" in text and "CLAIM" in text and "HYPOTHESIS" in text
    assert all(word not in text.upper() for word in ["BUY", "SELL", "LONG", "SHORT"])
