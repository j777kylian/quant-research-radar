from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine

from quant_research_radar.db import Base
from quant_research_radar.llm import FakeLLMClient
from quant_research_radar.pipeline import ingest_records
from quant_research_radar.replay import (
    filter_records_as_of,
    run_replay_day,
    utc_day_cutoff,
)
from quant_research_radar.sources import SourceRecord


def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import Session

    return Session(engine)


def test_replay_future_exclusion_for_market_and_papers() -> None:
    cutoff = datetime(2026, 8, 25, 23, 59, tzinfo=UTC)
    records = [
        SourceRecord(
            "MARKET",
            "hyperliquid",
            "before",
            "before",
            None,
            [],
            cutoff,
            "",
            {"asset": "BTC", "kind": "funding", "funding_rate": 0.1},
        ),
        SourceRecord(
            "MARKET",
            "hyperliquid",
            "after",
            "after",
            None,
            [],
            cutoff + timedelta(seconds=1),
            "",
            {"asset": "BTC", "kind": "funding", "funding_rate": 0.2},
        ),
        SourceRecord(
            "ACADEMIC",
            "arxiv",
            "paper",
            "paper",
            None,
            [],
            cutoff + timedelta(days=1),
            "",
            {},
        ),
    ]
    eligible = filter_records_as_of(records, cutoff)
    assert [record.external_id for record in eligible] == ["before"]


def test_replay_output_identity_and_fake_reproducibility(tmp_path: Path) -> None:
    db = session()
    cutoff = utc_day_cutoff(date(2026, 8, 25))
    records = [
        SourceRecord(
            "ACADEMIC",
            "arxiv",
            "paper",
            "paper",
            None,
            [],
            cutoff - timedelta(hours=1),
            "Evidence",
            {},
        ),
        SourceRecord(
            "MARKET",
            "hyperliquid",
            "funding",
            "BTC funding",
            None,
            [],
            cutoff - timedelta(hours=1),
            "",
            {"asset": "BTC", "kind": "funding", "funding_rate": 0.1},
        ),
    ]
    ingest_records(db, records)
    first = run_replay_day(
        db,
        FakeLLMClient(),
        tmp_path,
        date(2026, 8, 25),
        cutoff - timedelta(days=30),
        "fixture",
    )
    second = run_replay_day(
        db,
        FakeLLMClient(),
        tmp_path,
        date(2026, 8, 24),
        cutoff - timedelta(days=30),
        "fixture",
    )
    assert Path(first["reports"][0]).exists()
    assert Path(second["reports"][0]).exists()
    assert (tmp_path / "2026-08-25" / "audit.json").exists()
    assert (tmp_path / "2026-08-24" / "audit.json").exists()
    assert "AS_OF=2026-08-25" in (tmp_path / "2026-08-25" / "daily.md").read_text()
