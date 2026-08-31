from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import Base, RawArtifactReceipt
from quant_research_radar.pipeline import ingest_records
from quant_research_radar.raw_archive import RawArchive
from quant_research_radar.sources import SourceRecord


def test_direct_market_ingest_creates_bounded_collection_run(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    record = SourceRecord(
        "MARKET",
        "hyperliquid",
        "funding:SOL:1",
        "SOL funding",
        None,
        [],
        datetime(2026, 8, 30, tzinfo=UTC),
        "funding",
        {"asset": "SOL", "kind": "funding", "funding_rate": 0.1},
    )

    inserted, duplicates = ingest_records(
        session, [record], RawArchive(tmp_path / "raw")
    )

    receipt = session.query(RawArtifactReceipt).one()
    assert (inserted, duplicates) == (1, 0)
    assert receipt.collection_run_id is not None
