from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    EvidenceSource,
    RawArtifactReceipt,
    ResearchWork,
    WorkLocation,
)
from quant_research_radar.discovery import ingest_records
from quant_research_radar.raw_archive import RawArchive
from quant_research_radar.sources import SourceRecord


def test_ingest_deduplicates_versions_by_doi_but_preserves_locations() -> None:
    engine = create_engine("sqlite://")
    session = Session(engine)
    Base.metadata.create_all(engine)
    records = [
        SourceRecord(
            "ACADEMIC",
            "openalex",
            "openalex:1",
            "Funding in perpetual markets",
            "https://doi.org/10.1/funding",
            ["A"],
            datetime(2026, 8, 1, tzinfo=UTC),
            "market microstructure evidence",
            {"doi": "10.1/funding", "access_mode": "METADATA_ONLY"},
        ),
        SourceRecord(
            "PREPRINT",
            "arxiv",
            "arxiv:2608.1",
            "Funding in perpetual markets",
            "https://arxiv.org/abs/2608.1",
            ["A"],
            datetime(2026, 8, 2, tzinfo=UTC),
            "market microstructure evidence",
            {"doi": "10.1/funding", "access_mode": "OA_PREPRINT"},
        ),
    ]

    result = ingest_records(
        session, records, retrieved_at=datetime(2026, 8, 31, tzinfo=UTC)
    )

    assert result == {
        "discovered": 2,
        "source_items": 2,
        "canonical_works": 1,
        "locations": 2,
        "archive_failures": 0,
    }
    assert len(session.scalars(select(ResearchWork)).all()) == 1
    assert len(session.scalars(select(WorkLocation)).all()) == 2
    arxiv = session.scalar(
        select(EvidenceSource).where(EvidenceSource.source_name == "arxiv")
    )
    assert arxiv is not None
    assert arxiv.source_class == "ACADEMIC"
    assert arxiv.peer_review_status == "PREPRINT"
    assert arxiv.reliability_prior == "PREPRINT"


def test_ingest_archives_lawful_source_payload_and_links_receipt(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite://")
    session = Session(engine)
    Base.metadata.create_all(engine)
    record = SourceRecord(
        "ACADEMIC",
        "openalex",
        "openalex:archive",
        "Funding in perpetual markets",
        "https://api.openalex.org/works/archive",
        ["A"],
        datetime(2026, 8, 1, tzinfo=UTC),
        "market microstructure evidence",
        {"access_mode": "METADATA_ONLY", "source_payload": {"id": "archive"}},
    )

    ingest_records(
        session,
        [record],
        retrieved_at=datetime(2026, 8, 31, tzinfo=UTC),
        archive=RawArchive(tmp_path / "raw"),
    )

    receipt = session.scalars(select(RawArtifactReceipt)).one()
    assert receipt.provider == "openalex"
    assert receipt.source_item_id is not None
    assert receipt.archive_status == "ARCHIVED"
