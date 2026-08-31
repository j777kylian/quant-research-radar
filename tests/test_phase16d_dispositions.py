from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import Base, SourceItem, content_hash
from quant_research_radar.intelligence_v2 import AvailabilityBasis, run_intelligence_day


def test_replay_audit_explains_retained_and_rejected_source_items(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, tzinfo=UTC)
    session.add_all(
        [
            SourceItem(
                source_type="ACADEMIC",
                source_name="openalex",
                external_id="keep",
                canonical_url="https://example/keep",
                title="Funding in perpetual markets",
                authors=[],
                published_at=as_of - timedelta(days=1),
                retrieved_at=as_of + timedelta(days=1),
                raw_text="Market microstructure evidence.",
                raw_metadata={"access_mode": "METADATA_ONLY"},
                content_sha256=content_hash("keep", {}),
            ),
            SourceItem(
                source_type="PRACTITIONER",
                source_name="alpha-architect",
                external_id="future",
                canonical_url="https://example/future",
                title="Funding in perpetual markets",
                authors=[],
                published_at=as_of + timedelta(seconds=1),
                retrieved_at=as_of,
                raw_text="Market microstructure evidence.",
                raw_metadata={"access_mode": "PUBLIC_WEB"},
                content_sha256=content_hash("future", {}),
            ),
        ]
    )
    session.commit()

    audit = run_intelligence_day(
        session,
        tmp_path,
        as_of,
        availability_basis=AvailabilityBasis.SOURCE_NATIVE_REPLAY,
        persist=False,
    )

    retained, future = audit["source_dispositions"]
    assert retained["channel"] == "ACADEMIC"
    assert retained["disposition"] == "RETAINED"
    assert retained["published_at"] == (as_of - timedelta(days=1)).isoformat()
    assert retained["retrieved_at"] == (as_of + timedelta(days=1)).isoformat()
    assert retained["replay_availability_at"] == retained["published_at"]
    assert retained["access_mode"] == "METADATA_ONLY"
    assert retained["raw_artifact_id"] is None
    assert retained["linked_hypothesis_ids"] == []
    assert future["channel"] == "SOCIAL"
    assert future["disposition"] == "REJECTED_AVAILABILITY"
    assert future["reason_code"] == "PUBLISHED_AFTER_AS_OF"
