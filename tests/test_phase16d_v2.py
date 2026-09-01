from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    ChannelHypothesis,
    CollectionRun,
    MarketObservation,
    RawArtifact,
    RawArtifactReceipt,
    SourceItem,
    content_hash,
    normalize_utc,
)
from quant_research_radar.intelligence_v2 import (
    run_intelligence_day,
    run_intelligence_replay,
    run_phase18_intelligence_cycle,
)
from quant_research_radar.llm import FakeLLMClient


def test_legacy_day_entrypoint_delegates_to_canonical_phase18_cycle(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, tzinfo=UTC)

    canonical = run_phase18_intelligence_cycle(session, tmp_path / "canonical", as_of)
    legacy = run_intelligence_day(session, tmp_path / "legacy", as_of)

    assert legacy["contract_versions"] == canonical["contract_versions"]
    assert legacy["technical_status"] == canonical["technical_status"]


def test_empty_production_day_does_not_fabricate_critic_pass(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)

    audit = run_intelligence_day(session, tmp_path, datetime(2026, 8, 30, tzinfo=UTC))

    assert audit["critics"]["evidence_auditor"]["disposition"] == "NOT_RUN"
    assert audit["technical_status"] == "RESEARCH_UTILITY_INSUFFICIENT"


def test_canonical_cycle_quarantines_unarchived_source_before_drafting(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, tzinfo=UTC)
    item = SourceItem(
        source_type="ACADEMIC",
        source_name="openalex",
        external_id="empty-body",
        canonical_url="https://example.test/empty",
        title="Funding in perpetual markets",
        authors=[],
        published_at=as_of,
        retrieved_at=as_of,
        raw_text="",
        raw_metadata={},
        content_sha256=content_hash("", {}),
    )
    session.add(item)
    session.commit()

    audit = run_phase18_intelligence_cycle(session, tmp_path, as_of)

    assert audit["channels"]["ACADEMIC"]["retained"] == 0
    assert audit["channels"]["ACADEMIC"]["hypotheses_retained"] == 0


def test_v2_day_keeps_channels_separate_and_persists_market_h1(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    valuation = datetime(2026, 8, 30, 22, tzinfo=UTC)
    for hour in range(31):
        session.add(
            MarketObservation(
                asset="SOL",
                observed_at=valuation - timedelta(hours=hour),
                observation_kind="funding",
                funding_rate=1.0 if hour == 0 else 0.0,
                source_name="hyperliquid",
                retrieved_at=as_of,
            )
        )
        session.add(
            MarketObservation(
                asset="SOL",
                observed_at=valuation - timedelta(hours=hour),
                observation_kind="candle",
                mark_price=95.0 if hour == 24 else 100.0,
                source_name="hyperliquid",
                retrieved_at=as_of,
            )
        )
    session.add_all(
        [
            SourceItem(
                source_type="ACADEMIC",
                source_name="openalex",
                external_id="academic-1",
                canonical_url="https://doi.org/10.1/example",
                title="Funding in perpetual markets and return predictability",
                authors=["Researcher"],
                published_at=as_of - timedelta(days=1),
                retrieved_at=as_of,
                raw_text="This market microstructure study examines funding in perpetual futures.",
                raw_metadata={"doi": "10.1/example", "access_mode": "METADATA_ONLY"},
                content_sha256=content_hash("academic", {}),
            ),
            SourceItem(
                source_type="PRACTITIONER",
                source_name="alpha-architect",
                external_id="social-1",
                canonical_url="https://example.org/original",
                title="Funding in perpetual markets and return predictability",
                authors=[],
                published_at=as_of - timedelta(days=1),
                retrieved_at=as_of,
                raw_text="Public practitioner note on crypto market microstructure and funding.",
                raw_metadata={
                    "independence_key": "https://example.org/original",
                    "access_mode": "PUBLIC_WEB",
                },
                content_sha256=content_hash("social", {}),
            ),
        ]
    )
    session.commit()
    artifact = RawArtifact(
        content_sha256="f" * 64,
        media_type="application/json",
        byte_size=2,
        storage_uri="data/raw/objects/ff/" + "f" * 64,
    )
    run = CollectionRun(source="test", status="SUCCESS")
    session.add_all([artifact, run])
    session.flush()
    for source_item in session.query(SourceItem).all():
        session.add(
            RawArtifactReceipt(
                raw_artifact_id=artifact.id,
                provider=source_item.source_name,
                canonical_url=source_item.canonical_url,
                source_native_timestamp=source_item.published_at,
                retrieved_at=as_of,
                source_item_id=source_item.id,
                collection_run_id=run.id,
            )
        )
    for observation in session.query(MarketObservation).all():
        session.add(
            RawArtifactReceipt(
                raw_artifact_id=artifact.id,
                provider="hyperliquid",
                canonical_url=None,
                source_native_timestamp=observation.observed_at,
                retrieved_at=as_of,
                market_observation_id=observation.id,
                collection_run_id=run.id,
            )
        )
    session.commit()

    audit = run_intelligence_day(session, tmp_path, as_of, client=FakeLLMClient())

    persisted = session.query(ChannelHypothesis).all()
    assert len(persisted) == 3
    assert {item.analysis_mode for item in persisted} == {"PRODUCTION_LIVE"}
    assert {item.availability_basis for item in persisted} == {"RECEIPT_TIME"}
    assert {normalize_utc(item.as_of) for item in persisted if item.as_of} == {as_of}
    assert audit["channels"]["ACADEMIC"]["hypotheses_retained"] == 1
    assert audit["channels"]["SOCIAL"]["hypotheses_retained"] == 1
    assert audit["channels"]["MARKET"]["hypotheses_retained"] == 1
    assert audit["fusion"]["unified_hypotheses"] == 3
    assert {entry["novelty"] for entry in audit["knowledge"]["prior_context"]} == {
        "NEW"
    }
    assert audit["fusion"]["maturity"] == [
        "H1_STATISTICAL_HYPOTHESIS",
        "H1_STATISTICAL_HYPOTHESIS",
        "H1_STATISTICAL_HYPOTHESIS",
    ]
    assert "H3_CONVERGENT" not in audit["fusion"]["maturity"]
    assert audit["tutor"]["concepts"] == 3
    assert (tmp_path / "tutor.json").is_file()
    report = (tmp_path / "executive.md").read_text()
    assert "## Academic Radar" in report
    assert "## Social / Practitioner Radar" in report
    assert "## Market Radar" in report
    assert "## Fusion Radar" in report
    assert "## Tutor" in report
    replay = run_intelligence_replay(session, tmp_path / "replay", [as_of])
    assert replay["mode"] == "ACCELERATED_RECONSTRUCTIVE_REPLAY"
    assert (
        replay["daily_audits"][0]["availability_basis"]
        == "SOURCE_NATIVE_AVAILABILITY_TIME"
    )
