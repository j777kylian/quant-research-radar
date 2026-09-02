from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    CollectionRun,
    MarketObservation,
    RawArtifact,
    RawArtifactReceipt,
)
from quant_research_radar.event_study import (
    EventDatasetBuilder,
    EventStudyEngine,
    EventStudySpec,
    SpecIncompleteError,
    funding_spec_from_hypothesis,
)


def _spec() -> EventStudySpec:
    return replace(
        EventStudySpec.funding_v1(
            hypothesis_id="hypothesis-1",
            hypothesis_family_id="EXTREME_FUNDING_FORWARD_RETURN",
            created_at=datetime(2026, 8, 31, tzinfo=UTC),
            sample_start=datetime(2026, 7, 1, tzinfo=UTC),
            sample_end=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        assets=("BTC",),
    )


def _session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed_hour(session: Session, at: datetime, funding: float, price: float) -> None:
    session.add_all(
        [
            MarketObservation(
                asset="BTC",
                observed_at=at,
                observation_kind="funding",
                funding_rate=funding,
                source_name="hyperliquid",
                retrieved_at=at,
            ),
            MarketObservation(
                asset="BTC",
                observed_at=at,
                observation_kind="candle",
                mark_price=price,
                source_name="hyperliquid",
                retrieved_at=at,
            ),
        ]
    )


def _archive_replay_observations(session: Session) -> None:
    artifact = RawArtifact(
        content_sha256="a" * 64,
        media_type="application/json",
        byte_size=2,
        storage_uri="data/raw/objects/aa/" + "a" * 64,
    )
    run = CollectionRun(source="test", status="SUCCESS")
    session.add_all([artifact, run])
    session.flush()
    for observation in session.query(MarketObservation).all():
        session.add(
            RawArtifactReceipt(
                raw_artifact_id=artifact.id,
                provider="hyperliquid",
                canonical_url=None,
                source_native_timestamp=observation.observed_at,
                retrieved_at=observation.retrieved_at,
                market_observation_id=observation.id,
                collection_run_id=run.id,
                analysis_mode="ACCELERATED_RECONSTRUCTIVE_RESEARCH",
            )
        )


def test_incomplete_hypothesis_cannot_become_executable_spec() -> None:
    with pytest.raises(SpecIncompleteError, match="SPEC_INCOMPLETE"):
        funding_spec_from_hypothesis(
            {
                "condition": "funding percentile >= 90",
                "outcome": "",
                "universe": "BTC, ETH, SOL",
                "horizon": "24h",
                "required_data": ["funding", "candles"],
                "falsification_criterion": "difference is absent",
                "critic_disposition": "ACCEPT",
            },
            created_at=datetime(2026, 8, 31, tzinfo=UTC),
        )


def test_event_study_spec_has_immutable_identity() -> None:
    spec = EventStudySpec.funding_v1(
        hypothesis_id="hypothesis-1",
        hypothesis_family_id="EXTREME_FUNDING_FORWARD_RETURN",
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        sample_start=datetime(2026, 7, 1, tzinfo=UTC),
        sample_end=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert spec.spec_id
    assert spec.spec_hash == spec.spec_id


def test_future_funding_and_candle_cannot_change_past_event_qualification() -> None:
    session = _session()
    start = datetime(2026, 7, 1, tzinfo=UTC)
    for hour in range(31):
        _seed_hour(session, start + timedelta(hours=hour), float(hour), 100.0 + hour)
    _archive_replay_observations(session)
    session.commit()

    before = EventDatasetBuilder(session, _spec()).build()
    target = next(
        row for row in before if row.event_time == start + timedelta(hours=29)
    )
    _seed_hour(session, start + timedelta(hours=60), 1_000_000.0, 1_000_000.0)
    session.commit()
    after = EventDatasetBuilder(session, _spec()).build()
    repeated = next(row for row in after if row.event_time == target.event_time)

    assert repeated.treatment == target.treatment
    assert repeated.funding_percentile == target.funding_percentile
    assert target.outcomes[1] is not None
    assert target.outcomes[24] is None


def test_engine_writes_reproducible_artifacts_and_never_supports_without_critic(
    tmp_path,
) -> None:
    session = _session()
    start = datetime(2026, 7, 1, tzinfo=UTC)
    for hour in range(80):
        _seed_hour(
            session, start + timedelta(hours=hour), float(hour % 20), 100.0 + hour
        )
    _archive_replay_observations(session)
    session.commit()

    result = EventStudyEngine(session, _spec(), client=None).run(tmp_path)

    assert result["status"] != "SUPPORTED"
    assert result["critic"]["disposition"] == "NOT_RUN"
    artifact = tmp_path / result["run_id"]
    assert {
        "spec.json",
        "dataset_manifest.json",
        "summary.json",
        "statistical_results.json",
        "robustness.json",
        "negative_controls.json",
        "critic.json",
        "executive.md",
    } <= {path.name for path in artifact.iterdir()}
