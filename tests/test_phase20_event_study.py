from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    ChannelHypothesis,
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

HYPOTHESIS_ID = UUID("00000000-0000-0000-0000-000000000001")
FAMILY = "EXTREME_FUNDING_FORWARD_RETURN"


def _spec() -> EventStudySpec:
    return replace(
        EventStudySpec.funding_v1(
            hypothesis_id=str(HYPOTHESIS_ID),
            hypothesis_family_id=FAMILY,
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


def _ready_hypothesis(session: Session) -> None:
    session.add(
        ChannelHypothesis(
            id=HYPOTHESIS_ID,
            channel="MARKET",
            statement="funding study",
            mechanism=None,
            condition="funding percentile >= 90",
            outcome="forward return",
            universe="BTC",
            horizon="24h",
            expected_direction=None,
            required_data=["funding", "candles"],
            falsification_criterion="no difference",
            maturity="H3",
            status="TEST_READY",
            fingerprint=FAMILY,
            analysis_mode="ACCELERATED_RECONSTRUCTIVE_RESEARCH",
            availability_basis="SOURCE_NATIVE_AVAILABILITY_TIME",
            as_of=datetime(2026, 8, 1, tzinfo=UTC),
        )
    )


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


def _archive_replay_observations(
    session: Session, *, source_time_offset: timedelta = timedelta()
) -> None:
    artifact = RawArtifact(
        content_sha256="a" * 64,
        media_type="application/json",
        byte_size=2,
        storage_uri="data/raw/objects/aa/" + "a" * 64,
    )
    run = CollectionRun(
        source="test", status="SUCCESS", ended_at=datetime(2026, 8, 1, tzinfo=UTC)
    )
    session.add_all([artifact, run])
    session.flush()
    for observation in session.query(MarketObservation).all():
        session.add(
            RawArtifactReceipt(
                raw_artifact_id=artifact.id,
                provider="hyperliquid",
                canonical_url=None,
                source_native_timestamp=observation.observed_at
                + (
                    timedelta(hours=1)
                    if observation.observation_kind == "candle"
                    else timedelta()
                )
                + source_time_offset,
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


def test_unarchived_observations_are_not_event_or_outcome_inputs() -> None:
    session = _session()
    start = datetime(2026, 7, 1, tzinfo=UTC)
    for hour in range(32):
        _seed_hour(session, start + timedelta(hours=hour), float(hour), 100.0 + hour)
    session.commit()

    assert EventDatasetBuilder(session, _spec()).build() == ()


def test_future_source_native_receipts_are_not_replay_inputs() -> None:
    session = _session()
    start = datetime(2026, 7, 1, tzinfo=UTC)
    for hour in range(32):
        _seed_hour(session, start + timedelta(hours=hour), float(hour), 100.0 + hour)
    _archive_replay_observations(session, source_time_offset=timedelta(days=90))
    session.commit()

    assert EventDatasetBuilder(session, _spec()).build() == ()


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


def test_engine_rejects_unpersisted_hypothesis_spec(tmp_path) -> None:
    with pytest.raises(SpecIncompleteError, match="persisted TEST_READY"):
        EventStudyEngine(_session(), _spec(), client=None).run(tmp_path)


def test_engine_rejects_forged_production_mode_spec(tmp_path) -> None:
    session = _session()
    _ready_hypothesis(session)
    session.commit()
    with pytest.raises(SpecIncompleteError, match="persisted TEST_READY"):
        EventStudyEngine(
            session, replace(_spec(), analysis_mode="PRODUCTION_LIVE"), client=None
        ).run(tmp_path)


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
    _ready_hypothesis(session)
    session.commit()

    result = EventStudyEngine(session, _spec(), client=None).run(tmp_path)

    assert result["status"] != "SUPPORTED"
    assert result["critic"]["disposition"] == "NOT_RUN"
    artifact = tmp_path / result["run_id"]
    manifest = (artifact / "dataset_manifest.json").read_text(encoding="utf-8")
    assert '"source_receipt_ids"' in manifest
    repeat = EventStudyEngine(session, _spec(), client=None).run(tmp_path / "repeat")
    repeat_artifact = Path(tmp_path / "repeat" / repeat["run_id"])
    for name in (
        "spec.json",
        "dataset_manifest.json",
        "summary.json",
        "statistical_results.json",
        "robustness.json",
        "negative_controls.json",
        "critic.json",
        "executive.md",
    ):
        assert (artifact / name).read_bytes() == (repeat_artifact / name).read_bytes()
