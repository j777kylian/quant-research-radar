import warnings
from datetime import UTC, datetime, timedelta
from typing import cast

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    ChannelHypothesis,
    EvidenceLink,
    MarketObservation,
    RawArtifact,
    RawArtifactReceipt,
    normalize_utc,
)
from quant_research_radar.intelligence_v2 import (
    AvailabilityBasis,
    market_evidence,
    run_intelligence_day,
    run_intelligence_replay,
)
from quant_research_radar.llm import CriticOutput, FakeLLMClient


def _seed(session: Session, *, retrieved_at: datetime) -> None:
    valuation = datetime(2026, 8, 30, 22, tzinfo=UTC)
    for hour in range(31):
        timestamp = valuation - timedelta(hours=hour)
        session.add(
            MarketObservation(
                asset="SOL",
                observation_kind="funding",
                observed_at=timestamp,
                funding_rate=100.0 if hour == 0 else 0.0,
                source_name="hyperliquid",
                retrieved_at=retrieved_at,
            )
        )
        session.add(
            MarketObservation(
                asset="SOL",
                observation_kind="candle",
                observed_at=timestamp,
                mark_price=100 + hour,
                source_name="hyperliquid",
                retrieved_at=retrieved_at,
            )
        )
    session.commit()


def test_production_availability_rejects_later_receipt_but_replay_accepts_native_time() -> (
    None
):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))

    assert market_evidence(session, as_of, AvailabilityBasis.PRODUCTION_RECEIPT) == []
    replay = market_evidence(session, as_of, AvailabilityBasis.SOURCE_NATIVE_REPLAY)
    assert len(replay) == 1
    assert replay[0].metadata["availability_basis"] == "SOURCE_NATIVE_AVAILABILITY_TIME"
    assert normalize_utc(
        session.query(MarketObservation).first().retrieved_at
    ) == as_of + timedelta(days=1)


def test_production_recurrence_persists_one_occurrence_per_as_of(tmp_path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    first_as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=first_as_of)
    observation = session.query(MarketObservation).first()
    artifact = RawArtifact(
        content_sha256="a" * 64,
        media_type="application/json",
        byte_size=2,
        storage_uri="data/raw/objects/aa/" + "a" * 64,
    )
    session.add(artifact)
    session.flush()
    session.add(
        RawArtifactReceipt(
            raw_artifact_id=artifact.id,
            provider="hyperliquid",
            canonical_url=None,
            source_native_timestamp=observation.observed_at,
            retrieved_at=first_as_of,
            market_observation_id=observation.id,
        )
    )
    session.commit()

    second_as_of = first_as_of + timedelta(microseconds=1)
    run_intelligence_day(session, tmp_path / "day1", first_as_of)
    run_intelligence_day(session, tmp_path / "day2", second_as_of)

    rows = session.query(ChannelHypothesis).all()
    assert len(rows) == 2
    assert len({row.fingerprint for row in rows}) == 1
    assert {normalize_utc(row.as_of) for row in rows} == {first_as_of, second_as_of}
    assert all(
        link.raw_artifact_receipt_id is not None
        for link in session.query(EvidenceLink).all()
    )


def test_replay_does_not_persist_reconstructive_candidates(tmp_path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))

    run_intelligence_replay(session, tmp_path, [as_of])

    assert session.query(ChannelHypothesis).count() == 0


def test_replay_writes_nonproduction_candidate_ledger_with_traceable_evidence(
    tmp_path,
) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))

    run_intelligence_replay(session, tmp_path, [as_of])

    ledger = __import__("json").loads(
        (tmp_path / "replay-candidate-ledger.json").read_text()
    )
    candidate = ledger["candidates"][0]
    assert candidate["analysis_mode"] == "ACCELERATED_RECONSTRUCTIVE_REPLAY"
    assert candidate["availability_basis"] == "SOURCE_NATIVE_AVAILABILITY_TIME"
    assert candidate["origin_channel"] == "MARKET"
    assert candidate["evidence_ids"] == [
        "v2-market:SOL:2026-08-30T22:00:00+00:00:funding-extreme"
    ]
    assert candidate["critic"] == {
        "disposition": "NOT_RUN",
        "reason": "no replay critic client",
    }
    assert session.query(ChannelHypothesis).count() == 0


def test_replay_candidate_is_critic_reviewed_and_tutored_without_production_persistence(
    tmp_path,
) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))

    run_intelligence_replay(session, tmp_path, [as_of], client=FakeLLMClient())

    ledger = __import__("json").loads(
        (tmp_path / "replay-candidate-ledger.json").read_text()
    )
    candidate = ledger["candidates"][0]
    assert candidate["critic"]["disposition"] == "ACCEPT"
    assert candidate["tutor"]["non_evidentiary"] is True
    assert (tmp_path / "tutor.json").exists()
    assert session.query(ChannelHypothesis).count() == 0


class RecordingClient(FakeLLMClient):
    def __init__(self) -> None:
        self.critic_input = ""

    def critique(self, hypothesis: str):
        self.critic_input = hypothesis
        return super().critique(hypothesis)


def test_replay_critic_receives_structured_candidate_context(tmp_path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))
    client = RecordingClient()

    run_intelligence_replay(session, tmp_path, [as_of], client=client)

    context = __import__("json").loads(client.critic_input)["candidate"]
    assert context["origin_channel"] == "MARKET"
    assert context["evidence_ids"]
    assert context["evidence_provenance_ids"]
    assert context["evidence_independence_keys"]
    assert context["evidence_observed_at"]
    assert context["evidence_metadata"]
    assert context["recurrence_status"] == "NEW_CANDIDATE"
    assert context["condition"] and context["outcome"]
    assert context["universe"] and context["horizon"]
    assert context["required_data"] and context["falsification_criterion"]
    assert context["analysis_mode"] == "ACCELERATED_RECONSTRUCTIVE_REPLAY"


class MalformedClient(FakeLLMClient):
    def critique(self, hypothesis: str) -> CriticOutput:
        return cast(CriticOutput, {})


def test_malformed_replay_critic_output_fails_closed(tmp_path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))

    run_intelligence_replay(session, tmp_path, [as_of], client=MalformedClient())

    candidate = __import__("json").loads(
        (tmp_path / "replay-candidate-ledger.json").read_text()
    )["candidates"][0]
    assert candidate["critic"]["disposition"] == "REQUEST_DATA"
    assert candidate["tutor"] is None


class InvalidConstructedClient(FakeLLMClient):
    def critique(self, hypothesis: str) -> CriticOutput:
        return CriticOutput.model_construct(
            biases="not-a-list",
            confounders=None,
            failure_reasons=[],
            provenance_sufficient=True,
        )


def test_invalid_constructed_critic_output_fails_closed(tmp_path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run_intelligence_replay(
            session, tmp_path, [as_of], client=InvalidConstructedClient()
        )

    candidate = __import__("json").loads(
        (tmp_path / "replay-candidate-ledger.json").read_text()
    )["candidates"][0]
    assert candidate["critic"]["disposition"] == "REQUEST_DATA"
    assert candidate["tutor"] is None


class ExplodingClient(FakeLLMClient):
    def critique(self, hypothesis: str) -> CriticOutput:
        raise RuntimeError("provider transport exploded")


def test_replay_critic_transport_failure_fails_closed(tmp_path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))

    run_intelligence_replay(session, tmp_path, [as_of], client=ExplodingClient())

    candidate = __import__("json").loads(
        (tmp_path / "replay-candidate-ledger.json").read_text()
    )["candidates"][0]
    assert candidate["critic"] == {
        "disposition": "REQUEST_DATA",
        "reason": "critic structured output failed",
    }
    assert candidate["tutor"] is None


def test_replay_critics_report_noop_without_client(tmp_path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))

    audit = run_intelligence_replay(session, tmp_path, [as_of])["daily_audits"][0]

    assert audit["critics"]["evidence_auditor"]["disposition"] == "NOT_RUN"
    assert audit["tutor"]["concepts"] == 0


def test_replay_excludes_future_source_events_even_when_real_receipt_is_later() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))
    session.add(
        MarketObservation(
            asset="SOL",
            observation_kind="funding",
            observed_at=as_of + timedelta(hours=1),
            funding_rate=999.0,
            source_name="hyperliquid",
            retrieved_at=as_of + timedelta(days=1),
        )
    )
    session.commit()

    replay = market_evidence(session, as_of, AvailabilityBasis.SOURCE_NATIVE_REPLAY)
    assert "999.0" not in replay[0].body
