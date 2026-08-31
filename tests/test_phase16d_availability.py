from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    ChannelHypothesis,
    MarketObservation,
    normalize_utc,
)
from quant_research_radar.intelligence_v2 import (
    AvailabilityBasis,
    market_evidence,
    run_intelligence_replay,
)


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


def test_replay_does_not_persist_reconstructive_candidates(tmp_path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 30, 23, 59, 59, tzinfo=UTC)
    _seed(session, retrieved_at=as_of + timedelta(days=1))

    run_intelligence_replay(session, tmp_path, [as_of])

    assert session.query(ChannelHypothesis).count() == 0


def test_replay_critics_report_noop_instead_of_fabricated_pass(tmp_path) -> None:
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
