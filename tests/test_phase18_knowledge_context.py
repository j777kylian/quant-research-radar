from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import Base, ChannelHypothesis
from quant_research_radar.knowledge_context import Novelty, prior_research_context


def _row(
    session: Session, *, universe: str, at: datetime, mode: str = "PRODUCTION_LIVE"
) -> None:
    session.add(
        ChannelHypothesis(
            channel="MARKET",
            statement=f"{universe} funding predicts returns",
            condition="extreme positive funding",
            outcome="subsequent return",
            universe=universe,
            horizon="24h",
            expected_direction="negative",
            required_data=["funding", "return"],
            falsification_criterion="no difference",
            maturity="H1_STATISTICAL_HYPOTHESIS",
            fingerprint="funding-reversal",
            analysis_mode=mode,
            availability_basis="RECEIPT_TIME",
            as_of=at,
        )
    )


def test_prior_context_classifies_recurrence_and_asset_variants_without_replay_leakage() -> (
    None
):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    at = datetime(2026, 8, 30, tzinfo=UTC)
    _row(session, universe="SOL", at=at)
    _row(session, universe="BTC", at=at, mode="ACCELERATED_RECONSTRUCTIVE_REPLAY")
    session.commit()

    recurrent = prior_research_context(
        session, "funding-reversal", "SOL", at + timedelta(days=1), "MARKET"
    )
    variant = prior_research_context(
        session, "funding-reversal", "ETH", at + timedelta(days=1), "MARKET"
    )

    assert recurrent.novelty is Novelty.PERSISTENT_REGIME
    assert recurrent.occurrence_count == 1
    assert variant.novelty is Novelty.ASSET_VARIANT
    assert all(row.analysis_mode == "PRODUCTION_LIVE" for row in recurrent.related)


def test_prior_context_hides_future_history() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    at = datetime(2026, 8, 30, tzinfo=UTC)
    _row(session, universe="SOL", at=at + timedelta(days=1))
    session.commit()

    context = prior_research_context(session, "funding-reversal", "SOL", at, "MARKET")

    assert context.novelty is Novelty.NEW
    assert context.related == ()
