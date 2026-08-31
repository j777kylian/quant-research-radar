from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import Base, MarketObservation, SourceItem
from quant_research_radar.fast import run_fast_day


def db() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def add_market_history(session: Session, cutoff: datetime) -> None:
    valuation = cutoff.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    for asset in ("BTC", "ETH", "SOL"):
        for offset in range(31):
            observed_at = valuation - timedelta(hours=offset)
            session.add(
                MarketObservation(
                    asset=asset,
                    observed_at=observed_at,
                    observation_kind="funding",
                    funding_rate=1.0 if asset == "BTC" and offset == 0 else 0.0,
                    source_name="hyperliquid",
                    retrieved_at=cutoff + timedelta(days=30),
                )
            )
            session.add(
                MarketObservation(
                    asset=asset,
                    observed_at=observed_at,
                    observation_kind="candle",
                    mark_price=102.0 if asset == "BTC" and offset == 0 else 100.0,
                    source_name="hyperliquid",
                    retrieved_at=cutoff + timedelta(days=30),
                )
            )
    session.commit()


def test_fast_day_uses_native_availability_without_rewriting_receipts(
    tmp_path: Path,
) -> None:
    session = db()
    cutoff = datetime(2026, 8, 24, 23, 59, 59, 999999, tzinfo=UTC)
    add_market_history(session, cutoff)
    original_receipts = [
        row.retrieved_at for row in session.query(MarketObservation).all()
    ]

    audit = run_fast_day(session, tmp_path, cutoff.date())

    report = (tmp_path / "day-1-2026-08-24" / "daily.md").read_text()
    assert audit["mode"] == "ACCELERATED_RECONSTRUCTIVE_REPLAY"
    assert audit["pit_basis"] == "SOURCE_NATIVE_AVAILABILITY_TIME"
    assert audit["real_receipt_pit"] == "NOT_CLAIMED"
    assert audit["market_facts"] == 1
    assert "ACCELERATED RECONSTRUCTIVE REPLAY" in report
    assert "AS_OF=2026-08-24T23:59:59.999999+00:00" in report
    assert "BTC funding" in report
    assert [
        row.retrieved_at for row in session.query(MarketObservation).all()
    ] == original_receipts


def test_fast_day_rejects_partial_current_candle_and_off_topic_academia(
    tmp_path: Path,
) -> None:
    session = db()
    cutoff = datetime(2026, 8, 24, 23, 59, 59, 999999, tzinfo=UTC)
    add_market_history(session, cutoff)
    session.add(
        SourceItem(
            source_type="ACADEMIC",
            source_name="arxiv",
            external_id="quantum",
            canonical_url=None,
            title="Quantum funding dynamics",
            authors=[],
            published_at=cutoff - timedelta(hours=1),
            retrieved_at=cutoff + timedelta(days=1),
            raw_text="Quantum funding is not a perpetual-futures market study.",
            raw_metadata={"categories": ["quant-ph"]},
            content_sha256="a" * 64,
        )
    )
    session.add(
        MarketObservation(
            asset="BTC",
            observed_at=datetime(2026, 8, 24, 23, tzinfo=UTC),
            observation_kind="candle",
            mark_price=1000.0,
            source_name="hyperliquid",
            retrieved_at=cutoff + timedelta(days=30),
        )
    )
    session.commit()

    audit = run_fast_day(session, tmp_path, cutoff.date())

    report = (tmp_path / "day-1-2026-08-24" / "daily.md").read_text()
    assert audit["academic_items_retained"] == 0
    assert "Quantum funding" not in report
    assert "2026-08-24T23:00:00" not in report


def test_fast_day_marks_missing_support_unavailable(tmp_path: Path) -> None:
    session = db()
    audit = run_fast_day(session, tmp_path, datetime(2026, 8, 24, tzinfo=UTC).date())

    report = (tmp_path / "day-1-2026-08-24" / "daily.md").read_text()
    assert audit["market_data_status"] == "INSUFFICIENT_HISTORY"
    assert (
        "**UNAVAILABLE:** required source-native metric support is incomplete" in report
    )
    assert "No deterministic market FACT met" not in report


def test_fast_day_suppresses_prior_hypothesis_and_concept(tmp_path: Path) -> None:
    from quant_research_radar.llm import FakeLLMClient

    session = db()
    cutoff = datetime(2026, 8, 24, 23, 59, 59, tzinfo=UTC)
    add_market_history(session, cutoff)
    session.add(
        SourceItem(
            source_type="ACADEMIC",
            source_name="arxiv",
            external_id="funding-study",
            canonical_url=None,
            title="Funding-rate persistence in perpetual futures",
            authors=[],
            published_at=cutoff - timedelta(hours=1),
            retrieved_at=cutoff + timedelta(days=1),
            raw_text="Extreme perpetual funding rates predict subsequent crypto returns.",
            raw_metadata={"categories": ["q-fin.ST"]},
            content_sha256="b" * 64,
        )
    )
    session.commit()
    seen_hypotheses: set[str] = set()
    seen_concepts: set[str] = set()

    first = run_fast_day(
        session,
        tmp_path,
        cutoff.date(),
        client=FakeLLMClient(),
        seen_hypothesis_families=seen_hypotheses,
        seen_concepts=seen_concepts,
    )
    seen_hypotheses.add(first["hypotheses"][0]["statement"].lower())
    seen_concepts.add(first["concepts"][0]["name"])
    second = run_fast_day(
        session,
        tmp_path,
        (cutoff + timedelta(days=1)).date(),
        ordinal=2,
        client=FakeLLMClient(),
        seen_hypothesis_families=seen_hypotheses,
        seen_concepts=seen_concepts,
    )

    assert len(first["hypotheses"]) == len(first["concepts"]) == 1
    assert second["hypotheses"] == second["concepts"] == []
    assert len(second["repeated_hypotheses_from_prior_days"]) == 1
    assert len(second["repeated_concepts_from_prior_days"]) == 1
