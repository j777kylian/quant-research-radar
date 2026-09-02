from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    CollectionRun,
    MarketObservation,
    RawArtifact,
    RawArtifactReceipt,
)
from quant_research_radar.raw_archive import RawArchive
from quant_research_radar.sources import HyperliquidSource


def _seed_funding(
    session: Session, *, asset: str, rates: list[tuple[datetime, float]]
) -> None:
    """Persist reconstructive, receipt-bound funding observations for coverage_audit."""
    artifact = RawArtifact(
        content_sha256="b" * 64,
        media_type="application/json",
        byte_size=2,
        storage_uri="data/raw/objects/bb/" + "b" * 64,
    )
    run = CollectionRun(
        source="hyperliquid",
        status="SUCCESS",
        ended_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    session.add_all([artifact, run])
    session.flush()
    for at, rate in rates:
        observation = MarketObservation(
            asset=asset,
            observed_at=at,
            observation_kind="funding",
            funding_rate=rate,
            source_name="hyperliquid",
            retrieved_at=at,
        )
        session.add(observation)
        session.flush()
        session.add(
            RawArtifactReceipt(
                raw_artifact_id=artifact.id,
                provider="hyperliquid",
                canonical_url=None,
                source_native_timestamp=at,
                retrieved_at=at,
                market_observation_id=observation.id,
                collection_run_id=run.id,
                analysis_mode="ACCELERATED_RECONSTRUCTIVE_RESEARCH",
            )
        )
    session.commit()


def test_reconstructive_backfill_creates_completed_receipt_bound_run(tmp_path) -> None:
    from quant_research_radar.market_operations import run_historical_backfill

    class FixtureHyperliquid:
        name = "hyperliquid"
        assets = ("BTC", "ETH", "SOL")
        last_funding_diagnostics = {}

        def collect_history(self, _limit, *, start, end):
            return [
                HyperliquidSource._history_record(
                    asset, start + timedelta(hours=index), index
                )
                for asset in self.assets
                for index in range(int((end - start).total_seconds() // 3600) + 1)
            ]

        def collect_candles(self, _limit, *, start, end):
            return [
                HyperliquidSource._candle_record(
                    asset, start + timedelta(hours=index), index
                )
                for asset in self.assets
                for index in range(int((end - start).total_seconds() // 3600) + 1)
            ]

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    result = run_historical_backfill(
        session,
        FixtureHyperliquid(),
        RawArchive(tmp_path / "raw"),
        start=start,
        end=start + timedelta(hours=2),
        code_sha="test-sha",
    )

    run = session.get(CollectionRun, UUID(result["run_id"]))
    receipts = session.scalars(select(RawArtifactReceipt)).all()
    assert run is not None and run.status == "SUCCESS" and run.ended_at is not None
    assert len(receipts) == 15
    assert {receipt.analysis_mode for receipt in receipts} == {
        "ACCELERATED_RECONSTRUCTIVE_RESEARCH"
    }
    assert {receipt.collection_run_id for receipt in receipts} == {run.id}


def test_overlap_rerun_skips_completed_window(
    tmp_path,
) -> None:
    from quant_research_radar.market_operations import run_historical_backfill

    class OneHour:
        name = "hyperliquid"
        assets = ("BTC", "ETH", "SOL")
        last_funding_diagnostics = {}

        def collect_history(self, _limit, *, start, end):
            return [
                HyperliquidSource._history_record(asset, start, 0)
                for asset in self.assets
            ]

        def collect_candles(self, _limit, *, start, end):
            return [
                HyperliquidSource._candle_record(asset, start, 0)
                for asset in self.assets
            ]

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    archive = RawArchive(tmp_path / "raw")
    run_historical_backfill(
        session,
        OneHour(),
        archive,
        start=start,
        end=start + timedelta(minutes=1),
        code_sha="one",
    )
    second = run_historical_backfill(
        session,
        OneHour(),
        archive,
        start=start,
        end=start + timedelta(minutes=1),
        code_sha="two",
    )
    assert session.scalars(select(RawArtifactReceipt)).all()
    assert session.query(RawArtifactReceipt).count() == 3
    assert second["resumed_windows"] == 1


def test_funding_gap_detection_ignores_millisecond_jitter() -> None:
    from quant_research_radar.market_operations import coverage_audit

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    start = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    # Contiguous hourly funding with sub-second jitter around the hour boundary.
    jittered = [
        (start + timedelta(hours=index, milliseconds=(137 * index) % 1000), 0.00001)
        for index in range(48)
    ]
    _seed_funding(session, asset="BTC", rates=jittered)

    audit = coverage_audit(session, start=start, end=start + timedelta(hours=48))
    assert audit["assets"]["BTC"]["funding"]["gaps"] == 0

    # A genuine 2-hour hole must still surface as one gap.
    engine2 = create_engine("sqlite://")
    Base.metadata.create_all(engine2)
    session2 = Session(engine2)
    with_hole = [
        (start + timedelta(hours=index), 0.00001)
        for index in range(24)
        if index < 12 or index >= 14  # skip hour 12 and 13 -> one 2h gap
    ]
    _seed_funding(session2, asset="BTC", rates=with_hole)
    audit2 = coverage_audit(session2, start=start, end=start + timedelta(hours=24))
    assert audit2["assets"]["BTC"]["funding"]["gaps"] == 1


def test_extreme_funding_excludes_dominant_default_rate() -> None:
    from quant_research_radar.market_operations import coverage_audit

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    start = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    ceiling = 1.25e-05
    # 30 hours at the dominant default/ceiling rate, then one genuinely higher hour.
    rates = [(start + timedelta(hours=index), ceiling) for index in range(30)] + [
        (start + timedelta(hours=30), 2.5e-05)
    ]

    _seed_funding(session, asset="BTC", rates=rates)
    audit = coverage_audit(session, start=start, end=start + timedelta(hours=31))

    # The tied ceiling default is ordinary; only the single elevated hour is extreme.
    assert audit["assets"]["BTC"]["extreme_funding_observation_count"] == 1
    assert audit["assets"]["BTC"]["independent_extreme_funding_regime_count"] == 1
