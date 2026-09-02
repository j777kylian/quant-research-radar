from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import Base, CollectionRun, RawArtifactReceipt
from quant_research_radar.raw_archive import RawArchive
from quant_research_radar.sources import HyperliquidSource


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
    assert len(receipts) == 18
    assert {receipt.analysis_mode for receipt in receipts} == {
        "ACCELERATED_RECONSTRUCTIVE_RESEARCH"
    }
    assert {receipt.collection_run_id for receipt in receipts} == {run.id}


def test_overlap_rerun_reuses_observations_and_adds_reconstructive_receipts(
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
    run_historical_backfill(
        session,
        OneHour(),
        archive,
        start=start,
        end=start + timedelta(minutes=1),
        code_sha="two",
    )
    assert session.scalars(select(RawArtifactReceipt)).all()
    assert session.query(RawArtifactReceipt).count() == 12
