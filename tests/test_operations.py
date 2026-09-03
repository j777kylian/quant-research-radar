from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import Base, RawArtifactReceipt
from quant_research_radar.market_operations import run_live_market_collection
from quant_research_radar.operations import OperationsLock
from quant_research_radar.raw_archive import RawArchive
from quant_research_radar.sources import HyperliquidSource
from quant_research_radar.user_fit import (
    FIT_HIGH,
    FIT_LOW,
    FIT_MEDIUM,
    FIT_OUT_OF_SCOPE,
    low_frequency_fit,
)


def test_low_frequency_fit_buckets() -> None:
    assert low_frequency_fit("5 min") == FIT_OUT_OF_SCOPE
    assert low_frequency_fit("1h") == FIT_OUT_OF_SCOPE
    assert low_frequency_fit("3h") == FIT_LOW
    assert low_frequency_fit("12h") == FIT_MEDIUM
    assert low_frequency_fit("24h") == FIT_HIGH
    assert low_frequency_fit("1d") == FIT_HIGH
    assert low_frequency_fit("7d") == FIT_HIGH
    assert low_frequency_fit("30d") == FIT_HIGH
    assert low_frequency_fit(None) == FIT_OUT_OF_SCOPE


def test_operations_lock_acquire_release(tmp_path: Path) -> None:
    lock = OperationsLock(tmp_path)
    assert lock.acquire() is True
    # A second lock on the same root must not acquire.
    other = OperationsLock(tmp_path)
    assert other.acquire() is False
    lock.release()
    assert other.acquire() is True
    other.release()


def test_operations_lock_stale_recovery(tmp_path: Path) -> None:
    lock = OperationsLock(tmp_path)
    lock.dir.mkdir(parents=True, exist_ok=True)
    # Simulate a dead owner PID (999999 does not exist).
    lock.owner.write_text("999999 2026-01-01T00:00:00+00:00")
    fresh = OperationsLock(tmp_path)
    assert fresh.acquire() is True  # stale lock recovered
    fresh.release()


def test_market_daily_catchup_supports_48h_window(tmp_path: Path) -> None:
    class FixtureHyperliquid:
        name = "hyperliquid"
        assets = ("BTC", "ETH", "SOL")
        last_funding_diagnostics = {}

        def collect_history(self, _limit, *, start, end):
            return [
                HyperliquidSource._history_record(asset, start + timedelta(hours=i), i)
                for asset in self.assets
                for i in range(int((end - start).total_seconds() // 3600) + 1)
            ]

        def collect_candles(self, _limit, *, start, end):
            return [
                HyperliquidSource._candle_record(asset, start + timedelta(hours=i), i)
                for asset in self.assets
                for i in range(int((end - start).total_seconds() // 3600) + 1)
            ]

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=48)

    result = run_live_market_collection(
        session,
        FixtureHyperliquid(),
        RawArchive(tmp_path / "raw"),
        start=start,
        end=end,
        code_sha="test-sha",
    )

    assert result["analysis_mode"] == "PRODUCTION_LIVE"
    receipts = session.scalars(select(RawArtifactReceipt)).all()
    assert len(receipts) > 0
    assert {receipt.analysis_mode for receipt in receipts} == {"PRODUCTION_LIVE"}
