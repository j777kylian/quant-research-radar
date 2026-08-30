import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from quant_research_radar import live, pipeline
from quant_research_radar.db import Base, MarketObservation
from quant_research_radar.live import (
    _latest_persisted_completed_candle,
    _market_gate,
    run_live_cycle,
)
from quant_research_radar.llm import DeepSeekClient, FakeLLMClient
from quant_research_radar.pipeline import calculate_metrics, ingest_records
from quant_research_radar.replay import valuation_timestamp
from quant_research_radar.sources import HyperliquidSource


def test_market_gate_rejects_a_missing_completed_candle_inside_warmup() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 27, 10, 12, 18, tzinfo=UTC)
    valuation = datetime(2026, 8, 27, 9, tzinfo=UTC)
    start = valuation - timedelta(days=33)
    records = []
    for asset in HyperliquidSource.assets:
        records.extend(
            HyperliquidSource._history_record(
                asset, start + timedelta(hours=index), index
            )
            for index in range(33 * 24 + 1)
        )
        records.extend(
            HyperliquidSource._candle_record(
                asset, start + timedelta(hours=index), index
            )
            for index in range(33 * 24 + 1)
            if index != 2
        )
    ingest_records(session, records)
    calculate_metrics(session, as_of=as_of)

    gate, blockers = _market_gate(session, as_of)

    assert gate == "BLOCKED"
    assert any("candle coverage gap" in blocker for blocker in blockers)


def test_market_gate_requires_the_exact_completed_valuation_candle() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 27, 10, 12, 18, tzinfo=UTC)
    valuation = datetime(2026, 8, 27, 9, tzinfo=UTC)
    start = valuation - timedelta(days=33)
    records = []
    for asset in HyperliquidSource.assets:
        records.extend(
            HyperliquidSource._history_record(
                asset, start + timedelta(hours=index), index
            )
            for index in range(33 * 24 + 1)
        )
        records.extend(
            HyperliquidSource._candle_record(
                asset, start + timedelta(hours=index), index
            )
            for index in range(33 * 24)
        )
    ingest_records(session, records)
    calculate_metrics(session, as_of=as_of)

    gate, blockers = _market_gate(session, as_of)

    assert gate == "BLOCKED"
    assert any("candle coverage gap" in blocker for blocker in blockers)


def test_market_gate_blocks_when_exact_four_hour_anchor_is_missing() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    as_of = datetime(2026, 8, 27, 10, 12, 18, tzinfo=UTC)
    valuation = datetime(2026, 8, 27, 9, tzinfo=UTC)
    start = valuation - timedelta(days=33)
    records = []
    for asset in HyperliquidSource.assets:
        records.extend(
            HyperliquidSource._history_record(
                asset, start + timedelta(hours=index), index
            )
            for index in range(33 * 24 + 1)
        )
        records.extend(
            HyperliquidSource._candle_record(
                asset, start + timedelta(hours=index), index
            )
            for index in range(33 * 24 + 1)
            if index != 33 * 24 - 4
        )
    ingest_records(session, records)
    calculate_metrics(session, as_of=as_of)

    gate, blockers = _market_gate(session, as_of)

    assert gate == "BLOCKED"
    assert any("return_4h unavailable" in blocker for blocker in blockers)


def test_candle_collection_rejects_provider_rows_outside_completed_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = HyperliquidSource()
    start = datetime(2026, 8, 28, 22, tzinfo=UTC)
    partial = start + timedelta(hours=1)
    monkeypatch.setattr(
        source,
        "_post",
        lambda _payload: [
            {"t": int(partial.timestamp() * 1000), "c": "101"},
            {"t": int(start.timestamp() * 1000), "c": "100"},
        ],
    )

    candles = source.collect_candles(2, start=start, end=start)

    assert len(candles) == 3
    assert {c.published_at for c in candles} == {start}


def test_completed_candle_boundaries_are_strictly_point_in_time_safe() -> None:
    expected = datetime(2026, 8, 27, 9, tzinfo=UTC)
    assert (
        valuation_timestamp(datetime(2026, 8, 27, 10, 12, 18, tzinfo=UTC)) == expected
    )
    assert (
        valuation_timestamp(datetime(2026, 8, 27, 10, 59, 59, tzinfo=UTC)) == expected
    )
    assert valuation_timestamp(datetime(2026, 8, 27, 11, tzinfo=UTC)) == datetime(
        2026, 8, 27, 10, tzinfo=UTC
    )
    assert valuation_timestamp(
        datetime(2026, 8, 27, 11, 0, 0, 1, tzinfo=UTC)
    ) == datetime(2026, 8, 27, 10, tzinfo=UTC)
    assert valuation_timestamp(datetime(2026, 8, 27, tzinfo=UTC)) == datetime(
        2026, 8, 26, 23, tzinfo=UTC
    )


def test_blocked_live_cycle_writes_audit_without_using_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyHyperliquid:
        def collect_history(self, *_args, **_kwargs):
            return []

        def collect_candles(self, *_args, **_kwargs):
            return []

        last_funding_diagnostics = {}

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(live, "HyperliquidSource", EmptyHyperliquid)

    with pytest.raises(RuntimeError, match="LIVE_CYCLE_STATUS=BLOCKED"):
        run_live_cycle(
            Session(engine), cast(DeepSeekClient, object()), tmp_path, 1, "test-sha"
        )

    audit = json.loads((tmp_path / "cycle-1" / "audit.json").read_text())
    assert audit["cycle_technical_status"] == "BLOCKED"
    assert audit["deepseek_call_status"] == "NOT_CALLED_DUE_TO_GATE"
    assert audit["hypotheses_generated"] == 0


def test_cycle_two_reuses_the_live_database_without_duplicate_market_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DeterministicHyperliquid:
        last_funding_diagnostics = {}

        def collect_history(self, _limit, *, start, end, **_kwargs):
            return [
                HyperliquidSource._history_record(
                    asset, start + timedelta(hours=index), index
                )
                for asset in HyperliquidSource.assets
                for index in range(int((end - start).total_seconds() // 3600) + 1)
            ]

        def collect_candles(self, _limit, *, start, end, **_kwargs):
            return [
                HyperliquidSource._candle_record(
                    asset, start + timedelta(hours=index), index
                )
                for asset in HyperliquidSource.assets
                for index in range(int((end - start).total_seconds() // 3600) + 1)
            ]

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(live, "HyperliquidSource", DeterministicHyperliquid)
    first = run_live_cycle(
        Session(engine), cast(DeepSeekClient, FakeLLMClient()), tmp_path, 1, "test-sha"
    )
    rows_after_first = Session(engine).scalar(select(func.count(MarketObservation.id)))
    second = run_live_cycle(
        Session(engine), cast(DeepSeekClient, FakeLLMClient()), tmp_path, 2, "test-sha"
    )

    assert first["cycle_technical_status"] == "PASS"
    assert second["cycle_technical_status"] == "PASS"
    assert rows_after_first == Session(engine).scalar(
        select(func.count(MarketObservation.id))
    )
    assert second["counts"]["hyperliquid"] == 0


def test_cycle_two_collects_the_entire_gap_from_persisted_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingHyperliquid:
        last_funding_diagnostics = {}
        windows: list[tuple[datetime, datetime]] = []

        def collect_history(self, _limit, *, start, end, **_kwargs):
            self.windows.append((start, end))
            return [
                HyperliquidSource._history_record(
                    asset, start + timedelta(hours=index), index
                )
                for asset in HyperliquidSource.assets
                for index in range(int((end - start).total_seconds() // 3600) + 1)
            ]

        def collect_candles(self, _limit, *, start, end, **_kwargs):
            return [
                HyperliquidSource._candle_record(
                    asset, start + timedelta(hours=index), index
                )
                for asset in HyperliquidSource.assets
                for index in range(int((end - start).total_seconds() // 3600) + 1)
            ]

    first_now = datetime(2026, 8, 27, 23, 52, tzinfo=UTC)
    second_now = datetime(2026, 8, 28, 23, 56, tzinfo=UTC)

    class FrozenDatetime(datetime):
        values = [first_now, first_now, second_now, second_now]

        @classmethod
        def now(cls, tz=None):
            return cls.values.pop(0)

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(live, "HyperliquidSource", RecordingHyperliquid)
    monkeypatch.setattr(live, "datetime", FrozenDatetime)
    monkeypatch.setattr(pipeline, "utcnow", lambda: first_now)

    run_live_cycle(
        Session(engine), cast(DeepSeekClient, FakeLLMClient()), tmp_path, 1, "test-sha"
    )
    second = run_live_cycle(
        Session(engine), cast(DeepSeekClient, FakeLLMClient()), tmp_path, 2, "test-sha"
    )

    assert RecordingHyperliquid.windows[1] == (
        datetime(2026, 8, 27, 20, tzinfo=UTC),
        datetime(2026, 8, 28, 22, tzinfo=UTC),
    )
    assert second["cycle_technical_status"] == "PASS"
    assert second["counts"]["hyperliquid_candles"] == 72
    assert all(
        quality["missing_expected_1h_intervals"] == 0
        for quality in second["market_quality"].values()
    )


def test_stale_incremental_history_blocks_without_calling_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MustNotCollect:
        last_funding_diagnostics = {}

        def collect_history(self, *_args, **_kwargs):
            raise AssertionError("provider must not be called")

        def collect_candles(self, *_args, **_kwargs):
            raise AssertionError("provider must not be called")

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 30, 23, tzinfo=UTC)

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    stale = datetime(2026, 8, 27, 22, tzinfo=UTC)
    ingest_records(
        session,
        [
            HyperliquidSource._candle_record(asset, stale, 0)
            for asset in HyperliquidSource.assets
        ],
    )
    monkeypatch.setattr(live, "HyperliquidSource", MustNotCollect)
    monkeypatch.setattr(live, "datetime", FrozenDatetime)

    with pytest.raises(RuntimeError, match="LIVE_INCREMENTAL_HISTORY_TOO_STALE"):
        run_live_cycle(session, cast(DeepSeekClient, object()), tmp_path, 2, "test-sha")

    audit = json.loads((tmp_path / "cycle-2" / "audit.json").read_text())
    assert audit["cycle_technical_status"] == "BLOCKED"
    assert audit["source_status"]["hyperliquid_transport"] == "NOT_CALLED"
    assert audit["deepseek_call_status"] == "NOT_CALLED_DUE_TO_GATE"


def test_partial_persisted_asset_coverage_does_not_trigger_full_backfill() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    timestamp = datetime(2026, 8, 28, 22, tzinfo=UTC)
    ingest_records(
        session,
        [
            HyperliquidSource._candle_record(asset, timestamp, 0)
            for asset in ("BTC", "ETH")
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="LIVE_INCREMENTAL_COVERAGE_INCOMPLETE: missing candles for SOL",
    ):
        _latest_persisted_completed_candle(session)
