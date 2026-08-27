from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine

from quant_research_radar.db import Base
from quant_research_radar.llm import FakeLLMClient
from quant_research_radar.pipeline import ingest_records
from quant_research_radar.replay import (
    filter_records_as_of,
    funding_coverage,
    run_replay_day,
    utc_day_cutoff,
)
from quant_research_radar.sources import HyperliquidSource, SourceRecord


def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import Session

    return Session(engine)


def test_replay_future_exclusion_for_market_and_papers() -> None:
    cutoff = datetime(2026, 8, 25, 23, 59, tzinfo=UTC)
    records = [
        SourceRecord(
            "MARKET",
            "hyperliquid",
            "before",
            "before",
            None,
            [],
            cutoff,
            "",
            {"asset": "BTC", "kind": "funding", "funding_rate": 0.1},
        ),
        SourceRecord(
            "MARKET",
            "hyperliquid",
            "after",
            "after",
            None,
            [],
            cutoff + timedelta(seconds=1),
            "",
            {"asset": "BTC", "kind": "funding", "funding_rate": 0.2},
        ),
        SourceRecord(
            "ACADEMIC",
            "arxiv",
            "paper",
            "paper",
            None,
            [],
            cutoff + timedelta(days=1),
            "",
            {},
        ),
    ]
    eligible = filter_records_as_of(records, cutoff)
    assert [record.external_id for record in eligible] == ["before"]


def test_replay_output_identity_and_fake_reproducibility(tmp_path: Path) -> None:
    db = session()
    cutoff = utc_day_cutoff(date(2026, 8, 25))
    records = [
        SourceRecord(
            "ACADEMIC",
            "arxiv",
            "paper",
            "paper",
            None,
            [],
            cutoff - timedelta(hours=1),
            "Evidence",
            {},
        ),
        SourceRecord(
            "MARKET",
            "hyperliquid",
            "funding",
            "BTC funding",
            None,
            [],
            cutoff - timedelta(hours=1),
            "",
            {"asset": "BTC", "kind": "funding", "funding_rate": 0.1},
        ),
    ]
    ingest_records(db, records)
    first = run_replay_day(
        db,
        FakeLLMClient(),
        tmp_path,
        date(2026, 8, 25),
        cutoff - timedelta(days=30),
        "fixture",
    )
    second = run_replay_day(
        db,
        FakeLLMClient(),
        tmp_path,
        date(2026, 8, 24),
        cutoff - timedelta(days=30),
        "fixture",
    )
    assert Path(first["reports"][0]).exists()
    assert Path(second["reports"][0]).exists()
    assert (tmp_path / "2026-08-25" / "audit.json").exists()
    assert (tmp_path / "2026-08-24" / "audit.json").exists()
    assert "AS_OF=2026-08-25" in (tmp_path / "2026-08-25" / "daily.md").read_text()


def test_hyperliquid_history_propagates_bounded_window_and_excludes_future() -> None:
    class Response:
        def __init__(self, rows: list[dict[str, str | int]]) -> None:
            self.rows = rows

        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, str | int]]:
            return self.rows

    class Client:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def post(self, _endpoint: str, json: dict) -> Response:
            self.requests.append(json)
            start = datetime.fromtimestamp(json["startTime"] / 1000, UTC)
            end = datetime.fromtimestamp(json["endTime"] / 1000, UTC)
            asset = json["coin"]
            return Response(
                [
                    {
                        "coin": asset,
                        "time": int((start + timedelta(hours=1)).timestamp() * 1000),
                        "fundingRate": "0.001",
                    },
                    {
                        "coin": asset,
                        "time": int((end + timedelta(hours=1)).timestamp() * 1000),
                        "fundingRate": "0.002",
                    },
                ]
            )

    start = datetime(2026, 7, 25, tzinfo=UTC)
    end = datetime(2026, 8, 27, tzinfo=UTC)
    client = Client()
    records = HyperliquidSource(client=client).collect_history(1, start=start, end=end)
    assert [request["coin"] for request in client.requests] == ["BTC", "ETH", "SOL"]
    assert all(
        request["startTime"] == int(start.timestamp() * 1000)
        for request in client.requests
    )
    assert all(
        request["endTime"] == int(end.timestamp() * 1000) for request in client.requests
    )
    assert len(records) == 3
    assert all(record.published_at <= end for record in records)


def test_hyperliquid_history_limit_is_per_asset_and_not_30_day_coverage() -> None:
    source = HyperliquidSource()
    end = datetime(2026, 8, 27, tzinfo=UTC)
    funding = source.collect_history(800, offline=True, end=end)
    candles = source.collect_candles(800, offline=True, end=end)
    assert {record.raw_metadata["asset"] for record in funding} == {"BTC", "ETH", "SOL"}
    assert {record.raw_metadata["asset"] for record in candles} == {"BTC", "ETH", "SOL"}
    assert all(
        sum(record.raw_metadata["asset"] == asset for record in funding) == 6
        for asset in source.assets
    )
    assert all(
        sum(record.raw_metadata["asset"] == asset for record in candles) == 30
        for asset in source.assets
    )
    assert max(record.published_at for record in candles) - min(
        record.published_at for record in candles
    ) == timedelta(hours=29)


def test_funding_coverage_requires_all_assets_and_requested_window() -> None:
    db = session()
    start = datetime(2026, 7, 25, tzinfo=UTC)
    end = datetime(2026, 8, 27, tzinfo=UTC)
    records = [
        HyperliquidSource._history_record(asset, start, 0)
        for asset in ("BTC", "ETH", "SOL")
    ]
    records += [
        HyperliquidSource._history_record(asset, end, 1)
        for asset in ("BTC", "ETH", "SOL")
    ]
    ingest_records(db, records)
    coverage = funding_coverage(db, start, end)
    assert all(item["required_warmup_satisfied"] for item in coverage.values())
    assert all(item["requested_start"] == start for item in coverage.values())
    assert all(item["requested_end"] == end for item in coverage.values())


def test_replay_script_avoids_bash4_mapfile_and_interpolated_sha() -> None:
    script = Path("scripts/run_phase16a_replay.sh").read_text()
    assert "mapfile" not in script
    assert '"${SHA}"' not in script
    assert 'os.environ["PHASE16A_SHA"]' in script
