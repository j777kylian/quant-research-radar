import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select

from quant_research_radar.db import (
    Base,
    CollectionRun,
    get_phase16a_collection_run,
)
from quant_research_radar.llm import FakeLLMClient
from quant_research_radar.pipeline import ingest_records
from quant_research_radar.replay import (
    filter_records_as_of,
    funding_coverage,
    market_quality,
    parse_utc_timestamp,
    run_replay_day,
    utc_day_cutoff,
    valuation_timestamp,
    write_summary,
)
from quant_research_radar.sources import HyperliquidSource, SourceRecord


def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import Session

    return Session(engine)


def rows(start, end, missing=()):
    result = []
    for asset in ("BTC", "ETH", "SOL"):
        for i in range(int((end - start).total_seconds() // 3600) + 1):
            if i not in missing:
                result.append(
                    HyperliquidSource._history_record(
                        asset, start + timedelta(hours=i), i
                    )
                )
    return result


def diagnostics(cap=False):
    return {
        asset: {
            "funding_request_count": 3,
            "raw_records_returned": 1500,
            "eligible_records": 1200,
            "duplicate_records_removed": 2,
            "malformed_records": 0,
            "safety_cap_reached": cap,
            "pagination_termination_reason": "SAFETY_CAP" if cap else "REACHED_END",
        }
        for asset in ("BTC", "ETH", "SOL")
    }


def test_naive_and_aware_candle_timestamps_match_and_missing_counts_are_exact():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=23)
    db = session()
    records = [
        HyperliquidSource._candle_record("BTC", start + timedelta(hours=i), i)
        for i in range(24)
    ]
    records[0] = SourceRecord(
        records[0].source_type,
        records[0].source_name,
        records[0].external_id,
        records[0].title,
        records[0].canonical_url,
        records[0].authors,
        start.replace(tzinfo=None),
        records[0].raw_text,
        records[0].raw_metadata,
    )
    ingest_records(db, records)
    assert market_quality(db, start, end)["BTC"]["missing_expected_1h_intervals"] == 0
    db2 = session()
    db2_records = [
        HyperliquidSource._candle_record("BTC", start + timedelta(hours=i), i)
        for i in range(24)
        if i != 7
    ]
    ingest_records(db2, db2_records)
    assert market_quality(db2, start, end)["BTC"]["missing_expected_1h_intervals"] == 1


def test_valuation_timestamp_is_pit_safe_and_eod_does_not_require_235959():
    cutoff = datetime(2026, 8, 26, 23, 59, 59, 999999, tzinfo=UTC)
    assert valuation_timestamp(cutoff) == datetime(2026, 8, 26, 23, tzinfo=UTC)
    assert valuation_timestamp(datetime(2026, 8, 26, 23, tzinfo=UTC)) == datetime(
        2026, 8, 26, 22, tzinfo=UTC
    )


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
    cutoff = utc_day_cutoff(datetime(2026, 8, 25, tzinfo=UTC).date())
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
        datetime(2026, 8, 25, tzinfo=UTC).date(),
        cutoff - timedelta(days=30),
        "fixture",
        diagnostics(),
    )
    second = run_replay_day(
        db,
        FakeLLMClient(),
        tmp_path,
        datetime(2026, 8, 24, tzinfo=UTC).date(),
        cutoff - timedelta(days=30),
        "fixture",
        diagnostics(),
    )
    assert Path(first["reports"][0]).exists()
    assert Path(second["reports"][0]).exists()
    assert "AS_OF=2026-08-25" in (tmp_path / "2026-08-25" / "daily.md").read_text()


def test_hyperliquid_history_propagates_bounded_window_and_excludes_future():
    class Response:
        def __init__(self, rows):
            self.rows = rows

        def raise_for_status(self):
            pass

        def json(self):
            return self.rows

    class Client:
        def __init__(self):
            self.requests = []

        def post(self, _endpoint, json):
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


def test_hyperliquid_history_limit_is_per_asset_and_not_30_day_coverage():
    source = HyperliquidSource()
    end = datetime(2026, 8, 27, tzinfo=UTC)
    funding = source.collect_history(800, offline=True, end=end)
    candles = source.collect_candles(800, offline=True, end=end)
    assert {r.raw_metadata["asset"] for r in funding} == {"BTC", "ETH", "SOL"}
    assert {r.raw_metadata["asset"] for r in candles} == {"BTC", "ETH", "SOL"}
    assert all(
        sum(r.raw_metadata["asset"] == asset for r in funding) == 6
        for asset in source.assets
    )
    assert all(
        sum(r.raw_metadata["asset"] == asset for r in candles) == 30
        for asset in source.assets
    )
    assert max(r.published_at for r in candles) - min(
        r.published_at for r in candles
    ) == timedelta(hours=29)


def test_hyperliquid_funding_paginates_and_deduplicates_inclusive_boundaries():
    class Response:
        def __init__(self, rows):
            self.rows = rows

        def raise_for_status(self):
            pass

        def json(self):
            return self.rows

    class Client:
        def __init__(self):
            self.requests = []

        def post(self, _endpoint, json):
            self.requests.append(json)
            start_ms, asset, step = json["startTime"], json["coin"], 3_600_000
            rows = [
                {"coin": asset, "time": start_ms + i * step, "fundingRate": "0.001"}
                for i in range(500)
            ]
            if len(self.requests) % 3 == 1:
                rows[0] = {
                    "coin": asset,
                    "time": start_ms - step,
                    "fundingRate": "0.001",
                }
            return Response(rows)

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1100)
    client = Client()
    records = HyperliquidSource(client=client).collect_history(
        1200, start=start, end=end
    )
    assert len(client.requests) == 9
    assert {request["coin"] for request in client.requests} == {"BTC", "ETH", "SOL"}
    assert len({record.external_id for record in records}) == len(records)
    assert all(start <= record.published_at <= end for record in records)
    for asset in HyperliquidSource.assets:
        asset_requests = [r for r in client.requests if r["coin"] == asset]
        assert len(asset_requests) == 3
        assert asset_requests[1]["startTime"] > asset_requests[0]["startTime"]


def test_hyperliquid_pagination_fails_closed_on_non_advancing_page():
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"coin": "BTC", "time": 1_000, "fundingRate": "0.001"}] * 500

    class Client:
        def post(self, _endpoint, json):
            return Response()

    source = HyperliquidSource(client=Client())
    start = datetime.fromtimestamp(1, UTC)
    with pytest.raises(ValueError, match="Duplicate|Non-advancing"):
        source.collect_history(1200, start=start, end=start + timedelta(hours=1000))


def test_boundaries_and_jitter_pass():
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 3, tzinfo=UTC)
    db = session()
    ingest_records(
        db, rows(start + timedelta(milliseconds=25), end - timedelta(hours=1))
    )
    assert funding_coverage(db, start, end, diagnostics())["BTC"]["start_boundary_ok"]


def test_late_start_and_end_boundary():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=23, minutes=59, seconds=59)
    db = session()
    ingest_records(db, rows(start + timedelta(hours=1), end - timedelta(minutes=59)))
    result = funding_coverage(db, start, end, diagnostics())["BTC"]
    assert not result["start_boundary_ok"]
    assert result["end_boundary_ok"]


def test_missing_single_and_multiple_hours_are_detected():
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 8, tzinfo=UTC)
    db = session()
    ingest_records(db, rows(start, end, (3, 5, 6)))
    result = funding_coverage(db, start, end, diagnostics())["BTC"]
    assert result["missing_interval_count"] == 3
    assert not result["internal_continuity_ok"]


def test_safety_cap_fails_closed():
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 1, tzinfo=UTC)
    db = session()
    ingest_records(db, rows(start, end))
    result = funding_coverage(db, start, end, diagnostics(True))["BTC"]
    assert not result["required_warmup_satisfied"]
    assert "SAFETY_CAP_REACHED" in result["failure_reasons"]


def test_diagnostics_are_exposed_and_missing_fails():
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 1, tzinfo=UTC)
    db = session()
    ingest_records(db, rows(start, end))
    result = funding_coverage(db, start, end, diagnostics())
    assert result["ETH"]["pagination"]["funding_request_count"] == 3
    assert result["SOL"]["pagination"]["duplicate_records_removed"] == 2
    assert (
        "COLLECTION_DIAGNOSTICS_MISSING"
        in funding_coverage(db, start, end)["BTC"]["failure_reasons"]
    )


def test_diagnostics_process_boundary_and_interval_binding(tmp_path: Path):
    import json

    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 1, tzinfo=UTC)
    artifact = tmp_path / "diagnostics.json"
    artifact.write_text(
        json.dumps(
            {
                "run_id": "run",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "diagnostics": diagnostics(),
            }
        )
    )
    payload = json.loads(artifact.read_text())
    assert (
        payload["diagnostics"]["BTC"]["pagination_termination_reason"] == "REACHED_END"
    )
    assert payload["start"] == start.isoformat()
    assert payload["end"] != (end + timedelta(hours=1)).isoformat()


def test_explicit_run_binding_prevents_stale_and_same_interval_mixups():
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 1, tzinfo=UTC)
    db = session()
    old = CollectionRun(
        source="hyperliquid",
        phase16a_run_id="A",
        requested_start=start,
        requested_end=end,
        code_sha="sha",
        status="SUCCESS",
        diagnostics=diagnostics(),
    )
    current = CollectionRun(
        source="hyperliquid",
        phase16a_run_id="B",
        requested_start=start,
        requested_end=end,
        code_sha="sha",
        status="FAILED",
        diagnostics={},
    )
    db.add_all([old, current])
    db.commit()
    selected = db.scalar(
        select(CollectionRun).where(CollectionRun.phase16a_run_id == "B")
    )
    assert selected is current
    assert not selected.diagnostics


def test_phase16a_lookup_is_strict_and_timezone_normalized():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, 1, tzinfo=UTC)
    db = session()
    run = CollectionRun(
        source="hyperliquid",
        phase16a_run_id="run-a",
        requested_start=start.replace(tzinfo=None),
        requested_end=end.replace(tzinfo=None),
        code_sha="sha-a",
        status="SUCCESS",
        diagnostics=diagnostics(),
    )
    db.add(run)
    db.add(CollectionRun(source="arxiv", status="SUCCESS", diagnostics={}))
    db.add(CollectionRun(source="repec", status="DEGRADED", diagnostics={}))
    db.commit()
    assert (
        get_phase16a_collection_run(
            db,
            source="hyperliquid",
            phase16a_run_id="run-a",
            requested_start=start,
            requested_end=end,
            code_sha="sha-a",
        )
        is run
    )
    for _day in ("2026-01-01", "2026-01-02", "2026-01-03"):
        assert (
            get_phase16a_collection_run(
                db,
                source="hyperliquid",
                phase16a_run_id="run-a",
                requested_start=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                requested_end=datetime.fromisoformat("2026-01-01T01:00:00+00:00"),
                code_sha="sha-a",
            )
            is run
        )
    assert (
        get_phase16a_collection_run(
            db,
            source="hyperliquid",
            phase16a_run_id="wrong",
            requested_start=start,
            requested_end=end,
            code_sha="sha-a",
        )
        is None
    )
    assert (
        get_phase16a_collection_run(
            db,
            source="hyperliquid",
            phase16a_run_id="run-a",
            requested_start=start,
            requested_end=end,
            code_sha="wrong",
        )
        is None
    )
    assert (
        get_phase16a_collection_run(
            db,
            source="hyperliquid",
            phase16a_run_id="run-a",
            requested_start=start + timedelta(seconds=1),
            requested_end=end,
            code_sha="sha-a",
        )
        is None
    )
    assert (
        get_phase16a_collection_run(
            db,
            source="hyperliquid",
            phase16a_run_id="run-a",
            requested_start=start,
            requested_end=end,
            code_sha="sha-a",
            status="FAILED",
        )
        is None
    )
    run.diagnostics = {}
    assert not run.diagnostics


def test_collection_window_end_is_separate_from_replay_cutoff():
    start = datetime(2026, 7, 22, tzinfo=UTC)
    day1 = utc_day_cutoff(datetime(2026, 8, 24, tzinfo=UTC).date())
    day2 = utc_day_cutoff(datetime(2026, 8, 25, tzinfo=UTC).date())
    day3 = utc_day_cutoff(datetime(2026, 8, 26, tzinfo=UTC).date())
    db = session()
    run = CollectionRun(
        source="hyperliquid",
        phase16a_run_id="shared",
        requested_start=start,
        requested_end=day3,
        code_sha="sha",
        status="SUCCESS",
        diagnostics=diagnostics(),
    )
    db.add(run)
    db.commit()
    for replay_cutoff in (day1, day2, day3):
        assert (
            get_phase16a_collection_run(
                db,
                source="hyperliquid",
                phase16a_run_id="shared",
                requested_start=start,
                requested_end=day3,
                code_sha="sha",
            )
            is run
        )
        assert replay_cutoff <= day3
    assert (
        get_phase16a_collection_run(
            db,
            source="hyperliquid",
            phase16a_run_id="shared",
            requested_start=start,
            requested_end=day1,
            code_sha="sha",
        )
        is None
    )


def test_summary_timestamp_boundary_and_serialization(tmp_path: Path):
    requested_end = parse_utc_timestamp("2026-08-26T23:59:59.999999Z", "PHASE16A_END")
    assert requested_end == datetime(2026, 8, 26, 23, 59, 59, 999999, tzinfo=UTC)
    audit = {
        "replay_date": "2026-08-26",
        "warmup_start": "2026-07-22T00:00:00+00:00",
        "market_quality": {},
        "metric_availability": {},
        "hypotheses_generated": 0,
    }
    path = write_summary(
        tmp_path,
        requested_end,
        requested_end,
        [datetime(2026, 8, 26, tzinfo=UTC).date()],
        parse_utc_timestamp(audit["warmup_start"], "warmup_start"),
        [audit],
        "sha",
        requested_end=requested_end,
    )
    payload = json.loads(path.read_text())
    assert payload["run_identity"]["requested_end"] == requested_end.isoformat()
    assert payload["run_identity"]["requested_start"] == audit["warmup_start"]


def test_summary_timestamp_malformed_fails_clearly():
    with pytest.raises(ValueError, match="PHASE16A_END.*ISO-8601"):
        parse_utc_timestamp("not-a-timestamp", "PHASE16A_END")


def test_replay_script_keeps_two_clock_provenance_wiring():
    script = Path("scripts/run_phase16a_replay.sh").read_text()
    assert "mapfile" not in script
    assert '"${SHA}"' not in script
    assert 'os.environ["PHASE16A_SHA"]' in script
    assert script.count('--collection-end "$LATEST_REPLAY_CUTOFF"') == 2
    assert '--as-of "$CUTOFF" --collection-end "$LATEST_REPLAY_CUTOFF"' in script
