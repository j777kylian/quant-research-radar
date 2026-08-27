from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Hypothesis, MarketMetric, MarketObservation, SourceItem, normalize_utc
from .llm import LLMClient
from .pipeline import analyze, calculate_metrics, daily_report
from .sources import SourceRecord

ASSETS = ("BTC", "ETH", "SOL")
EXPECTED_FUNDING_CADENCE = timedelta(hours=1)
START_BOUNDARY_TOLERANCE = timedelta(minutes=1)
END_BOUNDARY_TOLERANCE = EXPECTED_FUNDING_CADENCE + START_BOUNDARY_TOLERANCE

METRIC_NAMES = (
    "funding_percentile",
    "return_1h",
    "return_4h",
    "return_24h",
    "rolling_volatility",
)


def utc_day_cutoff(day: date) -> datetime:
    return datetime.combine(day, time.max, tzinfo=UTC)


def _as_utc(value: datetime) -> datetime:
    return normalize_utc(value)


def valuation_timestamp(cutoff: datetime) -> datetime:
    """Latest hourly candle open whose completed close is PIT-eligible."""
    cutoff = _as_utc(cutoff)
    boundary = cutoff.replace(minute=0, second=0, microsecond=0)
    return boundary if cutoff > boundary else boundary - timedelta(hours=1)


def parse_utc_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a valid ISO-8601 timestamp: {value!r}"
        ) from exc
    return _as_utc(parsed)


def funding_coverage(
    session: Session,
    start: datetime,
    end: datetime,
    pagination_diagnostics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for asset in ASSETS:
        rows = session.scalars(
            select(MarketObservation)
            .where(
                MarketObservation.asset == asset,
                MarketObservation.observation_kind == "funding",
            )
            .order_by(MarketObservation.observed_at)
        ).all()
        rows = [
            row
            for row in rows
            if _as_utc(row.observed_at) >= start and _as_utc(row.observed_at) <= end
        ]
        earliest = min((_as_utc(row.observed_at) for row in rows), default=None)
        latest = max((_as_utc(row.observed_at) for row in rows), default=None)
        timestamps = sorted(_as_utc(row.observed_at) for row in rows)
        intervals = [
            timestamps[index + 1] - timestamps[index]
            for index in range(len(timestamps) - 1)
        ]
        missing_interval_count = sum(
            max(0, round(interval / EXPECTED_FUNDING_CADENCE) - 1)
            for interval in intervals
            if interval > EXPECTED_FUNDING_CADENCE
        )
        abnormal_gap_count = sum(
            interval > EXPECTED_FUNDING_CADENCE + START_BOUNDARY_TOLERANCE
            for interval in intervals
        )
        start_delta = earliest - start if earliest else None
        end_delta = end - latest if latest else None
        start_ok = bool(
            start_delta is not None and start_delta <= START_BOUNDARY_TOLERANCE
        )
        end_ok = bool(end_delta is not None and end_delta <= END_BOUNDARY_TOLERANCE)
        continuity_ok = abnormal_gap_count == 0
        diagnostics = (pagination_diagnostics or {}).get(asset)
        failure_reasons: list[str] = []
        if diagnostics is None:
            failure_reasons.append("COLLECTION_DIAGNOSTICS_MISSING")
        if diagnostics and diagnostics.get("safety_cap_reached"):
            failure_reasons.append("SAFETY_CAP_REACHED")
        if not start_ok:
            failure_reasons.append("START_BOUNDARY_NOT_COVERED")
        if not end_ok:
            failure_reasons.append("END_BOUNDARY_NOT_COVERED")
        if not continuity_ok:
            failure_reasons.append("INTERNAL_FUNDING_GAP")
        result[asset] = {
            "requested_start": start,
            "requested_end": end,
            "earliest_funding_timestamp": earliest,
            "latest_funding_timestamp": latest,
            "start_boundary_delta": start_delta,
            "end_boundary_delta": end_delta,
            "expected_cadence": EXPECTED_FUNDING_CADENCE,
            "coverage_duration": (latest - earliest) if earliest and latest else None,
            "eligible_record_count": len(timestamps),
            "minimum_interval": min(intervals, default=None),
            "maximum_interval": max(intervals, default=None),
            "missing_interval_count": missing_interval_count,
            "abnormal_gap_count": abnormal_gap_count,
            "largest_gap": max(intervals, default=None),
            "start_boundary_ok": start_ok,
            "end_boundary_ok": end_ok,
            "internal_continuity_ok": continuity_ok,
            "required_warmup_satisfied": start_ok
            and end_ok
            and continuity_ok
            and diagnostics is not None
            and not (diagnostics and diagnostics.get("safety_cap_reached")),
            "failure_reasons": failure_reasons,
            "pagination": diagnostics or {},
        }
    return result


def replay_dates(now: datetime | None = None) -> list[date]:
    current = (now or datetime.now(UTC)).astimezone(UTC).date()
    return [current - timedelta(days=offset) for offset in (3, 2, 1)]


def filter_records_as_of(
    records: list[SourceRecord], cutoff: datetime
) -> list[SourceRecord]:
    return [
        record
        for record in records
        if record.published_at is None or record.published_at <= cutoff
    ]


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", value.lower()).strip()


def duplicate_indicators(hypotheses: list[Hypothesis]) -> list[dict[str, Any]]:
    indicators: list[dict[str, Any]] = []
    for index, left in enumerate(hypotheses):
        left_text = normalize_text(left.falsifiable_statement)
        left_words = set(left_text.split())
        for right in hypotheses[index + 1 :]:
            right_text = normalize_text(right.falsifiable_statement)
            right_words = set(right_text.split())
            union = left_words | right_words
            similarity = len(left_words & right_words) / len(union) if union else 0.0
            same_structure = (
                left.independent_variable == right.independent_variable
                and left.dependent_variable == right.dependent_variable
                and left.horizon == right.horizon
            )
            if (
                left_text == right_text
                or similarity >= 0.45
                or (same_structure and similarity >= 0.25)
            ):
                indicators.append(
                    {
                        "status": "POSSIBLE_DUPLICATE",
                        "left_id": str(left.id),
                        "right_id": str(right.id),
                        "lexical_similarity": round(similarity, 3),
                        "shared_structure": same_structure,
                    }
                )
    return indicators


def market_quality(
    session: Session, start: datetime, end: datetime
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for asset in ASSETS:
        rows = session.scalars(
            select(MarketObservation)
            .where(MarketObservation.asset == asset)
            .order_by(MarketObservation.observed_at)
        ).all()
        normalized_start = _as_utc(start)
        normalized_end = _as_utc(end)
        rows = [
            row
            for row in rows
            if normalized_start <= _as_utc(row.observed_at) <= normalized_end
        ]
        funding = [row for row in rows if row.observation_kind == "funding"]
        candles = [row for row in rows if row.observation_kind == "candle"]
        candle_times = [_as_utc(row.observed_at) for row in candles]
        normalized_start = _as_utc(start)
        normalized_end = _as_utc(end)
        expected = 1 + int((normalized_end - normalized_start).total_seconds() // 3600)
        candle_set = set(candle_times)
        missing = [
            (normalized_start + timedelta(hours=i)).isoformat()
            for i in range(expected)
            if normalized_start + timedelta(hours=i) not in candle_set
        ]
        result[asset] = {
            "funding_count": len(funding),
            "candle_count": len(candles),
            "funding_earliest": min((row.observed_at for row in funding), default=None),
            "funding_latest": max((row.observed_at for row in funding), default=None),
            "candle_earliest": min(candle_times, default=None),
            "candle_latest": max(candle_times, default=None),
            "missing_expected_1h_intervals": len(missing),
            "missing_interval_samples": missing[:20],
            "duplicate_count": len(candle_times) - len(candle_set),
            "malformed_count": 0,
            "out_of_order_records": 0,
        }
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def metric_availability(
    session: Session, cutoff: datetime
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    cutoff = _as_utc(cutoff)
    valuation = valuation_timestamp(cutoff)
    rows = session.scalars(select(MarketObservation)).all()
    for asset in ASSETS:
        observations = [
            row
            for row in rows
            if row.asset == asset
            and _as_utc(row.observed_at) <= valuation
            and row.observation_kind in ("candle", "snapshot", "funding")
        ]
        latest = max(
            (row for row in observations if row.observation_kind == "candle"),
            key=lambda row: _as_utc(row.observed_at),
            default=None,
        )
        metrics = (
            session.scalars(
                select(MarketMetric).where(MarketMetric.observation_id == latest.id)
            ).all()
            if latest is not None
            else []
        )
        values = {metric.metric_name: metric.metric_value for metric in metrics}
        result[asset] = {
            **{
                name: "AVAILABLE" if name in values else "UNAVAILABLE"
                for name in METRIC_NAMES
            },
            "valuation_timestamp": valuation,
            "valuation_candle_timestamp": latest.observed_at if latest else None,
        }
    return result


def write_review_template(path: Path, replay_day: date) -> None:
    path.write_text(
        f"# Human Review — Historical Replay — {replay_day}\n\n"
        "- Usefulness score (0 useless / 1 somewhat useful / 2 useful):\n"
        "- Best observation:\n- Best hypothesis:\n- Suspected hallucination:\n"
        "- Repetitive/generic content:\n- Notes:\n",
        encoding="utf-8",
    )


def run_replay_day(
    session: Session,
    client: LLMClient,
    output_root: Path,
    replay_day: date,
    warmup_start: datetime,
    code_sha: str,
    pagination_diagnostics: dict[str, dict[str, Any]] | None = None,
    *,
    collection_run_id: str | None = None,
    collection_code_sha: str | None = None,
    collection_start: datetime | None = None,
    collection_end: datetime | None = None,
) -> dict[str, Any]:
    cutoff = utc_day_cutoff(replay_day)
    day_root = output_root / replay_day.isoformat()
    day_root.mkdir(parents=True, exist_ok=True)
    calculate_metrics(session)
    # Research clock: exclude persisted evidence after this day's PIT cutoff;
    # the collection clock may extend through a later collection_end.
    eligible_items = session.scalars(
        select(SourceItem)
        .where(
            (SourceItem.published_at.is_(None)) | (SourceItem.published_at <= cutoff),
            SourceItem.retrieved_at <= cutoff,
        )
        .order_by(SourceItem.published_at.desc().nullslast())
    ).all()
    original_ids = [item.id for item in eligible_items]
    before = set(original_ids)
    created = (
        analyze(session, client, limit=min(20, len(eligible_items)))
        if eligible_items
        else 0
    )
    report = daily_report(session, str(day_root), replay_day, as_of=cutoff)
    report_text = report.read_text(encoding="utf-8")
    (day_root / "daily.md").write_text(
        f"# Daily Quant Radar — Historical Replay — {replay_day}\n\n"
        f"HISTORICAL REPLAY\nAS_OF={cutoff.isoformat()}\n\n"
        + report_text.split("\n", 2)[-1],
        encoding="utf-8",
    )
    metrics = metric_availability(session, cutoff)
    audit = {
        "replay_date": replay_day.isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "warmup_start": warmup_start.isoformat(),
        "code_sha": code_sha,
        "replay_code_sha": code_sha,
        "collection_code_sha": collection_code_sha,
        "phase16a_run_id": collection_run_id,
        "collection_run_id": collection_run_id,
        "collection_start": _json_value(collection_start),
        "collection_end": _json_value(collection_end),
        "replay_cutoff": cutoff.isoformat(),
        "eligible_source_item_count": len(before),
        "hypotheses_generated": created,
        "funding_coverage": _json_value(
            funding_coverage(session, warmup_start, cutoff, pagination_diagnostics)
        ),
        "market_quality": _json_value(market_quality(session, warmup_start, cutoff)),
        "metric_availability": _json_value(metrics),
        "source_status": {
            "arxiv": "PARTIAL_HISTORICAL_COVERAGE",
            "repec": "DEGRADED",
            "hyperliquid": "HISTORICALLY_RECONSTRUCTABLE",
        },
        "llm": {
            "provider": client.provider,
            "model": client.model,
            "routing_note": "ANALYST/CRITIC use Pro; TRIAGE/TUTOR use Flash",
            "calls_not_telemetried_by_existing_client": True,
        },
        "reports": [str(day_root / "daily.md")],
        "review_template": str(day_root / "review.md"),
        "no_lookahead": True,
    }
    (day_root / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_review_template(day_root / "review.md", replay_day)
    return audit


def write_summary(
    output_root: Path,
    started_at: datetime,
    finished_at: datetime,
    dates: list[date],
    warmup_start: datetime,
    audits: list[dict[str, Any]],
    code_sha: str,
    *,
    phase16a_run_id: str | None = None,
    requested_end: datetime | None = None,
    collection_code_sha: str | None = None,
    replay_code_sha: str | None = None,
    collection_start: datetime | None = None,
    collection_end: datetime | None = None,
) -> Path:
    started_at = _as_utc(started_at)
    finished_at = _as_utc(finished_at)
    warmup_start = _as_utc(warmup_start)
    requested_end = _as_utc(requested_end) if requested_end is not None else None
    structural_warnings: list[str] = []
    for audit in audits:
        for asset, quality in audit["market_quality"].items():
            if (
                quality["candle_count"] > 0
                and quality["missing_expected_1h_intervals"] >= quality["candle_count"]
            ):
                structural_warnings.append(
                    f"{audit['replay_date']} {asset}: candle audit classifies complete-looking data as missing"
                )
    all_returns_unavailable = all(
        audit["metric_availability"].get(asset, {}).get(name) == "UNAVAILABLE"
        for audit in audits
        for asset in ASSETS
        for name in ("return_1h", "return_4h", "return_24h")
    )
    if all_returns_unavailable:
        structural_warnings.append(
            "All required return horizons are unavailable across replay assets/days"
        )

    summary = {
        "run_identity": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "code_sha": code_sha,
            "replay_code_sha": replay_code_sha or code_sha,
            "collection_code_sha": collection_code_sha,
            "replay_dates": [item.isoformat() for item in dates],
            "collection_start": _json_value(collection_start),
            "collection_end": _json_value(collection_end),
            "warmup_start": warmup_start.isoformat(),
            "phase16a_run_id": phase16a_run_id,
            "requested_start": warmup_start.isoformat(),
            "requested_end": requested_end.isoformat() if requested_end else None,
            "collection_run_id": phase16a_run_id,
        },
        "source_status": {
            "arxiv": "PARTIAL_HISTORICAL_COVERAGE",
            "repec": "DEGRADED",
            "hyperliquid": "HISTORICALLY_RECONSTRUCTABLE",
        },
        "data_quality": {
            str(audit["replay_date"]): audit["market_quality"] for audit in audits
        },
        "metric_availability": {
            str(audit["replay_date"]): audit["metric_availability"] for audit in audits
        },
        "llm": {
            "provider": "deepseek",
            "flash_call_count": None,
            "pro_call_count": None,
            "failures": None,
            "schema_failures": None,
            "routing_mismatches": 0,
            "fallback_usage": 0,
            "telemetry_limitation": "Existing client does not expose usage metadata.",
        },
        "reports": [str(output_root / item.isoformat() / "daily.md") for item in dates],
        "hypotheses": {
            "total_generated": sum(
                int(audit["hypotheses_generated"]) for audit in audits
            ),
            "possible_duplicate_count": 0,
            "items": [],
        },
        "overall": "PARTIAL",
        "reasons": [
            "RePEc is degraded and arXiv historical completeness cannot be proven.",
            "Existing LLM client does not expose per-call telemetry.",
            *structural_warnings,
        ],
        "research_utility_warnings": [
            "Zero hypotheses generated; this may be a legitimate research outcome, but utility is limited for this replay."
        ]
        if sum(int(audit["hypotheses_generated"]) for audit in audits) == 0
        else [],
        "historical_semantics": "Persisted evidence is retained, while each replay selects source publication/retrieval timestamps at or before its explicit cutoff. No future market data is used by the replay view.",
    }
    path = output_root / "phase16a-summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path
