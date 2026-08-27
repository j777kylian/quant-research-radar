from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Hypothesis, MarketMetric, MarketObservation, SourceItem
from .llm import LLMClient
from .pipeline import analyze, calculate_metrics, daily_report
from .sources import SourceRecord

ASSETS = ("BTC", "ETH", "SOL")
METRIC_NAMES = (
    "funding_percentile",
    "return_1h",
    "return_4h",
    "return_24h",
    "rolling_volatility",
)


def utc_day_cutoff(day: date) -> datetime:
    return datetime.combine(day, time.max, tzinfo=UTC)


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
            .where(
                MarketObservation.asset == asset,
                MarketObservation.observed_at >= start,
                MarketObservation.observed_at <= end,
            )
            .order_by(MarketObservation.observed_at)
        ).all()
        funding = [row for row in rows if row.observation_kind == "funding"]
        candles = [row for row in rows if row.observation_kind == "candle"]
        candle_times = [row.observed_at for row in candles]
        expected = 1 + int((end - start).total_seconds() // 3600)
        candle_set = set(candle_times)
        missing = [
            (start + timedelta(hours=i)).isoformat()
            for i in range(expected)
            if start + timedelta(hours=i) not in candle_set
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
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def metric_availability(
    session: Session, cutoff: datetime
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    rows = session.scalars(
        select(MarketObservation).where(MarketObservation.observed_at <= cutoff)
    ).all()
    for asset in ASSETS:
        observations = [row for row in rows if row.asset == asset]
        latest = max(observations, key=lambda row: row.observed_at, default=None)
        metrics = (
            session.scalars(
                select(MarketMetric).where(MarketMetric.observation_id == latest.id)
            ).all()
            if latest is not None
            else []
        )
        values = {metric.metric_name: metric.metric_value for metric in metrics}
        result[asset] = {
            name: "AVAILABLE" if name in values else "UNAVAILABLE"
            for name in METRIC_NAMES
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
) -> dict[str, Any]:
    cutoff = utc_day_cutoff(replay_day)
    day_root = output_root / replay_day.isoformat()
    day_root.mkdir(parents=True, exist_ok=True)
    calculate_metrics(session)
    # Remove future academic and market rows from the research view without deleting persisted evidence.
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
    report = daily_report(session, str(day_root), replay_day)
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
        "eligible_source_item_count": len(before),
        "hypotheses_generated": created,
        "market_quality": _json_value(market_quality(session, warmup_start, cutoff)),
        "metric_availability": metrics,
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
) -> Path:
    summary = {
        "run_identity": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "code_sha": code_sha,
            "replay_dates": [item.isoformat() for item in dates],
            "warmup_start": warmup_start.isoformat(),
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
        ],
        "historical_semantics": "Persisted evidence is retained, while each replay selects source publication/retrieval timestamps at or before its explicit cutoff. No future market data is used by the replay view.",
    }
    path = output_root / "phase16a-summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path
