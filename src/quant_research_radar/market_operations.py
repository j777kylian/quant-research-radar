"""Bounded receipt-safe market collection operations for Phase 2.0."""

from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from time import sleep
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (
    CollectionRun,
    CollectionStatus,
    MarketObservation,
    RawArtifactReceipt,
    normalize_utc,
    utcnow,
)
from .pipeline import ingest_records
from .raw_archive import RawArchive
from .sources import HyperliquidSource

RECONSTRUCTIVE_MODE = "ACCELERATED_RECONSTRUCTIVE_RESEARCH"
PRODUCTION_MODE = "PRODUCTION_LIVE"
ASSETS = ("BTC", "ETH", "SOL")
CHUNK = timedelta(days=7)


def safe_complete_hour(now: datetime | None = None) -> datetime:
    now = normalize_utc(now or datetime.now(UTC))
    return now.replace(minute=0, second=0, microsecond=0)


def default_backfill_start(end: datetime) -> datetime:
    end = normalize_utc(end)
    return end.replace(
        year=end.year - 1,
        day=min(end.day, calendar.monthrange(end.year - 1, end.month)[1]),
    )


def _windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    start, end = normalize_utc(start), normalize_utc(end)
    if start >= end:
        raise ValueError("collection start must be before end")
    windows = []
    cursor = start
    while cursor < end:
        next_end = min(cursor + CHUNK - timedelta(microseconds=1), end)
        windows.append((cursor, next_end))
        cursor = next_end + timedelta(microseconds=1)
    return windows


def _collect_with_retries(
    method: Any, limit: int, start: datetime, end: datetime
) -> list[Any]:
    for attempt in range(3):
        try:
            return list(method(limit, start=start, end=end))
        except Exception:
            if attempt == 2:
                raise
            sleep(2**attempt)
    raise AssertionError("unreachable")


def _run(
    session: Session,
    adapter: HyperliquidSource,
    archive: RawArchive,
    *,
    start: datetime,
    end: datetime,
    code_sha: str,
    mode: str,
) -> dict[str, Any]:
    if mode not in {RECONSTRUCTIVE_MODE, PRODUCTION_MODE}:
        raise ValueError("unsupported market collection analysis mode")
    start, end = normalize_utc(start), normalize_utc(end)
    run = CollectionRun(
        source=adapter.name,
        status="RUNNING",
        requested_start=start,
        requested_end=end,
        code_sha=code_sha,
        diagnostics={
            "analysis_mode": mode,
            "chunk_hours": int(CHUNK.total_seconds() // 3600),
            "completed_windows": [],
        },
    )
    session.add(run)
    session.flush()
    inserted = duplicates = retrieved = 0
    resumed_windows = 0
    try:
        for window_start, window_end in _windows(start, end):
            completed = session.scalars(
                select(CollectionRun).where(CollectionRun.source == adapter.name)
            ).all()
            window_key = [window_start.isoformat(), window_end.isoformat()]
            if any(
                item.diagnostics.get("analysis_mode") == mode
                and window_key in item.diagnostics.get("completed_windows", [])
                for item in completed
            ):
                resumed_windows += 1
                continue
            limit = int((window_end - window_start).total_seconds() // 3600) + 2
            funding = _collect_with_retries(
                adapter.collect_history, limit, window_start, window_end
            )
            candles = [
                record
                for record in _collect_with_retries(
                    adapter.collect_candles, limit, window_start, window_end
                )
                if record.published_at is not None
                and record.published_at + timedelta(hours=1) <= window_end
            ]
            for records in (funding, candles):
                added, skipped = ingest_records(
                    session,
                    records,
                    archive=archive,
                    collection_run_id=run.id,
                    analysis_mode=mode,
                )
                inserted += added
                duplicates += skipped
                retrieved += len(records)
            diagnostics = dict(run.diagnostics)
            diagnostics["completed_windows"] = [
                *diagnostics["completed_windows"],
                window_key,
            ]
            run.diagnostics = diagnostics
            session.commit()
        run.retrieved = retrieved
        run.inserted = inserted
        run.skipped_duplicates = duplicates
        run.status = "SUCCESS"
        run.ended_at = utcnow()
        session.commit()
    except BaseException as exc:
        session.rollback()
        failed = session.get(CollectionRun, run.id)
        if failed is not None:
            failed.status = "FAILED"
            failed.failed += 1
            failed.error_reason = type(exc).__name__
            failed.ended_at = utcnow()
            session.commit()
        raise
    return {
        "run_id": str(run.id),
        "inserted": inserted,
        "duplicates": duplicates,
        "retrieved": retrieved,
        "resumed_windows": resumed_windows,
        "analysis_mode": mode,
    }


def coverage_audit(
    session: Session, *, start: datetime, end: datetime, mode: str = RECONSTRUCTIVE_MODE
) -> dict[str, Any]:
    """Count only completed-run, source-time-bound records eligible for this mode."""
    start, end = normalize_utc(start), normalize_utc(end)
    receipts = session.execute(
        select(RawArtifactReceipt, CollectionRun, MarketObservation)
        .join(CollectionRun, RawArtifactReceipt.collection_run_id == CollectionRun.id)
        .join(
            MarketObservation,
            RawArtifactReceipt.market_observation_id == MarketObservation.id,
        )
        .where(
            RawArtifactReceipt.analysis_mode == mode,
            CollectionRun.status == CollectionStatus.SUCCESS.value,
            CollectionRun.ended_at.is_not(None),
            MarketObservation.observed_at >= start,
            MarketObservation.observed_at <= end,
        )
    ).all()
    qualified: dict[str, dict[str, dict[datetime, MarketObservation]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    receipt_counts: dict[tuple[str, str, datetime], int] = defaultdict(int)
    for receipt, _run, observation in receipts:
        at = normalize_utc(observation.observed_at)
        if receipt.source_native_timestamp is not None and normalize_utc(
            receipt.source_native_timestamp
        ) == at + (
            timedelta(hours=1)
            if observation.observation_kind == "candle"
            else timedelta()
        ):
            qualified[observation.asset][observation.observation_kind][at] = observation
            receipt_counts[(observation.asset, observation.observation_kind, at)] += 1
    assets: dict[str, Any] = {}
    for asset in ASSETS:
        funding = qualified[asset]["funding"]
        candles = qualified[asset]["candle"]
        ordered_funding = sorted(funding)
        extreme: list[datetime] = []
        for index, at in enumerate(ordered_funding):
            window = [
                funding[item].funding_rate
                for item in ordered_funding[max(0, index - 29) : index + 1]
            ]
            values = [value for value in window if value is not None]
            rate = funding[at].funding_rate
            # Strict percentile rank: extreme iff the current rate strictly exceeds
            # 90% of the trailing 30 same-asset funding observations. Ties are NOT
            # counted as exceeding, so a dominant ceiling/default rate is ordinary.
            if (
                rate is not None
                and values
                and sum(value < rate for value in values) / len(values) >= 0.9
            ):
                extreme.append(at)
        regimes = sum(
            index == 0 or at - extreme[index - 1] > timedelta(hours=1)
            for index, at in enumerate(extreme)
        )
        outcomes = {
            horizon: sum(
                (
                    anchor := at.replace(minute=0, second=0, microsecond=0)
                    - timedelta(hours=1)
                )
                in candles
                and anchor + timedelta(hours=horizon) in candles
                for at in extreme
            )
            for horizon in (1, 4, 24)
        }

        def summary(
            asset_name: str, kind: str, rows: dict[datetime, MarketObservation]
        ) -> dict[str, Any]:
            points = sorted(rows)
            # Funding timestamps carry millisecond jitter around the hour boundary;
            # gap detection must compare on the native hourly cadence, not raw
            # sub-second instants, or contiguous hourly events are misread as gaps.
            hours = [
                point.replace(minute=0, second=0, microsecond=0) for point in points
            ]
            gaps = sum(
                b - a > timedelta(hours=1)
                for a, b in zip(hours, hours[1:], strict=False)
            )
            return {
                "start": points[0].isoformat() if points else None,
                "end": points[-1].isoformat() if points else None,
                "row_count": len(points),
                "receipt_qualified_row_count": len(points),
                "gaps": gaps,
                "duplicate_count": sum(
                    max(0, receipt_counts[(asset_name, kind, at)] - 1) for at in points
                ),
            }

        assets[asset] = {
            "funding": summary(asset, "funding", funding),
            "candle": summary(asset, "candle", candles),
            "extreme_funding_observation_count": len(extreme),
            "independent_extreme_funding_regime_count": regimes,
            "eligible_outcomes": {str(key): value for key, value in outcomes.items()},
        }
    total_regimes = sum(
        value["independent_extreme_funding_regime_count"] for value in assets.values()
    )
    ready = total_regimes >= 5 and all(
        value["eligible_outcomes"]["24"] >= 5 for value in assets.values()
    )
    recommendation = "HISTORY_WINDOW_SUFFICIENT" if ready else "EXTEND_TO_18_MONTHS"
    return {
        "mode": mode,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "assets": assets,
        "EVENT_STUDY_DATA_READY": "YES" if ready else "NO",
        "history_window_recommendation": recommendation,
    }


def run_historical_backfill(
    session: Session,
    adapter: HyperliquidSource,
    archive: RawArchive,
    *,
    start: datetime,
    end: datetime,
    code_sha: str,
) -> dict[str, Any]:
    return _run(
        session,
        adapter,
        archive,
        start=start,
        end=end,
        code_sha=code_sha,
        mode=RECONSTRUCTIVE_MODE,
    )


def run_live_market_collection(
    session: Session,
    adapter: HyperliquidSource,
    archive: RawArchive,
    *,
    start: datetime,
    end: datetime,
    code_sha: str,
) -> dict[str, Any]:
    end = normalize_utc(end)
    if end > safe_complete_hour():
        raise ValueError(
            "live collection may not extend past the latest completed UTC hour"
        )
    return _run(
        session,
        adapter,
        archive,
        start=start,
        end=end,
        code_sha=code_sha,
        mode=PRODUCTION_MODE,
    )


def latest_production_candle(session: Session) -> datetime | None:
    """Conservative production watermark: min over assets of the latest candle.

    A Daily run collects from this watermark forward (with a one-hour overlap for
    dedup safety), so hourly resolution is preserved while the collector runs only
    once per day. Returns ``None`` when no production candle exists yet.
    """
    from sqlalchemy import func

    rows = session.execute(
        select(func.max(MarketObservation.observed_at))
        .where(
            MarketObservation.observation_kind == "candle",
            MarketObservation.asset.in_(ASSETS),
        )
        .group_by(MarketObservation.asset)
    ).scalars()
    present = [normalize_utc(value) for value in rows if value is not None]
    if len(present) < len(ASSETS):
        return None
    return min(present)
