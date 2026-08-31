from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import CollectionRun, MarketObservation, normalize_utc
from .llm import DeepSeekClient
from .pipeline import (
    analyze,
    calculate_metrics,
    daily_report,
    generate_market_observations,
    ingest,
    ingest_records,
)
from .replay import (
    ASSETS,
    METRIC_NAMES,
    market_quality,
    metric_availability,
    valuation_timestamp,
)
from .sources import ArxivSource, HyperliquidSource, RepecSource


def rendered_report_counts(report: str) -> dict[str, int]:
    return {
        "market_facts": report.count("**FACT:**"),
        "academic_claim_lines": report.count("- **CLAIM:**"),
        "hypotheses": report.count("- **HYPOTHESIS:**"),
    }


def _latest_persisted_completed_candle(session: Session) -> datetime | None:
    rows = session.execute(
        select(MarketObservation.asset, func.max(MarketObservation.observed_at))
        .where(
            MarketObservation.observation_kind == "candle",
            MarketObservation.asset.in_(ASSETS),
        )
        .group_by(MarketObservation.asset)
    ).all()
    present_assets = {asset for asset, _timestamp in rows}
    if not present_assets:
        return None
    missing_assets = set(ASSETS) - present_assets
    if missing_assets:
        raise RuntimeError(
            "LIVE_INCREMENTAL_COVERAGE_INCOMPLETE: missing candles for "
            + ", ".join(sorted(missing_assets))
        )
    return min(normalize_utc(timestamp) for _asset, timestamp in rows if timestamp)


def _market_gate(session: Session, as_of: datetime) -> tuple[str, list[str]]:
    valuation = valuation_timestamp(as_of)
    warmup_start = valuation - timedelta(days=get_settings().market_warmup_days)
    blockers: list[str] = []
    quality = market_quality(session, warmup_start, valuation)
    for asset in ASSETS:
        rows = session.scalars(
            select(MarketObservation).where(MarketObservation.asset == asset)
        ).all()
        funding = [
            row
            for row in rows
            if row.observation_kind == "funding"
            and normalize_utc(row.observed_at) <= valuation
        ]
        candles = [
            row
            for row in rows
            if row.observation_kind == "candle"
            and normalize_utc(row.observed_at) <= valuation
        ]
        if not funding:
            blockers.append(f"{asset}: no eligible funding history")
        if not candles:
            blockers.append(f"{asset}: no eligible completed candle")
        if quality[asset]["missing_expected_1h_intervals"]:
            blockers.append(f"{asset}: candle coverage gap")
    availability = metric_availability(session, as_of)
    for asset, values in availability.items():
        for name in METRIC_NAMES:
            if values[name] != "AVAILABLE":
                blockers.append(f"{asset}: {name} unavailable")
    return ("READY", []) if not blockers else ("BLOCKED", blockers)


def run_live_cycle(
    session: Session,
    client: DeepSeekClient,
    output_root: Path,
    cycle: int,
    code_sha: str,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    cycle_root = output_root / f"cycle-{cycle}"
    cycle_root.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    collection_end = valuation_timestamp(now)
    bootstrap_start = collection_end - timedelta(days=settings.market_warmup_days)
    statuses: dict[str, str] = {}
    counts: dict[str, int] = {}
    hyperliquid = HyperliquidSource()
    persisted_latest: datetime | None = None
    try:
        persisted_latest = _latest_persisted_completed_candle(session)
        if persisted_latest is not None:
            stale_by = collection_end - persisted_latest
            if stale_by > timedelta(hours=settings.live_incremental_max_hours):
                raise RuntimeError(
                    "LIVE_INCREMENTAL_HISTORY_TOO_STALE: "
                    f"{stale_by.total_seconds() / 3600:.1f}h"
                )
            bootstrap_start = persisted_latest - timedelta(
                hours=settings.live_bootstrap_overlap_hours
            )
        funding = hyperliquid.collect_history(
            max(1, int((collection_end - bootstrap_start).total_seconds() // 3600) + 1),
            start=bootstrap_start,
            end=collection_end,
        )
        candles = hyperliquid.collect_candles(
            max(1, int((collection_end - bootstrap_start).total_seconds() // 3600) + 1),
            start=bootstrap_start,
            end=collection_end,
        )
        collection_run = CollectionRun(
            source="hyperliquid",
            requested=len(funding) + len(candles),
            status="RUNNING",
            requested_start=bootstrap_start,
            requested_end=collection_end,
            code_sha=code_sha,
            diagnostics=hyperliquid.last_funding_diagnostics,
        )
        session.add(collection_run)
        session.flush()
        inserted_funding, _ = ingest_records(
            session, funding, collection_run_id=collection_run.id
        )
        inserted_candles, _ = ingest_records(
            session, candles, collection_run_id=collection_run.id
        )
        counts["hyperliquid_funding"] = inserted_funding
        counts["hyperliquid_candles"] = inserted_candles
        counts["hyperliquid"] = inserted_funding + inserted_candles
        statuses["hyperliquid_transport"] = "SUCCESS"
        statuses["market_data"] = "COLLECTED"
        collection_run.retrieved = len(funding) + len(candles)
        collection_run.inserted = inserted_funding + inserted_candles
        collection_run.status = "SUCCESS"
        collection_run.ended_at = datetime.now(UTC)
        session.commit()
        latest_receipt = session.scalar(
            select(MarketObservation.retrieved_at)
            .where(
                MarketObservation.observation_kind == "candle",
                MarketObservation.observed_at == collection_end,
                MarketObservation.source_name == "hyperliquid",
            )
            .order_by(MarketObservation.retrieved_at.desc())
            .limit(1)
        )
        if latest_receipt is not None:
            now = max(now, normalize_utc(latest_receipt))
    except Exception as exc:
        blocked = str(exc).startswith(
            (
                "LIVE_INCREMENTAL_HISTORY_TOO_STALE",
                "LIVE_INCREMENTAL_COVERAGE_INCOMPLETE",
            )
        )
        statuses["hyperliquid_transport"] = (
            "NOT_CALLED" if blocked else f"FAILED: {exc}"
        )
        statuses["market_data"] = "INSUFFICIENT_HISTORY" if blocked else "FAILED"
        audit = {
            "cycle": cycle,
            "as_of": now.isoformat(),
            "code_sha": code_sha,
            "database_identity": str(session.get_bind().engine.url),
            "source_status": statuses,
            "counts": counts,
            "hypotheses_generated": 0,
            "cycle_technical_status": "BLOCKED" if blocked else "FAILED",
            "blocker_reason": str(exc),
            "deepseek_call_status": "NOT_CALLED_DUE_TO_GATE",
            "no_lookahead": True,
        }
        (cycle_root / "audit.json").write_text(
            json.dumps(audit, indent=2, default=str), encoding="utf-8"
        )
        if blocked:
            raise RuntimeError("LIVE_CYCLE_STATUS=BLOCKED: " + str(exc)) from exc
        raise
    calculate_metrics(session, as_of=now)
    gate, blockers = _market_gate(session, now)
    statuses["market_data"] = "READY" if gate == "READY" else "INSUFFICIENT_HISTORY"
    if gate != "READY":
        audit = {
            "cycle": cycle,
            "as_of": now.isoformat(),
            "bootstrap_start": bootstrap_start.isoformat(),
            "collection_end": collection_end.isoformat(),
            "persisted_latest_completed_candle": (
                persisted_latest.isoformat() if persisted_latest is not None else None
            ),
            "latest_completed_candle": valuation_timestamp(now).isoformat(),
            "code_sha": code_sha,
            "database_identity": str(session.get_bind().engine.url),
            "source_status": statuses,
            "counts": counts,
            "market_quality": market_quality(session, bootstrap_start, collection_end),
            "metric_availability": metric_availability(session, now),
            "hypotheses_generated": 0,
            "deepseek_call_status": "NOT_CALLED_DUE_TO_GATE",
            "cycle_technical_status": "BLOCKED",
            "blocker_reason": blockers,
            "no_lookahead": True,
        }
        (cycle_root / "audit.json").write_text(
            json.dumps(audit, indent=2, default=str), encoding="utf-8"
        )
        raise RuntimeError("LIVE_CYCLE_STATUS=BLOCKED: " + "; ".join(blockers))
    market_observations = generate_market_observations(session, now)
    counts["market_observations"] = market_observations
    for name, adapter in (("arxiv", ArxivSource()), ("repec", RepecSource())):
        try:
            counts[name] = ingest(session, adapter, 10)
            statuses[name] = "SUCCESS" if counts[name] else "DEGRADED"
        except Exception as exc:
            statuses[name] = f"DEGRADED: {exc}"
            counts[name] = 0
    hypotheses = analyze(session, client, 20)
    report = daily_report(session, str(cycle_root), as_of=now)
    daily = cycle_root / "daily.md"
    daily.write_text(
        "# Daily Quant Radar — Live Smoke\n\n"
        f"LIVE CYCLE {cycle}\nAS_OF={now.isoformat()}\n\n"
        + report.read_text(encoding="utf-8").split("\n", 2)[-1],
        encoding="utf-8",
    )
    rendered = rendered_report_counts(daily.read_text(encoding="utf-8"))
    audit = {
        "cycle": cycle,
        "as_of": now.isoformat(),
        "bootstrap_start": bootstrap_start.isoformat(),
        "collection_end": collection_end.isoformat(),
        "persisted_latest_completed_candle": (
            persisted_latest.isoformat() if persisted_latest is not None else None
        ),
        "latest_completed_candle": collection_end.isoformat(),
        "code_sha": code_sha,
        "database_identity": str(session.get_bind().engine.url),
        "source_status": statuses,
        "counts": counts,
        "hypotheses_generated": hypotheses,
        "rendered_content": rendered,
        "market_quality": market_quality(session, bootstrap_start, collection_end),
        "metric_availability": metric_availability(session, now),
        "deepseek_call_status": "CALLED",
        "cycle_technical_status": "PASS",
        "no_lookahead": True,
        "report": str(daily),
    }
    (cycle_root / "audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    return audit


def write_live_summary(root: Path, audits: list[dict[str, Any]], code_sha: str) -> Path:
    summary = {
        "phase": "1.6B",
        "code_sha": code_sha,
        "database_identity": audits[0]["database_identity"] if audits else None,
        "cycles": audits,
        "cross_cycle_state": "SAME_ISOLATED_DATABASE",
        "report_generation": all(bool(audit.get("report")) for audit in audits),
        "llm_telemetry_limitation": True,
    }
    path = root / "phase16b-summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def write_live_review(root: Path) -> Path:
    path = root / "human-review.md"
    path.write_text(
        "# Phase 1.6B Human Review\n\n"
        "## Cycle 1\n"
        "Usefulness: 0 / 1 / 2\n"
        "Best market observation:\nBest hypothesis:\nEvidence quality:\n"
        "Potential hallucination:\nGeneric/repetitive content:\nUseful quant concept:\n"
        "Would this have surfaced something I otherwise might not investigate?\nNotes:\n\n"
        "## Cycle 2\n"
        "Usefulness: 0 / 1 / 2\n"
        "Best market observation:\nBest hypothesis:\nEvidence quality:\n"
        "Potential hallucination:\nGeneric/repetitive content:\nUseful quant concept:\n"
        "Would this have surfaced something I otherwise might not investigate?\nNotes:\n\n"
        "## Aggregate\n"
        "Average usefulness:\nRepeated hypotheses:\nMost promising research theme:\n"
        "Main weakness:\nWould I read this tomorrow?\n"
        "Proceed to 5–7 day observation: YES / NO / UNSURE\n",
        encoding="utf-8",
    )
    return path
