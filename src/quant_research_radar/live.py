from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from .llm import DeepSeekClient
from .pipeline import analyze, calculate_metrics, daily_report, ingest
from .replay import market_quality, metric_availability
from .sources import ArxivSource, HyperliquidSource, RepecSource


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
    statuses: dict[str, str] = {}
    counts: dict[str, int] = {}
    hyperliquid = HyperliquidSource()
    try:
        counts["hyperliquid"] = ingest(session, hyperliquid, 30)
        statuses["hyperliquid"] = "SUCCESS"
    except Exception as exc:
        statuses["hyperliquid"] = f"FAILED: {exc}"
        raise
    for name, adapter in (("arxiv", ArxivSource()), ("repec", RepecSource())):
        try:
            counts[name] = ingest(session, adapter, 10)
            statuses[name] = "SUCCESS" if counts[name] else "DEGRADED"
        except Exception as exc:
            statuses[name] = f"DEGRADED: {exc}"
            counts[name] = 0
    calculate_metrics(session)
    hypotheses = analyze(session, client, 20)
    report = daily_report(session, str(cycle_root), as_of=now)
    daily = cycle_root / "daily.md"
    daily.write_text(
        "# Daily Quant Radar — Live Smoke\n\n"
        f"LIVE CYCLE {cycle}\nAS_OF={now.isoformat()}\n\n"
        + report.read_text(encoding="utf-8").split("\n", 2)[-1],
        encoding="utf-8",
    )
    audit = {
        "cycle": cycle,
        "as_of": now.isoformat(),
        "code_sha": code_sha,
        "database_identity": str(session.get_bind().engine.url),
        "source_status": statuses,
        "counts": counts,
        "hypotheses_generated": hypotheses,
        "market_quality": market_quality(
            session, now.replace(hour=0, minute=0, second=0, microsecond=0), now
        ),
        "metric_availability": metric_availability(session, now),
        "no_lookahead": True,
        "llm": {
            "provider": client.provider,
            "model": client.model,
            "telemetry_limitation": True,
        },
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
        "## Cycle 1\nUsefulness: 0 / 1 / 2\n\n"
        "## Cycle 2\nUsefulness: 0 / 1 / 2\n\n"
        "New observation I would otherwise have missed:\n\nBest hypothesis:\n\n"
        "Suspected hallucination:\n\nRepeated/generic content:\n\nOperational issues:\n\n"
        "Would I want to read this tomorrow?\n",
        encoding="utf-8",
    )
    return path
