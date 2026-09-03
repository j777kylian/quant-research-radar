"""Final-state Daily/Weekly report rendering from persisted state only.

Separation of concerns:
  * run_daily/run_weekly persist the FINAL run state first;
  * reporting builds a structured summary from persisted rows/artifacts;
  * the human report is rendered from that summary (never from transient state).

Regeneration reads the DB + intelligence audit.json and performs NO network or
LLM calls and mutates no scientific state.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (
    ChannelHypothesis,
    CollectionRun,
    DailyRun,
    EventStudyResultRecord,
    WeeklyRun,
    utcnow,
)
from .scheduler import BEIJING_TZ
from .user_fit import FIT_HIGH, FIT_OUT_OF_SCOPE, low_frequency_fit

SUCCESS = "SUCCESS"
DEGRADED = "DEGRADED"
FAILED = "FAILED"
PARTIAL = "PARTIAL"
SKIPPED = "SKIPPED"


def _day_bounds(logical_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(logical_date, time.min, tzinfo=BEIJING_TZ)
    return start, start + timedelta(days=1)


def _find_market_run(session: Session, daily: DailyRun) -> CollectionRun | None:
    """The hyperliquid production CollectionRun executed inside this Daily."""
    return session.scalars(
        select(CollectionRun)
        .where(CollectionRun.source == "hyperliquid")
        .where(CollectionRun.started_at >= daily.started_at - timedelta(minutes=5))
        .where(
            CollectionRun.started_at
            <= (daily.ended_at or daily.started_at) + timedelta(minutes=5)
        )
        .order_by(CollectionRun.started_at)
        .limit(1)
    ).first()


def _intelligence_audit(
    daily: DailyRun, intelligence_root: Path | None = None
) -> dict[str, Any] | None:
    if intelligence_root is not None:
        path = intelligence_root / "audit.json"
    elif daily.report_path:
        root = Path(daily.report_path).resolve().parent
        path = root / "intelligence" / "audit.json"
    else:
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _hypotheses_for_day(
    session: Session, logical_date: date
) -> list[ChannelHypothesis]:
    start, end = _day_bounds(logical_date)
    return list(
        session.scalars(
            select(ChannelHypothesis)
            .where(ChannelHypothesis.as_of >= start)
            .where(ChannelHypothesis.as_of < end)
            .order_by(ChannelHypothesis.created_at)
        ).all()
    )


def _prior_empirical(session: Session, family: str) -> dict[str, Any] | None:
    row = session.scalars(
        select(EventStudyResultRecord)
        .where(EventStudyResultRecord.hypothesis_family_id == family)
        .order_by(EventStudyResultRecord.created_at.desc())
        .limit(1)
    ).first()
    if row is None:
        return None
    return {
        "disposition": row.disposition,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def collect_daily_snapshot(
    session: Session, daily_run_id: Any, intelligence_root: Path | None = None
) -> dict[str, Any] | None:
    daily = session.get(DailyRun, daily_run_id)
    if daily is None:
        return None
    market_run = _find_market_run(session, daily)
    audit = _intelligence_audit(daily, intelligence_root)
    hypotheses = _hypotheses_for_day(session, daily.logical_date)

    market: dict[str, Any] = {"status": daily.market_status}
    if market_run is not None:
        market.update(
            {
                "start": (
                    market_run.requested_start.isoformat()
                    if market_run.requested_start
                    else None
                ),
                "end": (
                    market_run.requested_end.isoformat()
                    if market_run.requested_end
                    else None
                ),
                "retrieved": market_run.retrieved,
                "inserted": market_run.inserted,
                "duplicates": market_run.skipped_duplicates,
                "analysis_mode": (market_run.diagnostics or {}).get(
                    "analysis_mode", "PRODUCTION_LIVE"
                ),
                "watermark": daily.source_health.get("market", {}).get("end"),
            }
        )
    elif daily.source_health.get("market"):
        market.update(daily.source_health["market"])

    academic = daily.source_health.get("academic", {})
    practitioner = daily.source_health.get("practitioner", {})

    # Recurrence is canonical in the intelligence audit (novelty vs PRIOR
    # knowledge across days). Within-window fingerprint grouping is only a
    # fallback when the audit artifact is unavailable.
    audit_new_families = list((audit or {}).get("new_hypothesis_families") or [])
    novelty_counts: dict[str, int] = {}
    if audit is not None:
        for entry in audit.get("knowledge", {}).get("prior_context", []):
            key = str(entry.get("novelty", "UNKNOWN"))
            novelty_counts[key] = novelty_counts.get(key, 0) + 1
    if audit_new_families:
        new_family_list = audit_new_families
        recurrent_count = novelty_counts.get("RECURRENT", 0)
        new_count = novelty_counts.get("NEW", 0)
        all_families = new_family_list
    else:
        # Fallback: no audit artifact — families seen once within the day are
        # reported as new (recurrence vs prior days is then not recoverable).
        within_day: dict[str, list[ChannelHypothesis]] = {}
        for hypothesis in hypotheses:
            within_day.setdefault(hypothesis.fingerprint, []).append(hypothesis)
        new_family_list = [fp for fp, rows in within_day.items() if len(rows) == 1]
        recurrent_count = len([fp for fp, rows in within_day.items() if len(rows) > 1])
        new_count = len(new_family_list)
        all_families = list(within_day)

    knowledge: dict[str, Any] = {"prior_context_novelty": novelty_counts}

    low_fit = [h for h in hypotheses if low_frequency_fit(h.horizon) == FIT_HIGH]
    out_of_scope = [
        h for h in hypotheses if low_frequency_fit(h.horizon) == FIT_OUT_OF_SCOPE
    ]

    return {
        "logical_date": daily.logical_date.isoformat(),
        "final_status": daily.status,
        "component_statuses": {
            "market": daily.market_status,
            "academic": daily.academic_status,
            "practitioner": daily.practitioner_status,
            "analysis": daily.analysis_status,
            "knowledge": daily.knowledge_status,
            "audit": daily.audit_status,
        },
        "code_sha": daily.code_sha,
        "started_at": daily.started_at.isoformat() if daily.started_at else None,
        "ended_at": daily.ended_at.isoformat() if daily.ended_at else None,
        "market": market,
        "academic_sources": academic,
        "practitioner_sources": practitioner,
        "research": {
            "channel_hypotheses": len(hypotheses),
            "new_families": new_family_list,
            "new_count": new_count,
            "recurrent_count": recurrent_count,
            "high_fit_count": len(low_fit),
            "out_of_scope_count": len(out_of_scope),
            "technical_status": (audit or {}).get("technical_status"),
            "critics": {
                k: v.get("disposition")
                for k, v in ((audit or {}).get("critics") or {}).items()
            },
            "new_hypothesis_families_audit": (audit or {}).get(
                "new_hypothesis_families"
            ),
        },
        "knowledge": knowledge,
        "failure_reasons": list(daily.failure_reasons or []),
        "report_path": daily.report_path,
        "prior_empirical": prior_empirical_for_families(session, all_families),
    }


def prior_empirical_for_families(
    session: Session, families: list[str]
) -> list[dict[str, Any]]:
    out = []
    for fp in families:
        prior = _prior_empirical(session, fp)
        if prior is not None:
            prior["family"] = fp
            out.append(prior)
    return out


def daily_conclusion(new_count: int, recurrent_count: int, high_fit_count: int) -> str:
    if high_fit_count == 0 and new_count == 0 and recurrent_count == 0:
        return "No high-priority low-frequency research candidate emerged today."
    parts = []
    if new_count:
        parts.append(
            f"{new_count} new hypothesis famil{'y' if new_count == 1 else 'ies'}"
        )
    if recurrent_count:
        parts.append(
            f"{recurrent_count} recurrent famil{'y' if recurrent_count == 1 else 'ies'}"
        )
    if high_fit_count:
        parts.append(
            f"{high_fit_count} high-fit (1d-30d) candidate{'s' if high_fit_count != 1 else ''}"
        )
    return (
        "Today's research: "
        + "; ".join(parts)
        + ". In-sample evidence only; no trading signal."
    )


def render_daily_markdown(summary: dict[str, Any]) -> str:
    comp = summary.get("component_statuses", {})
    market = summary.get("market", {})
    research = summary.get("research", {})
    lines = [
        "# Daily Quant Research Radar",
        "",
        "## Date / Run",
        f"- **Logical Beijing date:** {summary.get('logical_date')}",
        f"- **Final run status:** {summary.get('final_status')}",
        f"- **Started / ended:** {summary.get('started_at')} / {summary.get('ended_at')}",
        f"- **Code SHA:** {summary.get('code_sha')}",
        "",
        "## System Health",
        f"- Market: {comp.get('market')} | Academic: {comp.get('academic')} | "
        f"Practitioner: {comp.get('practitioner')} | Analysis: {comp.get('analysis')} | "
        f"Knowledge: {comp.get('knowledge')} | Audit: {comp.get('audit')}",
        "",
        "## Market Collection",
        f"- **Status:** {market.get('status')}",
        f"- **Interval:** {market.get('start', 'n/a')} → {market.get('end', 'n/a')}",
        f"- **Retrieved / inserted / duplicates:** {market.get('retrieved', 0)} / "
        f"{market.get('inserted', 0)} / {market.get('duplicates', 0)}",
        f"- **Mode:** {market.get('analysis_mode', 'n/a')}",
    ]
    if summary.get("failure_reasons"):
        lines += ["", "## Critical Errors", ""] + [
            f"- {reason}" for reason in summary["failure_reasons"]
        ]
    lines += [
        "",
        "## Research Collection",
        "- **Academic:** " + _source_line(summary.get("academic_sources", {})),
        "- **Practitioner:** " + _source_line(summary.get("practitioner_sources", {})),
        "",
        "## Research Intelligence",
        f"- Channel hypotheses retained: {research.get('channel_hypotheses', 0)}",
        f"- New hypothesis families: {research.get('new_count', len(research.get('new_families', [])))}",
        f"- Recurrent families: {research.get('recurrent_count', 0)}",
        f"- Technical status: {research.get('technical_status', 'n/a')}",
        f"- Critics: {_critic_line(research.get('critics', {}))}",
        "",
        "## Today's Important Findings",
    ]
    new_families = research.get("new_families", [])
    new_count = research.get("new_count", len(new_families))
    if new_families:
        for family in new_families[:3]:
            lines.append(f"- New: `{family}`")
        if new_count == 0 and research.get("recurrent_count", 0):
            lines.append(
                f"- Recurrent this run (prior knowledge): {research.get('recurrent_count', 0)} hypotheses."
            )
    elif research.get("channel_hypotheses", 0):
        lines.append("- Recurrence/evidence accumulation only; no new family today.")
    else:
        lines.append("- No important finding today (no new or recurrent hypotheses).")
    lines += [
        "",
        "## Low-Frequency Fit",
        f"- High-fit (1d-30d) candidates: {research.get('high_fit_count', 0)}",
        f"- Out-of-scope (<2h) hypotheses retained scientifically: {research.get('out_of_scope_count', 0)}",
        "",
        "## Knowledge Updates",
    ]
    knowledge = summary.get("knowledge", {})
    novelty = knowledge.get("prior_context_novelty", {})
    lines.append(
        "- Prior-context novelty counts: "
        + (", ".join(f"{k}={v}" for k, v in novelty.items()) if novelty else "n/a")
        + f"; new hypothesis families this run: {new_count}"
    )
    lines += [
        "",
        "## Daily Conclusion",
        daily_conclusion(
            new_count,
            research.get("recurrent_count", 0),
            research.get("high_fit_count", 0),
        ),
        "",
        f"_Final report generated {utcnow().isoformat()} from persisted state._",
        "",
    ]
    return "\n".join(lines)


def _source_line(statuses: dict[str, Any]) -> str:
    if not statuses:
        return "per-source run outcomes not persisted for this run (pre-fix); aggregate status shown above"
    return ", ".join(f"{name}={value}" for name, value in sorted(statuses.items()))


def _critic_line(critics: dict[str, Any]) -> str:
    if not critics:
        return "n/a"
    return ", ".join(f"{k}={v}" for k, v in sorted(critics.items()))


# ---------------------------------------------------------------------------
# Weekly
# ---------------------------------------------------------------------------


def collect_weekly_snapshot(
    session: Session, weekly_run_id: Any
) -> dict[str, Any] | None:
    weekly = session.get(WeeklyRun, weekly_run_id)
    if weekly is None:
        return None
    week_start = weekly.week_saturday - timedelta(days=6)
    dailies = list(
        session.scalars(
            select(DailyRun)
            .where(DailyRun.logical_date >= week_start)
            .where(DailyRun.logical_date <= weekly.week_saturday)
            .order_by(DailyRun.logical_date)
        ).all()
    )
    return {
        "week_saturday": weekly.week_saturday.isoformat(),
        "final_status": weekly.status,
        "included_daily_dates": list(weekly.included_daily_dates or []),
        "daily_statuses": {d.logical_date.isoformat(): d.status for d in dailies},
        "hypotheses_count": len(weekly.priorities or []),
        "priorities": list(weekly.priorities or []),
        "low_frequency_fit": weekly.low_frequency_fit or {},
        "failure_reasons": list(weekly.failure_reasons or []),
        "report_path": weekly.report_path,
    }


def render_weekly_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Weekly Quant Research Review",
        "",
        "## Week",
        f"- **Week ending (Beijing Saturday):** {summary.get('week_saturday')}",
        f"- **Final Weekly status:** {summary.get('final_status')}",
        f"- **Included Daily runs:** {', '.join(summary.get('included_daily_dates', [])) or 'none'}",
    ]
    missing = [
        d
        for d in summary.get("included_daily_dates", [])
        if summary.get("daily_statuses", {}).get(d) not in (None, "SUCCESS")
    ]
    if missing:
        lines.append(f"- **Degraded/missing Daily runs:** {', '.join(missing)}")
    lines += ["", "## System / Source Health", ""]
    statuses = summary.get("daily_statuses", {})
    lines.append(
        "- Daily statuses: "
        + (
            ", ".join(f"{d}={v}" for d, v in sorted(statuses.items()))
            if statuses
            else "n/a"
        )
    )
    priorities = summary.get("priorities", [])[:5]
    lines += ["", "## Top Research Priorities", ""]
    if not priorities:
        lines.append(
            "- No high-priority candidates this week (valid zero-priority week)."
        )
    for index, item in enumerate(priorities, start=1):
        lines.append(
            f"{index}. **{item.get('fit', '?')}** — {str(item.get('statement', ''))[:180]}"
        )
        lines.append(
            f"   - family: `{item.get('hypothesis_family')}` | horizon: {item.get('horizon')} | "
            f"maturity: {item.get('maturity')} | prior empirical: {item.get('prior_empirical_disposition')}"
        )
    fit = summary.get("low_frequency_fit", {})
    lines += [
        "",
        "## Low-Frequency Research Fit",
        f"- High-fit (1d-30d): {len(fit.get('high_fit', []))} | "
        f"Out-of-scope retained: {len(fit.get('out_of_scope', []))}",
        "",
        "## Important Negative / Rejected Findings",
        "- Review preserves rejections and negative empirical results as prior context only.",
        "",
        "## Recommended Next Research Actions",
    ]
    if not priorities:
        lines.append(
            "- Continue evidence accumulation; no Event Study recommended this week."
        )
    else:
        lines.append(
            "- Candidate Event Studies are listed as recommendations only; never auto-run."
        )
    lines += [
        "",
        f"_Final report generated {utcnow().isoformat()} from persisted state._",
        "",
    ]
    return "\n".join(lines)
