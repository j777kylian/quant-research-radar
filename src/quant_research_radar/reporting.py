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
from .user_fit import (
    FIT_HIGH,
    FIT_OUT_OF_SCOPE,
    low_frequency_fit,
    parse_horizon_endpoints,
)

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

    snapshot = {
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
    _humanize(snapshot, session, daily, hypotheses, audit)
    return snapshot


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


# ---------------------------------------------------------------------------
# Human research brief (readable translation of machine state)
# ---------------------------------------------------------------------------

_ACADEMIC_PROVIDERS = {"openalex", "crossref", "arxiv", "nber", "repec"}
_PRACTITIONER_PROVIDERS = {"alpha-architect", "man-institute", "aqr"}


def _human_title(hypothesis: ChannelHypothesis) -> str:
    statement = (hypothesis.statement or "").strip().rstrip(".")
    if len(statement) > 8:
        return statement
    universe = hypothesis.universe or "market"
    return f"{universe} {hypothesis.outcome or 'research question'}".strip()


def _human_question(hypothesis: ChannelHypothesis) -> str:
    condition = (hypothesis.condition or "").lower()
    outcome = (hypothesis.outcome or "").lower()
    return f"Does {condition} change the {outcome}?"


def _humanize(
    snapshot: dict[str, Any],
    session: Session,
    daily: DailyRun,
    hypotheses: list[ChannelHypothesis],
    audit: dict[str, Any] | None,
) -> None:
    """Attach a readable brief layer; machine identities stay in the snapshot."""
    critic_reasons: dict[str, str] = {}
    for name, critic in ((daily.llm_summary or {}).get("critics") or {}).items():
        reason = critic.get("reason") if isinstance(critic, dict) else None
        if reason:
            critic_reasons[name] = str(reason)

    research = snapshot.get("research", {})
    new_families = research.get("new_families", [])
    audit_inputs = (audit or {}).get("fusion", {}).get("input_channels", [])

    findings: list[dict[str, Any]] = []
    family_tails = [family.split("|", 1)[-1] for family in new_families]

    def _matched_new(hypothesis: ChannelHypothesis) -> bool:
        return any(tail in (hypothesis.fingerprint or "") for tail in family_tails)

    # Findings are the NEW families (audit-canonical). For MARKET families we
    # show the concrete hypothesis; the single aggregated ACADEMIC family is
    # shown once with its count, not once per source item.
    new_market = [h for h in hypotheses if h.channel == "MARKET" and _matched_new(h)]
    new_academic = [
        h for h in hypotheses if h.channel == "ACADEMIC" and _matched_new(h)
    ]

    def _finding(hypothesis: ChannelHypothesis) -> dict[str, Any]:
        readable = {
            "title": _human_title(hypothesis),
            "question": _human_question(hypothesis),
            "universe": hypothesis.universe,
            "horizon": hypothesis.horizon,
            "horizon_endpoints": parse_horizon_endpoints(hypothesis.horizon),
            "channel": hypothesis.channel,
            "novelty": "NEW",
            "status": f"{hypothesis.maturity} / {hypothesis.status}",
            "disposition": research.get("technical_status", "n/a"),
            "reason": next(iter(critic_reasons.values()), None),
            "limitation": (
                "Single-channel support; the critics request independent "
                "evidence and methodological definition before any Event Study."
            ),
            "next": "Continue evidence accumulation; not yet recommended for an Event Study.",
        }
        return readable

    findings.extend(_finding(h) for h in new_market[:2])
    if new_academic:
        academic_family = new_families[0].split("|", 1)[-1] if new_families else ""
        findings.append(
            {
                "title": "New aggregated academic family on subsequent return / "
                "volatility distributions",
                "question": (
                    "Does a pre-specified measurable condition derived from retained "
                    "source evidence change the subsequent return or volatility "
                    "distribution? (aggregated family; see Research Inputs for the "
                    "underlying works)"
                ),
                "universe": "per-source universe (pre-specified)",
                "horizon": "not pre-specified",
                "horizon_endpoints": [],
                "channel": "ACADEMIC",
                "novelty": "NEW",
                "status": "H1_STATISTICAL_HYPOTHESIS / DISCOVERED (aggregated)",
                "disposition": research.get("technical_status", "n/a"),
                "reason": next(iter(critic_reasons.values()), None),
                "limitation": (
                    "Metadata-only evidence; no new paper body was retrieved today. "
                    "Aggregates the academic channel hypotheses into one family."
                ),
                "next": "Wait for new paper retrieval with archived content.",
                "family_tail": academic_family,
            }
        )

    # Research inputs: works retrieved in the window plus supporting items.
    start, end = _day_bounds(daily.logical_date)
    from .db import SourceItem

    def _items(providers: set[str]) -> list[dict[str, Any]]:
        rows = session.scalars(
            select(SourceItem)
            .where(SourceItem.retrieved_at >= start)
            .where(SourceItem.retrieved_at < end)
            .where(SourceItem.source_name.in_(sorted(providers)))
            .order_by(SourceItem.retrieved_at)
        ).all()
        items = []
        for item in rows[:5]:
            from .synthesis import evidence_depth

            depth = evidence_depth(item)
            summary = (
                "Only metadata is available; methodology and results cannot be reliably summarized."
                if depth == "METADATA_ONLY"
                else f"Persisted content depth: {depth}; summary is bounded to archived material."
            )
            items.append(
                {
                    "title": item.title,
                    "source": item.source_name,
                    "url": item.canonical_url,
                    "published_at": (
                        item.published_at.isoformat() if item.published_at else None
                    ),
                    "evidence_depth": depth,
                    "content_summary": summary,
                }
            )
        return items

    academic_items = _items(_ACADEMIC_PROVIDERS)
    practitioner_items = _items(_PRACTITIONER_PROVIDERS)

    # Market: structured funding summary over the window (never narrative).
    from .db import MarketObservation

    market_rows = session.scalars(
        select(MarketObservation)
        .where(MarketObservation.observed_at >= start)
        .where(MarketObservation.observed_at < end)
    ).all()
    funding: dict[str, list[float]] = {}
    for row in market_rows:
        if row.observation_kind != "funding" or row.funding_rate is None:
            continue
        funding.setdefault(row.asset, []).append(row.funding_rate)
    market_inputs: list[dict[str, Any]] = []
    for asset, rates in sorted(funding.items()):
        market_inputs.append(
            {
                "asset": asset,
                "hours": len(rates),
                "mean_funding": round(sum(rates) / len(rates), 8),
                "max_abs_funding": round(max(abs(r) for r in rates), 8),
            }
        )

    conclusion = (
        "No material market-state change triggered a high-priority research candidate."
        if not market_inputs
        else "Market inputs summarized from structured funding observations only."
    )

    snapshot["human"] = {
        "critic_reasons": critic_reasons,
        "inputs": {
            "academic": academic_items,
            "practitioner": practitioner_items,
            "market": market_inputs,
            "market_conclusion": conclusion,
        },
        "findings": findings,
        "findings_total": len(hypotheses),
        "audit_input_channels": audit_inputs,
    }


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
    human = summary.get("human", {})
    inputs = human.get("inputs", {})
    findings = human.get("findings", [])
    critic_reasons = human.get("critic_reasons", {})
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
        "## Today's Research Inputs",
        "",
        "### Academic",
    ]
    academic_inputs = inputs.get("academic", [])
    if academic_inputs:
        for item in academic_inputs:
            lines.append(f"- **{item.get('title')}**")
            lines.append(
                f"  - Source: {item.get('source')} | published: "
                f"{item.get('published_at') or 'n/a'} | {item.get('content_summary')}"
            )
            if item.get("url"):
                lines.append(f"  - Link: {item.get('url')}")
    else:
        lines.append("- No new academic works were retrieved today.")
    lines += ["", "### Practitioner"]
    practitioner_inputs = inputs.get("practitioner", [])
    if practitioner_inputs:
        for item in practitioner_inputs:
            lines.append(f"- **{item.get('title')}**")
            lines.append(
                f"  - Publisher: {item.get('source')} | published: "
                f"{item.get('published_at') or 'n/a'} | {item.get('content_summary')}"
            )
            if item.get("url"):
                lines.append(f"  - Link: {item.get('url')}")
    else:
        lines.append("- No new practitioner items were retrieved today.")
    lines += ["", "### Market"]
    market_inputs = inputs.get("market", [])
    if market_inputs:
        for item in market_inputs:
            lines.append(
                f"- **{item.get('asset')}** — {item.get('hours')}h of funding "
                f"observations; mean {item.get('mean_funding')}, "
                f"max |funding| {item.get('max_abs_funding')}."
            )
    else:
        lines.append(
            "- " + inputs.get("market_conclusion", "No notable market-state change.")
        )
    lines += [
        "",
        "## Topic Briefs",
    ]
    if findings:
        for finding in findings[:5]:
            lines.append(f"### {finding.get('title')}")
            lines.append(f"- **Research question:** {finding.get('question')}")
            lines.append(
                f"- **Today's evidence:** {finding.get('channel')} channel; "
                f"novelty {finding.get('novelty')} | scientific status {finding.get('status')}"
            )
            lines.append(
                "- **Method:** Persisted structured observation/hypothesis; no new empirical result is claimed here."
            )
            lines.append(
                "- **Result:** The hypothesis is retained for research; it is not evidence of causality or a trading signal."
            )
            endpoints = finding.get("horizon_endpoints") or []
            if endpoints:
                fit_text = ", ".join(
                    f"{e.get('value')}{e.get('unit')} → {e.get('fit')}"
                    for e in endpoints
                )
                lines.append(f"  - Horizon endpoints: {fit_text}")
            lines.append(f"  - Limitation: {finding.get('limitation')}")
            lines.append(f"  - Next: {finding.get('next')}")
    elif research.get("channel_hypotheses", 0):
        lines.append(
            "- No new family today; recurrent hypotheses remain under monitoring."
        )
    else:
        lines.append("- No important finding today (no new or recurrent hypotheses).")
    if critic_reasons:
        lines += ["", "## Critic Reasons (why the research gate asked for more data)"]
        for name, reason in critic_reasons.items():
            bounded = (
                reason if len(reason) <= 420 else reason[:420].rsplit(" ", 1)[0] + " …"
            )
            lines.append(f"- **{name.replace('_', ' ').title()}:** {bounded}")
    lines += [
        "",
        "## Negative / Inconclusive / Blocked Findings",
    ]
    blocked = research.get("channel_hypotheses", 0) - research.get("new_count", 0)
    if research.get("technical_status") == "CRITIC_REQUEST_DATA" or blocked > 0:
        lines.append(
            "- All retained hypotheses are blocked from promotion: critics request "
            "additional independent evidence or methodological definition "
            "(see Critic Reasons). No rejection was issued today."
        )
    else:
        lines.append("- None today.")
    lines += [
        "",
        "## Low-Frequency Fit",
        f"- Visible high-fit (1d-30d) hypotheses: {research.get('high_fit_count', 0)}",
        f"- Out-of-scope (<2h only) hypotheses retained scientifically: "
        f"{research.get('out_of_scope_count', 0)}",
        "",
        "## Knowledge Updates",
    ]
    knowledge = summary.get("knowledge", {})
    novelty = knowledge.get("prior_context_novelty", {})
    lines.append(
        "- Prior-context novelty counts: "
        + (", ".join(f"{k}={v}" for k, v in novelty.items()) if novelty else "n/a")
        + f"; new hypothesis families this run: {research.get('new_count', 0)}"
    )
    lines += [
        "",
        "## Daily Conclusion",
        _human_conclusion(summary),
        "",
        f"_Final report generated {utcnow().isoformat()} from persisted state._",
        "",
    ]
    return "\n".join(lines)


def _human_conclusion(summary: dict[str, Any]) -> str:
    research = summary.get("research", {})
    human = summary.get("human", {})
    findings = human.get("findings", [])
    new_count = research.get("new_count", 0)
    recurrent = research.get("recurrent_count", 0)
    high_fit = research.get("high_fit_count", 0)
    if not findings and new_count == 0 and recurrent == 0:
        return (
            "No useful research progress today: no new hypothesis families, no "
            "recurrence requiring attention, and no market-state change that "
            "triggered a candidate."
        )
    if findings:
        top = findings[0]
        return (
            f"Most important today: {top.get('title')} ({top.get('novelty')}). "
            f"{new_count} new and {recurrent} recurrent families; "
            f"{high_fit} visible 1d-30d candidate(s). "
            "Critics require more evidence or methodological definition before "
            "any Event Study is recommended; no candidate is test-ready yet."
        )
    return daily_conclusion(new_count, recurrent, high_fit)


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
