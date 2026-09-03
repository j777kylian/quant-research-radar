"""Publication + delivery orchestration downstream of Daily/Weekly runs.

Called only after the underlying research report is finalized. Delivery or
publication failure never degrades research status; retries are idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .db import (
    DailyRun,
    DeliveryRecord,
    PublicationDraft,
    PublicationRecord,
    WeeklyRun,
)
from .delivery import daily_digest_text, send_discord, send_email, weekly_digest_text
from .publishing import (
    create_draft,
    find_recent_duplicate,
    render_effect_chart,
    select_daily_candidates,
    select_weekly_candidates,
)
from .x_client import publish_draft, x_mode


def after_daily(
    session: Session,
    settings: Settings,
    *,
    daily_run_id: str,
    logical_date: str,
    market_summary: dict[str, Any],
    report_path: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    """Private delivery + public publication evaluation for a completed Daily."""
    result: dict[str, Any] = {"delivery": {}, "publication": {}}

    record = session.get(DailyRun, __import__("uuid").UUID(daily_run_id))
    research_complete = record is not None and record.status == "SUCCESS"
    summary = {
        "logical_date": logical_date,
        "status": "SUCCESS" if research_complete else "DEGRADED",
        "market": market_summary,
        "failure_reasons": list(record.failure_reasons) if record else [],
        "high_fit_count": 0,
    }
    email_result = send_email(
        session,
        settings,
        run_kind="DAILY",
        run_date=logical_date,
        subject=f"Quant Radar Daily {logical_date}",
        body_text=daily_digest_text(summary, str(report_path) if report_path else None),
    )
    discord_result = send_discord(
        session,
        settings,
        run_kind="DAILY",
        run_date=logical_date,
        content=daily_digest_text(summary, str(report_path) if report_path else None),
    )
    result["delivery"] = {
        "email": email_result.status,
        "discord": discord_result.status,
        "research_status_unchanged": research_complete,
    }

    if settings.publication_mode == "DISABLED" or not research_complete:
        result["publication"] = {
            "status": "SKIPPED",
            "reason": "mode disabled or research incomplete",
        }
        return result
    candidates = select_daily_candidates(
        session, daily_run_id=daily_run_id, logical_date=logical_date
    )
    if not candidates:
        result["publication"] = {
            "status": "ZERO_POST",
            "reason": "no eligible candidate",
        }
        return result
    drafts = []
    for candidate in candidates:
        empirical = dict(candidate.evidence)
        empirical.setdefault("disposition", "INCONCLUSIVE")
        structured = {"0": 0.0}  # deterministic copy uses no invented numbers
        draft, rejection = create_draft(
            session,
            candidate,
            empirical=empirical,
            structured_numbers=structured,
            language=settings.publication_language,
        )
        if draft is None:
            result["publication"] = {"status": "REJECTED", "reason": rejection}
            continue
        if (
            find_recent_duplicate(session, draft.text) is not None
            and draft.id != _draft_by_key(session, draft.idempotence_key).id
        ):
            result["publication"] = {"status": "SKIP_DUPLICATE_TOPIC"}
            continue
        visual_path = render_effect_chart(
            output_root / "visuals",
            structured_numbers=_visual_numbers(
                session, empirical.get("event_study_result_id")
            ),
            title="Extreme vs ordinary funding: 24h forward-return difference",
            sample_note="historical reconstructive sample",
        )
        draft.visual_ids = [visual_path.name]
        session.commit()
        mode = x_mode(settings)
        if mode == "AUTO_PUBLISH":
            x_result = publish_draft(
                draft,
                settings,
                research_run_complete=research_complete,
                already_published=_already_posted(session, draft),
                media_path=str(visual_path),
            )
            session.add(
                PublicationRecord(
                    draft_id=draft.id,
                    platform="X",
                    status=x_result.status,
                    external_post_id=x_result.external_post_id,
                    failure_reason=x_result.reason,
                )
            )
            session.commit()
            result["publication"] = {
                "status": x_result.status,
                "reason": x_result.reason,
            }
        else:
            result["publication"] = {"status": "DRAFT_ONLY", "draft_id": str(draft.id)}
        drafts.append(str(draft.id))
    return result


def after_weekly(
    session: Session,
    settings: Settings,
    *,
    weekly_run_id: str,
    week_saturday: str,
    report_path: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {"delivery": {}, "publication": {}}
    record = session.get(WeeklyRun, __import__("uuid").UUID(weekly_run_id))
    research_complete = record is not None and record.status == "SUCCESS"
    summary = {
        "week_saturday": week_saturday,
        "status": "SUCCESS" if research_complete else "DEGRADED",
        "included_daily_dates": list(record.included_daily_dates) if record else [],
        "hypothesis_count": len(record.priorities) if record else 0,
        "priorities": list(record.priorities) if record else [],
        "failure_reasons": list(record.failure_reasons) if record else [],
    }
    email_result = send_email(
        session,
        settings,
        run_kind="WEEKLY",
        run_date=week_saturday,
        subject=f"Quant Radar Weekly {week_saturday}",
        body_text=weekly_digest_text(
            summary, str(report_path) if report_path else None
        ),
    )
    discord_result = send_discord(
        session,
        settings,
        run_kind="WEEKLY",
        run_date=week_saturday,
        content=weekly_digest_text(summary, str(report_path) if report_path else None)[
            :900
        ],
    )
    result["delivery"] = {
        "email": email_result.status,
        "discord": discord_result.status,
    }

    if settings.publication_mode == "DISABLED" or not research_complete:
        result["publication"] = {"status": "SKIPPED"}
        return result
    for candidate in select_weekly_candidates(
        session, weekly_run_id=weekly_run_id, week_saturday=week_saturday
    ):
        draft, rejection = create_draft(
            session,
            candidate,
            empirical=None,
            structured_numbers={"0": 0.0},
            language=settings.publication_language,
        )
        if draft is None:
            result["publication"] = {"status": "REJECTED", "reason": rejection}
            continue
        result["publication"] = {
            "status": "DRAFT_ONLY" if x_mode(settings) != "AUTO_PUBLISH" else "GATED",
            "draft_id": str(draft.id),
        }
    return result


def _draft_by_key(session: Session, key: str) -> PublicationDraft:
    return session.scalars(
        select(PublicationDraft).where(PublicationDraft.idempotence_key == key)
    ).one()


def _already_posted(session: Session, draft: PublicationDraft) -> bool:
    """True if an accepted X post already exists for this draft (crash-safe)."""
    return (
        session.scalar(
            select(PublicationRecord)
            .where(PublicationRecord.draft_id == draft.id)
            .where(PublicationRecord.platform == "X")
            .where(PublicationRecord.status == "PUBLISHED")
        )
        is not None
    )


def _visual_numbers(session: Session, event_study_result_id: Any) -> dict[str, float]:
    """Chart values come only from structured result records; missing → zeros."""
    from uuid import UUID

    from .db import EventStudyResultRecord

    if event_study_result_id is None:
        return {"treatment": 0.0, "ordinary": 0.0}
    key = (
        event_study_result_id
        if isinstance(event_study_result_id, UUID)
        else UUID(str(event_study_result_id))
    )
    result = session.get(EventStudyResultRecord, key)
    if result is None:
        return {"treatment": 0.0, "ordinary": 0.0}
    pooled = (result.effects or {}).get("observation", {}).get("POOLED:24h", {})
    treatment = float((pooled.get("treatment") or {}).get("mean") or 0.0)
    baseline = float((pooled.get("baseline") or {}).get("mean") or 0.0)
    return {"treatment": treatment, "ordinary": baseline}


def publication_status(session: Session) -> dict[str, Any]:
    drafts = session.scalars(select(PublicationDraft)).all()
    publications = session.scalars(select(PublicationRecord)).all()
    deliveries = session.scalars(select(DeliveryRecord)).all()
    last_failure = next(
        (p.failure_reason for p in reversed(publications) if p.failure_reason), None
    )
    return {
        "pending_draft_count": sum(
            1
            for d in drafts
            if d.id not in {p.draft_id for p in publications if p.status == "PUBLISHED"}
        ),
        "published_count": sum(1 for p in publications if p.status == "PUBLISHED"),
        "last_publication_failure": last_failure,
        "last_email": next(
            (d.status for d in reversed(deliveries) if d.channel == "EMAIL"), None
        ),
        "last_discord": next(
            (d.status for d in reversed(deliveries) if d.channel == "DISCORD"), None
        ),
    }
