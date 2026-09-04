"""Publication + delivery orchestration downstream of Daily/Weekly runs.

Called only after the underlying research report is finalized. Delivery or
publication failure never degrades research status; retries are idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .config import Settings
from .db import (
    DailyRun,
    DailySocialEditorialPackage,
    DeliveryRecord,
    PublicationDraft,
    PublicationRecord,
    TopicBrief,
    WeeklyRun,
    WeeklySocialEditorialPackage,
)
from .delivery import (
    daily_digest_text,
    daily_discord_text,
    send_discord,
    send_email,
    weekly_digest_text,
)
from .publishing import (
    create_draft,
    render_effect_chart,
    select_daily_candidates,
    select_editorial_daily_candidate,
    select_weekly_candidates,
)
from .x_client import publish_draft, x_mode


def _persist_social_package_impl(
    session: Session,
    daily: DailyRun,
    candidates: list[Any],
    selected: Any | None,
    recommendation: str,
    reason: str,
    output_root: Path,
    draft_text: str | None = None,
) -> None:
    """Persist one deterministic editorial package; never touches science."""
    existing = session.scalar(
        select(DailySocialEditorialPackage).where(
            DailySocialEditorialPackage.logical_date == daily.logical_date
        )
    )
    payload = {
        "logical_date": daily.logical_date.isoformat(),
        "source_run_id": str(daily.id),
        "recommendation": recommendation,
        "selection_reason": reason,
        "selected_candidate_id": str(selected.id)
        if selected is not None and selected.id
        else None,
        "candidate_themes": [
            {"category": c.category, "title": c.title, "score": c.publication_value}
            for c in candidates
        ],
        "topic_brief_ids": [
            str(t.id)
            for t in session.scalars(
                select(TopicBrief).where(TopicBrief.source_run_id == str(daily.id))
            ).all()
        ],
    }
    package_dir = output_root / "social" / daily.logical_date.isoformat()
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / "summary.json"
    package_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if draft_text:
        (package_dir / "draft.md").write_text(draft_text, encoding="utf-8")
    if existing is None:
        existing = DailySocialEditorialPackage(
            logical_date=daily.logical_date,
            source_run_id=str(daily.id),
            topic_brief_ids=payload["topic_brief_ids"],
            candidates=payload["candidate_themes"],
            selected_candidate_id=payload["selected_candidate_id"],
            recommendation=recommendation,
            selection_reason=reason,
            content_format="THREAD" if len(candidates) > 3 else "SHORT_POST",
            draft_text=draft_text,
            source_bundle={},
            output_path=str(package_dir),
        )
        session.add(existing)
    else:
        existing.topic_brief_ids = cast(list[str], payload["topic_brief_ids"])
        existing.candidates = cast(list[dict[str, Any]], payload["candidate_themes"])
        existing.selected_candidate_id = cast(
            str | None, payload["selected_candidate_id"]
        )
        existing.recommendation = recommendation
        existing.selection_reason = reason
        existing.draft_text = draft_text
        existing.output_path = str(package_dir)
    session.commit()


def _persist_social_package(
    session: Session,
    daily: DailyRun,
    candidates: list[Any],
    selected: Any | None,
    recommendation: str,
    reason: str,
    output_root: Path,
    draft_text: str | None = None,
) -> bool:
    """Best-effort downstream persistence; never changes research outcome."""
    try:
        _persist_social_package_impl(
            session,
            daily,
            candidates,
            selected,
            recommendation,
            reason,
            output_root,
            draft_text,
        )
    except (OSError, SQLAlchemyError):
        session.rollback()
        return False
    return True


def _persist_weekly_social_package(
    session: Session,
    weekly: WeeklyRun,
    candidates: list[Any],
    recommendation: str,
    reason: str,
    output_root: Path,
    draft_text: str | None = None,
) -> None:
    package_dir = output_root / "social" / f"weekly-{weekly.week_saturday.isoformat()}"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "summary.json").write_text(
        json.dumps(
            {
                "week_saturday": weekly.week_saturday.isoformat(),
                "source_run_id": str(weekly.id),
                "recommendation": recommendation,
                "reason": reason,
                "candidates": [
                    {
                        "category": c.category,
                        "title": c.title,
                        "score": c.publication_value,
                    }
                    for c in candidates
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    existing = session.scalar(
        select(WeeklySocialEditorialPackage).where(
            WeeklySocialEditorialPackage.week_saturday == weekly.week_saturday
        )
    )
    values = dict(
        candidates=[
            {"category": c.category, "title": c.title, "score": c.publication_value}
            for c in candidates
        ],
        recommendation=recommendation,
        selection_reason=reason,
        draft_text=draft_text,
        output_path=str(package_dir),
    )
    if existing is None:
        session.add(
            WeeklySocialEditorialPackage(
                week_saturday=weekly.week_saturday,
                source_run_id=str(weekly.id),
                content_format="THREAD",
                **values,
            )
        )
    else:
        for key, value in values.items():
            setattr(existing, key, value)
    session.commit()


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
    # Daily Email must carry the Human Brief, not an operational summary: merge
    # the persisted final structured summary (research + human layers).
    from .reporting import collect_daily_snapshot

    if record is not None:
        snapshot = collect_daily_snapshot(session, record.id)
        if snapshot:
            summary.update(
                {
                    "high_fit_count": snapshot.get("research", {}).get(
                        "high_fit_count", 0
                    ),
                    "new_count": snapshot.get("research", {}).get("new_count", 0),
                    "recurrent_count": snapshot.get("research", {}).get(
                        "recurrent_count", 0
                    ),
                    "findings": snapshot.get("human", {}).get("findings", []),
                    "inputs": snapshot.get("human", {}).get("inputs", {}),
                    "conclusion": _daily_conclusion_text(snapshot),
                    "critic_reasons": snapshot.get("human", {}).get(
                        "critic_reasons", {}
                    ),
                }
            )
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
        content=daily_discord_text(summary),
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
        if record is not None and research_complete:
            _persist_social_package(
                session,
                record,
                [],
                None,
                "SKIP",
                "publication mode disabled",
                output_root,
            )
        return result
    assert record is not None
    from .synthesis import synthesize_daily_topics

    synthesize_daily_topics(session, record.id)
    candidates = select_daily_candidates(
        session, daily_run_id=daily_run_id, logical_date=logical_date
    )
    if not candidates:
        result["publication"] = {
            "status": "ZERO_POST",
            "reason": "no eligible candidate",
            "pool": [],
        }
        _persist_social_package(
            session,
            record,
            candidates,
            None,
            "SKIP",
            "No sufficiently informative, source-grounded, non-duplicative research story today.",
            output_root,
        )
        return result
    selected, editorial = select_editorial_daily_candidate(candidates)
    if selected is None:
        result["publication"] = {
            "status": "ZERO_POST",
            "reason": editorial.get("reason", "no publishable candidate"),
            "pool": [c.category for c in candidates],
        }
        _persist_social_package(
            session,
            record,
            candidates,
            None,
            "SKIP",
            editorial.get("reason", "no publishable candidate"),
            output_root,
        )
        return result
    empirical = dict(selected.evidence)
    # Study-shape gates (disposition + structured numbers) apply only to
    # event-study-backed candidates; paper/process candidates must not inherit
    # a forced INCONCLUSIVE or a numeric claim gate they cannot satisfy.
    has_study = bool(empirical.get("event_study_result_id"))
    if has_study:
        empirical.setdefault("disposition", "INCONCLUSIVE")
    structured = {"0": 0.0} if has_study else {}
    draft, rejection = create_draft(
        session,
        selected,
        empirical=empirical,
        structured_numbers=structured,
        language=settings.publication_language,
    )
    if draft is None:
        result["publication"] = {
            "status": "REJECTED",
            "reason": rejection,
            "category": selected.category,
        }
        _persist_social_package(
            session,
            record,
            candidates,
            selected,
            "SKIP",
            rejection or "copy verification failed",
            output_root,
        )
        return result
    visual_path: Path | None = None
    if has_study:
        visual_path = render_effect_chart(
            output_root / "visuals",
            structured_numbers=_visual_numbers(
                session, empirical.get("event_study_result_id")
            ),
            title="Extreme vs ordinary funding: 24h forward-return difference",
            sample_note="historical reconstructive sample",
        )
    draft.visual_ids = [visual_path.name] if visual_path is not None else []
    session.commit()
    mode = x_mode(settings)
    if mode == "AUTO_PUBLISH":
        x_result = publish_draft(
            draft,
            settings,
            research_run_complete=research_complete,
            already_published=_already_posted(session, draft),
            media_path=str(visual_path) if visual_path is not None else None,
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
        result["publication"] = {
            "status": "DRAFT_ONLY",
            "draft_id": str(draft.id),
            "category": selected.category,
            "editorial": editorial,
            "pool": [c.category for c in candidates],
        }
    _persist_social_package(
        session,
        record,
        candidates,
        selected,
        result["publication"]["status"],
        editorial.get("reason", "highest publication value"),
        output_root,
        draft.text,
    )
    return result


def _daily_conclusion_text(snapshot: dict[str, Any]) -> str:
    from .reporting import _human_conclusion

    try:
        return _human_conclusion(snapshot)
    except Exception:
        return ""


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
        if record is not None and research_complete:
            _persist_weekly_social_package(
                session, record, [], "SKIP", "publication mode disabled", output_root
            )
        return result
    weekly_candidates = select_weekly_candidates(
        session, weekly_run_id=weekly_run_id, week_saturday=week_saturday
    )
    for candidate in weekly_candidates:
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
    if record is not None:
        _persist_weekly_social_package(
            session,
            record,
            weekly_candidates,
            result["publication"].get("status", "SKIP"),
            "weekly research package generated",
            output_root,
        )
    return result


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
