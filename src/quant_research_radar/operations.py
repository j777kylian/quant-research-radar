"""Autonomous Daily/Weekly operations for the low-frequency Quant Research Radar.

These are the two canonical unattended entry points. They reuse the existing
Phase 1.8 production intelligence path, the bounded market collector, and the
Phase 1.6D discovery/archive pipeline; they do not implement a second engine.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import (
    ChannelHypothesis,
    CollectionRun,
    DailyRun,
    EventStudyResultRecord,
    WeeklyRun,
    make_engine,
    make_session_factory,
    normalize_utc,
    utcnow,
)
from .discovery import ingest_records as ingest_discovery_records
from .intelligence_v2 import run_phase18_intelligence_cycle
from .llm import LLMClient, build_deepseek_client
from .market_operations import (
    latest_production_candle,
    run_live_market_collection,
    safe_complete_hour,
)
from .raw_archive import RawArchive
from .scheduler import beijing_date, beijing_now
from .sources import (
    ArxivSource,
    CrossrefSource,
    InstitutionalHtmlSource,
    InstitutionalRssSource,
    NberSource,
    OpenAlexSource,
    PractitionerRssSource,
    RepecSource,
    collect_isolated,
    source_registry,
)
from .user_fit import (
    FIT_HIGH,
    FIT_MEDIUM,
    FIT_OUT_OF_SCOPE,
    low_frequency_fit,
)

ACADEMIC_SOURCES = ("openalex", "crossref", "arxiv", "nber", "repec")
PRACTITIONER_SOURCES = ("alpha-architect", "man-institute", "aqr")
DAILY_OVERLAP_HOURS = 1


@dataclass(frozen=True)
class ComponentStatus:
    status: str
    detail: str = ""


class OperationsLock:
    """Cross-process directory lock with safe stale recovery.

    Shared by Daily, Weekly, manual market collection, and the historical
    backfill so they cannot write conflicting state concurrently.
    """

    def __init__(self, root: Path) -> None:
        self.dir = root / ".operations.lock"
        self.owner = self.dir / "owner"

    def acquire(self) -> bool:
        for _ in range(2):
            try:
                self.dir.mkdir(parents=True, exist_ok=False)
                self.owner.write_text(f"{os.getpid()} {utcnow().isoformat()}")
                return True
            except FileExistsError:
                if self._stale():
                    self._clear()
                    continue
                return False
        return False

    def release(self) -> None:
        try:
            self.owner.unlink(missing_ok=True)
            self.dir.rmdir()
        except OSError:
            pass

    def _stale(self) -> bool:
        try:
            pid_text, _ = self.owner.read_text().split(maxsplit=1)
            pid = int(pid_text)
            try:
                os.kill(pid, 0)
            except OSError:
                return True  # owner process is gone
        except (ValueError, OSError, FileNotFoundError):
            return True
        return False

    def _clear(self) -> None:
        self.owner.unlink(missing_ok=True)
        try:
            self.dir.rmdir()
        except OSError:
            pass

    def __enter__(self) -> OperationsLock:
        if not self.acquire():
            raise RuntimeError("operations lock is held by another process")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def build_client(settings: Settings) -> LLMClient | None:
    """Production DeepSeek client when a credential is configured, else None."""
    if not settings.deepseek_api_key:
        return None
    return build_deepseek_client(
        settings.deepseek_api_key,
        settings.llm_flash_model,
        settings.llm_pro_model,
        settings.deepseek_base_url,
        settings.llm_timeout_seconds,
        settings.http_retries,
    )


def _research_adapters(settings: Settings) -> tuple[list[Any], list[Any]]:
    retrieved_at = utcnow()
    academic = [
        OpenAlexSource(now=lambda: retrieved_at),
        CrossrefSource(),
        ArxivSource(lookback_days=settings.arxiv_lookback_days),
        NberSource(),
        RepecSource(),
    ]
    practitioner = [
        PractitionerRssSource(),  # alpha-architect
        InstitutionalHtmlSource(
            name="man-institute", endpoint="https://www.man.com/insights"
        ),
        InstitutionalRssSource(
            name="aqr", endpoint="https://www.aqr.com/Insights/Research"
        ),
    ]
    return academic, practitioner


def _collect_research(
    session: Any,
    adapters: list[Any],
    *,
    limit: int,
    archive: RawArchive,
    code_sha: str,
) -> tuple[dict[str, str], dict[str, int]]:
    records, status = collect_isolated(adapters, limit)
    run = CollectionRun(
        source="daily-research",
        requested=limit,
        status="RUNNING",
        code_sha=code_sha,
        diagnostics={"kind": "daily-research"},
    )
    session.add(run)
    session.flush()
    try:
        result = ingest_discovery_records(
            session,
            records,
            retrieved_at=utcnow(),
            archive=archive,
            collection_run_id=run.id,
        )
    except Exception:
        run.status = "FAILED"
        session.commit()
        raise
    run.status = "SUCCESS"
    run.ended_at = utcnow()
    session.commit()
    return status, result


def _market_catch_up(
    session: Any,
    *,
    archive: RawArchive,
    code_sha: str,
) -> dict[str, Any]:
    from .sources import HyperliquidSource

    end = safe_complete_hour()
    watermark = latest_production_candle(session)
    if watermark is None:
        start = end - timedelta(days=1)
    else:
        start = watermark - timedelta(hours=DAILY_OVERLAP_HOURS)
    if start >= end:
        return {
            "status": "UP_TO_DATE",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "inserted": 0,
        }
    result = run_live_market_collection(
        session,
        HyperliquidSource(),
        archive,
        start=start,
        end=end,
        code_sha=code_sha,
    )
    return {
        **result,
        "status": "COLLECTED",
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


def _last_daily_date(session: Session) -> date | None:
    return session.scalars(
        select(DailyRun.logical_date)
        .where(DailyRun.status == "SUCCESS")
        .order_by(DailyRun.logical_date.desc())
        .limit(1)
    ).first()


def _last_weekly_saturday(session: Session) -> date | None:
    return session.scalars(
        select(WeeklyRun.week_saturday)
        .where(WeeklyRun.status == "SUCCESS")
        .order_by(WeeklyRun.week_saturday.desc())
        .limit(1)
    ).first()


def run_daily(
    session: Any,
    *,
    logical_date: date,
    code_sha: str,
    output_root: Path,
    client: LLMClient | None,
    archive_root: Path,
    settings: Settings,
) -> dict[str, Any]:
    """Canonical Daily cycle: collect, analyze, persist, audit, report."""
    from .intelligence_v2 import AvailabilityBasis

    daily_root = output_root / logical_date.isoformat()
    daily_root.mkdir(parents=True, exist_ok=True)
    record = DailyRun(
        logical_date=logical_date,
        status="RUNNING",
        code_sha=code_sha,
    )
    session.add(record)
    session.commit()
    reasons: list[str] = []
    market: dict[str, Any] = {}
    academic_status: dict[str, str] = {}
    practitioner_status: dict[str, str] = {}
    intelligence: dict[str, Any] = {}
    try:
        archive = RawArchive(archive_root)
        market = _market_catch_up(session, archive=archive, code_sha=code_sha)
        record.market_status = "SUCCESS"
    except Exception as exc:
        market = {"status": "FAILED", "detail": str(exc)}
        record.market_status = "FAILED"
        reasons.append(f"market: {exc}")

    for label, adapters, attr in (
        ("academic", _research_adapters(settings)[0], "academic_status"),
        ("practitioner", _research_adapters(settings)[1], "practitioner_status"),
    ):
        try:
            status, _ = _collect_research(
                session,
                adapters,
                limit=10,
                archive=RawArchive(archive_root),
                code_sha=code_sha,
            )
            setattr(
                record,
                attr,
                "SUCCESS" if all(v == "READY" for v in status.values()) else "DEGRADED",
            )
            if label == "academic":
                academic_status = status
            else:
                practitioner_status = status
        except Exception as exc:
            setattr(record, attr, "FAILED")
            reasons.append(f"{label}: {exc}")

    as_of = beijing_now()
    try:
        intelligence = run_phase18_intelligence_cycle(
            session,
            daily_root / "intelligence",
            as_of,
            availability_basis=AvailabilityBasis.PRODUCTION_RECEIPT,
            persist=True,
            client=client,
        )
        record.analysis_status = "SUCCESS"
        record.knowledge_status = "SUCCESS"
        record.llm_summary = {
            "technical_status": intelligence.get("technical_status"),
            "critics": intelligence.get("critics", {}),
            "deepseek_used": client is not None,
        }
    except Exception as exc:
        record.analysis_status = "FAILED"
        record.knowledge_status = "PENDING"
        reasons.append(f"intelligence: {exc}")
        intelligence = {"technical_status": "FAILED", "detail": str(exc)}

    # Finalize durable state FIRST (whole-dict assignment so JSON mutates persist).
    record.audit_status = "SUCCESS" if not reasons else "FAILED"
    record.source_health = {
        "market": market,
        "academic": academic_status,
        "practitioner": practitioner_status,
    }
    record.failure_reasons = reasons
    record.status = (
        "SUCCESS"
        if not reasons
        else ("PARTIAL" if reasons and not _fatal(reasons) else "FAILED")
    )
    record.ended_at = utcnow()
    session.commit()

    # THEN render the canonical final report from persisted state (never RUNNING).
    from .reporting import collect_daily_snapshot, render_daily_markdown

    snapshot = collect_daily_snapshot(session, record.id)
    report_path = _write_final_report(
        daily_root, "report.md", snapshot, render_daily_markdown
    )
    summary_path = _write_final_summary(daily_root, snapshot)
    record.report_path = str(report_path)
    session.commit()
    return {
        "logical_date": logical_date.isoformat(),
        "status": record.status,
        "daily_run_id": str(record.id),
        "report": str(report_path),
        "summary": str(summary_path),
        "market": market,
        "intelligence_technical_status": intelligence.get("technical_status"),
        "failure_reasons": reasons,
    }


def _fatal(reasons: list[str]) -> bool:
    """A run is FAILED only if market or analysis failed outright."""
    return any(
        reason.startswith("market:") or reason.startswith("intelligence:")
        for reason in reasons
    )


def _write_final_report(
    root: Path, name: str, snapshot: dict[str, Any] | None, renderer: Any
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    if snapshot is None:
        path.write_text(
            "# Report unavailable\n\nRun state not found.\n", encoding="utf-8"
        )
        return path
    path.write_text(renderer(snapshot), encoding="utf-8")
    return path


def _write_final_summary(root: Path, snapshot: dict[str, Any] | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "summary.json"
    path.write_text(
        json.dumps(snapshot, default=str, sort_keys=True, indent=2), encoding="utf-8"
    )
    return path


def _rank_priorities(
    session: Any, hypotheses: list[ChannelHypothesis]
) -> list[dict[str, Any]]:
    ranked = []
    for hypothesis in hypotheses:
        family = hypothesis.fingerprint
        prior = session.scalar(
            select(EventStudyResultRecord)
            .where(EventStudyResultRecord.hypothesis_family_id == family)
            .order_by(EventStudyResultRecord.created_at.desc())
            .limit(1)
        )
        ranked.append(
            {
                "hypothesis_family": family,
                "statement": hypothesis.statement,
                "horizon": hypothesis.horizon,
                "fit": low_frequency_fit(hypothesis.horizon),
                "maturity": hypothesis.maturity,
                "prior_empirical_disposition": prior.disposition if prior else None,
                "mechanism": hypothesis.mechanism,
            }
        )
    ranked.sort(
        key=lambda item: (item["fit"] == FIT_HIGH, item["fit"] == FIT_MEDIUM),
        reverse=True,
    )
    return ranked[:5]


def run_weekly(
    session: Any,
    *,
    week_saturday: date,
    code_sha: str,
    output_root: Path,
    client: LLMClient | None,
) -> dict[str, Any]:
    """Canonical Weekly review over the seven Daily cycles ending at week_saturday."""
    weekly_root = output_root / f"week-{week_saturday.isoformat()}"
    weekly_root.mkdir(parents=True, exist_ok=True)
    record = WeeklyRun(week_saturday=week_saturday, status="RUNNING", code_sha=code_sha)
    session.add(record)
    session.commit()
    reasons: list[str] = []

    week_start = week_saturday - timedelta(days=6)
    dailies = session.scalars(
        select(DailyRun)
        .where(
            DailyRun.logical_date >= week_start,
            DailyRun.logical_date <= week_saturday,
        )
        .order_by(DailyRun.logical_date)
    ).all()
    record.included_daily_dates = [d.logical_date.isoformat() for d in dailies]
    missing = [
        (week_start + timedelta(days=offset)).isoformat()
        for offset in range(7)
        if (week_start + timedelta(days=offset)).isoformat()
        not in record.included_daily_dates
    ]

    hypotheses = session.scalars(
        select(ChannelHypothesis)
        .where(
            ChannelHypothesis.as_of
            >= normalize_utc(
                datetime.combine(week_start, datetime.min.time(), tzinfo=UTC)
            )
        )
        .order_by(ChannelHypothesis.created_at)
    ).all()
    families: dict[str, int] = {}
    for hypothesis in hypotheses:
        families[hypothesis.fingerprint] = families.get(hypothesis.fingerprint, 0) + 1
    recurrent = {k: v for k, v in families.items() if v > 1}

    priorities = _rank_priorities(session, hypotheses)
    high_fit = [p for p in priorities if p["fit"] == FIT_HIGH]
    out_of_scope = [p for p in priorities if p["fit"] == FIT_OUT_OF_SCOPE]

    record.priorities = priorities
    record.low_frequency_fit = {
        "high_fit": high_fit,
        "out_of_scope": out_of_scope,
    }
    record.failure_reasons = reasons
    record.status = (
        "SUCCESS" if not reasons else ("PARTIAL" if not _fatal(reasons) else "FAILED")
    )
    record.ended_at = utcnow()
    session.commit()

    # THEN render the canonical final report from persisted state.
    from .reporting import collect_weekly_snapshot, render_weekly_markdown

    snapshot = collect_weekly_snapshot(session, record.id)
    report = _write_final_report(
        weekly_root, "report.md", snapshot, render_weekly_markdown
    )
    summary_path = _write_final_summary(weekly_root, snapshot)
    record.report_path = str(report)
    session.commit()
    return {
        "week_saturday": week_saturday.isoformat(),
        "status": record.status,
        "weekly_run_id": str(record.id),
        "report": str(report),
        "summary": str(summary_path),
        "included_daily_dates": record.included_daily_dates,
        "missing_daily_dates": missing,
        "hypothesis_count": len(hypotheses),
        "recurrent_families": recurrent,
        "priorities": priorities,
    }


def regenerate_report(
    session: Session,
    *,
    logical_date: str | None = None,
    week_saturday: str | None = None,
    output_root: Path,
) -> dict[str, Any]:
    """Regenerate a final report from persisted state ONLY (no research rerun).

    Refuses to render a RUNNING run; renders the canonical report.md +
    summary.json from the durable Daily/Weekly record.
    """
    from .reporting import (
        collect_daily_snapshot,
        collect_weekly_snapshot,
        render_daily_markdown,
        render_weekly_markdown,
    )

    if week_saturday:
        weekly_record: WeeklyRun | None = session.scalars(
            select(WeeklyRun)
            .where(WeeklyRun.week_saturday == date.fromisoformat(week_saturday))
            .order_by(WeeklyRun.started_at.desc())
            .limit(1)
        ).first()
        if weekly_record is None:
            raise ValueError(f"no weekly run for {week_saturday}")
        if weekly_record.status == "RUNNING":
            raise RuntimeError("refusing to render a RUNNING weekly as final")
        root = output_root / f"week-{week_saturday}"
        snapshot = collect_weekly_snapshot(session, weekly_record.id)
        report = _write_final_report(
            root, "report.md", snapshot, render_weekly_markdown
        )
        summary_path = _write_final_summary(root, snapshot)
        return {
            "kind": "WEEKLY",
            "week_saturday": week_saturday,
            "report": str(report),
            "summary": str(summary_path),
            "network_calls": 0,
            "llm_calls": 0,
        }
    if logical_date:
        daily_record: DailyRun | None = session.scalars(
            select(DailyRun)
            .where(DailyRun.logical_date == date.fromisoformat(logical_date))
            .order_by(DailyRun.started_at.desc())
            .limit(1)
        ).first()
        if daily_record is None:
            raise ValueError(f"no daily run for {logical_date}")
        if daily_record.status == "RUNNING":
            raise RuntimeError("refusing to render a RUNNING daily as final")
        root = output_root / logical_date
        snapshot = collect_daily_snapshot(session, daily_record.id)
        report = _write_final_report(root, "report.md", snapshot, render_daily_markdown)
        summary_path = _write_final_summary(root, snapshot)
        return {
            "kind": "DAILY",
            "logical_date": logical_date,
            "report": str(report),
            "summary": str(summary_path),
            "network_calls": 0,
            "llm_calls": 0,
        }
    raise ValueError("provide logical_date or week_saturday")


def scheduler_tick(
    session: Session,
    *,
    settings: Settings,
    output_root: Path,
    archive_root: Path,
    code_sha: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One lightweight scheduler tick: run only what is logically due.

    Makes no provider/LLM call when nothing is due. Saturday ordering is enforced:
    Daily completes before Weekly is considered.
    """
    from .scheduler import compute_due

    bj = beijing_now(now)
    due = compute_due(bj, _last_daily_date(session), _last_weekly_saturday(session))
    ran: list[str] = []
    root = output_root / "scheduler"
    root.mkdir(parents=True, exist_ok=True)
    lock = OperationsLock(output_root)
    if due.daily_due:
        if lock.acquire():
            try:
                client = build_client(settings)
                run_daily(
                    session,
                    logical_date=due.daily_date,
                    code_sha=code_sha,
                    output_root=output_root / "daily",
                    client=client,
                    archive_root=archive_root,
                    settings=settings,
                )
                ran.append("daily")
            finally:
                lock.release()
    last_daily = _last_daily_date(session)
    if due.weekly_due and last_daily is not None and last_daily >= due.weekly_saturday:
        if lock.acquire():
            try:
                client = build_client(settings)
                run_weekly(
                    session,
                    week_saturday=due.weekly_saturday,
                    code_sha=code_sha,
                    output_root=output_root / "weekly",
                    client=client,
                )
                ran.append("weekly")
            finally:
                lock.release()
    return {
        "beijing_time": bj.isoformat(),
        "ran": ran,
        "daily_due": due.daily_due,
        "weekly_due": due.weekly_due,
        "daily_date": due.daily_date.isoformat(),
        "weekly_saturday": due.weekly_saturday.isoformat(),
    }


def ops_status(
    session: Any, *, now: datetime | None = None, settings: Any = None
) -> dict[str, Any]:
    """Human/machine-readable operational health snapshot.

    Distinguishes SOURCE CAPABILITY (configured adapter state) from the last
    Daily run's per-source RETRIEVAL OUTCOME, and reports a deterministic next
    Daily due date even when today's Daily is already complete.
    """
    from .scheduler import compute_due

    settings = settings or get_settings()
    bj = beijing_now(now)
    last_daily = _last_daily_date(session)
    last_weekly = _last_weekly_saturday(session)
    due = compute_due(bj, last_daily, last_weekly)
    watermark = latest_production_candle(session)
    registry = {
        entry["source_name"]: entry["adapter_status"] for entry in source_registry()
    }
    from .publication_ops import publication_status

    # Deterministic next Daily due date (next 18:30 Beijing boundary).
    from .scheduler import DAILY_DUE_TIME

    if bj.time() < DAILY_DUE_TIME:
        next_daily = bj.date().isoformat()
    else:
        next_daily = (bj.date() + timedelta(days=1)).isoformat()

    last_daily_row = None
    last_run_source_outcomes: dict[str, str] = {}
    if last_daily is not None:
        last_daily_row = session.scalars(
            select(DailyRun)
            .where(DailyRun.logical_date == last_daily)
            .order_by(DailyRun.started_at.desc())
            .limit(1)
        ).first()
        if last_daily_row is not None:
            for group in ("academic", "practitioner"):
                outcomes = last_daily_row.source_health.get(group, {})
                if isinstance(outcomes, dict):
                    last_run_source_outcomes.update(outcomes)

    return {
        "beijing_time": bj.isoformat(),
        "next_daily_due": next_daily,
        "last_daily_date": last_daily.isoformat() if last_daily else None,
        "last_daily_status": last_daily_row.status if last_daily_row else None,
        "last_weekly_saturday": last_weekly.isoformat() if last_weekly else None,
        "market_watermark": watermark.isoformat() if watermark else None,
        "daily_due": due.daily_due,
        "weekly_due": due.weekly_due,
        "database_url": str(session.get_bind().engine.url).replace(":pass", ":****"),
        "source_capability": {
            name: registry.get(name, "READY")
            for name in (*ACADEMIC_SOURCES, *PRACTITIONER_SOURCES)
        },
        "last_daily_source_run_outcomes": last_run_source_outcomes,
        "publishing": {
            "x_mode": settings.publication_mode,
            **publication_status(session),
        },
    }


def _main() -> None:
    import argparse

    from .cli import migrate_database

    parser = argparse.ArgumentParser(prog="quant-radar-ops")
    parser.add_argument("command", choices=["daily", "weekly", "status", "tick"])
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--output-root", default="outputs/operations")
    parser.add_argument("--archive-root", default="data/raw")
    parser.add_argument("--code-sha", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--now", type=datetime.fromisoformat, default=None)
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url
    migrate_database(database_url)
    session = make_session_factory(make_engine(database_url))()

    if args.command == "status":
        print(
            json.dumps(ops_status(session, now=args.now), default=str, sort_keys=True)
        )
        return

    if args.command == "tick":
        from .scheduler import compute_due

        bj = beijing_now(args.now)
        due = compute_due(bj, _last_daily_date(session), _last_weekly_saturday(session))
        print(
            json.dumps(
                {
                    "beijing_time": bj.isoformat(),
                    "daily_due": due.daily_due,
                    "weekly_due": due.weekly_due,
                },
                sort_keys=True,
            )
        )
        return

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    lock = OperationsLock(root)
    if not args.force and not lock.acquire():
        print(json.dumps({"status": "LOCKED"}))
        return
    try:
        client = build_client(settings)
        if args.command == "daily":
            logical_date = beijing_date(args.now)
            result = run_daily(
                session,
                logical_date=logical_date,
                code_sha=args.code_sha,
                output_root=root / "daily",
                client=client,
                archive_root=Path(args.archive_root),
                settings=settings,
            )
        else:
            from .scheduler import most_recent_saturday

            saturday = most_recent_saturday(beijing_date(args.now))
            result = run_weekly(
                session,
                week_saturday=saturday,
                code_sha=args.code_sha,
                output_root=root / "weekly",
                client=client,
            )
        print(json.dumps(result, default=str, sort_keys=True))
    finally:
        lock.release()


if __name__ == "__main__":
    _main()
