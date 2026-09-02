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

ACADEMIC_SOURCES = ("openalex", "crossref", "arxiv", "nber", "repec")
PRACTITIONER_SOURCES = ("alpha-architect", "man-institute", "aqr")
DAILY_OVERLAP_HOURS = 1

# Low-frequency actionability classification (P0 strategy preference).
FIT_OUT_OF_SCOPE = "OUT_OF_SCOPE_FOR_USER"
FIT_LOW = "LOW_FIT"
FIT_MEDIUM = "MEDIUM_FIT"
FIT_HIGH = "HIGH_FIT"


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
    return {**result, "status": "COLLECTED"}


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
    try:
        archive = RawArchive(archive_root)
        market = _market_catch_up(session, archive=archive, code_sha=code_sha)
        record.market_status = "SUCCESS"
        record.source_health["market"] = market
    except Exception as exc:
        market = {"status": "FAILED", "detail": str(exc)}
        record.market_status = "FAILED"
        reasons.append(f"market: {exc}")
        record.source_health["market"] = market

    academic_status: dict[str, str] = {}
    practitioner_status: dict[str, str] = {}
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
    record.source_health["academic"] = academic_status
    record.source_health["practitioner"] = practitioner_status

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

    report_path = _write_daily_report(
        daily_root,
        logical_date,
        record,
        market,
        academic_status,
        practitioner_status,
        reasons,
    )
    record.audit_status = "SUCCESS"
    record.report_path = str(report_path)
    record.failure_reasons = reasons
    record.status = "SUCCESS" if not reasons else "PARTIAL"
    record.ended_at = utcnow()
    session.commit()
    return {
        "logical_date": logical_date.isoformat(),
        "status": record.status,
        "report": str(report_path),
        "market": market,
        "intelligence_technical_status": intelligence.get("technical_status"),
        "failure_reasons": reasons,
    }


def _write_daily_report(
    root: Path,
    logical_date: date,
    record: DailyRun,
    market: dict[str, Any],
    academic: dict[str, str],
    practitioner: dict[str, str],
    reasons: list[str],
) -> Path:
    degraded_academic = [k for k, v in academic.items() if v != "READY"]
    degraded_practitioner = [k for k, v in practitioner.items() if v != "READY"]
    lines = [
        "# Daily Research Report",
        "",
        f"- **Logical Beijing date:** {logical_date.isoformat()}",
        f"- **Run status:** {record.status}",
        f"- **Market collection:** {market.get('status')} "
        f"({market.get('start', 'n/a')} → {market.get('end', 'n/a')}, "
        f"inserted={market.get('inserted', 0)})",
        f"- **Academic sources:** {'degraded: ' + ', '.join(degraded_academic) if degraded_academic else 'all ready'}",
        f"- **Practitioner sources:** {'degraded: ' + ', '.join(degraded_practitioner) if degraded_practitioner else 'all ready'}",
        f"- **Analysis:** {record.analysis_status}",
        f"- **Knowledge:** {record.knowledge_status}",
        f"- **LLM:** {'DeepSeek configured' if record.llm_summary.get('deepseek_used') else 'no client (critics NOT_RUN)'}",
    ]
    if reasons:
        lines += ["", "## Failures", ""] + [f"- {reason}" for reason in reasons]
    lines += ["", f"_Generated {utcnow().isoformat()}_", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def low_frequency_fit(horizon: str | None) -> str:
    """Map a hypothesis horizon string to the user's actionability-fit bucket."""
    if not horizon:
        return FIT_OUT_OF_SCOPE
    text = horizon.lower()
    minutes: float | None = None
    if "min" in text:
        try:
            minutes = float(
                "".join(ch for ch in text.split("min")[0] if ch.isdigit() or ch == ".")
                or 0
            )
        except ValueError:
            minutes = None
    elif "h" in text and "d" not in text:
        try:
            minutes = (
                float(
                    "".join(
                        ch for ch in text.split("h")[0] if ch.isdigit() or ch == "."
                    )
                    or 0
                )
                * 60
            )
        except ValueError:
            minutes = None
    elif "d" in text:
        try:
            minutes = (
                float(
                    "".join(
                        ch for ch in text.split("d")[0] if ch.isdigit() or ch == "."
                    )
                    or 0
                )
                * 24
                * 60
            )
        except ValueError:
            minutes = None
    if minutes is None:
        return FIT_OUT_OF_SCOPE
    hours = minutes / 60
    if hours < 2:
        return FIT_OUT_OF_SCOPE
    if hours < 12:
        return FIT_LOW
    if hours < 24:
        return FIT_MEDIUM
    return FIT_HIGH


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

    report = _write_weekly_report(
        weekly_root,
        week_saturday,
        record,
        hypotheses,
        recurrent,
        priorities,
        high_fit,
        out_of_scope,
        missing,
        reasons,
    )
    record.priorities = priorities
    record.low_frequency_fit = {
        "high_fit": high_fit,
        "out_of_scope": out_of_scope,
    }
    record.report_path = str(report)
    record.failure_reasons = reasons
    record.status = "SUCCESS" if not reasons else "PARTIAL"
    record.ended_at = utcnow()
    session.commit()
    return {
        "week_saturday": week_saturday.isoformat(),
        "status": record.status,
        "report": str(report),
        "included_daily_dates": record.included_daily_dates,
        "missing_daily_dates": missing,
        "hypothesis_count": len(hypotheses),
        "recurrent_families": recurrent,
        "priorities": priorities,
    }


def _write_weekly_report(
    root: Path,
    week_saturday: date,
    record: WeeklyRun,
    hypotheses: list[ChannelHypothesis],
    recurrent: dict[str, int],
    priorities: list[dict[str, Any]],
    high_fit: list[dict[str, Any]],
    out_of_scope: list[dict[str, Any]],
    missing: list[str],
    reasons: list[str],
) -> Path:
    lines = [
        "# Weekly Research Review",
        "",
        f"**Week of:** {week_saturday.isoformat()}",
        f"**Status:** {record.status}",
        "",
        "## System health",
        f"- Included Daily runs: {len(record.included_daily_dates)}",
        f"- Missing Daily runs: {', '.join(missing) if missing else 'none'}",
        f"- New/recurrent hypotheses: {len(hypotheses)} total, "
        f"{len(recurrent)} recurrent families",
        "",
        "## Low-frequency fit",
        f"- HIGH-FIT candidates: {len(high_fit)}",
        f"- OUT-OF-SCOPE (scientifically retained): {len(out_of_scope)}",
        "",
        "## Top research priorities",
    ]
    if priorities:
        for index, priority in enumerate(priorities, start=1):
            lines += [
                f"{index}. **{priority['hypothesis_family']}** "
                f"(fit={priority['fit']}, horizon={priority['horizon']})",
                f"   - {priority['statement']}",
                f"   - prior empirical: {priority['prior_empirical_disposition']}",
            ]
    else:
        lines.append("_No priorities this week._")
    if reasons:
        lines += ["", "## Failures", ""] + [f"- {r}" for r in reasons]
    lines += ["", f"_Generated {utcnow().isoformat()}_", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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


def ops_status(session: Any, *, now: datetime | None = None) -> dict[str, Any]:
    """Human/machine-readable operational health snapshot."""
    from .scheduler import compute_due

    bj = beijing_now(now)
    last_daily = _last_daily_date(session)
    last_weekly = _last_weekly_saturday(session)
    due = compute_due(bj, last_daily, last_weekly)
    watermark = latest_production_candle(session)
    registry = {
        entry["source_name"]: entry["adapter_status"] for entry in source_registry()
    }
    return {
        "beijing_time": bj.isoformat(),
        "next_daily_due": due.daily_date.isoformat() if due.daily_due else None,
        "last_daily_date": last_daily.isoformat() if last_daily else None,
        "last_weekly_saturday": last_weekly.isoformat() if last_weekly else None,
        "market_watermark": watermark.isoformat() if watermark else None,
        "daily_due": due.daily_due,
        "weekly_due": due.weekly_due,
        "database_url": str(session.get_bind().engine.url).replace(":pass", ":****"),
        "supported_sources": {
            name: registry.get(name, "READY")
            for name in (*ACADEMIC_SOURCES, *PRACTITIONER_SOURCES)
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
