from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import cast

from alembic import command
from alembic.config import Config
from sqlalchemy import select

from .config import Settings, get_settings
from .db import (
    CollectionRun,
    get_phase16a_collection_run,
    make_engine,
    make_session_factory,
    normalize_utc,
)
from .llm import (
    DeepSeekClient,
    FakeLLMClient,
    LLMClient,
    ModelRouter,
    build_deepseek_client,
)
from .pipeline import (
    calculate_metrics,
    daily_report,
    ingest,
    ingest_records,
    weekly_report,
)
from .replay import (
    funding_coverage,
    parse_utc_timestamp,
    run_replay_day,
    write_summary,
)
from .sources import (
    ArxivSource,
    CrossrefSource,
    HyperliquidSource,
    InstitutionalHtmlSource,
    InstitutionalRssSource,
    NberSource,
    OpenAlexSource,
    PractitionerRssSource,
    RepecSource,
    SourceAdapter,
    collect_isolated,
)


def migrate_database(database_url: str) -> None:
    engine = make_engine(database_url)
    with engine.begin() as connection:
        tables = (
            set(
                connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).scalars()
            )
            if engine.dialect.name == "sqlite"
            else set()
        )
        if tables and "alembic_version" not in tables:
            from alembic.runtime.migration import MigrationContext

            context = MigrationContext.configure(connection)
            if "collection_runs" in tables:
                # Legacy replay DBs were created by create_all with the current
                # tables, so stamp the last accepted revision before applying 0004.
                context._ensure_version_table()
                connection.exec_driver_sql(
                    "INSERT INTO alembic_version (version_num) VALUES ('0003_phase15c')"
                )
            else:
                raise RuntimeError("Unversioned database has no recognizable schema")
    command.upgrade(_alembic_config(database_url), "head")


def _alembic_config(database_url: str) -> Config:
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _phase18_client(settings: Settings) -> DeepSeekClient | None:
    """DeepSeek client when a credential is present; absence leaves Critic/Tutor NOT_RUN.

    The provider flag (``llm_provider``) is a legacy safe default and must not
    disable the real DeepSeek credential when one is configured: role routing is
    already pinned to the DeepSeek flash/pro models in the environment.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="quant-radar")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db")
    collect = sub.add_parser("collect")
    collect.add_argument("source", choices=["arxiv", "repec", "hyperliquid"])
    collect.add_argument("--limit", type=int, default=None)
    collect.add_argument("--offline", action="store_true")
    collect.add_argument("--history", action="store_true")
    collect.add_argument("--start", type=datetime.fromisoformat, default=None)
    collect.add_argument("--end", type=datetime.fromisoformat, default=None)
    collect.add_argument("--phase16a-run-id", default=None)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--limit", type=int, default=20)
    sub.add_parser("run-daily").add_argument("--offline", action="store_true")
    routing = sub.add_parser("validate-llm-routing")
    routing.add_argument(
        "--artifact", default="outputs/validation/phase15c-llm-routing.json"
    )
    routing.add_argument("--database-url", default="sqlite:///phase15c-validation.db")
    routing.add_argument("--live", action="store_true")
    validate = sub.add_parser("validate-live")
    validate.add_argument("--limit", type=int, default=5)
    validate.add_argument(
        "--artifact", default="outputs/validation/phase15b-live-validation.json"
    )
    report = sub.add_parser("report")
    report.add_argument("kind", choices=["daily", "weekly"])
    replay = sub.add_parser("replay")
    replay.add_argument("--date", type=date.fromisoformat, required=True)
    replay.add_argument("--as-of", type=datetime.fromisoformat, required=True)
    replay.add_argument(
        "--collection-end",
        type=datetime.fromisoformat,
        required=True,
        help="UTC end of the persisted Phase 1.6A collection window; separate from --as-of",
    )
    replay.add_argument("--output-dir", default="outputs/replay")
    replay.add_argument("--provider", choices=["fake", "deepseek"], default=None)
    replay.add_argument("--warmup-start", type=datetime.fromisoformat, default=None)
    replay.add_argument("--collection-code-sha", required=True)
    replay.add_argument("--replay-code-sha", required=True)
    replay.add_argument("--coverage-only", action="store_true")
    replay.add_argument("--phase16a-run-id", default=None)
    live_parser = sub.add_parser("live-cycle")
    live_parser.add_argument("--output-dir", required=True)
    live_parser.add_argument("--cycle", type=int, required=True)
    live_parser.add_argument("--database-url", required=True)
    live_parser.add_argument("--code-sha", required=True)
    fast = sub.add_parser("phase16c-fast")
    fast.add_argument("--output-dir", required=True)
    fast.add_argument("--database-url", required=True)
    fast.add_argument("--provider", choices=["fake", "deepseek"], default="deepseek")
    v2_replay = sub.add_parser("phase16d-replay")
    v2_replay.add_argument("--output-dir", required=True)
    v2_replay.add_argument("--database-url", required=True)
    v2_replay.add_argument("--start-date", type=date.fromisoformat, required=True)
    v2_replay.add_argument("--days", type=int, default=7)
    v2_replay.add_argument("--provider", choices=["fake", "deepseek"], default="fake")
    v2_discover = sub.add_parser("phase16d-discover")
    v2_discover.add_argument("--output", required=True)
    v2_discover.add_argument("--database-url", required=True)
    v2_discover.add_argument("--limit", type=int, default=100)
    knowledge = sub.add_parser("knowledge")
    knowledge.add_argument("action", choices=["search", "show"])
    knowledge.add_argument("value")
    knowledge.add_argument("--database-url", required=True)
    knowledge.add_argument(
        "--scope",
        choices=["PRODUCTION", "REPLAY", "ALL_WITH_PROVENANCE"],
        default="PRODUCTION",
    )
    knowledge.add_argument("--channel")
    knowledge.add_argument("--maturity")
    knowledge.add_argument("--as-of", type=datetime.fromisoformat)
    operations = sub.add_parser("phase20-market-collect")
    operations.add_argument("mode", choices=["historical", "live", "audit"])
    operations.add_argument("--database-url", required=True)
    operations.add_argument("--start", type=datetime.fromisoformat)
    operations.add_argument("--end", type=datetime.fromisoformat)
    operations.add_argument("--archive-root", default="data/raw")
    operations.add_argument("--audit-output")
    operations.add_argument("--code-sha", default="")
    ops = sub.add_parser("ops")
    ops.add_argument(
        "action",
        choices=["daily", "weekly", "status", "tick", "daily-report", "weekly-report"],
    )
    ops.add_argument("--database-url", default=None)
    ops.add_argument("--output-root", default="outputs/operations")
    ops.add_argument("--archive-root", default="data/raw")
    ops.add_argument("--code-sha", default="")
    ops.add_argument("--force", action="store_true")
    ops.add_argument("--now", type=datetime.fromisoformat, default=None)
    ops.add_argument("--date", default=None)
    publish = sub.add_parser("publish")
    publish.add_argument(
        "action", choices=["latest", "list", "show", "daily", "history", "evaluate"]
    )
    publish.add_argument("value", nargs="?", default=None)
    publish.add_argument("--date", default=None)
    publish.add_argument("--database-url", default=None)
    publish.add_argument("--output-root", default="outputs/operations")
    publish.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    event_study = sub.add_parser("event-study")
    event_study.add_argument("action", choices=["run", "list", "show"])
    event_study.add_argument("--database-url", required=True)
    event_study.add_argument("--output-dir", default="outputs/event-study")
    event_study.add_argument("--start", type=datetime.fromisoformat)
    event_study.add_argument("--end", type=datetime.fromisoformat)
    event_study.add_argument("--hypothesis-id")
    event_study.add_argument("--run-id")
    event_study.add_argument(
        "--provider", choices=["fake", "configured"], default="configured"
    )
    summary = sub.add_parser("rebuild-phase16a-summary")
    summary.add_argument("--output-dir", default="outputs/replay")
    summary.add_argument("--phase16a-run-id", default=None)
    summary.add_argument(
        "--requested-end",
        type=lambda value: parse_utc_timestamp(value, "requested_end"),
        default=None,
    )
    summary.add_argument(
        "--warmup-start",
        type=lambda value: parse_utc_timestamp(value, "warmup_start"),
        default=None,
    )
    summary.add_argument("--code-sha", default="unknown")
    summary.add_argument("--collection-code-sha", required=True)
    summary.add_argument("--replay-code-sha", required=True)
    summary.add_argument(
        "--collection-end",
        required=True,
        type=lambda value: parse_utc_timestamp(value, "collection_end"),
    )
    summary.add_argument(
        "--collection-start",
        required=False,
        type=lambda value: parse_utc_timestamp(value, "collection_start"),
    )
    args = parser.parse_args()
    settings = get_settings()
    if args.command == "publish":
        from .db import DailyRun, DailySocialEditorialPackage, PublicationDraft

        database_url = args.database_url or settings.database_url
        migrate_database(database_url)
        session = make_session_factory(make_engine(database_url))()
        packages = select(DailySocialEditorialPackage).order_by(
            DailySocialEditorialPackage.logical_date.desc()
        )
        if args.action in {"latest", "list", "history"}:
            package_rows = session.scalars(packages).all()
            if args.date:
                package_rows = [
                    row
                    for row in package_rows
                    if row.logical_date.isoformat() == args.date
                ]
            if args.action == "latest":
                package_rows = package_rows[:1]
            if args.action == "history":
                draft_rows = session.scalars(
                    select(PublicationDraft).order_by(
                        PublicationDraft.created_at.desc()
                    )
                ).all()
                print(
                    json.dumps(
                        [
                            {
                                "id": str(row.id),
                                "created_at": row.created_at,
                                "text": row.text,
                            }
                            for row in draft_rows
                        ],
                        default=str,
                        sort_keys=True,
                    )
                )
            else:
                rows = [
                    {
                        "id": str(row.id),
                        "date": row.logical_date.isoformat(),
                        "recommendation": row.recommendation,
                        "format": row.content_format,
                        "selected": row.selected_candidate_id,
                        "path": row.output_path,
                    }
                    for row in package_rows
                ]
                if args.json:
                    print(json.dumps(rows, sort_keys=True))
                else:
                    for row in rows:
                        print(
                            f"{row['date']}  {row['recommendation']:<7}  {row['format']:<10}  {row['id']}"
                        )
            return
        package = None
        uuid_value = False
        if args.action == "show" and args.value:
            try:
                package = session.get(
                    DailySocialEditorialPackage, uuid.UUID(args.value)
                )
                uuid_value = True
            except ValueError:
                pass
        target_date = args.date if uuid_value else (args.date or args.value)
        if package is None and not target_date:
            parser.error("publish daily/show/evaluate requires --date or value")
        try:
            parsed_date = date.fromisoformat(target_date) if target_date else None
        except ValueError:
            parser.error("date must use YYYY-MM-DD")
        if package is None:
            package = session.scalar(
                select(DailySocialEditorialPackage).where(
                    DailySocialEditorialPackage.logical_date == parsed_date
                )
            )
        if args.action in {"daily", "evaluate"} and (
            package is None or package.recommendation not in {"PUBLISH", "SKIP"}
        ):
            daily = session.scalar(
                select(DailyRun).where(
                    DailyRun.logical_date == date.fromisoformat(target_date)
                )
            )
            if daily is None or daily.status == "RUNNING":
                parser.error("completed Daily required for historical evaluation")
            from .publication_ops import _persist_social_package
            from .publishing import (
                classify_policy,
                select_daily_candidates,
                select_editorial_daily_candidate,
            )
            from .synthesis import synthesize_daily_topics

            synthesize_daily_topics(session, daily.id)
            candidates = select_daily_candidates(
                session, daily_run_id=str(daily.id), logical_date=target_date
            )
            selected, editorial = select_editorial_daily_candidate(candidates)
            draft = None
            rejection = "no publishable candidate"
            if selected is not None:
                from .publishing import create_draft

                empirical = dict(selected.evidence)
                has_study = bool(empirical.get("event_study_result_id"))
                if has_study:
                    empirical.setdefault("disposition", "INCONCLUSIVE")
                draft, rejection_value = create_draft(
                    session,
                    selected,
                    empirical=empirical,
                    structured_numbers={"0": 0.0} if has_study else {},
                    language=settings.publication_language,
                )
                rejection = rejection_value or "copy verification failed"
            recommendation = (
                "PUBLISH" if draft is not None and draft.text.strip() else "SKIP"
            )
            _persist_social_package(
                session,
                daily,
                candidates,
                selected,
                recommendation,
                editorial.get("reason", rejection or "historical evaluation"),
                Path(args.output_root),
                draft.text if draft else None,
                {
                    "policy": classify_policy(selected) if selected else None,
                    "verification": {
                        "status": "PASS" if draft else "FAIL",
                        "reason": rejection,
                    },
                    "copy_quality": {"status": "PASS" if draft else "FAIL"},
                    "draft_id": str(draft.id) if draft else None,
                    "source_bundle": draft.source_bundle if draft else {},
                    "score": selected.publication_value if selected else None,
                    "selected": {"id": str(selected.id), "title": selected.title}
                    if selected
                    else {},
                    "source_ids": [
                        str(v)
                        for k, v in (draft.source_bundle if draft else {}).items()
                        if k.endswith("_id") and v
                    ],
                    "visual_ids": list(draft.visual_ids) if draft else [],
                },
            )
            package = session.scalar(
                select(DailySocialEditorialPackage).where(
                    DailySocialEditorialPackage.logical_date == daily.logical_date
                )
            )
        if package is None:
            parser.error("no social package for date")
        output = {
            "id": str(package.id),
            "date": package.logical_date,
            "recommendation": package.recommendation,
            "format": package.content_format,
            "reason": package.selection_reason,
            "candidates": package.candidates,
            "draft": package.draft_text,
            "path": package.output_path,
        }
        if args.json:
            print(json.dumps(output, default=str, indent=2, sort_keys=True))
        else:
            print(f"{output['date']} — {output['recommendation']}")
            print(f"Reason: {output['reason']}")
            print(f"Package: {output['id']}")
            draft_text = output["draft"]
            if isinstance(draft_text, str) and draft_text:
                print(f"Draft: {len(draft_text)} characters")
        return
    if args.command == "ops":
        from .operations import (
            OperationsLock,
            build_client,
            ops_status,
            run_daily,
            run_weekly,
        )
        from .scheduler import beijing_date, most_recent_saturday

        database_url = args.database_url or settings.database_url
        migrate_database(database_url)
        session = make_session_factory(make_engine(database_url))()
        if args.action == "status":
            print(
                json.dumps(
                    ops_status(session, now=args.now), default=str, sort_keys=True
                )
            )
            return
        if args.action == "tick":
            from .operations import scheduler_tick

            print(
                json.dumps(
                    scheduler_tick(
                        session,
                        settings=settings,
                        output_root=Path(args.output_root),
                        archive_root=Path(args.archive_root),
                        code_sha=args.code_sha,
                        now=args.now,
                    ),
                    default=str,
                    sort_keys=True,
                )
            )
            return
        if args.action in ("daily-report", "weekly-report"):
            from .operations import regenerate_report

            base = Path(args.output_root) / (
                "daily" if args.action == "daily-report" else "weekly"
            )
            print(
                json.dumps(
                    regenerate_report(
                        session,
                        logical_date=args.date
                        if args.action == "daily-report"
                        else None,
                        week_saturday=args.date
                        if args.action == "weekly-report"
                        else None,
                        output_root=base,
                    ),
                    default=str,
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
            ops_client = build_client(settings)
            if args.action == "daily":
                ops_result = run_daily(
                    session,
                    logical_date=beijing_date(args.now),
                    code_sha=args.code_sha,
                    output_root=root / "daily",
                    client=ops_client,
                    archive_root=Path(args.archive_root),
                    settings=settings,
                )
                if ops_result.get("status") == "SUCCESS":
                    from .publication_ops import after_daily

                    ops_result["downstream"] = after_daily(
                        session,
                        settings,
                        daily_run_id=str(ops_result.get("daily_run_id", "")),
                        logical_date=str(ops_result.get("logical_date")),
                        market_summary=ops_result.get("market", {}),
                        report_path=Path(ops_result["report"])
                        if ops_result.get("report")
                        else None,
                        output_root=root / "publication",
                    )
            else:
                ops_result = run_weekly(
                    session,
                    week_saturday=most_recent_saturday(beijing_date(args.now)),
                    code_sha=args.code_sha,
                    output_root=root / "weekly",
                    client=ops_client,
                )
                if ops_result.get("status") == "SUCCESS":
                    from .publication_ops import after_weekly

                    ops_result["downstream"] = after_weekly(
                        session,
                        settings,
                        weekly_run_id=str(ops_result.get("weekly_run_id", "")),
                        week_saturday=str(ops_result.get("week_saturday")),
                        report_path=Path(ops_result["report"])
                        if ops_result.get("report")
                        else None,
                        output_root=root / "publication",
                    )
            print(json.dumps(ops_result, default=str, sort_keys=True))
        finally:
            lock.release()
        return
    if args.command == "phase20-market-collect":
        from .market_operations import (
            coverage_audit,
            default_backfill_start,
            run_historical_backfill,
            run_live_market_collection,
            safe_complete_hour,
        )
        from .raw_archive import RawArchive

        if args.mode != "audit" and not args.code_sha:
            raise SystemExit("BLOCKED: --code-sha is required for collection")
        safe_end = safe_complete_hour()
        end = normalize_utc(args.end) if args.end else safe_end
        if end > safe_end:
            raise SystemExit(
                "BLOCKED: --end must not exceed the latest completed UTC hour"
            )
        start = (
            normalize_utc(args.start)
            if args.start
            else (
                default_backfill_start(end)
                if args.mode in {"historical", "audit"}
                else end - timedelta(hours=2)
            )
        )
        migrate_database(args.database_url)
        session = make_session_factory(make_engine(args.database_url))()
        if args.mode == "audit":
            print(
                json.dumps(
                    coverage_audit(session, start=start, end=end), sort_keys=True
                )
            )
            return
        runner = (
            run_historical_backfill
            if args.mode == "historical"
            else run_live_market_collection
        )
        collection_result = runner(
            session,
            HyperliquidSource(),
            RawArchive(Path(args.archive_root)),
            start=start,
            end=end,
            code_sha=args.code_sha,
        )
        collection_result["coverage_audit"] = coverage_audit(
            session,
            start=start,
            end=end,
            mode=(
                "ACCELERATED_RECONSTRUCTIVE_RESEARCH"
                if args.mode == "historical"
                else "PRODUCTION_LIVE"
            ),
        )
        if args.audit_output:
            audit_path = Path(args.audit_output)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(
                    collection_result["coverage_audit"], indent=2, sort_keys=True
                ),
                encoding="utf-8",
            )
        print(json.dumps(collection_result, sort_keys=True))
        return
    if args.command == "event-study":
        from .db import ChannelHypothesis, EventStudyResultRecord
        from .event_study import EventStudyEngine, funding_spec_from_hypothesis

        migrate_database(args.database_url)
        session = make_session_factory(make_engine(args.database_url))()
        if args.action == "list":
            event_rows = session.scalars(
                select(EventStudyResultRecord).order_by(
                    EventStudyResultRecord.created_at
                )
            ).all()
            print(
                json.dumps(
                    [
                        {
                            "result_id": str(row.id),
                            "run_id": row.run_id,
                            "spec_id": row.spec_id,
                            "hypothesis_family_id": row.hypothesis_family_id,
                            "disposition": row.disposition,
                            "created_at": row.created_at,
                        }
                        for row in event_rows
                    ],
                    default=str,
                    sort_keys=True,
                )
            )
            return
        if args.action == "show":
            if not args.run_id:
                raise SystemExit("event-study show requires --run-id")
            event_result = session.scalar(
                select(EventStudyResultRecord).where(
                    EventStudyResultRecord.run_id == args.run_id
                )
            )
            if event_result is None:
                raise SystemExit("event-study run not found")
            print(
                json.dumps(
                    {
                        "result_id": str(event_result.id),
                        "run_id": event_result.run_id,
                        "spec_id": event_result.spec_id,
                        "disposition": event_result.disposition,
                        "effects": event_result.effects,
                        "robustness": event_result.robustness,
                        "methodology_critic": event_result.methodology_critic,
                        "artifact_uri": event_result.artifact_uri,
                    },
                    default=str,
                    sort_keys=True,
                )
            )
            return
        if args.start is None or args.end is None or not args.hypothesis_id:
            raise SystemExit(
                "event-study run requires --start, --end, and --hypothesis-id"
            )
        try:
            hypothesis = session.get(ChannelHypothesis, uuid.UUID(args.hypothesis_id))
        except ValueError as exc:
            raise SystemExit("event-study hypothesis-id must be a UUID") from exc
        if hypothesis is None:
            raise SystemExit("event-study hypothesis not found")
        if hypothesis.status != "TEST_READY":
            raise SystemExit("event-study hypothesis is not durably TEST_READY")
        start, end = normalize_utc(args.start), normalize_utc(args.end)
        client: LLMClient | None = (
            FakeLLMClient() if args.provider == "fake" else _phase18_client(settings)
        )
        spec = funding_spec_from_hypothesis(
            {
                "hypothesis_id": str(hypothesis.id),
                "hypothesis_family_id": hypothesis.fingerprint,
                "condition": hypothesis.condition,
                "outcome": hypothesis.outcome,
                "universe": hypothesis.universe,
                "horizon": hypothesis.horizon,
                "required_data": hypothesis.required_data,
                "falsification_criterion": hypothesis.falsification_criterion,
                "critic_disposition": "ACCEPT",
            },
            created_at=end,
            sample_start=start,
            sample_end=end,
        )
        print(
            json.dumps(
                EventStudyEngine(session, spec, client=client).run(
                    Path(args.output_dir)
                ),
                default=str,
                sort_keys=True,
            )
        )
        return
    if args.command == "knowledge":
        from .retrieval import hypothesis_lineage, search_hypotheses

        migrate_database(args.database_url)
        session = make_session_factory(make_engine(args.database_url))()
        knowledge_result = (
            search_hypotheses(
                session,
                args.value,
                scope=args.scope,
                channel=args.channel,
                maturity=args.maturity,
                as_of=args.as_of,
            )
            if args.action == "search"
            else hypothesis_lineage(
                session, args.value, scope=args.scope, as_of=args.as_of
            )
        )
        print(json.dumps(knowledge_result, indent=2, sort_keys=True, default=str))
        return
    if args.command == "phase16d-discover":
        if not 1 <= args.limit <= 100:
            raise SystemExit("--limit must be between 1 and 100")
        from .discovery import ingest_records as ingest_phase16d_records
        from .raw_archive import RawArchive

        migrate_database(args.database_url)
        session = make_session_factory(make_engine(args.database_url))()
        retrieved_at = datetime.now(UTC)
        collection_run = CollectionRun(
            source="phase16d-discover",
            requested=args.limit,
            status="RUNNING",
            diagnostics={
                "retrieval_scope": {
                    "adapters": [
                        "openalex",
                        "crossref",
                        "arxiv",
                        "nber",
                        "repec",
                        "alpha-architect",
                        "man-institute",
                        "aqr",
                    ],
                    "per_adapter_limit": args.limit,
                }
            },
        )
        session.add(collection_run)
        session.flush()
        records, source_status = collect_isolated(
            [
                OpenAlexSource(now=lambda: retrieved_at),
                CrossrefSource(),
                ArxivSource(lookback_days=settings.arxiv_lookback_days),
                NberSource(),
                RepecSource(),
                PractitionerRssSource(),
                InstitutionalHtmlSource(
                    name="man-institute", endpoint="https://www.man.com/insights"
                ),
                InstitutionalRssSource(
                    name="aqr", endpoint="https://www.aqr.com/Insights/Research"
                ),
            ],
            args.limit,
        )
        discovery_result = ingest_phase16d_records(
            session,
            records,
            retrieved_at=retrieved_at,
            archive=RawArchive(Path("data/raw")),
            collection_run_id=collection_run.id,
        )
        collection_run.retrieved = discovery_result["discovered"]
        collection_run.inserted = discovery_result["source_items"]
        collection_run.failed = discovery_result["archive_failures"]
        collection_run.status = "SUCCESS" if not collection_run.failed else "DEGRADED"
        collection_run.ended_at = datetime.now(UTC)
        session.commit()
        artifact = Path(args.output)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(
                {
                    "phase": "1.6D",
                    "retrieved_at": retrieved_at.isoformat(),
                    "source_status": source_status
                    | {
                        "ssrn": "UNAVAILABLE_EXTERNAL_ACCESS",
                        "x": "UNAVAILABLE_EXTERNAL_CREDENTIAL",
                    },
                    "result": discovery_result,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps({"artifact": str(artifact), **discovery_result}, sort_keys=True)
        )
        return
    if args.command == "phase16d-replay":
        if not 1 <= args.days <= 7:
            raise SystemExit("--days must be between 1 and 7")
        from .intelligence_v2 import run_intelligence_replay

        engine = make_engine(args.database_url)
        session = make_session_factory(engine)()
        days = [
            datetime.combine(
                args.start_date + timedelta(days=index), time.max, tzinfo=UTC
            )
            for index in range(args.days)
        ]
        v2_client: LLMClient
        if args.provider == "fake":
            v2_client = FakeLLMClient()
        else:
            if not settings.deepseek_api_key:
                raise SystemExit(
                    "DEEPSEEK_API_KEY is required for DeepSeek replay review"
                )
            v2_client = DeepSeekClient(
                settings.deepseek_api_key,
                ModelRouter(settings.llm_flash_model, settings.llm_pro_model),
                settings.deepseek_base_url,
                settings.llm_timeout_seconds,
                settings.http_retries,
            )
        replay_result = run_intelligence_replay(
            session, Path(args.output_dir), days, client=v2_client
        )
        print(
            json.dumps(
                {
                    "summary": str(Path(args.output_dir) / "phase16d-summary.json"),
                    "technical_success_count": replay_result["technical_success_count"],
                    "mode": replay_result["mode"],
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "phase16c-fast":
        from .fast import run_fast_walk_forward

        fast_client: LLMClient
        if args.provider == "fake":
            fast_client = FakeLLMClient()
        else:
            if not settings.deepseek_api_key:
                raise SystemExit("DEEPSEEK_API_KEY is required for Phase 1.6C-FAST")
            fast_client = DeepSeekClient(
                settings.deepseek_api_key,
                ModelRouter(settings.llm_flash_model, settings.llm_pro_model),
                settings.deepseek_base_url,
                settings.llm_timeout_seconds,
                settings.http_retries,
            )
        engine = make_engine(args.database_url)
        session = make_session_factory(engine)()
        fast_summary = run_fast_walk_forward(
            session, Path(args.output_dir), fast_client
        )
        print(
            json.dumps(
                {
                    "summary": str(
                        Path(args.output_dir) / "phase16c-fast-summary.json"
                    ),
                    "technical_success_count": fast_summary["technical_success_count"],
                    "market_fact_count": fast_summary["cross_day"]["market_fact_count"],
                    "hypothesis_count": fast_summary["cross_day"]["hypothesis_count"],
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "live-cycle":
        if not settings.deepseek_api_key:
            raise SystemExit("DEEPSEEK_API_KEY is required for live cycle")
        from .live import run_live_cycle

        engine = make_engine(args.database_url)
        session = make_session_factory(engine)()
        client = DeepSeekClient(
            settings.deepseek_api_key,
            ModelRouter(settings.llm_flash_model, settings.llm_pro_model),
            settings.deepseek_base_url,
            settings.llm_timeout_seconds,
            settings.http_retries,
        )
        print(
            json.dumps(
                run_live_cycle(
                    session, client, Path(args.output_dir), args.cycle, args.code_sha
                ),
                default=str,
            )
        )
        return
    if args.command == "rebuild-phase16a-summary":
        root = Path(args.output_dir)
        dates = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", path.name)
        )
        replay_days = [date.fromisoformat(item) for item in dates]
        if not replay_days:
            raise SystemExit("BLOCKED: no replay-day directories found")
        audits = [
            json.loads((root / day.isoformat() / "audit.json").read_text())
            for day in replay_days
        ]
        requested_end = args.requested_end
        if requested_end is None:
            raise SystemExit("BLOCKED: --requested-end is required")
        warmup_start = args.warmup_start
        if warmup_start is None:
            warmup_start = parse_utc_timestamp(
                str(audits[0]["warmup_start"]), "warmup_start"
            )
        path = write_summary(
            root,
            datetime.now(UTC),
            datetime.now(UTC),
            replay_days,
            warmup_start,
            audits,
            args.replay_code_sha,
            phase16a_run_id=args.phase16a_run_id,
            requested_end=requested_end,
            collection_code_sha=args.collection_code_sha,
            collection_start=args.collection_start,
            collection_end=args.collection_end,
        )
        print(f"SUMMARY={path}")
        return
    if args.command == "replay":
        cutoff = args.as_of.astimezone(UTC)
        collection_end = args.collection_end.astimezone(UTC)
        if args.provider == "fake" or (
            args.provider is None and settings.llm_provider == "fake"
        ):
            replay_client: FakeLLMClient | DeepSeekClient = FakeLLMClient()
        else:
            if not settings.deepseek_api_key:
                raise SystemExit("DEEPSEEK_API_KEY is required for live replay")
            replay_client = DeepSeekClient(
                settings.deepseek_api_key,
                ModelRouter(settings.llm_flash_model, settings.llm_pro_model),
                settings.deepseek_base_url,
                settings.llm_timeout_seconds,
                settings.http_retries,
            )
        warmup_start = args.warmup_start or cutoff - timedelta(days=30)
        database_url = settings.database_url
        migrate_database(database_url)
        engine = make_engine(database_url)
        session = make_session_factory(engine)()
        diagnostics_run = get_phase16a_collection_run(
            session,
            source="hyperliquid",
            phase16a_run_id=args.phase16a_run_id,
            requested_start=warmup_start,
            requested_end=collection_end,
            collection_code_sha=args.collection_code_sha,
        )
        if diagnostics_run is None or not diagnostics_run.diagnostics:
            raise SystemExit(
                "BLOCKED: matching Hyperliquid funding diagnostics are missing"
            )
        coverage = funding_coverage(
            session, warmup_start, cutoff, diagnostics_run.diagnostics
        )
        print(json.dumps(coverage, indent=2, default=str))
        if args.coverage_only:
            if not all(item["required_warmup_satisfied"] for item in coverage.values()):
                raise SystemExit(
                    "PARTIAL: required bounded funding warm-up is unavailable"
                )
            return
        audit = run_replay_day(
            session,
            replay_client,
            Path(args.output_dir),
            args.date,
            warmup_start,
            args.replay_code_sha,
            pagination_diagnostics=diagnostics_run.diagnostics,
            collection_run_id=diagnostics_run.phase16a_run_id,
            collection_code_sha=diagnostics_run.code_sha,
            collection_start=diagnostics_run.requested_start,
            collection_end=diagnostics_run.requested_end,
        )
        print(json.dumps(audit, indent=2, default=str))
        return
    database_url = (
        args.database_url
        if args.command == "validate-llm-routing"
        else settings.database_url
    )
    migrate_database(database_url)
    engine = make_engine(database_url)
    session = make_session_factory(engine)()

    if args.command == "validate-llm-routing":
        router = ModelRouter(settings.llm_flash_model, settings.llm_pro_model)
        roles = ["TRIAGE", "TUTOR", "ANALYST", "CRITIC"]
        configured: dict[str, object] = {}
        result: dict[str, object] = {
            "validation_timestamp": datetime.now(UTC).isoformat(),
            "provider": "deepseek",
            "endpoint_type": "chat_completions",
            "mock": {"roles": configured},
        }
        for role in roles:
            config = router.resolve(role)
            configured[role] = {
                "requested_model": config.model,
                "thinking": config.thinking,
                "reasoning_effort": config.reasoning_effort,
                "max_output_tokens": config.max_output_tokens,
                "status": "PASS",
            }
        if args.live:
            if not settings.deepseek_api_key:
                result["live_status"] = "NOT_RUN_NO_CREDENTIAL"
            else:
                live_client = DeepSeekClient(
                    settings.deepseek_api_key,
                    router,
                    settings.deepseek_base_url,
                    settings.llm_timeout_seconds,
                    settings.http_retries,
                )
                live: dict[str, object] = {}
                for role in roles:
                    config = router.resolve(role)
                    try:
                        if role == "TRIAGE":
                            live_client.triage(
                                "Liquidity screening", "A bounded market observation."
                            )
                        elif role == "TUTOR":
                            live_client.tutor("Explain point-in-time data.")
                        elif role == "ANALYST":
                            live_client.analyze(
                                "A bounded research question",
                                "A source-reported result.",
                            )
                        else:
                            live_client.critique(
                                "A candidate hypothesis about a measured relationship."
                            )
                        live[role] = {
                            "requested_model": config.model,
                            "actual_model": None,
                            "thinking": config.thinking,
                            "reasoning_effort": config.reasoning_effort,
                            "schema": "PASS",
                            "result": "PASS",
                        }
                    except Exception as exc:
                        live[role] = {
                            "requested_model": config.model,
                            "actual_model": None,
                            "thinking": config.thinking,
                            "reasoning_effort": config.reasoning_effort,
                            "schema": "FAIL",
                            "result": "FAIL",
                            "failure": str(exc),
                        }
                result["live"] = {"roles": live}
                result["live_status"] = (
                    "PASS"
                    if all(
                        isinstance(item, dict) and item["result"] == "PASS"
                        for item in live.values()
                    )
                    else "PARTIAL"
                )
        result["overall"] = result.get("live_status", "PASS") if args.live else "PASS"
        artifact = Path(args.artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "init-db":
        print("Database initialized")
    elif args.command == "collect":
        adapters = {
            "arxiv": ArxivSource(lookback_days=settings.arxiv_lookback_days),
            "repec": RepecSource(),
            "hyperliquid": HyperliquidSource(),
        }
        limit = args.limit or getattr(settings, f"{args.source}_max_items", 10)
        adapter = cast("SourceAdapter", adapters[args.source])
        if args.history and args.source == "hyperliquid":
            source = cast(HyperliquidSource, adapter)
            history_records = source.collect_history(
                limit, offline=args.offline, start=args.start, end=args.end
            )
            run = CollectionRun(
                source="hyperliquid",
                requested=limit,
                status="RUNNING",
                requested_start=args.start,
                requested_end=args.end,
                code_sha=os.environ.get("PHASE16A_SHA", "unknown"),
                phase16a_run_id=args.phase16a_run_id
                or os.environ.get("PHASE16A_RUN_ID"),
                diagnostics=source.last_funding_diagnostics,
            )
            session.add(run)
            session.flush()
            inserted, duplicates = ingest_records(
                session, history_records, collection_run_id=run.id
            )
            inserted_candles, candle_duplicates = ingest_records(
                session,
                source.collect_candles(
                    limit, offline=args.offline, start=args.start, end=args.end
                ),
                collection_run_id=run.id,
            )
            run.retrieved = len(history_records)
            run.inserted = inserted + inserted_candles
            run.skipped_duplicates = duplicates + candle_duplicates
            run.status = "SUCCESS"
            run.ended_at = datetime.now(UTC)
            session.commit()
            print(
                json.dumps(
                    {
                        "funding_inserted": inserted,
                        "funding_duplicates": duplicates,
                        "candle_inserted": inserted_candles,
                        "candle_duplicates": candle_duplicates,
                    }
                )
            )
        else:
            print(f"Inserted {ingest(session, adapter, limit, args.offline)} records")
        calculate_metrics(session, datetime.now(UTC))
    elif args.command == "analyze":
        from .intelligence_v2 import run_phase18_intelligence_cycle

        audit = run_phase18_intelligence_cycle(
            session,
            Path(settings.report_output_dir) / "phase18-analyze",
            datetime.now(UTC),
            client=_phase18_client(settings),
        )
        print(json.dumps(audit, default=str, sort_keys=True))
    elif args.command == "run-daily":
        arxiv, repec, hyperliquid = (
            ArxivSource(lookback_days=settings.arxiv_lookback_days),
            RepecSource(),
            HyperliquidSource(lookback_hours=48),
        )
        ingest(session, arxiv, 5, args.offline)
        ingest(session, repec, 5, args.offline)
        ingest(session, hyperliquid, 5, args.offline)
        ingest_records(session, hyperliquid.collect_history(30, offline=args.offline))
        ingest_records(session, hyperliquid.collect_candles(30, offline=args.offline))
        calculate_metrics(session, datetime.now(UTC))
        from .intelligence_v2 import run_phase18_intelligence_cycle

        run_phase18_intelligence_cycle(
            session,
            Path(settings.report_output_dir) / "phase18-daily",
            datetime.now(UTC),
            client=_phase18_client(settings),
        )
        print(
            daily_report(session, settings.report_output_dir, as_of=datetime.now(UTC))
        )
    elif args.command == "report":
        print(
            daily_report(session, settings.report_output_dir, as_of=datetime.now(UTC))
            if args.kind == "daily"
            else weekly_report(session, settings.report_output_dir)
        )


if __name__ == "__main__":
    main()
