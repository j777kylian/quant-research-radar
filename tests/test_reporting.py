"""Reporting-semantics regressions: final state, interval, sources, regen."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    ChannelHypothesis,
    CollectionRun,
    DailyRun,
    WeeklyRun,
)
from quant_research_radar.operations import regenerate_report
from quant_research_radar.reporting import (
    collect_daily_snapshot,
    daily_conclusion,
    render_daily_markdown,
    render_weekly_markdown,
)


def _session(tmp_path: Path) -> tuple[Session, Path]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine), tmp_path


def _daily(
    session: Session,
    *,
    status: str = "SUCCESS",
    when: date | None = None,
    root: Path | None = None,
) -> DailyRun:
    base = root or Path("/tmp/qrr-reporting-isolated")
    run = DailyRun(
        logical_date=when or date(2026, 9, 3),
        status=status,
        started_at=datetime.now(UTC) - timedelta(minutes=2),
        ended_at=datetime.now(UTC),
        code_sha="test-sha",
        market_status="SUCCESS",
        academic_status="DEGRADED",
        practitioner_status="DEGRADED",
        analysis_status="SUCCESS",
        knowledge_status="SUCCESS",
        audit_status="SUCCESS",
        failure_reasons=[],
        report_path=str(base / "daily" / "2026-09-03" / "report.md"),
    )
    session.add(run)
    session.commit()
    return run


def _market_run(session: Session, daily: DailyRun) -> None:
    session.add(
        CollectionRun(
            source="hyperliquid",
            status="SUCCESS",
            requested_start=datetime(2026, 9, 2, 10, 0, tzinfo=UTC),
            requested_end=datetime(2026, 9, 3, 10, 0, tzinfo=UTC),
            requested=0,
            retrieved=144,
            inserted=132,
            skipped_duplicates=12,
            started_at=daily.started_at,
            ended_at=daily.ended_at,
            diagnostics={"analysis_mode": "PRODUCTION_LIVE"},
        )
    )
    session.commit()


def test_final_report_shows_success_not_running(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    daily = _daily(session, root=tmp_path)
    _market_run(session, daily)
    snapshot = collect_daily_snapshot(session, daily.id)
    assert snapshot["final_status"] == "SUCCESS"
    text = render_daily_markdown(snapshot)
    assert "Final run status:** SUCCESS" in text
    assert "RUNNING" not in text


def test_degraded_daily_is_explicit(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    daily = DailyRun(
        logical_date=date(2026, 9, 3),
        status="PARTIAL",
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        code_sha="s",
        market_status="FAILED",
        academic_status="DEGRADED",
        practitioner_status="SUCCESS",
        analysis_status="SUCCESS",
        knowledge_status="SUCCESS",
        audit_status="FAILED",
        failure_reasons=["market: connection refused"],
    )
    session.add(daily)
    session.commit()
    snapshot = collect_daily_snapshot(session, daily.id)
    text = render_daily_markdown(snapshot)
    assert "Final run status:** PARTIAL" in text
    assert "connection refused" in text


def test_market_interval_renders_from_collection_run(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    daily = _daily(session, root=tmp_path)
    _market_run(session, daily)
    snapshot = collect_daily_snapshot(session, daily.id)
    market = snapshot["market"]
    # SQLite strips tzinfo on read; compare the instant part.
    assert market["start"] and market["start"].startswith("2026-09-02T10:00:00")
    assert market["end"] and market["end"].startswith("2026-09-03T10:00:00")
    assert market["retrieved"] == 144 and market["inserted"] == 132
    assert market["duplicates"] == 12
    assert market["analysis_mode"] == "PRODUCTION_LIVE"
    rendered = render_daily_markdown(snapshot)
    assert "2026-09-02T10:00:00" in rendered and "2026-09-03T10:00:00" in rendered


def test_no_market_provenance_does_not_fake_interval(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    daily = _daily(session, root=tmp_path)
    daily.source_health = {}  # nothing persisted
    session.commit()
    snapshot = collect_daily_snapshot(session, daily.id)
    # Without a CollectionRun the interval must stay unset, never invented.
    assert snapshot["market"].get("start") is None or "n/a" in render_daily_markdown(
        snapshot
    )


def test_zero_hypothesis_day_is_valid(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    daily = _daily(session, root=tmp_path)
    _market_run(session, daily)
    snapshot = collect_daily_snapshot(session, daily.id)
    assert snapshot["research"]["channel_hypotheses"] == 0
    assert "no new or recurrent hypotheses" in render_daily_markdown(snapshot).lower()


def test_recurrent_hypothesis_labeled_recurrent(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    daily = _daily(session, root=tmp_path)
    audit_dir = Path(daily.report_path).resolve().parent / "intelligence"
    audit_dir.mkdir(parents=True)
    (audit_dir / "audit.json").write_text(
        json.dumps(
            {
                "new_hypothesis_families": ["market|family-a|btc|4h and 24h"],
                "technical_status": "CRITIC_REQUEST_DATA",
                "critics": {"methodology_critic": {"disposition": "REQUEST_DATA"}},
                "knowledge": {
                    "prior_context": [
                        {"novelty": "NEW", "occurrence_count": 0},
                        {"novelty": "RECURRENT", "occurrence_count": 2},
                    ]
                },
            }
        )
    )
    session.add(
        ChannelHypothesis(
            channel="MARKET",
            statement="s",
            condition="c",
            outcome="24h return",
            universe="BTC perpetual",
            horizon="4h and 24h",
            falsification_criterion="none",
            maturity="H1_STATISTICAL_HYPOTHESIS",
            status="DISCOVERED",
            fingerprint="market|family-a|btc|4h and 24h",
            analysis_mode="PRODUCTION_LIVE",
            availability_basis="RECEIPT_TIME",
            as_of=datetime(2026, 9, 3, 10, 0, tzinfo=UTC),
        )
    )
    session.commit()
    snapshot = collect_daily_snapshot(session, daily.id)
    research = snapshot["research"]
    assert research["new_count"] == 1
    assert research["recurrent_count"] == 1
    # "4h and 24h" must parse as LOW (4h), NOT 4 days high-fit (the 'd' in 'and').
    assert research["high_fit_count"] == 0


def test_source_capability_vs_run_outcome_distinct(tmp_path: Path) -> None:
    from quant_research_radar.operations import ops_status

    session, _ = _session(tmp_path)
    daily = _daily(session, root=tmp_path)
    daily.source_health = {
        "academic": {"nber": "DEGRADED", "openalex": "READY"},
        "practitioner": {"aqr": "DEGRADED"},
    }
    session.commit()
    status = ops_status(
        session, settings=type("S", (), {"publication_mode": "DRAFT_ONLY"})()
    )
    # Capability comes from the registry; run outcomes from the last DailyRun.
    assert "source_capability" in status and "last_daily_source_run_outcomes" in status
    assert status["last_daily_source_run_outcomes"].get("nber") == "DEGRADED"


def test_regenerate_uses_persisted_state_no_research_mutation(tmp_path: Path) -> None:
    session, tmp = _session(tmp_path)
    daily = _daily(session, root=tmp_path)
    _market_run(session, daily)
    before = session.scalars(select(ChannelHypothesis)).all()
    result = regenerate_report(
        session, logical_date="2026-09-03", output_root=tmp / "daily"
    )
    assert result["network_calls"] == 0 and result["llm_calls"] == 0
    report_path = Path(result["report"])
    assert report_path.exists()
    assert "Final run status:** SUCCESS" in report_path.read_text()
    # No new hypotheses/occurrences were created by regeneration.
    assert session.scalars(select(ChannelHypothesis)).all() == before


def test_regenerate_refuses_running_run(tmp_path: Path) -> None:
    session, tmp = _session(tmp_path)
    _daily(session, status="RUNNING", root=tmp_path)
    try:
        regenerate_report(session, logical_date="2026-09-03", output_root=tmp / "daily")
        raise AssertionError("must refuse RUNNING final render")
    except RuntimeError as error:
        assert "RUNNING" in str(error)


def test_weekly_markdown_lists_included_and_bounded(tmp_path: Path) -> None:
    session, _ = _session(tmp_path)
    weekly = WeeklyRun(
        week_saturday=date(2026, 9, 5),
        status="SUCCESS",
        code_sha="s",
        included_daily_dates=["2026-09-01", "2026-09-02", "2026-09-03"],
        priorities=[
            {
                "fit": "HIGH_FIT",
                "hypothesis_family": "market|f",
                "statement": "x",
                "horizon": "24h",
                "maturity": "H1",
                "prior_empirical_disposition": None,
            }
        ],
    )
    session.add(weekly)
    session.commit()
    from quant_research_radar.reporting import collect_weekly_snapshot

    snapshot = collect_weekly_snapshot(session, weekly.id)
    text = render_weekly_markdown(snapshot)
    assert "2026-09-03" in text
    assert "Top Research Priorities" in text
    assert len(snapshot["priorities"]) <= 5


def test_daily_conclusion_zero_output() -> None:
    assert (
        "No high-priority low-frequency research candidate emerged today"
        in daily_conclusion(0, 0, 0)
    )
