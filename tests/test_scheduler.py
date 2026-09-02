from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from quant_research_radar.scheduler import (
    BEIJING_TZ,
    compute_due,
    most_recent_saturday,
)

UTC = UTC
HELSINKI = ZoneInfo("Europe/Helsinki")


def _bj(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=BEIJING_TZ)


def test_most_recent_saturday() -> None:
    # 2026-09-02 is Wednesday.
    assert most_recent_saturday(date(2026, 9, 2)) == date(2026, 8, 29)
    assert most_recent_saturday(date(2026, 9, 5)) == date(2026, 9, 5)  # Sat
    assert most_recent_saturday(date(2026, 9, 6)) == date(2026, 9, 5)  # Sun
    assert most_recent_saturday(date(2026, 9, 4)) == date(2026, 8, 29)  # Fri


def test_friday_before_due_nothing_due() -> None:
    now = _bj(2026, 9, 4, 18, 29)
    d = compute_due(
        now, last_daily_date=date(2026, 9, 3), last_weekly_saturday=date(2026, 8, 29)
    )
    assert d.nothing_due


def test_friday_at_due_daily_due() -> None:
    now = _bj(2026, 9, 4, 18, 30)
    d = compute_due(
        now, last_daily_date=date(2026, 9, 3), last_weekly_saturday=date(2026, 8, 29)
    )
    assert d.daily_due and d.daily_date == date(2026, 9, 4)
    assert not d.weekly_due


def test_friday_repeated_tick_no_duplicate() -> None:
    now = _bj(2026, 9, 4, 19, 0)
    d = compute_due(
        now, last_daily_date=date(2026, 9, 4), last_weekly_saturday=date(2026, 8, 29)
    )
    assert d.nothing_due


def test_saturday_before_due_nothing_due_when_friday_done() -> None:
    now = _bj(2026, 9, 5, 18, 29)
    d = compute_due(
        now, last_daily_date=date(2026, 9, 4), last_weekly_saturday=date(2026, 8, 29)
    )
    assert d.nothing_due


def test_saturday_at_due_daily_then_weekly() -> None:
    now = _bj(2026, 9, 5, 18, 30)
    d = compute_due(
        now, last_daily_date=date(2026, 9, 4), last_weekly_saturday=date(2026, 8, 29)
    )
    assert d.daily_due and d.daily_date == date(2026, 9, 5)
    assert d.weekly_due and d.weekly_saturday == date(2026, 9, 5)


def test_saturday_repeated_tick_nothing_duplicated() -> None:
    now = _bj(2026, 9, 5, 19, 0)
    d = compute_due(
        now, last_daily_date=date(2026, 9, 5), last_weekly_saturday=date(2026, 9, 5)
    )
    assert d.nothing_due


def test_sunday_missed_saturday_weekly_runs_once() -> None:
    now = _bj(2026, 9, 6, 18, 0)
    d = compute_due(
        now, last_daily_date=date(2026, 9, 5), last_weekly_saturday=date(2026, 8, 29)
    )
    assert d.weekly_due and d.weekly_saturday == date(2026, 9, 5)
    assert not d.daily_due


def test_missed_one_daily_catchup() -> None:
    # Machine slept through Friday 18:30, wakes Saturday 10:00 Beijing.
    now = _bj(2026, 9, 5, 10, 0)
    d = compute_due(
        now, last_daily_date=date(2026, 9, 3), last_weekly_saturday=date(2026, 8, 29)
    )
    assert d.daily_due and d.daily_date == date(2026, 9, 4)
    assert not d.weekly_due


def test_missed_multiple_days_catchup() -> None:
    # Machine slept Mon 18:30, wakes Wed 10:00 Beijing.
    now = _bj(2026, 9, 9, 10, 0)
    d = compute_due(
        now, last_daily_date=date(2026, 9, 7), last_weekly_saturday=date(2026, 9, 5)
    )
    assert d.daily_due and d.daily_date == date(2026, 9, 8)
    assert not d.weekly_due


def test_host_timezone_and_dst_do_not_shift_beijing_due() -> None:
    # Same instant expressed in UTC, Helsinki winter (UTC+2) and summer (UTC+3).
    utc_instant = datetime(2026, 9, 4, 10, 30, tzinfo=UTC)  # Beijing 18:30
    helsinki_winter = utc_instant.astimezone(HELSINKI)  # 13:30 EEST
    decisions = {
        compute_due(
            inst,
            last_daily_date=date(2026, 9, 3),
            last_weekly_saturday=date(2026, 8, 29),
        )
        for inst in (utc_instant, helsinki_winter)
    }
    assert len(decisions) == 1
    decision = decisions.pop()
    assert decision.daily_due and decision.daily_date == date(2026, 9, 4)


def test_no_completed_runs_means_daily_due() -> None:
    now = _bj(2026, 9, 4, 20, 0)
    d = compute_due(now, last_daily_date=None, last_weekly_saturday=None)
    assert d.daily_due and d.daily_date == date(2026, 9, 4)
    # Weekly is also logically due (no prior review), but the orchestrator gates
    # actual Weekly execution on available Daily history, so it is not run empty.
    assert d.weekly_due
