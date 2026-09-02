"""Asia/Shanghai-anchored operations scheduling, independent of the host timezone.

The host Mac may sit in another timezone and may enter/leave DST. All due-state
decisions are computed from the logical clock in Asia/Shanghai (which has no
DST), never from host-local wall-clock time, so the Beijing 18:30 schedule does
not drift when the host timezone or DST changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

BEIJING_TZ: ZoneInfo = ZoneInfo("Asia/Shanghai")

DAILY_DUE_TIME: time = time(18, 30)
WEEKLY_DUE_TIME: time = time(18, 30)
WEEKLY_WEEKDAY: int = 5  # Saturday (Monday == 0)

# A scheduler tick is a cheap state check, never a collection/LLM run.
TICK_INTERVAL_MINUTES: int = 30


@dataclass(frozen=True)
class DueDecision:
    daily_due: bool
    weekly_due: bool
    daily_date: date  # logical Beijing date the Daily run would target
    weekly_saturday: date  # the Saturday the Weekly review would cover

    @property
    def nothing_due(self) -> bool:
        return not self.daily_due and not self.weekly_due


def beijing_now(now: datetime | None = None) -> datetime:
    """Return the current instant expressed in the Asia/Shanghai clock."""
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        raise ValueError("beijing_now requires a timezone-aware instant")
    return instant.astimezone(BEIJING_TZ)


def beijing_date(now: datetime | None = None) -> date:
    return beijing_now(now).date()


def most_recent_saturday(day: date) -> date:
    """The Saturday on or before ``day`` (the week-ending anchor)."""
    return day - timedelta(days=(day.weekday() - WEEKLY_WEEKDAY) % 7)


def compute_due(
    now: datetime,
    last_daily_date: date | None,
    last_weekly_saturday: date | None,
) -> DueDecision:
    """Compute what is logically due at ``now`` (any timezone-aware instant).

    ``last_daily_date`` is the most recent Daily run's logical Beijing date.
    ``last_weekly_saturday`` is the most recent Weekly review's week-ending
    Saturday. Both may be ``None`` when nothing has ever completed.

    Daily is due for the most recent logical Beijing date whose 18:30 has passed
    and which has not yet completed. Weekly is due for the most recent Saturday
    (after its 18:30 on Saturday itself) that has not yet completed.
    """
    bj = beijing_now(now)
    today = bj.date()
    past_daily_due = bj.time() >= DAILY_DUE_TIME

    daily_date = today if past_daily_due else today - timedelta(days=1)
    daily_due = last_daily_date is None or daily_date > last_daily_date

    saturday = most_recent_saturday(today)
    weekly_due = last_weekly_saturday is None or saturday > last_weekly_saturday
    if today.weekday() == WEEKLY_WEEKDAY and not past_daily_due:
        # On Saturday the Weekly review is due only after 18:30, and always
        # after the Saturday Daily has completed.
        weekly_due = False

    return DueDecision(
        daily_due=daily_due,
        weekly_due=weekly_due,
        daily_date=daily_date,
        weekly_saturday=saturday,
    )
