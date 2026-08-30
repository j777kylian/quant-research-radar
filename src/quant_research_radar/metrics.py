from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from math import sqrt


def _ordered(
    values: Iterable[tuple[datetime, float | None]],
) -> list[tuple[datetime, float]]:
    return sorted(
        ((timestamp, value) for timestamp, value in values if value is not None),
        key=lambda x: x[0],
    )


def funding_percentile(
    history: Iterable[tuple[datetime, float | None]], at: datetime, window: int = 30
) -> float | None:
    values = [value for timestamp, value in _ordered(history) if timestamp <= at][
        -window:
    ]
    if not values:
        return None
    current = values[-1]
    return 100.0 * sum(value <= current for value in values) / len(values)


def return_at(prices: dict[datetime, float], at: datetime, hours: int) -> float | None:
    current = prices.get(at)
    previous = prices.get(at - timedelta(hours=hours))
    if current is None or previous is None or previous == 0:
        return None
    return current / previous - 1.0


def rolling_volatility(
    prices: dict[datetime, float], at: datetime, window: int = 24
) -> float | None:
    ordered = [
        (at - timedelta(hours=offset), prices.get(at - timedelta(hours=offset)))
        for offset in range(window, -1, -1)
    ]
    if any(price is None for _, price in ordered):
        return None
    returns: list[float] = []
    for (_, previous), (_, current) in zip(ordered, ordered[1:], strict=False):
        assert previous is not None and current is not None
        if current <= 0 or previous <= 0:
            continue
        returns.append(current / previous - 1.0)
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    return sqrt(sum((value - mean) ** 2 for value in returns) / (len(returns) - 1))
