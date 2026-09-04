"""Low-frequency user actionability classification (leaf module, no deps).

Endpoint-aware: a free-form horizon such as "4h and 24h" is decomposed into
per-endpoint fits so the 24h endpoint is not suppressed by the 4h endpoint.
User actionability fit is deliberately separate from scientific strength.
"""

import re
from typing import Any

FIT_OUT_OF_SCOPE = "OUT_OF_SCOPE_FOR_USER"
FIT_LOW = "LOW_FIT"
FIT_MEDIUM = "MEDIUM_FIT"
FIT_HIGH = "HIGH_FIT"

_FIT_RANK = {FIT_OUT_OF_SCOPE: 0, FIT_LOW: 1, FIT_MEDIUM: 2, FIT_HIGH: 3}

_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|hours?|hrs?|h|days?|d)\b",
    re.IGNORECASE,
)


def _fit_for_minutes(minutes: float) -> str:
    hours = minutes / 60
    if hours < 2:
        return FIT_OUT_OF_SCOPE
    if hours < 12:
        return FIT_LOW
    if hours < 24:
        return FIT_MEDIUM
    return FIT_HIGH


def parse_horizon_endpoints(horizon: str | None) -> list[dict[str, Any]]:
    """Split a free-form horizon into structured per-endpoint fits.

    Returns records like [{"value": 4, "unit": "h", "fit": "LOW_FIT"}].
    Ranges ("4h and 24h", "1h / 4h / 24h") yield one record per endpoint;
    unparseable text yields an empty list (caller decides fallback).
    """
    if not horizon:
        return []
    records: list[dict[str, Any]] = []
    seen: set[tuple[float, str]] = set()
    for match in _UNIT.finditer(horizon.lower()):
        number = float(match.group(1))
        unit = match.group(2)
        if unit.startswith("d"):
            minutes = number * 1440
            unit_label = "d"
        elif unit.startswith("h"):
            minutes = number * 60
            unit_label = "h"
        else:
            minutes = number
            unit_label = "min"
        key = (minutes, unit_label)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "value": number,
                "unit": unit_label,
                "minutes": minutes,
                "fit": _fit_for_minutes(minutes),
            }
        )
    return records


def low_frequency_fit(horizon: str | None) -> str:
    """Best (most actionable) endpoint fit for a horizon string.

    A hypothesis stays visible when ANY endpoint falls in the user's preferred
    1d-30d band (e.g. "4h and 24h" is not suppressed by its 4h endpoint).
    """
    if not horizon:
        return FIT_OUT_OF_SCOPE
    endpoints = parse_horizon_endpoints(horizon)
    if not endpoints:
        return FIT_OUT_OF_SCOPE
    best = max(endpoints, key=lambda item: _FIT_RANK[str(item["fit"])])
    return str(best["fit"])
