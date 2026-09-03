"""Low-frequency user actionability classification (leaf module, no deps)."""

import re

# Low-frequency actionability classification (1d-30d preferred horizon).
FIT_OUT_OF_SCOPE = "OUT_OF_SCOPE_FOR_USER"
FIT_LOW = "LOW_FIT"
FIT_MEDIUM = "MEDIUM_FIT"
FIT_HIGH = "HIGH_FIT"

_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|hours?|hrs?|h|days?|d)\b",
    re.IGNORECASE,
)


def low_frequency_fit(horizon: str | None) -> str:
    """Map a hypothesis horizon string to the user's actionability-fit bucket.

    Tokenizes real unit words so that ranges such as "4h and 24h" do not
    misparse the "d" inside "and". The SMALLEST duration mentioned is used as
    the required reaction time (most conservative). User actionability fit is
    deliberately separate from scientific strength: a short-horizon finding is
    demoted, never deleted.
    """
    if not horizon:
        return FIT_OUT_OF_SCOPE
    smallest: float | None = None
    for match in _UNIT.finditer(horizon.lower()):
        number = float(match.group(1))
        unit = match.group(2)
        minutes = number * 60 if unit.startswith("h") else number
        if unit.startswith("d"):
            minutes = number * 1440
        smallest = minutes if smallest is None else min(smallest, minutes)
    if smallest is None:
        return FIT_OUT_OF_SCOPE
    hours = smallest / 60
    if hours < 2:
        return FIT_OUT_OF_SCOPE
    if hours < 12:
        return FIT_LOW
    if hours < 24:
        return FIT_MEDIUM
    return FIT_HIGH
