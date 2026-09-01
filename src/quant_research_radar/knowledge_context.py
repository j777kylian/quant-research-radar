"""Typed, scope-safe prior research context for the Phase 1.8 engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import ChannelHypothesis, normalize_utc


class Novelty(StrEnum):
    NEW = "NEW"
    RECURRENT = "RECURRENT"
    ASSET_VARIANT = "ASSET_VARIANT"
    PERSISTENT_REGIME = "PERSISTENT_REGIME"
    PREVIOUSLY_REJECTED = "PREVIOUSLY_REJECTED"
    KNOWN_DATA_GAP = "KNOWN_DATA_GAP"
    POSSIBLE_DUPLICATE = "POSSIBLE_DUPLICATE"


@dataclass(frozen=True)
class PriorResearchContext:
    """Prior records only: never a source-evidence or support-count input."""

    related: tuple[ChannelHypothesis, ...]
    novelty: Novelty
    occurrence_count: int

    @property
    def hypothesis_ids(self) -> list[str]:
        return [str(row.id) for row in self.related]


def prior_research_context(
    session: Session,
    fingerprint: str,
    universe: str,
    as_of: datetime,
    channel: str,
) -> PriorResearchContext:
    """Production-only past records, bounded by the candidate's explicit PIT clock."""
    rows = tuple(
        session.scalars(
            select(ChannelHypothesis)
            .where(
                ChannelHypothesis.analysis_mode == "PRODUCTION_LIVE",
                ChannelHypothesis.fingerprint == fingerprint,
                ChannelHypothesis.as_of < normalize_utc(as_of),
            )
            .order_by(ChannelHypothesis.as_of, ChannelHypothesis.created_at)
        ).all()
    )
    if not rows:
        return PriorResearchContext(rows, Novelty.NEW, 0)
    if any(row.maturity == "REJECTED" for row in rows):
        novelty = Novelty.PREVIOUSLY_REJECTED
    elif any(row.universe == universe for row in rows):
        novelty = (
            Novelty.PERSISTENT_REGIME if channel == "MARKET" else Novelty.RECURRENT
        )
    else:
        novelty = Novelty.ASSET_VARIANT
    return PriorResearchContext(rows, novelty, len(rows))
