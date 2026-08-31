from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Channel(StrEnum):
    ACADEMIC = "ACADEMIC"
    SOCIAL = "SOCIAL"
    MARKET = "MARKET"


class Maturity(StrEnum):
    H0 = "H0_OBSERVATION"
    H1 = "H1_STATISTICAL_HYPOTHESIS"
    H2 = "H2_MECHANISM_BACKED"
    H3 = "H3_CONVERGENT"


class Evidence(BaseModel):
    evidence_id: str
    channel: Channel
    title: str
    body: str
    provenance_id: str
    independence_key: str
    observed_at: object
    metadata: dict[str, Any] = Field(default_factory=dict)


class HypothesisDraft(BaseModel):
    statement: str
    origin: Channel
    maturity: Maturity
    mechanism: str | None = None
    condition: str
    outcome: str
    universe: str
    horizon: str
    expected_direction: str | None = None
    required_data: list[str]
    falsification_criterion: str
    evidence_ids: list[str]
    supporting_channels: list[Channel]
    independence_keys: list[str]
    semantic_claim_key: str | None = None

    @classmethod
    def from_market(cls, evidence: Evidence) -> HypothesisDraft:
        asset = str(evidence.metadata["asset"])
        percentile = float(evidence.metadata["funding_percentile"])
        if percentile < 90:
            raise ValueError(
                "market evidence does not meet the extreme funding threshold"
            )
        return cls(
            statement=(
                f"Extreme {asset} funding changes the distribution of subsequent "
                "returns relative to ordinary funding."
            ),
            origin=Channel.MARKET,
            maturity=Maturity.H1,
            condition=f"{asset} funding percentile >= 90",
            outcome="subsequent 4h, 24h return distribution",
            universe=f"{asset} perpetual",
            horizon="4h and 24h",
            required_data=["PIT-safe funding rate", "completed 1h candles"],
            falsification_criterion=(
                "Conditional subsequent-return distributions are indistinguishable "
                "from the pre-specified ordinary-funding baseline."
            ),
            evidence_ids=[evidence.evidence_id],
            supporting_channels=[Channel.MARKET],
            independence_keys=[evidence.independence_key],
            semantic_claim_key=f"funding-extreme:{asset.lower()}:return-distribution",
        )

    @classmethod
    def from_research(cls, evidence: Evidence, channel: Channel) -> HypothesisDraft:
        """Convert retained single-channel evidence into a testable, non-causal H1."""
        if evidence.channel != channel:
            raise ValueError(f"{channel.value} analyzer received foreign evidence")
        title = evidence.title.strip()
        if not title or not evidence.body.strip():
            raise ValueError("research evidence requires title and supplied content")
        topic = str(evidence.metadata.get("research_topic") or title)
        return cls(
            statement=(
                f"The measurable market relationship described by '{topic}' is associated "
                "with a non-baseline subsequent-return distribution."
            ),
            origin=channel,
            maturity=Maturity.H1,
            condition="Pre-specified measurable condition derived from retained source evidence",
            outcome="subsequent return or volatility distribution",
            universe="Universe and instrument class specified before empirical testing",
            horizon="Pre-specified during empirical design",
            required_data=["source-defined condition", "market outcome data"],
            falsification_criterion=(
                "The pre-specified conditional distribution is indistinguishable from "
                "the declared baseline in a suitably powered test."
            ),
            evidence_ids=[evidence.evidence_id],
            supporting_channels=[channel],
            independence_keys=[evidence.independence_key],
            semantic_claim_key=str(evidence.metadata.get("semantic_claim_key") or "")
            or None,
        )


class FusionInput(BaseModel):
    draft: HypothesisDraft
    evidence_independence_keys: list[str]


class UnifiedHypothesis(HypothesisDraft):
    independent_evidence_count: int


def _channel_only(evidence: list[Evidence], channel: Channel) -> None:
    if any(item.channel != channel for item in evidence):
        raise ValueError(
            f"{channel.value} analyzer accepts {channel.value} evidence only"
        )


def _market_only(evidence: list[Evidence]) -> None:
    _channel_only(evidence, Channel.MARKET)


def _analyze_research(
    evidence: list[Evidence], channel: Channel
) -> list[HypothesisDraft]:
    _channel_only(evidence, channel)
    drafts: list[HypothesisDraft] = []
    for item in evidence:
        try:
            drafts.append(HypothesisDraft.from_research(item, channel))
        except ValueError:
            continue
    return drafts


def analyze_academic(evidence: list[Evidence]) -> list[HypothesisDraft]:
    return _analyze_research(evidence, Channel.ACADEMIC)


def analyze_social(evidence: list[Evidence]) -> list[HypothesisDraft]:
    return _analyze_research(evidence, Channel.SOCIAL)


def analyze_market(evidence: list[Evidence]) -> list[HypothesisDraft]:
    """Generate H1 hypotheses from deterministic market observations only."""
    _market_only(evidence)
    drafts: list[HypothesisDraft] = []
    for item in evidence:
        try:
            drafts.append(HypothesisDraft.from_market(item))
        except (KeyError, TypeError, ValueError):
            continue
    return drafts


def _fingerprint(draft: HypothesisDraft) -> tuple[str, str, str, str]:
    """Only an explicit source-validated semantic key permits cross-channel fusion."""
    semantic_key = (
        draft.semantic_claim_key
        or f"unfused:{draft.origin.value}:{draft.evidence_ids[0]}"
    )
    return (
        semantic_key.lower(),
        draft.outcome.lower(),
        draft.universe.lower(),
        draft.horizon.lower(),
    )


def fuse_hypotheses(inputs: list[FusionInput]) -> list[UnifiedHypothesis]:
    """Fuse normalized channel outputs without inventing new source evidence."""
    grouped: dict[tuple[str, str, str, str], list[FusionInput]] = {}
    for item in inputs:
        grouped.setdefault(_fingerprint(item.draft), []).append(item)
    unified: list[UnifiedHypothesis] = []
    for group in grouped.values():
        base = group[0].draft
        channels = sorted({entry.draft.origin for entry in group}, key=str)
        evidence_ids = [
            evidence_id for entry in group for evidence_id in entry.draft.evidence_ids
        ]
        independence_keys = sorted(
            {key for entry in group for key in entry.evidence_independence_keys}
        )
        maturity = base.maturity
        if len(channels) >= 2 and len(independence_keys) >= 2:
            maturity = Maturity.H3
        elif any(entry.draft.mechanism for entry in group):
            maturity = Maturity.H2
        payload = base.model_dump()
        payload.update(
            maturity=maturity,
            evidence_ids=evidence_ids,
            supporting_channels=channels,
            independence_keys=independence_keys,
            independent_evidence_count=len(independence_keys),
        )
        unified.append(UnifiedHypothesis(**payload))
    return unified
