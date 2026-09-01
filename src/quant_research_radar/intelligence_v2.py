from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (
    ChannelHypothesis,
    EvidenceLink,
    MarketObservation,
    RawArtifactReceipt,
    SourceItem,
    UnifiedHypothesisMember,
    UnifiedHypothesisRecord,
    content_hash,
    normalize_utc,
)
from .intelligence import (
    Channel,
    Evidence,
    FusionInput,
    HypothesisDraft,
    analyze_academic,
    analyze_market,
    analyze_social,
    fuse_hypotheses,
)
from .knowledge_context import prior_research_context
from .llm import CriticOutput, LLMClient
from .metrics import funding_percentile, return_at
from .research_contracts import ContractRole, contract_for, validate_contract_output

MODE = "ACCELERATED_RECONSTRUCTIVE_REPLAY"
PIT_BASIS = "SOURCE_NATIVE_AVAILABILITY_TIME"
REAL_RECEIPT_PIT = "NOT_CLAIMED"


class AvailabilityBasis(StrEnum):
    PRODUCTION_RECEIPT = "RECEIPT_TIME"
    SOURCE_NATIVE_REPLAY = "SOURCE_NATIVE_AVAILABILITY_TIME"


def _valuation(as_of: datetime) -> datetime:
    return normalize_utc(as_of).replace(minute=0, second=0, microsecond=0) - timedelta(
        hours=1
    )


def market_evidence(
    session: Session, as_of: datetime, availability_basis: AvailabilityBasis
) -> list[Evidence]:
    """Build V2 market H0 evidence without mutating collection receipt clocks."""
    as_of = normalize_utc(as_of)
    valuation = _valuation(as_of)
    evidence: list[Evidence] = []
    for asset in ("BTC", "ETH", "SOL"):
        rows = session.scalars(
            select(MarketObservation).where(
                MarketObservation.asset == asset,
                MarketObservation.source_name == "hyperliquid",
            )
        ).all()
        rows = [
            row
            for row in rows
            if (
                availability_basis == AvailabilityBasis.SOURCE_NATIVE_REPLAY
                or normalize_utc(row.retrieved_at) <= as_of
            )
            and (
                (
                    row.observation_kind == "funding"
                    and normalize_utc(row.observed_at) <= valuation
                )
                or (
                    row.observation_kind == "candle"
                    and normalize_utc(row.observed_at) + timedelta(hours=1) <= as_of
                    and normalize_utc(row.observed_at) <= valuation
                )
            )
        ]
        funding = [
            (normalize_utc(row.observed_at), row.funding_rate)
            for row in rows
            if row.observation_kind == "funding"
        ]
        prices = {
            normalize_utc(row.observed_at): row.mark_price
            for row in rows
            if row.observation_kind == "candle" and row.mark_price is not None
        }
        percentile = funding_percentile(funding, valuation)
        result = return_at(prices, valuation, 24)
        if (
            percentile is None
            or result is None
            or percentile < 90
            or abs(result) < 0.01
        ):
            continue
        evidence_id = f"v2-market:{asset}:{valuation.isoformat()}:funding-extreme"
        evidence.append(
            Evidence(
                evidence_id=evidence_id,
                channel=Channel.MARKET,
                title=f"{asset} extreme funding with 24h return move",
                body=f"funding_percentile={percentile}; return_24h={result}",
                provenance_id=evidence_id,
                independence_key=f"hyperliquid:{asset}:{valuation.isoformat()}",
                observed_at=valuation,
                metadata={
                    "asset": asset,
                    "funding_percentile": percentile,
                    "return_24h": result,
                    "as_of": normalize_utc(as_of).isoformat(),
                    "market_observation_ids": [str(row.id) for row in rows],
                    "availability_basis": availability_basis.value,
                    "real_receipt_pit": (
                        "CLAIMED"
                        if availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT
                        else REAL_RECEIPT_PIT
                    ),
                },
            )
        )
    return evidence


def _quant_relevant(item: SourceItem) -> bool:
    """Strict deterministic gate: requires a specific market topic, not generic ML/finance."""
    text = f"{item.title} {item.raw_text}".lower()
    topic_pairs = (
        ("funding", "perpetual"),
        ("market", "microstructure"),
        ("return", "predictability"),
        ("realized", "volatility"),
        ("limit", "order book"),
        ("event", "study"),
        ("crypto", "market"),
        ("derivative", "liquidity"),
    )
    return any(all(term in text for term in pair) for pair in topic_pairs)


def _source_dispositions(
    session: Session, as_of: datetime, availability_basis: AvailabilityBasis
) -> list[dict[str, Any]]:
    dispositions: list[dict[str, Any]] = []
    for channel, source_types in (
        (Channel.ACADEMIC, {"ACADEMIC", "PREPRINT"}),
        (Channel.SOCIAL, {"PRACTITIONER", "SOCIAL"}),
    ):
        for item in session.scalars(
            select(SourceItem).where(SourceItem.source_type.in_(source_types))
        ).all():
            receipt = session.scalar(
                select(RawArtifactReceipt).where(
                    RawArtifactReceipt.source_item_id == item.id
                )
            )
            disposition, reason = (
                ("RETAINED", None)
                if receipt is not None
                else ("RETAINED_LEGACY_UNARCHIVED", "NO_RAW_ARCHIVE_RECEIPT")
            )
            if item.published_at is None or normalize_utc(item.published_at) > as_of:
                disposition, reason = "REJECTED_AVAILABILITY", "PUBLISHED_AFTER_AS_OF"
            elif (
                availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT
                and normalize_utc(item.retrieved_at) > as_of
            ):
                disposition, reason = "REJECTED_AVAILABILITY", "RETRIEVED_AFTER_AS_OF"
            elif not _quant_relevant(item):
                disposition, reason = "REJECTED_RELEVANCE", "STRICT_TOPIC_GATE"
            dispositions.append(
                {
                    "channel": channel.value,
                    "source_item_id": str(item.id),
                    "external_id": item.external_id,
                    "canonical_url": item.canonical_url,
                    "published_at": normalize_utc(item.published_at).isoformat()
                    if item.published_at
                    else None,
                    "retrieved_at": normalize_utc(item.retrieved_at).isoformat(),
                    "replay_availability_at": normalize_utc(
                        item.published_at
                    ).isoformat()
                    if item.published_at
                    else None,
                    "access_mode": str(
                        item.raw_metadata.get("access_mode", "METADATA_ONLY")
                    ),
                    "raw_artifact_id": str(receipt.raw_artifact_id)
                    if receipt
                    else None,
                    "linked_hypothesis_ids": [],
                    "disposition": disposition,
                    "reason_code": reason,
                }
            )
    return sorted(
        dispositions, key=lambda item: (str(item["channel"]), str(item["external_id"]))
    )


def channel_evidence(
    session: Session,
    channel: Channel,
    as_of: datetime,
    availability_basis: AvailabilityBasis,
) -> list[Evidence]:
    """Admit only channel-matching retained source evidence before first-pass analysis."""
    source_types = {
        Channel.ACADEMIC: {"ACADEMIC", "PREPRINT"},
        Channel.SOCIAL: {"PRACTITIONER", "SOCIAL"},
    }
    if channel not in source_types:
        raise ValueError("channel_evidence only supports academic or social channels")
    as_of = normalize_utc(as_of)
    items = session.scalars(
        select(SourceItem).where(SourceItem.source_type.in_(source_types[channel]))
    ).all()
    eligible: list[Evidence] = []
    for item in items:
        if item.published_at is None or normalize_utc(item.published_at) > as_of:
            continue
        if (
            availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT
            and normalize_utc(item.retrieved_at) > as_of
        ):
            continue
        if not _quant_relevant(item):
            continue
        receipt = session.scalar(
            select(RawArtifactReceipt)
            .where(
                RawArtifactReceipt.source_item_id == item.id,
                RawArtifactReceipt.retrieved_at <= as_of,
                RawArtifactReceipt.collection_run_id.is_not(None),
                RawArtifactReceipt.analysis_mode == "PRODUCTION_LIVE",
            )
            .order_by(
                RawArtifactReceipt.retrieved_at.desc(), RawArtifactReceipt.id.desc()
            )
        )
        if (
            availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT
            and receipt is None
        ):
            continue
        access_mode = str(item.raw_metadata.get("access_mode", "METADATA_ONLY"))
        evidence_id = f"source-item:{item.id}"
        eligible.append(
            Evidence(
                evidence_id=evidence_id,
                channel=channel,
                title=item.title,
                body=item.raw_text,
                provenance_id=item.external_id,
                independence_key=str(
                    item.raw_metadata.get("independence_key")
                    or item.raw_metadata.get("doi")
                    or item.canonical_url
                    or item.external_id
                ),
                observed_at=normalize_utc(item.published_at),
                metadata={
                    "source_item_id": str(item.id),
                    "source_identity": item.source_name,
                    "canonical_identity": item.canonical_url or item.external_id,
                    "raw_artifact_receipt_id": str(receipt.id) if receipt else None,
                    "raw_artifact_id": str(receipt.raw_artifact_id)
                    if receipt
                    else None,
                    "collection_run_id": str(receipt.collection_run_id)
                    if receipt
                    else None,
                    "receipt_retrieved_at": normalize_utc(
                        receipt.retrieved_at
                    ).isoformat()
                    if receipt
                    else None,
                    "source_native_availability_at": normalize_utc(
                        receipt.source_native_timestamp
                    ).isoformat()
                    if receipt and receipt.source_native_timestamp
                    else normalize_utc(item.published_at).isoformat(),
                    "analysis_mode": receipt.analysis_mode if receipt else None,
                    "access_mode": access_mode,
                    "research_topic": item.title,
                    "availability_basis": availability_basis.value,
                },
            )
        )
    return eligible


def _persist_source_item(
    session: Session, evidence: Evidence, as_of: datetime
) -> SourceItem:
    existing = session.scalar(
        select(SourceItem).where(
            SourceItem.source_type == "MARKET",
            SourceItem.external_id == evidence.evidence_id,
        )
    )
    if existing is not None:
        return existing
    item = SourceItem(
        source_type="MARKET",
        source_name="quant-radar-v2-market",
        external_id=evidence.evidence_id,
        canonical_url=None,
        title=evidence.title,
        authors=[],
        published_at=evidence.observed_at,
        retrieved_at=as_of,
        raw_text=evidence.body,
        raw_metadata=evidence.metadata | {"provenance_id": evidence.provenance_id},
        content_sha256=content_hash(evidence.body, evidence.metadata),
    )
    session.add(item)
    session.flush()
    return item


def _persistence_fingerprint(draft: HypothesisDraft) -> str:
    semantic = draft.semantic_claim_key or draft.evidence_ids[0]
    return "|".join([semantic, draft.outcome, draft.universe, draft.horizon]).lower()


def _persist_market_hypothesis(
    session: Session, draft: HypothesisDraft, evidence: Evidence, as_of: datetime
) -> ChannelHypothesis:
    fingerprint = _persistence_fingerprint(draft)
    existing = session.scalar(
        select(ChannelHypothesis).where(
            ChannelHypothesis.channel == Channel.MARKET.value,
            ChannelHypothesis.fingerprint == fingerprint,
            ChannelHypothesis.analysis_mode == "PRODUCTION_LIVE",
            ChannelHypothesis.availability_basis
            == AvailabilityBasis.PRODUCTION_RECEIPT.value,
            ChannelHypothesis.as_of == as_of,
        )
    )
    if existing is not None:
        return existing
    source_item = _persist_source_item(session, evidence, as_of)
    observation_ids = [
        uuid.UUID(value)
        for value in evidence.metadata.get("market_observation_ids", [])
        if isinstance(value, str)
    ]
    receipts = session.scalars(
        select(RawArtifactReceipt)
        .where(
            RawArtifactReceipt.market_observation_id.in_(observation_ids),
            RawArtifactReceipt.retrieved_at <= as_of,
            RawArtifactReceipt.analysis_mode == "PRODUCTION_LIVE",
            RawArtifactReceipt.collection_run_id.is_not(None),
        )
        .order_by(RawArtifactReceipt.retrieved_at.desc(), RawArtifactReceipt.id.desc())
    ).all()
    if len({receipt.market_observation_id for receipt in receipts}) != len(
        observation_ids
    ):
        raise ValueError("market evidence requires archived receipts for every input")
    receipt = receipts[0] if receipts else None
    if receipt is None:
        raise ValueError(
            "market evidence requires an archived receipt bound to a collection run"
        )
    hypothesis = ChannelHypothesis(
        channel=draft.origin.value,
        statement=draft.statement,
        mechanism=draft.mechanism,
        condition=draft.condition,
        outcome=draft.outcome,
        universe=draft.universe,
        horizon=draft.horizon,
        expected_direction=draft.expected_direction,
        required_data=draft.required_data,
        falsification_criterion=draft.falsification_criterion,
        maturity=draft.maturity.value,
        fingerprint=fingerprint,
        analysis_mode="PRODUCTION_LIVE",
        availability_basis=AvailabilityBasis.PRODUCTION_RECEIPT.value,
        as_of=as_of,
    )
    session.add(hypothesis)
    session.flush()
    session.add(
        EvidenceLink(
            channel_hypothesis_id=hypothesis.id,
            source_item_id=source_item.id,
            relation="ORIGIN",
            channel=Channel.MARKET.value,
            independence_key=evidence.independence_key,
            raw_artifact_receipt_id=receipt.id if receipt else None,
        )
    )
    session.flush()
    return hypothesis


def _persist_hypothesis(
    session: Session, draft: HypothesisDraft, evidence: Evidence, as_of: datetime
) -> ChannelHypothesis:
    if draft.origin == Channel.MARKET:
        return _persist_market_hypothesis(session, draft, evidence, as_of)
    fingerprint = _persistence_fingerprint(draft)
    existing = session.scalar(
        select(ChannelHypothesis).where(
            ChannelHypothesis.channel == draft.origin.value,
            ChannelHypothesis.fingerprint == fingerprint,
            ChannelHypothesis.analysis_mode == "PRODUCTION_LIVE",
            ChannelHypothesis.availability_basis
            == AvailabilityBasis.PRODUCTION_RECEIPT.value,
            ChannelHypothesis.as_of == as_of,
        )
    )
    if existing is not None:
        return existing
    source_item_id = evidence.metadata.get("source_item_id")
    if not isinstance(source_item_id, str):
        raise ValueError("non-market evidence must bind an existing source item")
    source_item = session.get(SourceItem, uuid.UUID(source_item_id))
    if source_item is None:
        raise ValueError("non-market evidence source item is missing")
    receipt = session.scalar(
        select(RawArtifactReceipt)
        .where(
            RawArtifactReceipt.source_item_id == source_item.id,
            RawArtifactReceipt.retrieved_at <= as_of,
            RawArtifactReceipt.analysis_mode == "PRODUCTION_LIVE",
        )
        .order_by(RawArtifactReceipt.retrieved_at.desc(), RawArtifactReceipt.id.desc())
    )
    if receipt is None or receipt.collection_run_id is None:
        raise ValueError(
            "source evidence requires an archived receipt bound to a collection run"
        )
    hypothesis = ChannelHypothesis(
        channel=draft.origin.value,
        statement=draft.statement,
        mechanism=draft.mechanism,
        condition=draft.condition,
        outcome=draft.outcome,
        universe=draft.universe,
        horizon=draft.horizon,
        expected_direction=draft.expected_direction,
        required_data=draft.required_data,
        falsification_criterion=draft.falsification_criterion,
        maturity=draft.maturity.value,
        fingerprint=fingerprint,
        analysis_mode="PRODUCTION_LIVE",
        availability_basis=AvailabilityBasis.PRODUCTION_RECEIPT.value,
        as_of=as_of,
    )
    session.add(hypothesis)
    session.flush()
    session.add(
        EvidenceLink(
            channel_hypothesis_id=hypothesis.id,
            source_item_id=source_item.id,
            relation="ORIGIN",
            channel=draft.origin.value,
            independence_key=evidence.independence_key,
            raw_artifact_receipt_id=receipt.id if receipt else None,
        )
    )
    session.flush()
    return hypothesis


def _persist_unified(
    session: Session, draft: HypothesisDraft, member: ChannelHypothesis
) -> UnifiedHypothesisRecord:
    fingerprint = _persistence_fingerprint(draft)
    unified = session.scalar(
        select(UnifiedHypothesisRecord).where(
            UnifiedHypothesisRecord.fingerprint == fingerprint
        )
    )
    if unified is None:
        unified = UnifiedHypothesisRecord(
            fingerprint=fingerprint,
            statement=draft.statement,
            maturity=draft.maturity.value,
            supporting_channels=[
                channel.value for channel in draft.supporting_channels
            ],
            independent_evidence_count=len(draft.independence_keys),
            priority=50 + 10 * len(draft.supporting_channels),
        )
        session.add(unified)
        session.flush()
    if (
        session.scalar(
            select(UnifiedHypothesisMember).where(
                UnifiedHypothesisMember.unified_hypothesis_id == unified.id,
                UnifiedHypothesisMember.channel_hypothesis_id == member.id,
            )
        )
        is None
    ):
        session.add(
            UnifiedHypothesisMember(
                unified_hypothesis_id=unified.id, channel_hypothesis_id=member.id
            )
        )
    return unified


def _write_report(root: Path, audit: dict[str, Any], drafts: list[Any]) -> None:
    lines = [
        "# Quant Research Radar — Phase 1.6D",
        "",
        f"MODE={MODE}",
        f"PIT_BASIS={PIT_BASIS}",
        f"REAL_RECEIPT_PIT={REAL_RECEIPT_PIT}",
        "",
        "## Academic Radar",
        f"Retained hypotheses: {audit['channels']['ACADEMIC']['hypotheses_retained']}",
        "",
        "## Social / Practitioner Radar",
        f"Retained hypotheses: {audit['channels']['SOCIAL']['hypotheses_retained']}",
        "",
        "## Market Radar",
        f"Retained hypotheses: {audit['channels']['MARKET']['hypotheses_retained']}",
        "",
        "## Fusion Radar",
        f"Unified hypotheses: {audit['fusion']['unified_hypotheses']}",
    ]
    for draft in drafts:
        lines += [
            f"- **HYPOTHESIS:** {draft.statement}",
            f"  - maturity: {draft.maturity.value}; condition: {draft.condition}; outcome: {draft.outcome}",
            f"  - universe: {draft.universe}; horizon: {draft.horizon}",
            f"  - falsification: {draft.falsification_criterion}",
        ]
    lines += ["", "## Tutor"]
    if audit["tutor"]["concepts"]:
        lines += [
            "- Generated concepts are stored in `tutor.json` and are educational only.",
            "- Tutor output is not evidence and never enters fusion or critic inputs.",
        ]
    else:
        lines.append(
            "- No Tutor concepts generated; no accepted persistent hypothesis artifact."
        )
    (root / "executive.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _critic_evidence_packet(
    session: Session,
    draft: HypothesisDraft,
    evidence: Evidence,
    as_of: datetime,
    availability_basis: AvailabilityBasis,
) -> dict[str, Any]:
    """Bounded, receipt-grounded evidence supplied to the runtime Critic."""
    metadata = evidence.metadata
    receipt_mode = (
        "PRODUCTION_LIVE"
        if availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT
        else MODE
    )
    analysis_mode = receipt_mode
    real_receipt_pit = (
        "CLAIMED"
        if availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT
        else REAL_RECEIPT_PIT
    )
    if draft.origin == Channel.MARKET:
        observation_ids = [
            uuid.UUID(value)
            for value in metadata.get("market_observation_ids", [])
            if isinstance(value, str)
        ]
        receipt_statement = select(RawArtifactReceipt).where(
            RawArtifactReceipt.market_observation_id.in_(observation_ids),
            RawArtifactReceipt.collection_run_id.is_not(None),
            RawArtifactReceipt.analysis_mode == receipt_mode,
        )
        if availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT:
            receipt_statement = receipt_statement.where(
                RawArtifactReceipt.retrieved_at <= as_of
            )
        receipts = session.scalars(
            receipt_statement.order_by(
                RawArtifactReceipt.retrieved_at.desc(), RawArtifactReceipt.id.desc()
            )
        ).all()
        return {
            "source_identity": "hyperliquid",
            "canonical_identity": [
                f"market-observation:{value}" for value in observation_ids
            ],
            "raw_artifact_receipt_id": [str(receipt.id) for receipt in receipts],
            "raw_artifact_id": [str(receipt.raw_artifact_id) for receipt in receipts],
            "collection_run_id": [
                str(receipt.collection_run_id) for receipt in receipts
            ],
            "retrieved_at": [
                normalize_utc(receipt.retrieved_at).isoformat() for receipt in receipts
            ],
            "source_native_availability_at": [
                normalize_utc(receipt.source_native_timestamp).isoformat()
                if receipt.source_native_timestamp
                else None
                for receipt in receipts
            ],
            "access_mode": "PUBLIC_API",
            "independence_key": evidence.independence_key,
            "reviewable_summary": evidence.body[:4_000],
            "analysis_mode": analysis_mode,
            "availability_basis": availability_basis.value,
            "pit_qualifications": {
                "real_receipt_pit": real_receipt_pit,
                "as_of": as_of.isoformat(),
            },
        }
    return {
        "source_identity": metadata["source_identity"],
        "canonical_identity": metadata["canonical_identity"],
        "raw_artifact_receipt_id": metadata["raw_artifact_receipt_id"],
        "raw_artifact_id": metadata["raw_artifact_id"],
        "collection_run_id": metadata["collection_run_id"],
        "retrieved_at": metadata["receipt_retrieved_at"],
        "source_native_availability_at": metadata["source_native_availability_at"],
        "access_mode": metadata["access_mode"],
        "independence_key": evidence.independence_key,
        "reviewable_summary": evidence.body[:4_000],
        "analysis_mode": analysis_mode,
        "availability_basis": availability_basis.value,
        "pit_qualifications": {
            "real_receipt_pit": real_receipt_pit,
            "as_of": as_of.isoformat(),
        },
    }


def _critic_results(
    session: Session,
    pairs: list[tuple[HypothesisDraft, Evidence]],
    unified: list[Any],
    *,
    as_of: datetime,
    availability_basis: AvailabilityBasis,
    persist: bool,
    client: LLMClient | None = None,
) -> dict[str, dict[str, str]]:
    if not persist:
        return {
            name: {
                "disposition": "NOT_RUN",
                "reason": "replay candidates are non-persistent",
            }
            for name in ("evidence_auditor", "methodology_critic", "fusion_critic")
        }
    if not pairs:
        return {
            name: {"disposition": "NOT_RUN", "reason": "no eligible candidates"}
            for name in ("evidence_auditor", "methodology_critic", "fusion_critic")
        }
    if client is None:
        return {
            name: {"disposition": "NOT_RUN", "reason": "no production critic client"}
            for name in ("evidence_auditor", "methodology_critic", "fusion_critic")
        }
    try:
        review = CriticOutput.model_validate(
            client.critique(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "hypothesis_contract": {
                                    "statement": draft.statement,
                                    "condition": draft.condition,
                                    "outcome": draft.outcome,
                                    "universe": draft.universe,
                                    "horizon": draft.horizon,
                                    "expected_direction": draft.expected_direction,
                                    "required_data": draft.required_data,
                                    "falsification_criterion": draft.falsification_criterion,
                                    "semantic_claim_key": draft.semantic_claim_key,
                                },
                                "evidence": _critic_evidence_packet(
                                    session, draft, evidence, as_of, availability_basis
                                ),
                            }
                            for draft, evidence in pairs
                        ],
                    },
                    sort_keys=True,
                )
            ).model_dump()
        )
    except Exception:
        return {
            name: {
                "disposition": "REQUEST_DATA",
                "reason": "critic structured output failed",
            }
            for name in ("evidence_auditor", "methodology_critic", "fusion_critic")
        }
    disposition = "ACCEPT" if review.provenance_sufficient else "REQUEST_DATA"
    return {
        name: {"disposition": disposition, "reason": "; ".join(review.failure_reasons)}
        for name in ("evidence_auditor", "methodology_critic", "fusion_critic")
    }


def _draft_family(draft: HypothesisDraft) -> str:
    return "|".join(
        [draft.origin.value, draft.outcome, draft.universe, draft.horizon]
    ).lower()


def _validate_phase18_channel_draft(draft: HypothesisDraft, evidence: Evidence) -> None:
    """Compatibility wrapper: existing deterministic generators enter contracts here."""
    if draft.origin == Channel.MARKET:
        validate_contract_output(
            ContractRole.MARKET_ANALYST,
            {
                "state_type": "EXTREME_STATE",
                "condition": draft.condition,
                "outcome": draft.outcome,
                "universe": draft.universe,
                "horizon": draft.horizon,
                "baseline": draft.falsification_criterion,
                "direction": draft.expected_direction,
                "source_evidence_ids": draft.evidence_ids,
                "falsification_criterion": draft.falsification_criterion,
            },
        )
    elif draft.origin == Channel.ACADEMIC:
        validate_contract_output(
            ContractRole.ACADEMIC_ANALYST,
            {
                "research_question": evidence.title,
                "actual_evidence": evidence.body,
                "causal_status": "CORRELATIONAL",
                "analysis_confidence": "ABSTRACT_ONLY",
                "source_evidence_ids": draft.evidence_ids,
                "source_access_mode": str(
                    evidence.metadata.get("access_mode", "METADATA_ONLY")
                ),
                "limitations": ["Source content may be metadata or abstract only."],
                "testable_radar_hypothesis": None,
            },
        )
    else:
        validate_contract_output(
            ContractRole.PRACTITIONER_SOCIAL_ANALYST,
            {
                "claim": draft.statement,
                "original_source": True,
                "independence_key": evidence.independence_key,
                "supplied_evidence": evidence.body,
                "reproducibility": "UNKNOWN",
                "source_evidence_ids": draft.evidence_ids,
            },
        )


def _validate_phase18_fusion(drafts: list[Any]) -> None:
    for draft in drafts:
        validate_contract_output(
            ContractRole.FUSION_ANALYST,
            {
                "semantic_equivalence": "SAME_FAMILY",
                "prior_research_context_ids": [],
                "fresh_evidence_ids": draft.evidence_ids,
                "context_is_evidence": False,
            },
        )


def _contract_versions() -> dict[str, str]:
    return {role.value: contract_for(role).version for role in ContractRole}


def _validate_phase18_critics(critics: dict[str, dict[str, str]]) -> None:
    for result in critics.values():
        disposition = result["disposition"]
        validate_contract_output(
            ContractRole.METHODOLOGY_CRITIC,
            {
                "disposition": {
                    "ACCEPT": "ACCEPT_FOR_EMPIRICAL_TEST",
                    "REJECT": "REJECT",
                    "NOT_RUN": "REQUEST_DATA",
                }.get(disposition, "REQUEST_DATA"),
                "structured_reasons": [result.get("reason", disposition)],
                "provenance_sufficient": disposition == "ACCEPT",
            },
        )


def _validate_phase18_tutor(tutor_concepts: list[dict[str, str]]) -> None:
    for concept in tutor_concepts:
        validate_contract_output(
            ContractRole.TUTOR,
            {
                "why_this_matters": concept["topic"],
                "how_it_would_be_tested": concept["hypothesis"],
                "non_evidentiary": True,
            },
        )


def run_phase18_intelligence_cycle(
    session: Session,
    output_root: Path,
    as_of: datetime,
    *,
    availability_basis: AvailabilityBasis = AvailabilityBasis.PRODUCTION_RECEIPT,
    seen_families: set[str] | None = None,
    persist: bool = True,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Minimal full-stack V2 day: independent empty channels cannot block market H1."""
    as_of = normalize_utc(as_of)
    if persist and availability_basis != AvailabilityBasis.PRODUCTION_RECEIPT:
        raise ValueError(
            "reconstructive replay candidates must not enter production persistence"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    market_evidence_items = market_evidence(session, as_of, availability_basis)
    academic_evidence_items = channel_evidence(
        session, Channel.ACADEMIC, as_of, availability_basis
    )
    social_evidence_items = channel_evidence(
        session, Channel.SOCIAL, as_of, availability_basis
    )
    market_drafts = analyze_market(market_evidence_items)
    academic_drafts = analyze_academic(academic_evidence_items)
    social_drafts = analyze_social(social_evidence_items)
    evidence_by_id = {
        evidence.evidence_id: evidence
        for evidence in [
            *market_evidence_items,
            *academic_evidence_items,
            *social_evidence_items,
        ]
    }
    channel_pairs = [
        (draft, evidence_by_id[draft.evidence_ids[0]])
        for draft in [*market_drafts, *academic_drafts, *social_drafts]
    ]
    for draft, evidence in channel_pairs:
        _validate_phase18_channel_draft(draft, evidence)
    raw_drafts = [draft for draft, _evidence in channel_pairs]
    prior_contexts = [
        prior_research_context(
            session,
            _persistence_fingerprint(draft),
            draft.universe,
            as_of,
            draft.origin.value,
        )
        for draft in raw_drafts
    ]
    known_families = seen_families if seen_families is not None else set()
    repeated_families = sorted(
        {_draft_family(draft) for draft in raw_drafts} & known_families
    )
    if not persist:
        channel_pairs = [
            pair
            for pair in channel_pairs
            if _draft_family(pair[0]) not in known_families
        ]
    market_drafts = [
        draft for draft, _evidence in channel_pairs if draft.origin == Channel.MARKET
    ]
    academic_drafts = [
        draft for draft, _evidence in channel_pairs if draft.origin == Channel.ACADEMIC
    ]
    social_drafts = [
        draft for draft, _evidence in channel_pairs if draft.origin == Channel.SOCIAL
    ]
    if persist:
        persisted = [
            _persist_hypothesis(session, draft, evidence, as_of)
            for draft, evidence in channel_pairs
        ]
    else:
        persisted = []
    unified_drafts = fuse_hypotheses(
        [
            FusionInput(draft=draft, evidence_independence_keys=draft.independence_keys)
            for draft, _evidence in channel_pairs
        ]
    )
    _validate_phase18_fusion(unified_drafts)
    for unified_draft in unified_drafts if persist else []:
        members = [
            member
            for draft, member in zip(channel_pairs, persisted, strict=True)
            if _persistence_fingerprint(draft[0])
            == _persistence_fingerprint(unified_draft)
        ]
        if not members:
            continue
        unified = _persist_unified(session, unified_draft, members[0])
        for member in members[1:]:
            if (
                session.scalar(
                    select(UnifiedHypothesisMember).where(
                        UnifiedHypothesisMember.unified_hypothesis_id == unified.id,
                        UnifiedHypothesisMember.channel_hypothesis_id == member.id,
                    )
                )
                is None
            ):
                session.add(
                    UnifiedHypothesisMember(
                        unified_hypothesis_id=unified.id,
                        channel_hypothesis_id=member.id,
                    )
                )
    if persist:
        session.commit()
    critics = _critic_results(
        session,
        channel_pairs,
        unified_drafts,
        as_of=as_of,
        availability_basis=availability_basis,
        persist=persist,
        client=client,
    )
    _validate_phase18_critics(critics)
    tutor_accepted = persist and all(
        result["disposition"] == "ACCEPT" for result in critics.values()
    )
    tutor_concepts = (
        [
            {
                "hypothesis": draft.statement,
                "topic": "research hypothesis testing",
            }
            for draft in unified_drafts
        ]
        if tutor_accepted and client is None
        else (
            [
                {
                    "hypothesis": draft.statement,
                    "topic": concept.name,
                }
                for draft in unified_drafts
                for concept in client.tutor(draft.statement).concepts
            ]
            if tutor_accepted and client is not None
            else []
        )
    )
    _validate_phase18_tutor(tutor_concepts)
    audit: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "mode": (
            "PRODUCTION_LIVE"
            if availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT
            else MODE
        ),
        "availability_basis": availability_basis.value,
        "pit_basis": availability_basis.value,
        "real_receipt_pit": (
            "CLAIMED"
            if availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT
            else REAL_RECEIPT_PIT
        ),
        "contract_versions": _contract_versions(),
        "channels": {
            "ACADEMIC": {
                "discovered": len(academic_evidence_items),
                "retained": len(academic_evidence_items),
                "hypotheses_retained": len(academic_drafts),
                "analyzer_input_channels": ["ACADEMIC"],
            },
            "SOCIAL": {
                "discovered": len(social_evidence_items),
                "retained": len(social_evidence_items),
                "hypotheses_retained": len(social_drafts),
                "analyzer_input_channels": ["SOCIAL"],
            },
            "MARKET": {
                "hypotheses_retained": len(market_drafts),
                "analyzer_input_channels": ["MARKET"],
            },
        },
        "source_dispositions": _source_dispositions(session, as_of, availability_basis),
        "knowledge": {
            "prior_context": [
                {
                    "hypothesis_ids": context.hypothesis_ids,
                    "novelty": context.novelty.value,
                    "occurrence_count": context.occurrence_count,
                    "fresh_evidence_ids": [],
                }
                for context in prior_contexts
            ],
            "fresh_evidence_isolated": True,
        },
        "fusion": {
            "unified_hypotheses": len(unified_drafts),
            "maturity": [draft.maturity.value for draft in unified_drafts],
            "input_channels": [
                [channel.value for channel in draft.supporting_channels]
                for draft in unified_drafts
            ],
        },
        "new_hypothesis_families": sorted(
            {_draft_family(draft) for draft, _evidence in channel_pairs}
        ),
        "repeated_hypothesis_families": repeated_families,
        "market_regimes": [
            {
                "family": family,
                "status": "PERSISTENT_REGIME",
                "occurrence_count": 1,
            }
            for family in repeated_families
            if family.startswith("market|")
        ],
        "critics": critics,
        "replay_candidates": (
            [
                {
                    "origin_channel": draft.origin.value,
                    "statement": draft.statement,
                    "maturity": draft.maturity.value,
                    "condition": draft.condition,
                    "outcome": draft.outcome,
                    "universe": draft.universe,
                    "horizon": draft.horizon,
                    "mechanism": draft.mechanism,
                    "required_data": draft.required_data,
                    "falsification_criterion": draft.falsification_criterion,
                    "evidence_ids": draft.evidence_ids,
                    "evidence_provenance_ids": [evidence.provenance_id],
                    "evidence_independence_keys": [evidence.independence_key],
                    "evidence_observed_at": [str(evidence.observed_at)],
                    "evidence_metadata": [evidence.metadata],
                    "critic_evidence": _critic_evidence_packet(
                        session,
                        draft,
                        evidence,
                        as_of,
                        availability_basis,
                    ),
                    "semantic_claim_key": draft.semantic_claim_key,
                    "recurrence_status": "NEW_CANDIDATE",
                }
                for draft, evidence in channel_pairs
            ]
            if not persist
            else []
        ),
        "tutor": {
            "concepts": len(tutor_concepts),
            "evidence_source": False,
            "artifact": "tutor.json" if tutor_concepts else None,
        },
        "technical_status": (
            "RESEARCH_UTILITY_INSUFFICIENT"
            if not channel_pairs
            else (
                "CRITIC_NOT_RUN"
                if any(
                    result["disposition"] == "NOT_RUN" for result in critics.values()
                )
                else (
                    "READY"
                    if all(
                        result["disposition"] == "ACCEPT" for result in critics.values()
                    )
                    else "CRITIC_REQUEST_DATA"
                )
            )
        ),
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    if tutor_concepts:
        (output_root / "tutor.json").write_text(
            json.dumps({"non_evidentiary": True, "concepts": tutor_concepts}, indent=2),
            encoding="utf-8",
        )
    _write_report(output_root, audit, unified_drafts)
    return audit


def run_intelligence_day(
    session: Session,
    output_root: Path,
    as_of: datetime,
    *,
    availability_basis: AvailabilityBasis = AvailabilityBasis.PRODUCTION_RECEIPT,
    seen_families: set[str] | None = None,
    persist: bool = True,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper; all Phase 1.6D callers use the Phase 1.8 cycle."""
    return run_phase18_intelligence_cycle(
        session,
        output_root,
        as_of,
        availability_basis=availability_basis,
        seen_families=seen_families,
        persist=persist,
        client=client,
    )


def _review_replay_candidate(
    candidate: dict[str, Any], client: LLMClient | None
) -> dict[str, Any]:
    packet = candidate.get("critic_evidence")
    required = (
        "source_identity",
        "canonical_identity",
        "raw_artifact_receipt_id",
        "collection_run_id",
        "retrieved_at",
        "source_native_availability_at",
        "access_mode",
        "independence_key",
        "reviewable_summary",
        "analysis_mode",
        "availability_basis",
        "pit_qualifications",
    )
    if client is None:
        return candidate | {
            "critic": {"disposition": "NOT_RUN", "reason": "no replay critic client"},
            "tutor": None,
        }
    if not isinstance(packet, dict) or any(not packet.get(field) for field in required):
        return candidate | {
            "critic": {
                "disposition": "REQUEST_DATA",
                "reason": "critic evidence packet is incomplete",
            },
            "tutor": None,
        }
    try:
        review = client.critique(
            json.dumps(
                {
                    "review_contract": [
                        "evidence provenance",
                        "channel origin",
                        "condition measurability",
                        "outcome measurability",
                        "universe",
                        "horizon",
                        "mechanism strength",
                        "alternative explanations",
                        "required data",
                        "falsification criterion",
                        "duplicate recurrence",
                        "evidence independence",
                        "overclaiming",
                        "empirical-test suitability",
                    ],
                    "candidate": candidate,
                },
                sort_keys=True,
            )
        )
        review = CriticOutput.model_validate(review.model_dump())
    except Exception:
        return candidate | {
            "critic": {
                "disposition": "REQUEST_DATA",
                "reason": "critic structured output failed",
            },
            "tutor": None,
        }
    if not review.provenance_sufficient:
        return candidate | {
            "critic": {
                "disposition": "REQUEST_DATA",
                "reason": "; ".join(review.failure_reasons)
                or "provenance insufficient",
                "confounders": review.confounders,
                "biases": review.biases,
            },
            "tutor": None,
        }
    tutor: dict[str, Any] | None = None
    try:
        tutor = {
            "non_evidentiary": True,
            **client.tutor(candidate["statement"]).model_dump(),
        }
    except Exception:
        tutor = None
    return candidate | {
        "critic": {
            "disposition": "ACCEPT",
            "reason": "structured replay review completed",
            "confounders": review.confounders,
            "biases": review.biases,
        },
        "tutor": tutor,
    }


def run_intelligence_replay(
    session: Session,
    output_root: Path,
    days: list[datetime],
    *,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Sequential V2 reconstructive replay; deliberately never changes source receipt clocks."""
    output_root.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    run_id = output_root.name
    seen_families: set[str] = set()
    for ordinal, as_of in enumerate(days, start=1):
        day_root = (
            output_root / f"day-{ordinal}-{normalize_utc(as_of).date().isoformat()}"
        )
        audit = run_intelligence_day(
            session,
            day_root,
            as_of,
            availability_basis=AvailabilityBasis.SOURCE_NATIVE_REPLAY,
            seen_families=seen_families,
            persist=False,
        )
        seen_families.update(audit["new_hypothesis_families"])
        audit["ordinal"] = ordinal
        for index, candidate in enumerate(audit.pop("replay_candidates"), start=1):
            reviewed = _review_replay_candidate(
                {
                    **candidate,
                    "replay_candidate_id": f"{run_id}:{ordinal}:{index}",
                    "run_id": run_id,
                    "pseudo_day": normalize_utc(as_of).isoformat(),
                    "analysis_mode": MODE,
                    "availability_basis": AvailabilityBasis.SOURCE_NATIVE_REPLAY.value,
                },
                client,
            )
            candidates.append(reviewed)
            for evidence_id in reviewed["evidence_ids"]:
                if not evidence_id.startswith("source-item:"):
                    continue
                source_item_id = evidence_id.removeprefix("source-item:")
                for disposition in audit["source_dispositions"]:
                    if disposition["source_item_id"] == source_item_id:
                        disposition["linked_hypothesis_ids"].append(
                            reviewed["replay_candidate_id"]
                        )
        audits.append(audit)
    summary = {
        "phase": "1.6D",
        "mode": MODE,
        "availability_basis": AvailabilityBasis.SOURCE_NATIVE_REPLAY.value,
        "real_receipt_pit": REAL_RECEIPT_PIT,
        "daily_audits": audits,
        "technical_success_count": sum(
            audit["technical_status"] == "READY" for audit in audits
        ),
        "research_utility_candidates": sum(
            audit["fusion"]["unified_hypotheses"] for audit in audits
        ),
    }
    tutor_outputs = [
        {"replay_candidate_id": candidate["replay_candidate_id"], **candidate["tutor"]}
        for candidate in candidates
        if candidate["tutor"] is not None
    ]
    if tutor_outputs:
        (output_root / "tutor.json").write_text(
            json.dumps(
                {"non_evidentiary": True, "candidates": tutor_outputs}, indent=2
            ),
            encoding="utf-8",
        )
    (output_root / "replay-candidate-ledger.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "analysis_mode": MODE,
                "availability_basis": AvailabilityBasis.SOURCE_NATIVE_REPLAY.value,
                "real_receipt_pit": REAL_RECEIPT_PIT,
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_root / "phase16d-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
