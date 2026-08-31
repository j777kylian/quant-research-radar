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
from .llm import CriticOutput, LLMClient
from .metrics import funding_percentile, return_at

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
            disposition, reason = "RETAINED", None
            if item.published_at is None or normalize_utc(item.published_at) > as_of:
                disposition, reason = "REJECTED_AVAILABILITY", "PUBLISHED_AFTER_AS_OF"
            elif (
                availability_basis == AvailabilityBasis.PRODUCTION_RECEIPT
                and normalize_utc(item.retrieved_at) > as_of
            ):
                disposition, reason = "REJECTED_AVAILABILITY", "RETRIEVED_AFTER_AS_OF"
            elif not _quant_relevant(item):
                disposition, reason = "REJECTED_RELEVANCE", "STRICT_TOPIC_GATE"
            receipt = session.scalar(
                select(RawArtifactReceipt).where(
                    RawArtifactReceipt.source_item_id == item.id
                )
            )
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
        )
    )
    if existing is not None:
        return existing
    source_item = _persist_source_item(session, evidence, as_of)
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


def _critic_results(
    pairs: list[tuple[HypothesisDraft, Evidence]], unified: list[Any], *, persist: bool
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
    evidence_ok = all(
        draft.evidence_ids and evidence.provenance_id for draft, evidence in pairs
    )
    method_ok = all(
        draft.condition and draft.required_data and draft.falsification_criterion
        for draft, _evidence in pairs
    )
    fusion_ok = all(
        draft.maturity.value != "H3_CONVERGENT" or draft.semantic_claim_key
        for draft in unified
    )
    return {
        "evidence_auditor": {"disposition": "ACCEPT" if evidence_ok else "REJECT"},
        "methodology_critic": {"disposition": "ACCEPT" if method_ok else "REJECT"},
        "fusion_critic": {"disposition": "ACCEPT" if fusion_ok else "REJECT"},
    }


def _draft_family(draft: HypothesisDraft) -> str:
    return "|".join(
        [draft.origin.value, draft.outcome, draft.universe, draft.horizon]
    ).lower()


def run_intelligence_day(
    session: Session,
    output_root: Path,
    as_of: datetime,
    *,
    availability_basis: AvailabilityBasis = AvailabilityBasis.PRODUCTION_RECEIPT,
    seen_families: set[str] | None = None,
    persist: bool = True,
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
    channel_pairs = [
        *zip(market_drafts, market_evidence_items, strict=True),
        *zip(academic_drafts, academic_evidence_items, strict=True),
        *zip(social_drafts, social_evidence_items, strict=True),
    ]
    raw_drafts = [draft for draft, _evidence in channel_pairs]
    known_families = seen_families if seen_families is not None else set()
    repeated_families = sorted(
        {_draft_family(draft) for draft in raw_drafts} & known_families
    )
    channel_pairs = [
        pair for pair in channel_pairs if _draft_family(pair[0]) not in known_families
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
    critics = _critic_results(channel_pairs, unified_drafts, persist=persist)
    tutor_accepted = persist and all(
        result["disposition"] == "ACCEPT" for result in critics.values()
    )
    tutor_concepts = (
        [
            {"hypothesis": draft.statement, "topic": "research hypothesis testing"}
            for draft in unified_drafts
        ]
        if tutor_accepted
        else []
    )
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
                "READY"
                if all(result["disposition"] != "REJECT" for result in critics.values())
                else "BLOCKED"
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


def _review_replay_candidate(
    candidate: dict[str, Any], client: LLMClient | None
) -> dict[str, Any]:
    if client is None:
        return candidate | {
            "critic": {"disposition": "NOT_RUN", "reason": "no replay critic client"},
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
    except (AttributeError, TypeError, ValueError):
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
    except (AttributeError, TypeError, ValueError):
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
