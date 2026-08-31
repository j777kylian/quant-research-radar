"""Read-only Phase 1.7 retrieval over authoritative relational records."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .db import (
    ChannelHypothesis,
    EvidenceLink,
    RawArtifact,
    RawArtifactReceipt,
    UnifiedHypothesisMember,
    UnifiedHypothesisRecord,
    normalize_utc,
)

Scope = Literal["PRODUCTION", "REPLAY", "ALL_WITH_PROVENANCE"]


def search_hypotheses(
    session: Session,
    query: str,
    *,
    scope: Scope = "PRODUCTION",
    channel: str | None = None,
    maturity: str | None = None,
    as_of: datetime | None = None,
) -> list[dict[str, Any]]:
    """Portable lexical baseline; caller input remains bound parameters."""
    query = query.strip()
    if scope not in {"PRODUCTION", "REPLAY", "ALL_WITH_PROVENANCE"}:
        raise ValueError("unsupported knowledge scope")
    if not query or len(query) > 500:
        raise ValueError("query must contain 1..500 characters")
    statement = select(ChannelHypothesis)
    if scope == "PRODUCTION":
        statement = statement.where(
            ChannelHypothesis.analysis_mode == "PRODUCTION_LIVE"
        )
    elif scope == "REPLAY":
        statement = statement.where(
            ChannelHypothesis.analysis_mode != "PRODUCTION_LIVE"
        )
    if channel:
        statement = statement.where(ChannelHypothesis.channel == channel.upper())
    if maturity:
        statement = statement.where(ChannelHypothesis.maturity == maturity)
    if as_of:
        statement = statement.where(ChannelHypothesis.as_of <= normalize_utc(as_of))
    needle = f"%{query.lower()}%"
    statement = statement.where(
        or_(
            ChannelHypothesis.statement.ilike(needle),
            ChannelHypothesis.condition.ilike(needle),
            ChannelHypothesis.outcome.ilike(needle),
            ChannelHypothesis.universe.ilike(needle),
        )
    ).order_by(ChannelHypothesis.as_of.desc(), ChannelHypothesis.created_at.desc())
    rows = []
    for hypothesis in session.scalars(statement).all():
        unified = session.scalar(
            select(UnifiedHypothesisRecord)
            .join(UnifiedHypothesisMember)
            .where(UnifiedHypothesisMember.channel_hypothesis_id == hypothesis.id)
        )
        rows.append(
            {
                "entity_type": "CHANNEL_HYPOTHESIS",
                "entity_id": str(hypothesis.id),
                "title": hypothesis.statement,
                "matched_field": "statement/condition/outcome/universe",
                "channel": hypothesis.channel,
                "maturity": hypothesis.maturity,
                "analysis_mode": hypothesis.analysis_mode,
                "availability_basis": hypothesis.availability_basis,
                "as_of": normalize_utc(hypothesis.as_of).isoformat()
                if hypothesis.as_of
                else None,
                "unified_hypothesis_id": str(unified.id) if unified else None,
            }
        )
    return rows


def hypothesis_lineage(session: Session, hypothesis_id: str) -> dict[str, Any]:
    hypothesis = session.get(ChannelHypothesis, uuid.UUID(hypothesis_id))
    if hypothesis is None:
        raise ValueError("hypothesis not found")
    unified = session.scalar(
        select(UnifiedHypothesisRecord)
        .join(UnifiedHypothesisMember)
        .where(UnifiedHypothesisMember.channel_hypothesis_id == hypothesis.id)
    )
    members: list[ChannelHypothesis] = []
    if unified:
        members = list(
            session.scalars(
                select(ChannelHypothesis)
                .join(UnifiedHypothesisMember)
                .where(UnifiedHypothesisMember.unified_hypothesis_id == unified.id)
                .order_by(ChannelHypothesis.as_of, ChannelHypothesis.created_at)
            ).all()
        )
    links = []
    for link in session.scalars(
        select(EvidenceLink).where(EvidenceLink.channel_hypothesis_id == hypothesis.id)
    ).all():
        receipt = (
            session.get(RawArtifactReceipt, link.raw_artifact_receipt_id)
            if link.raw_artifact_receipt_id
            else None
        )
        artifact = (
            session.get(RawArtifact, receipt.raw_artifact_id) if receipt else None
        )
        links.append(
            {
                "relation": link.relation,
                "source_item_id": str(link.source_item_id),
                "independence_key": link.independence_key,
                "raw_status": "ARCHIVED" if artifact else "LEGACY_UNARCHIVED",
                "receipt_id": str(receipt.id) if receipt else None,
                "raw_artifact_id": str(artifact.id) if artifact else None,
                "sha256": artifact.content_sha256 if artifact else None,
                "storage_uri": artifact.storage_uri if artifact else None,
                "collection_run_id": str(receipt.collection_run_id)
                if receipt and receipt.collection_run_id
                else None,
            }
        )
    return {
        "hypothesis": {
            "id": str(hypothesis.id),
            "statement": hypothesis.statement,
            "channel": hypothesis.channel,
            "maturity": hypothesis.maturity,
            "analysis_mode": hypothesis.analysis_mode,
            "availability_basis": hypothesis.availability_basis,
            "as_of": normalize_utc(hypothesis.as_of).isoformat()
            if hypothesis.as_of
            else None,
        },
        "family": {
            "id": str(unified.id),
            "fingerprint": unified.fingerprint,
            "statement": unified.statement,
        }
        if unified
        else None,
        "occurrences": [
            {
                "id": str(row.id),
                "as_of": normalize_utc(row.as_of).isoformat() if row.as_of else None,
                "maturity": row.maturity,
                "channel": row.channel,
                "analysis_mode": row.analysis_mode,
            }
            for row in members
        ],
        "evidence": links,
    }
