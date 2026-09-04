from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def normalize_utc(value: datetime) -> datetime:
    """Represent a datetime as a timezone-aware UTC instant for comparisons."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ClaimType(StrEnum):
    FACT = "FACT"
    CLAIM = "CLAIM"
    OPINION = "OPINION"
    RESULT = "RESULT"


class HypothesisStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    FORMALIZED = "FORMALIZED"
    DATA_AVAILABLE = "DATA_AVAILABLE"
    TEST_READY = "TEST_READY"
    TESTING = "TESTING"
    REJECTED = "REJECTED"
    PROMISING = "PROMISING"
    BACKTESTED = "BACKTESTED"
    WALK_FORWARD = "WALK_FORWARD"
    PAPER = "PAPER"


class CollectionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"


class AnalysisRole(StrEnum):
    TRIAGE = "TRIAGE"
    EXTRACTION = "EXTRACTION"
    HYPOTHESIS_CANDIDATE = "HYPOTHESIS_CANDIDATE"
    TUTOR = "TUTOR"
    ANALYST = "ANALYST"
    CRITIC = "CRITIC"
    WEEKLY_REVIEW = "WEEKLY_REVIEW"


class RawArtifact(Base):
    """Immutable content-addressed raw object; never overwritten when a source changes."""

    __tablename__ = "raw_artifacts"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    content_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    media_type: Mapped[str] = mapped_column(String(100))
    byte_size: Mapped[int] = mapped_column(Integer)
    storage_uri: Mapped[str] = mapped_column(String(1000), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class RawArtifactReceipt(Base):
    """One lawful retrieval event; many receipts may reference one immutable object."""

    __tablename__ = "raw_artifact_receipts"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    raw_artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("raw_artifacts.id"), index=True
    )
    provider: Mapped[str] = mapped_column(String(100), index=True)
    canonical_url: Mapped[str | None] = mapped_column(String(1000))
    source_native_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    parser_version: Mapped[str] = mapped_column(String(50), default="1")
    archive_status: Mapped[str] = mapped_column(String(30), default="ARCHIVED")
    analysis_mode: Mapped[str] = mapped_column(String(50), default="PRODUCTION_LIVE")
    source_item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("source_items.id"), index=True
    )
    market_observation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("market_observations.id"), index=True
    )
    collection_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("collection_runs.id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class SourceItem(Base):
    __tablename__ = "source_items"
    __table_args__ = (UniqueConstraint("source_type", "external_id"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_type: Mapped[str] = mapped_column(String(40))
    source_name: Mapped[str] = mapped_column(String(100))
    external_id: Mapped[str] = mapped_column(String(255))
    canonical_url: Mapped[str | None] = mapped_column(String(1000))
    title: Mapped[str] = mapped_column(String(1000))
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    raw_text: Mapped[str] = mapped_column(Text, default="")
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    content_sha256: Mapped[str] = mapped_column(String(64))
    ingestion_version: Mapped[str] = mapped_column(String(32), default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class EvidenceSource(Base):
    """Registry entry for an adapter/venue, not a claim of source truth."""

    __tablename__ = "evidence_sources"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_name: Mapped[str] = mapped_column(String(100), unique=True)
    source_class: Mapped[str] = mapped_column(String(30))
    venue: Mapped[str | None] = mapped_column(String(300))
    peer_review_status: Mapped[str] = mapped_column(String(40), default="UNKNOWN")
    domain_tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    access_mode: Mapped[str] = mapped_column(String(30))
    reliability_prior: Mapped[str] = mapped_column(String(40))
    provenance_class: Mapped[str] = mapped_column(String(40), default="PUBLIC")
    adapter_status: Mapped[str] = mapped_column(String(20), default="READY")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ResearchWork(Base):
    """Canonical work identity shared by journal/preprint/repository locations."""

    __tablename__ = "research_works"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    canonical_identity: Mapped[str] = mapped_column(String(500), unique=True)
    normalized_title: Mapped[str] = mapped_column(String(1000))
    doi: Mapped[str | None] = mapped_column(String(300), unique=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(100))
    ssrn_id: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class WorkLocation(Base):
    __tablename__ = "work_locations"
    __table_args__ = (UniqueConstraint("work_id", "source_item_id"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    work_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("research_works.id"), index=True
    )
    source_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_items.id"), index=True
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evidence_sources.id"), index=True
    )
    access_mode: Mapped[str] = mapped_column(String(30))
    version_label: Mapped[str | None] = mapped_column(String(100))
    is_primary: Mapped[bool] = mapped_column(default=False)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class ChannelHypothesis(Base):
    """Independent pre-fusion hypothesis emitted by one evidence channel."""

    __tablename__ = "channel_hypotheses"
    __table_args__ = (
        UniqueConstraint(
            "channel", "fingerprint", "analysis_mode", "availability_basis", "as_of"
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    channel: Mapped[str] = mapped_column(String(20), index=True)
    statement: Mapped[str] = mapped_column(Text)
    mechanism: Mapped[str | None] = mapped_column(Text)
    condition: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(Text)
    universe: Mapped[str] = mapped_column(String(500))
    horizon: Mapped[str] = mapped_column(String(100))
    expected_direction: Mapped[str | None] = mapped_column(String(100))
    required_data: Mapped[list[str]] = mapped_column(JSON, default=list)
    falsification_criterion: Mapped[str] = mapped_column(Text)
    maturity: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(30), default="DISCOVERED")
    fingerprint: Mapped[str] = mapped_column(String(1000), index=True, default="")
    analysis_mode: Mapped[str] = mapped_column(String(50), default="PRODUCTION_LIVE")
    availability_basis: Mapped[str] = mapped_column(String(60), default="RECEIPT_TIME")
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class EvidenceLink(Base):
    """Provenance link; relation is ORIGIN, SUPPORT, or CHALLENGE."""

    __tablename__ = "evidence_links"
    __table_args__ = (
        UniqueConstraint("channel_hypothesis_id", "source_item_id", "relation"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    channel_hypothesis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("channel_hypotheses.id"), index=True
    )
    source_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_items.id"), index=True
    )
    relation: Mapped[str] = mapped_column(String(20))
    channel: Mapped[str] = mapped_column(String(20))
    independence_key: Mapped[str] = mapped_column(String(500), index=True)
    raw_artifact_receipt_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("raw_artifact_receipts.id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class UnifiedHypothesisRecord(Base):
    __tablename__ = "unified_hypotheses"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    fingerprint: Mapped[str] = mapped_column(String(1000), unique=True)
    statement: Mapped[str] = mapped_column(Text)
    maturity: Mapped[str] = mapped_column(String(40))
    supporting_channels: Mapped[list[str]] = mapped_column(JSON, default=list)
    independent_evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class UnifiedHypothesisMember(Base):
    __tablename__ = "unified_hypothesis_members"
    __table_args__ = (
        UniqueConstraint("unified_hypothesis_id", "channel_hypothesis_id"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    unified_hypothesis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("unified_hypotheses.id"), index=True
    )
    channel_hypothesis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("channel_hypotheses.id"), index=True
    )


class UserFeedback(Base):
    __tablename__ = "user_feedback"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    unified_hypothesis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("unified_hypotheses.id"), index=True
    )
    preference: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class Claim(Base):
    __tablename__ = "claims"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_items.id"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str] = mapped_column(String(20))
    evidence_level: Mapped[str] = mapped_column(String(40), default="UNASSESSED")
    evidence_excerpt: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str | None] = mapped_column(String(100))
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Hypothesis(Base):
    __tablename__ = "hypotheses"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("source_items.id")
    )
    title: Mapped[str] = mapped_column(String(500))
    observation: Mapped[str] = mapped_column(Text)
    mechanism: Mapped[str] = mapped_column(Text)
    falsifiable_statement: Mapped[str] = mapped_column(Text)
    independent_variable: Mapped[str] = mapped_column(Text)
    dependent_variable: Mapped[str] = mapped_column(Text)
    universe: Mapped[str] = mapped_column(String(500))
    horizon: Mapped[str] = mapped_column(String(100))
    required_data: Mapped[list[str]] = mapped_column(JSON, default=list)
    confounders: Mapped[list[str]] = mapped_column(JSON, default=list)
    biases: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(
        String(30), default=HypothesisStatus.DISCOVERED.value
    )
    component_scores: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    penalties: Mapped[list[str]] = mapped_column(JSON, default=list)
    score: Mapped[int] = mapped_column(Integer, default=0)
    scoring_explanation: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Concept(Base):
    __tablename__ = "concepts"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    beginner_explanation: Mapped[str] = mapped_column(Text)
    technical_explanation: Mapped[str] = mapped_column(Text)
    formula: Mapped[str | None] = mapped_column(Text)
    example: Mapped[str] = mapped_column(Text, default="")
    difficulty: Mapped[str] = mapped_column(String(30), default="BEGINNER")
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    target_type: Mapped[str] = mapped_column(String(30))
    target_id: Mapped[uuid.UUID] = mapped_column()
    review_period: Mapped[str] = mapped_column(String(20))
    label: Mapped[str] = mapped_column(String(20))
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class EventStudySpecRecord(Base):
    """Immutable serialized Phase 2.0 study contract."""

    __tablename__ = "event_study_specs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    hypothesis_id: Mapped[str] = mapped_column(String(100), index=True)
    hypothesis_family_id: Mapped[str] = mapped_column(String(200), index=True)
    spec_version: Mapped[str] = mapped_column(String(30))
    spec_hash: Mapped[str] = mapped_column(String(64), unique=True)
    immutable_spec: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EventStudyRun(Base):
    __tablename__ = "event_study_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    spec_id: Mapped[str] = mapped_column(ForeignKey("event_study_specs.id"), index=True)
    hypothesis_id: Mapped[str] = mapped_column(String(100), index=True)
    analysis_mode: Mapped[str] = mapped_column(String(60))
    availability_basis: Mapped[str] = mapped_column(String(80))
    real_receipt_pit: Mapped[str] = mapped_column(String(30))
    data_lineage: Mapped[dict[str, Any]] = mapped_column(JSON)
    code_sha: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class EventStudyResultRecord(Base):
    __tablename__ = "event_study_results"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("event_study_runs.id"), unique=True, index=True
    )
    spec_id: Mapped[str] = mapped_column(ForeignKey("event_study_specs.id"), index=True)
    hypothesis_id: Mapped[str] = mapped_column(String(100), index=True)
    hypothesis_family_id: Mapped[str] = mapped_column(String(200), index=True)
    disposition: Mapped[str] = mapped_column(String(30), index=True)
    treatment_count: Mapped[int] = mapped_column(Integer)
    baseline_count: Mapped[int] = mapped_column(Integer)
    regime_count: Mapped[int] = mapped_column(Integer)
    effects: Mapped[dict[str, Any]] = mapped_column(JSON)
    robustness: Mapped[dict[str, Any]] = mapped_column(JSON)
    methodology_critic: Mapped[dict[str, Any]] = mapped_column(JSON)
    artifact_uri: Mapped[str] = mapped_column(String(1000))
    code_sha: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class MarketObservation(Base):
    __tablename__ = "market_observations"
    __table_args__ = (UniqueConstraint("asset", "observed_at", "observation_kind"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    asset: Mapped[str] = mapped_column(String(30), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    observation_kind: Mapped[str] = mapped_column(String(30), default="snapshot")
    funding_rate: Mapped[float | None] = mapped_column(Float)
    mark_price: Mapped[float | None] = mapped_column(Float)
    open_interest: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    source_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_name: Mapped[str] = mapped_column(String(100), default="hyperliquid")
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class MarketMetric(Base):
    __tablename__ = "market_metrics"
    __table_args__ = (UniqueConstraint("observation_id", "metric_name"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    observation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("market_observations.id"), index=True
    )
    metric_name: Mapped[str] = mapped_column(String(80))
    metric_value: Mapped[float] = mapped_column(Float)
    calculation_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class CollectionRun(Base):
    __tablename__ = "collection_runs"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(50))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    requested: Mapped[int] = mapped_column(Integer, default=0)
    retrieved: Mapped[int] = mapped_column(Integer, default=0)
    inserted: Mapped[int] = mapped_column(Integer, default=0)
    updated: Mapped[int] = mapped_column(Integer, default=0)
    skipped_duplicates: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        String(20), default=CollectionStatus.SUCCESS.value
    )
    error_reason: Mapped[str | None] = mapped_column(Text)
    requested_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    requested_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    code_sha: Mapped[str | None] = mapped_column(String(64))
    diagnostics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    phase16a_run_id: Mapped[str | None] = mapped_column(String(100), index=True)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    role: Mapped[str] = mapped_column(String(20), default=AnalysisRole.ANALYST.value)
    provider: Mapped[str] = mapped_column(String(80))
    model_name: Mapped[str] = mapped_column(String(100))
    requested_model_tier: Mapped[str | None] = mapped_column(String(20))
    actual_model_name: Mapped[str | None] = mapped_column(String(100))
    thinking_enabled: Mapped[bool | None] = mapped_column()
    reasoning_effort: Mapped[str | None] = mapped_column(String(20))
    fallback_used: Mapped[bool] = mapped_column(default=False)
    prompt_version: Mapped[str] = mapped_column(String(30), default="1")
    schema_version: Mapped[str] = mapped_column(String(30))
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="RUNNING")
    failure_reason: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DailyRun(Base):
    """One canonical Daily operations cycle, keyed by logical Beijing date."""

    __tablename__ = "daily_runs"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    logical_date: Mapped[date] = mapped_column(Date, index=True)
    status: Mapped[str] = mapped_column(String(20), default="RUNNING")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    code_sha: Mapped[str] = mapped_column(String(64))
    market_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    academic_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    practitioner_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    analysis_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    knowledge_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    audit_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    report_path: Mapped[str | None] = mapped_column(String(1000))
    failure_reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_health: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    llm_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class WeeklyRun(Base):
    """One canonical Weekly review, keyed by week-ending Saturday."""

    __tablename__ = "weekly_runs"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    week_saturday: Mapped[date] = mapped_column(Date, index=True)
    status: Mapped[str] = mapped_column(String(20), default="RUNNING")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    code_sha: Mapped[str] = mapped_column(String(64))
    included_daily_dates: Mapped[list[str]] = mapped_column(JSON, default=list)
    report_path: Mapped[str | None] = mapped_column(String(1000))
    failure_reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    priorities: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    low_frequency_fit: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# ---------------------------------------------------------------------------
# Publication / delivery domain.
#
# READ-ONLY toward research conclusions: these tables reference research
# artifacts but must never be used to mutate hypothesis status, empirical
# results, critic dispositions, knowledge strength, or research ranking.
# ---------------------------------------------------------------------------


class TopicBrief(Base):
    """Versioned interpretive artifact derived from finalized research state."""

    __tablename__ = "topic_briefs"
    __table_args__ = (
        UniqueConstraint(
            "source_run_id", "topic_id", "topic_version", name="uq_topic_brief_version"
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    topic_id: Mapped[str] = mapped_column(String(200), index=True)
    topic_version: Mapped[str] = mapped_column(String(30), default="1")
    logical_date: Mapped[date] = mapped_column(Date, index=True)
    source_run_id: Mapped[str] = mapped_column(String(64), index=True)
    source_kind: Mapped[str] = mapped_column(String(20), default="DAILY")
    human_title: Mapped[str] = mapped_column(String(500))
    packet: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    brief: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    input_packet_hash: Mapped[str] = mapped_column(String(64))
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    model_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class DailySocialEditorialPackage(Base):
    """Durable daily editorial evaluation; one package even for SKIP."""

    __tablename__ = "daily_social_packages"
    __table_args__ = (UniqueConstraint("logical_date", name="uq_social_package_date"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    logical_date: Mapped[date] = mapped_column(Date, index=True)
    source_run_id: Mapped[str] = mapped_column(String(64), index=True)
    topic_brief_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    selected_candidate_id: Mapped[str | None] = mapped_column(String(64))
    recommendation: Mapped[str] = mapped_column(String(30), default="SKIP")
    selection_reason: Mapped[str] = mapped_column(Text, default="")
    content_format: Mapped[str] = mapped_column(String(30), default="SHORT_POST")
    draft_text: Mapped[str | None] = mapped_column(Text)
    source_bundle: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output_path: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class WeeklySocialEditorialPackage(Base):
    """Durable weekly editorial evaluation."""

    __tablename__ = "weekly_social_packages"
    __table_args__ = (
        UniqueConstraint("week_saturday", name="uq_weekly_social_package_date"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    week_saturday: Mapped[date] = mapped_column(Date, index=True)
    source_run_id: Mapped[str] = mapped_column(String(64), index=True)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    recommendation: Mapped[str] = mapped_column(String(30), default="SKIP")
    selection_reason: Mapped[str] = mapped_column(Text, default="")
    content_format: Mapped[str] = mapped_column(String(30), default="THREAD")
    draft_text: Mapped[str | None] = mapped_column(Text)
    output_path: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class PublicationCandidate(Base):
    """Something in a completed research cycle worth public evaluation."""

    __tablename__ = "publication_candidates"
    __table_args__ = (
        UniqueConstraint(
            "source_run_id", "category", "title", name="uq_candidate_identity"
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_run_id: Mapped[str] = mapped_column(String(64), index=True)
    source_kind: Mapped[str] = mapped_column(String(20))  # DAILY | WEEKLY
    category: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(500))
    summary: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    publication_value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class PublicationDraft(Base):
    """Rendered public copy with claims, source bundle, and policy decision."""

    __tablename__ = "publication_drafts"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("publication_candidates.id"), index=True
    )
    policy: Mapped[str] = mapped_column(String(30))
    language: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(Text)
    claims: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    source_bundle: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    visual_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    idempotence_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class PublicationRecord(Base):
    """Immutable publication history; corrections create new linked records."""

    __tablename__ = "publications"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    draft_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("publication_drafts.id"), index=True
    )
    platform: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(
        String(30)
    )  # PUBLISHED/REJECTED/FAILED/SKIPPED_*
    external_post_id: Mapped[str | None] = mapped_column(String(100))
    failure_reason: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class DeliveryRecord(Base):
    """One private-delivery attempt identity (idempotent per channel+run)."""

    __tablename__ = "delivery_records"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    channel: Mapped[str] = mapped_column(String(20))  # EMAIL | DISCORD
    run_kind: Mapped[str] = mapped_column(String(20))  # DAILY | WEEKLY | ALERT
    run_date: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20))  # SENT | FAILED | SKIPPED
    idempotence_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


def content_hash(text: str, metadata: dict[str, Any] | None = None) -> str:
    value = text + str(sorted((metadata or {}).items()))
    return hashlib.sha256(value.encode()).hexdigest()


def make_engine(url: str) -> Engine:
    return create_engine(url, future=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_phase16a_collection_run(
    session: Session,
    *,
    source: str,
    phase16a_run_id: str,
    requested_start: datetime,
    requested_end: datetime,
    collection_code_sha: str,
    status: str = CollectionStatus.SUCCESS.value,
) -> CollectionRun | None:
    """Resolve exactly one Phase 1.6A collection run by explicit provenance."""
    start = normalize_utc(requested_start)
    end = normalize_utc(requested_end)
    candidates = session.scalars(
        select(CollectionRun).where(
            CollectionRun.source == source,
            CollectionRun.phase16a_run_id == phase16a_run_id,
            CollectionRun.code_sha == collection_code_sha,
            CollectionRun.status == status,
        )
    ).all()
    for run in candidates:
        if (
            run.requested_start is not None
            and run.requested_end is not None
            and normalize_utc(run.requested_start) == start
            and normalize_utc(run.requested_end) == end
        ):
            return run
    return None


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
