from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
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
    code_sha: str,
    status: str = CollectionStatus.SUCCESS.value,
) -> CollectionRun | None:
    """Resolve exactly one Phase 1.6A collection run by explicit provenance."""
    start = normalize_utc(requested_start)
    end = normalize_utc(requested_end)
    candidates = session.scalars(
        select(CollectionRun).where(
            CollectionRun.source == source,
            CollectionRun.phase16a_run_id == phase16a_run_id,
            CollectionRun.code_sha == code_sha,
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
