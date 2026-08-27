from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (
    AnalysisRun,
    Claim,
    ClaimType,
    CollectionRun,
    Concept,
    Hypothesis,
    MarketMetric,
    MarketObservation,
    SourceItem,
    content_hash,
    utcnow,
)
from .llm import LLMClient
from .metrics import funding_percentile, return_at, rolling_volatility
from .sources import SourceAdapter, SourceRecord


def ingest(
    session: Session, adapter: SourceAdapter, limit: int, offline: bool = False
) -> int:
    run = CollectionRun(source=adapter.name, requested=limit, started_at=utcnow())
    session.add(run)
    session.flush()
    try:
        records = adapter.collect(limit, offline)
        run.retrieved = len(records)
        inserted = 0
        for record in records:
            existing = session.scalar(
                select(SourceItem).where(
                    SourceItem.source_type == record.source_type,
                    SourceItem.external_id == record.external_id,
                )
            )
            digest = content_hash(record.raw_text, record.raw_metadata)
            if existing:
                run.skipped_duplicates += 1
                if existing.content_sha256 != digest:
                    existing.raw_text = record.raw_text
                    existing.raw_metadata = record.raw_metadata
                    existing.content_sha256 = digest
                    existing.updated_at = utcnow()
                    run.updated += 1
                continue
            session.add(
                SourceItem(
                    source_type=record.source_type,
                    source_name=record.source_name,
                    external_id=record.external_id,
                    canonical_url=record.canonical_url,
                    title=record.title,
                    authors=record.authors,
                    published_at=record.published_at,
                    retrieved_at=utcnow(),
                    raw_text=record.raw_text,
                    raw_metadata=record.raw_metadata,
                    content_sha256=digest,
                )
            )
            inserted += 1
            if record.source_type == "MARKET":
                _persist_market(session, record)
        run.inserted = inserted
        run.status = "SUCCESS"
        session.commit()
        return inserted
    except Exception as exc:
        session.rollback()
        run = CollectionRun(
            source=adapter.name,
            requested=limit,
            retrieved=0,
            failed=1,
            status="DEGRADED" if adapter.name == "repec" else "FAILED",
            error_reason=str(exc),
            started_at=utcnow(),
            ended_at=utcnow(),
        )
        session.add(run)
        session.commit()
        if adapter.name not in ("repec", "arxiv"):
            raise
        return 0
    finally:
        if run in session:
            run.ended_at = utcnow()
            session.commit()


def ingest_records(session: Session, records: list[SourceRecord]) -> tuple[int, int]:
    inserted = 0
    duplicates = 0
    for record in records:
        existing = session.scalar(
            select(SourceItem).where(
                SourceItem.source_type == record.source_type,
                SourceItem.external_id == record.external_id,
            )
        )
        if existing:
            duplicates += 1
            continue
        session.add(
            SourceItem(
                source_type=record.source_type,
                source_name=record.source_name,
                external_id=record.external_id,
                canonical_url=record.canonical_url,
                title=record.title,
                authors=record.authors,
                published_at=record.published_at,
                retrieved_at=utcnow(),
                raw_text=record.raw_text,
                raw_metadata=record.raw_metadata,
                content_sha256=content_hash(record.raw_text, record.raw_metadata),
            )
        )
        if record.source_type == "MARKET":
            _persist_market(session, record)
        inserted += 1
    session.commit()
    return inserted, duplicates


def _persist_market(session: Session, record: SourceRecord) -> None:
    metadata = record.raw_metadata
    timestamp = record.published_at or utcnow()
    kind = str(metadata.get("kind", "snapshot"))
    if session.scalar(
        select(MarketObservation).where(
            MarketObservation.asset == metadata["asset"],
            MarketObservation.observed_at == timestamp,
            MarketObservation.observation_kind == kind,
        )
    ):
        return
    session.add(
        MarketObservation(
            asset=str(metadata["asset"]),
            observed_at=timestamp,
            observation_kind=kind,
            funding_rate=metadata.get("funding_rate"),
            mark_price=metadata.get("mark_price", metadata.get("close")),
            open_interest=metadata.get("open_interest"),
            volume=metadata.get("volume"),
            source_payload=metadata,
            retrieved_at=utcnow(),
        )
    )


def calculate_metrics(
    session: Session, window: int = 30, volatility_window: int = 24
) -> int:
    observations = session.scalars(
        select(MarketObservation).order_by(
            MarketObservation.asset, MarketObservation.observed_at
        )
    ).all()
    by_asset: dict[str, list[MarketObservation]] = {}
    for observation in observations:
        by_asset.setdefault(observation.asset, []).append(observation)
    count = 0
    for _asset, rows in by_asset.items():
        funding = [
            (r.observed_at, r.funding_rate)
            for r in rows
            if r.observation_kind in ("funding", "snapshot")
        ]
        prices = {
            r.observed_at: r.mark_price
            for r in rows
            if r.mark_price is not None and r.observation_kind in ("candle", "snapshot")
        }
        for row in rows:
            values = {
                "funding_percentile": funding_percentile(
                    funding, row.observed_at, window
                ),
                "return_1h": return_at(prices, row.observed_at, 1),
                "return_4h": return_at(prices, row.observed_at, 4),
                "return_24h": return_at(prices, row.observed_at, 24),
                "rolling_volatility": rolling_volatility(
                    prices, row.observed_at, volatility_window
                ),
            }
            for name, value in values.items():
                if value is None or session.scalar(
                    select(MarketMetric).where(
                        MarketMetric.observation_id == row.id,
                        MarketMetric.metric_name == name,
                    )
                ):
                    continue
                session.add(
                    MarketMetric(
                        observation_id=row.id,
                        metric_name=name,
                        metric_value=value,
                        calculation_metadata={
                            "pit_cutoff": row.observed_at.isoformat(),
                            "window": window
                            if name == "funding_percentile"
                            else volatility_window,
                        },
                    )
                )
                count += 1
    session.commit()
    return count


def score_hypothesis() -> tuple[dict[str, int], list[str], int]:
    components = {
        "economic_mechanism": 15,
        "evidence_quality": 10,
        "data_availability": 17,
        "executability": 16,
        "alpha_half_life": 12,
        "simplicity": 8,
    }
    penalties = ["regime dependence unresolved"]
    return components, penalties, max(0, sum(components.values()) - len(penalties) * 3)


def analyze(session: Session, client: LLMClient, limit: int = 20) -> int:
    items = session.scalars(
        select(SourceItem).order_by(SourceItem.created_at.desc()).limit(limit)
    ).all()
    run = AnalysisRun(
        role="ANALYST",
        provider=client.provider,
        model_name=client.model,
        prompt_version=client.prompt_version,
        schema_version="1",
        item_count=len(items),
        started_at=utcnow(),
    )
    session.add(run)
    created = 0
    for item in items:
        try:
            triage = client.triage(item.title, item.raw_text)
            if not triage.retain:
                continue
            analyst = client.analyze(item.title, item.raw_text)
            critic = client.critique(analyst.possible_hypothesis)
            components, penalties, score = score_hypothesis()
            session.add(
                Claim(
                    source_item_id=item.id,
                    text=analyst.reported_finding,
                    claim_type=ClaimType.CLAIM.value,
                    evidence_level="SOURCE_REPORTED",
                    evidence_excerpt=item.raw_text[:500],
                    confidence=1.0,
                    model_name=client.model,
                )
            )
            session.add(
                Hypothesis(
                    source_item_id=item.id,
                    title=analyst.core_question,
                    observation=analyst.reported_finding,
                    mechanism=analyst.mechanism,
                    falsifiable_statement=analyst.possible_hypothesis,
                    independent_variable="funding rate",
                    dependent_variable="subsequent return",
                    universe=analyst.universe,
                    horizon=analyst.horizon,
                    required_data=analyst.required_data,
                    confounders=critic.confounders,
                    biases=critic.biases,
                    component_scores=components,
                    penalties=penalties,
                    score=score,
                    scoring_explanation="Explicit component sum minus 3 points per listed penalty.",
                )
            )
            for concept in client.tutor(analyst.possible_hypothesis).concepts:
                if not session.scalar(
                    select(Concept).where(Concept.name == concept.name)
                ):
                    session.add(
                        Concept(
                            name=concept.name,
                            beginner_explanation=concept.beginner_explanation,
                            technical_explanation=concept.technical_definition,
                            example=concept.example,
                        )
                    )
            created += 1
            run.success_count += 1
        except Exception:
            run.failure_count += 1
    run.ended_at = utcnow()
    session.commit()
    return created


def daily_report(
    session: Session, output_dir: str, report_date: date | None = None
) -> Path:
    report_date = report_date or datetime.now(UTC).date()
    items = session.scalars(
        select(SourceItem).order_by(SourceItem.created_at.desc()).limit(3)
    ).all()
    hypotheses = session.scalars(
        select(Hypothesis).order_by(Hypothesis.score.desc()).limit(3)
    ).all()
    concepts = session.scalars(
        select(Concept).order_by(Concept.last_seen_at.desc()).limit(2)
    ).all()
    lines = [f"# Daily Quant Radar — {report_date}", "", "## Market Observation", ""]
    market = [item for item in items if item.source_type == "MARKET"]
    if not market:
        lines.append("UNAVAILABLE — no market observation was collected.")
    for item in market:
        lines += [
            f"**FACT:** {item.title}",
            f"- source: {item.canonical_url or item.external_id}",
            f"- evidence timestamp: {item.published_at or 'UNAVAILABLE'}",
            "",
            "**INTERPRETATION:** Raw market evidence is displayed without converting it into a trading instruction.",
            "",
        ]
    lines += ["## Academic Research", ""]
    academic_items = [i for i in items if i.source_type == "ACADEMIC"][:2]
    if not academic_items:
        lines.append("- **CLAIM:** No academic item retained in this bounded run.")
    for item in academic_items:
        lines.append(
            f"- **CLAIM:** {item.title}; evidence excerpt: {item.raw_text[:300]}"
        )
    lines += ["", "## Hypothesis Candidates", ""]
    for hypothesis in hypotheses:
        lines.append(
            f"- **HYPOTHESIS:** {hypothesis.falsifiable_statement} Status: {hypothesis.status}."
        )
    lines += ["", "## Concepts", ""]
    for concept in concepts:
        lines.append(
            f"- **{concept.name}:** {concept.beginner_explanation} {concept.example}"
        )
    lines += ["", "No execution instructions are generated.", ""]
    path = Path(output_dir) / f"daily-{report_date}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def weekly_report(
    session: Session, output_dir: str, end_date: date | None = None
) -> Path:
    end_date = end_date or datetime.now(UTC).date()
    hypotheses = session.scalars(
        select(Hypothesis).order_by(Hypothesis.score.desc()).limit(10)
    ).all()
    lines = [
        f"# Weekly Quant Radar — week ending {end_date}",
        "",
        "## Top research candidates",
        "",
    ]
    lines.extend(
        f"- **HYPOTHESIS {h.id}:** {h.title} — {h.score}/100 ({h.status})"
        for h in hypotheses
    )
    lines += ["", "Research candidate != trade.", ""]
    path = Path(output_dir) / f"weekly-{end_date}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
