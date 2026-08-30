from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

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
    normalize_utc,
    utcnow,
)
from .llm import LLMClient
from .metrics import funding_percentile, return_at, rolling_volatility
from .sources import SourceAdapter, SourceRecord


def _valuation_timestamp(cutoff: datetime) -> datetime:
    cutoff = cutoff.astimezone(UTC)
    return cutoff.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


TOPIC_TERMS = (
    "asset pricing",
    "cryptocurrency",
    "crypto",
    "defi",
    "event study",
    "funding rate",
    "market microstructure",
    "market return",
    "order flow",
    "perpetual",
    "prediction market",
    "realized volatility",
    "return predictability",
)
TOPIC_CATEGORIES = ("q-fin", "econ", "stat")


def _valid_topic_category(category: object) -> bool:
    value = str(category)
    return any(
        value == root or value.startswith(f"{root}.") and len(value) > len(root) + 1
        for root in TOPIC_CATEGORIES
    )


def academic_relevant(title: str, text: str, metadata: Any = None) -> bool:
    haystack = f"{title} {text}".lower()
    categories = (
        metadata
        if isinstance(metadata, list)
        else (metadata or {}).get("categories", [])
    )
    return any(_valid_topic_category(category) for category in categories) and any(
        re.search(rf"\b{re.escape(term)}\b", haystack) for term in TOPIC_TERMS
    )


def _metric_values(
    session: Session, observation: MarketObservation
) -> dict[str, float]:
    return {
        metric.metric_name: metric.metric_value
        for metric in session.scalars(
            select(MarketMetric).where(MarketMetric.observation_id == observation.id)
        )
    }


def _receipt_safe_values(
    session: Session, observation: MarketObservation, as_of: datetime
) -> dict[str, float | None]:
    rows = session.scalars(
        select(MarketObservation).where(MarketObservation.asset == observation.asset)
    ).all()
    rows = [row for row in rows if normalize_utc(row.retrieved_at) <= as_of]
    funding = [
        (row.observed_at, row.funding_rate)
        for row in rows
        if row.observation_kind in ("funding", "snapshot")
    ]
    prices = {
        row.observed_at: row.mark_price
        for row in rows
        if row.observation_kind == "candle" and row.mark_price is not None
    }
    return {
        "funding_percentile": funding_percentile(funding, observation.observed_at),
        "return_24h": return_at(prices, observation.observed_at, 24),
    }


def _trusted_metric_values(
    session: Session, observation: MarketObservation, as_of: datetime
) -> dict[str, float] | None:
    if (
        observation.source_name != "hyperliquid"
        or normalize_utc(observation.retrieved_at) > as_of
    ):
        return None
    metrics = session.scalars(
        select(MarketMetric).where(MarketMetric.observation_id == observation.id)
    ).all()
    values: dict[str, float] = {}
    expected_cutoff = normalize_utc(observation.observed_at).isoformat()
    for metric in metrics:
        if metric.metric_name not in {"funding_percentile", "return_24h"}:
            continue
        if normalize_utc(metric.calculated_at) > as_of:
            return None
        try:
            metric_cutoff = normalize_utc(
                datetime.fromisoformat(metric.calculation_metadata["pit_cutoff"])
            ).isoformat()
            support_cutoff = normalize_utc(
                datetime.fromisoformat(
                    metric.calculation_metadata["support_receipt_cutoff"]
                )
            )
        except (KeyError, TypeError, ValueError):
            return None
        if metric_cutoff != expected_cutoff or support_cutoff > as_of:
            return None
        values[metric.metric_name] = metric.metric_value
    receipt_safe = _receipt_safe_values(session, observation, as_of)
    for name in ("funding_percentile", "return_24h"):
        value = receipt_safe[name]
        if value is None or values.get(name) != value:
            return None
    return values


def generate_market_observations(session: Session, as_of: datetime) -> int:
    as_of = normalize_utc(as_of)
    valuation = _valuation_timestamp(as_of)
    created = 0
    candles = session.scalars(
        select(MarketObservation).where(
            MarketObservation.observation_kind == "candle",
            MarketObservation.observed_at == valuation,
        )
    ).all()
    for candle in candles:
        metrics = _trusted_metric_values(session, candle, as_of)
        if metrics is None:
            continue
        percentile = metrics.get("funding_percentile")
        return_24h = metrics.get("return_24h")
        if percentile is None or return_24h is None:
            continue
        if not (percentile >= 90 and abs(return_24h) >= 0.01):
            continue
        external_id = f"market-observation:{candle.asset}:{valuation.isoformat()}:extreme-funding-24h"
        if session.scalar(
            select(SourceItem).where(SourceItem.external_id == external_id)
        ):
            continue
        metadata = {
            "asset": candle.asset,
            "as_of": valuation.isoformat(),
            "observation_rule": "EXTREME_FUNDING_WITH_24H_MOVE",
            "metric_values": metrics,
            "market_observation_id": str(candle.id),
            "source_name": candle.source_name,
        }
        direction = "negative" if return_24h < 0 else "positive"
        session.add(
            SourceItem(
                source_type="MARKET",
                source_name="quant-radar-metric-rule",
                external_id=external_id,
                canonical_url=None,
                title=(
                    f"{candle.asset} funding is in the upper {percentile:.0f}th percentile "
                    f"while its 24h return is {direction}"
                ),
                authors=[],
                published_at=valuation,
                retrieved_at=normalize_utc(candle.retrieved_at),
                raw_text=(
                    f"PIT-safe metric observation at {valuation.isoformat()}: "
                    f"funding percentile={percentile:.2f}; return_24h={return_24h:.6f}."
                ),
                raw_metadata=metadata,
                content_sha256=content_hash("market observation", metadata),
            )
        )
        created += 1
    session.commit()
    return created


def ingest(
    session: Session,
    adapter: SourceAdapter,
    limit: int,
    offline: bool = False,
    *,
    requested_start: datetime | None = None,
    requested_end: datetime | None = None,
    code_sha: str | None = None,
) -> int:
    run = CollectionRun(
        source=adapter.name,
        requested=limit,
        started_at=utcnow(),
        requested_start=requested_start,
        requested_end=requested_end,
        code_sha=code_sha,
    )
    session.add(run)
    session.flush()
    try:
        records = adapter.collect(limit, offline)
        diagnostics = getattr(adapter, "last_funding_diagnostics", None)
        if diagnostics:
            run.diagnostics = diagnostics
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
            requested_start=requested_start,
            requested_end=requested_end,
            code_sha=code_sha,
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
    session: Session,
    as_of: datetime,
    window: int = 30,
    volatility_window: int = 24,
) -> int:
    cutoff = normalize_utc(as_of)
    observations = session.scalars(
        select(MarketObservation).order_by(
            MarketObservation.asset, MarketObservation.observed_at
        )
    ).all()
    by_asset: dict[str, list[MarketObservation]] = {}
    for observation in observations:
        if normalize_utc(observation.retrieved_at) <= cutoff:
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
            if r.mark_price is not None and r.observation_kind == "candle"
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
                existing = session.scalar(
                    select(MarketMetric).where(
                        MarketMetric.observation_id == row.id,
                        MarketMetric.metric_name == name,
                    )
                )
                if (
                    existing
                    and existing.calculation_metadata.get("support_receipt_cutoff")
                    == cutoff.isoformat()
                ):
                    continue
                if value is None:
                    if existing:
                        session.delete(existing)
                    continue
                metadata = {
                    "pit_cutoff": row.observed_at.isoformat(),
                    "support_receipt_cutoff": cutoff.isoformat(),
                    "window": window
                    if name == "funding_percentile"
                    else volatility_window,
                }
                if existing:
                    existing.metric_value = value
                    existing.calculated_at = cutoff
                    existing.calculation_metadata = metadata
                    continue
                session.add(
                    MarketMetric(
                        observation_id=row.id,
                        metric_name=name,
                        metric_value=value,
                        calculated_at=cutoff,
                        calculation_metadata=metadata,
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


def hypothesis_quality_ok(
    statement: str,
    independent_variable: str,
    dependent_variable: str,
    universe: str,
    horizon: str,
    required_data: list[str],
) -> bool:
    return bool(
        statement.strip()
        and independent_variable.strip()
        and dependent_variable.strip()
        and universe.strip()
        and horizon.strip()
        and required_data
        and any(
            term
            in f"{statement} {independent_variable} {dependent_variable} {universe} {' '.join(required_data)}".lower()
            for term in TOPIC_TERMS
        )
    )


def analyze(session: Session, client: LLMClient, limit: int = 20) -> int:
    candidates = session.scalars(
        select(SourceItem)
        .where(SourceItem.source_type == "ACADEMIC")
        .order_by(SourceItem.created_at.desc())
        .limit(limit)
    ).all()
    items = [
        item
        for item in candidates
        if academic_relevant(item.title, item.raw_text, item.raw_metadata)
    ]
    route = client.router.resolve("ANALYST") if hasattr(client, "router") else None
    run = AnalysisRun(
        role="ANALYST",
        provider=client.provider,
        model_name=client.model,
        requested_model_tier=route.tier.value if route else None,
        actual_model_name=route.model if route else client.model,
        thinking_enabled=route.thinking if route else None,
        reasoning_effort=route.reasoning_effort if route else None,
        prompt_version=client.prompt_version,
        schema_version="1",
        item_count=len(items),
        started_at=utcnow(),
    )
    session.add(run)
    created = 0
    pro_calls = 0
    pro_limit = 3
    for item in items:
        try:
            triage = client.triage(item.title, item.raw_text)
            if not (
                triage.retain
                and triage.relevance_score >= 60
                and triage.testability_score >= 60
            ):
                continue
            if route and pro_calls >= pro_limit:
                raise ValueError("Pro analyst budget exhausted")
            analyst = client.analyze(item.title, item.raw_text)
            if not hypothesis_quality_ok(
                analyst.possible_hypothesis,
                "funding rate",
                "subsequent return",
                analyst.universe,
                analyst.horizon,
                analyst.required_data,
            ):
                continue
            critic = client.critique(analyst.possible_hypothesis)
            if not critic.provenance_sufficient:
                continue
            pro_calls += 1 if route else 0
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
    run.status = (
        "SUCCESS"
        if run.failure_count == 0
        else ("PARTIAL" if run.success_count else "FAILED")
    )
    run.ended_at = utcnow()
    session.commit()
    return created


def daily_report(
    session: Session,
    output_dir: str,
    report_date: date | None = None,
    as_of: datetime | None = None,
) -> Path:
    report_date = report_date or datetime.now(UTC).date()
    if as_of is None:
        raise ValueError("daily_report requires an explicit as_of cutoff")
    as_of = normalize_utc(as_of)
    items = session.scalars(
        select(SourceItem)
        .where(
            SourceItem.retrieved_at <= as_of,
            (SourceItem.published_at.is_(None)) | (SourceItem.published_at <= as_of),
        )
        .order_by(SourceItem.created_at.desc())
    ).all()
    hypotheses = session.scalars(
        select(Hypothesis)
        .where(Hypothesis.created_at <= as_of)
        .order_by(Hypothesis.score.desc())
    ).all()
    source_by_id = {item.id: item for item in items}
    hypotheses = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.source_item_id in source_by_id
        and academic_relevant(
            source_by_id[hypothesis.source_item_id].title,
            source_by_id[hypothesis.source_item_id].raw_text,
            source_by_id[hypothesis.source_item_id].raw_metadata,
        )
    ][:3]
    market = [
        item
        for item in items
        if item.source_type == "MARKET"
        and item.source_name == "quant-radar-metric-rule"
        and item.raw_metadata.get("as_of", "")
        == _valuation_timestamp(as_of).isoformat()
    ][:3]
    academic_items = [
        item
        for item in items
        if item.source_type == "ACADEMIC"
        and academic_relevant(item.title, item.raw_text, item.raw_metadata)
    ][:2]
    lines = [f"# Daily Quant Radar — {report_date}", "", "## Market Observation", ""]
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
    academic_items = [
        item
        for item in items
        if item.source_type == "ACADEMIC"
        and academic_relevant(item.title, item.raw_text, item.raw_metadata)
    ][:2]
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
    if market:
        lines.append(
            "- **Funding percentile:** A rank of the current funding rate within its trailing window; it identifies unusually crowded positioning without being a trading signal."
        )
    if hypotheses:
        lines.append(
            "- **Falsification criterion:** The pre-specified result that would reject a hypothesis when tested on point-in-time data."
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
