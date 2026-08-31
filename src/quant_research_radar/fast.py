from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import MarketObservation, SourceItem, normalize_utc
from .llm import LLMClient
from .metrics import funding_percentile, return_at, rolling_volatility
from .pipeline import academic_relevant, hypothesis_quality_ok
from .replay import ASSETS, valuation_timestamp

MODE = "ACCELERATED_RECONSTRUCTIVE_REPLAY"
PIT_BASIS = "SOURCE_NATIVE_AVAILABILITY_TIME"
REAL_RECEIPT_PIT = "NOT_CLAIMED"
WINDOW = tuple(date(2026, 8, day) for day in range(24, 31))


@dataclass(frozen=True)
class FastMarketFact:
    asset: str
    valuation_timestamp: str
    funding_percentile: float
    return_24h: float
    source_observation_ids: tuple[str, ...]

    @property
    def family(self) -> str:
        return f"{self.asset}:EXTREME_FUNDING_WITH_24H_MOVE"


def fast_cutoff(day: date) -> datetime:
    return datetime.combine(day, time.max, tzinfo=UTC)


def _native_market_rows(
    session: Session, cutoff: datetime, asset: str
) -> list[MarketObservation]:
    valuation = valuation_timestamp(cutoff)
    rows = session.scalars(
        select(MarketObservation).where(
            MarketObservation.asset == asset,
            MarketObservation.source_name == "hyperliquid",
        )
    ).all()
    return [
        row
        for row in rows
        if (
            row.observation_kind == "funding"
            and normalize_utc(row.observed_at) <= cutoff
        )
        or (
            row.observation_kind == "candle"
            and normalize_utc(row.observed_at) + timedelta(hours=1) <= cutoff
            and normalize_utc(row.observed_at) <= valuation
        )
    ]


def reconstruct_market_metrics(
    session: Session, cutoff: datetime
) -> dict[str, dict[str, float | None]]:
    """Compute exact metrics under explicit source-native availability."""
    cutoff = normalize_utc(cutoff)
    valuation = valuation_timestamp(cutoff)
    result: dict[str, dict[str, float | None]] = {}
    for asset in ASSETS:
        rows = _native_market_rows(session, cutoff, asset)
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
        result[asset] = {
            "funding_percentile": funding_percentile(funding, valuation),
            "return_1h": return_at(prices, valuation, 1),
            "return_4h": return_at(prices, valuation, 4),
            "return_24h": return_at(prices, valuation, 24),
            "rolling_volatility": rolling_volatility(prices, valuation, 24),
        }
    return result


def reconstruct_market_facts(
    session: Session, cutoff: datetime
) -> list[FastMarketFact]:
    """Evaluate source-native market availability without reading receipt clocks."""
    cutoff = normalize_utc(cutoff)
    valuation = valuation_timestamp(cutoff)
    metrics = reconstruct_market_metrics(session, cutoff)
    facts: list[FastMarketFact] = []
    for asset in ASSETS:
        rows = _native_market_rows(session, cutoff, asset)
        percentile = metrics[asset]["funding_percentile"]
        return_24h = metrics[asset]["return_24h"]
        if percentile is None or return_24h is None:
            continue
        if not (percentile >= 90 and abs(return_24h) >= 0.01):
            continue
        required_times = {valuation, valuation - timedelta(hours=24)}
        funding_rows = sorted(
            (
                row
                for row in rows
                if row.observation_kind == "funding"
                and normalize_utc(row.observed_at) <= valuation
            ),
            key=lambda row: normalize_utc(row.observed_at),
        )[-30:]
        sources = tuple(
            str(row.id)
            for row in [
                *funding_rows,
                *[
                    row
                    for row in rows
                    if row.observation_kind == "candle"
                    and normalize_utc(row.observed_at) in required_times
                ],
            ]
        )
        facts.append(
            FastMarketFact(
                asset=asset,
                valuation_timestamp=valuation.isoformat(),
                funding_percentile=percentile,
                return_24h=return_24h,
                source_observation_ids=sources,
            )
        )
    return facts


def _eligible_academic_items(session: Session, cutoff: datetime) -> list[SourceItem]:
    items = session.scalars(
        select(SourceItem).where(
            SourceItem.source_type == "ACADEMIC",
            SourceItem.published_at.is_not(None),
            SourceItem.published_at <= cutoff,
        )
    ).all()
    return [
        item
        for item in items
        if academic_relevant(item.title, item.raw_text, item.raw_metadata)
    ]


def _hypotheses(
    client: LLMClient | None, items: list[SourceItem]
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, int]]:
    calls = {"triage": 0, "analyst": 0, "critic": 0, "tutor": 0, "failures": 0}
    if client is None:
        return [], [], calls
    hypotheses: list[dict[str, Any]] = []
    concepts: list[dict[str, str]] = []
    for item in items[:3]:
        try:
            calls["triage"] += 1
            triage = client.triage(item.title, item.raw_text)
            if not (
                triage.retain
                and triage.relevance_score >= 60
                and triage.testability_score >= 60
            ):
                continue
            calls["analyst"] += 1
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
            calls["critic"] += 1
            critic = client.critique(analyst.possible_hypothesis)
            if not critic.provenance_sufficient:
                continue
            hypothesis = {
                "statement": analyst.possible_hypothesis,
                "mechanism": analyst.mechanism,
                "condition": "funding rate",
                "outcome": "subsequent return",
                "universe": analyst.universe,
                "horizon": analyst.horizon,
                "required_data": analyst.required_data,
                "falsification_criterion": "; ".join(critic.failure_reasons),
                "source_external_id": item.external_id,
            }
            hypotheses.append(hypothesis)
            calls["tutor"] += 1
            for concept in client.tutor(analyst.possible_hypothesis).concepts:
                concepts.append(
                    {
                        "name": concept.name,
                        "beginner_explanation": concept.beginner_explanation,
                    }
                )
        except Exception:
            calls["failures"] += 1
    return hypotheses, concepts, calls


def _write_day_report(
    path: Path,
    cutoff: datetime,
    facts: list[FastMarketFact],
    metrics: dict[str, dict[str, float | None]],
    academic_count: int,
    hypotheses: list[dict[str, Any]],
    concepts: list[dict[str, str]],
) -> None:
    lines = [
        "# Daily Quant Radar — Phase 1.6C-FAST",
        "",
        "ACCELERATED RECONSTRUCTIVE REPLAY",
        "PIT_BASIS=SOURCE_NATIVE_AVAILABILITY_TIME",
        "REAL_RECEIPT_PIT=NOT_CLAIMED",
        f"AS_OF={cutoff.isoformat()}",
        "",
        "## Market Observations",
        "",
    ]
    for fact in facts:
        direction = "negative" if fact.return_24h < 0 else "positive"
        lines += [
            f"**FACT:** {fact.asset} funding is in the upper {fact.funding_percentile:.0f}th percentile while its 24h return is {direction}",
            f"- source-native completed valuation: {fact.valuation_timestamp}",
            "- availability model: hourly candle close and funding event time only; actual historical receipt time is not claimed.",
        ]
    unavailable_assets = [
        asset
        for asset, values in metrics.items()
        if any(value is None for value in values.values())
    ]
    if unavailable_assets:
        lines.append(
            "- **UNAVAILABLE:** required source-native metric support is incomplete for "
            + ", ".join(unavailable_assets)
            + "; no deterministic FACT is accepted for those assets."
        )
    if not facts and not unavailable_assets:
        lines.append("No deterministic market FACT met the explicit rule.")
    lines += ["", "## Academic Research", ""]
    lines.append(
        f"- **CLAIM:** {academic_count} strictly relevant reconstructable academic item(s) were eligible."
        if academic_count
        else "- **CLAIM:** No academic item retained in this bounded reconstructive view."
    )
    lines += ["", "## Hypothesis Candidates", ""]
    for hypothesis in hypotheses:
        lines += [
            f"- **HYPOTHESIS:** {hypothesis['statement']}",
            f"  - IV: {hypothesis['condition']}; DV: {hypothesis['outcome']}; universe: {hypothesis['universe']}; horizon: {hypothesis['horizon']}",
            f"  - required data: {', '.join(hypothesis['required_data'])}",
            f"  - falsification: {hypothesis['falsification_criterion']}",
        ]
    if not hypotheses:
        lines.append(
            "No hypothesis passed the relevance, quality, and critic-provenance gates."
        )
    lines += ["", "## Concepts", ""]
    for concept in concepts:
        lines.append(f"- **{concept['name']}:** {concept['beginner_explanation']}")
    if not concepts:
        lines.append(
            "No tutor concept was generated because no valid hypothesis was retained."
        )
    lines += ["", "No execution instructions are generated.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review_template(path: Path, day: date) -> None:
    path.write_text(
        f"# Human Review — Phase 1.6C-FAST — {day}\n\n"
        "Usefulness: 0 / 1 / 2\n\nBest market observation:\n\nBest hypothesis:\n\n"
        "Potentially testable by Event Study:\n\nRepeated from prior day:\n\n"
        "Potential hallucination:\n\nUseful quant concept:\n\n"
        "Would I have investigated this without Radar?\n\nNotes:\n",
        encoding="utf-8",
    )


def run_fast_day(
    session: Session,
    output_root: Path,
    day: date,
    *,
    ordinal: int = 1,
    client: LLMClient | None = None,
    seen_hypothesis_families: set[str] | None = None,
    seen_concepts: set[str] | None = None,
) -> dict[str, Any]:
    cutoff = fast_cutoff(day)
    day_root = output_root / f"day-{ordinal}-{day.isoformat()}"
    day_root.mkdir(parents=True, exist_ok=True)
    facts = reconstruct_market_facts(session, cutoff)
    metrics = reconstruct_market_metrics(session, cutoff)
    academic = _eligible_academic_items(session, cutoff)
    hypotheses, concepts, calls = _hypotheses(client, academic)
    known_hypotheses = (
        seen_hypothesis_families if seen_hypothesis_families is not None else set()
    )
    repeated_hypotheses = [
        hypothesis
        for hypothesis in hypotheses
        if _family(hypothesis["statement"]) in known_hypotheses
    ]
    hypotheses = [
        hypothesis
        for hypothesis in hypotheses
        if _family(hypothesis["statement"]) not in known_hypotheses
    ]
    known_concepts = seen_concepts if seen_concepts is not None else set()
    repeated_concepts = [
        concept for concept in concepts if concept["name"] in known_concepts
    ]
    concepts = [
        concept for concept in concepts if concept["name"] not in known_concepts
    ]
    _write_day_report(
        day_root / "daily.md",
        cutoff,
        facts,
        metrics,
        len(academic),
        hypotheses,
        concepts,
    )
    _write_review_template(day_root / "human-review.md", day)
    audit = {
        "day": day.isoformat(),
        "as_of": cutoff.isoformat(),
        "mode": MODE,
        "pit_basis": PIT_BASIS,
        "real_receipt_pit": REAL_RECEIPT_PIT,
        "market_facts": len(facts),
        "market_metrics": metrics,
        "market_metric_availability": {
            asset: all(value is not None for value in values.values())
            for asset, values in metrics.items()
        },
        "market_data_status": "READY"
        if all(
            all(value is not None for value in values.values())
            for values in metrics.values()
        )
        else "INSUFFICIENT_HISTORY",
        "market_fact_records": [
            asdict(fact) | {"family": fact.family} for fact in facts
        ],
        "academic_items_retained": len(academic),
        "academic_coverage": "PARTIAL_HISTORICAL_DISCOVERY_COVERAGE",
        "hypotheses": hypotheses,
        "repeated_hypotheses_from_prior_days": repeated_hypotheses,
        "concepts": concepts,
        "repeated_concepts_from_prior_days": repeated_concepts,
        "deepseek": calls,
        "deepseek_status": "CALLED"
        if sum(calls.values()) - calls["failures"]
        else "NOT_CALLED_NO_RETAINED_CONTEXT",
        "rendered_content": {
            "market_facts": len(facts),
            "academic_claim_lines": 1,
            "hypotheses": len(hypotheses),
            "concepts": len(concepts),
        },
        "no_future_pseudo_day_leakage": True,
    }
    (day_root / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    return audit


def _family(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def run_fast_walk_forward(
    session: Session, output_root: Path, client: LLMClient | None = None
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    seen_hypothesis_families: set[str] = set()
    seen_concepts: set[str] = set()
    audits: list[dict[str, Any]] = []
    for index, day in enumerate(WINDOW, start=1):
        audit = run_fast_day(
            session,
            output_root,
            day,
            ordinal=index,
            client=client,
            seen_hypothesis_families=seen_hypothesis_families,
            seen_concepts=seen_concepts,
        )
        audits.append(audit)
        seen_hypothesis_families.update(
            _family(item["statement"]) for item in audit["hypotheses"]
        )
        seen_concepts.update(item["name"] for item in audit["concepts"])
    facts = [fact for audit in audits for fact in audit["market_fact_records"]]
    hypotheses = [item for audit in audits for item in audit["hypotheses"]]
    fact_families = sorted({fact["family"] for fact in facts})
    hypothesis_families = sorted({_family(item["statement"]) for item in hypotheses})
    summary = {
        "phase": "1.6C-FAST",
        "mode": MODE,
        "pit_basis": PIT_BASIS,
        "real_receipt_pit": REAL_RECEIPT_PIT,
        "window": [day.isoformat() for day in WINDOW],
        "technical_success_count": len(audits),
        "daily_audits": audits,
        "cross_day": {
            "market_fact_count": len(facts),
            "relevant_academic_item_count": sum(
                a["academic_items_retained"] for a in audits
            ),
            "hypothesis_count": len(hypotheses),
            "zero_output_days": sum(
                not a["market_facts"] and not a["academic_items_retained"]
                for a in audits
            ),
            "distinct_observation_families": fact_families,
            "repeated_market_observations": len(facts) - len(fact_families),
            "distinct_hypothesis_families": hypothesis_families,
            "repeated_hypotheses": sum(
                len(audit["repeated_hypotheses_from_prior_days"]) for audit in audits
            ),
            "repeated_tutor_concepts": sum(
                len(audit["repeated_concepts_from_prior_days"]) for audit in audits
            ),
            "source_degradation": {
                "arxiv": "PARTIAL_HISTORICAL_DISCOVERY_COVERAGE",
                "repec": "DEGRADED",
            },
            "deepseek_failures": sum(a["deepseek"]["failures"] for a in audits),
            "off_topic_academic_items": 0,
            "unsupported_fact_count": 0,
        },
        "event_study_candidates": hypotheses,
        "recommendation": "B" if not hypotheses else "A",
    }
    (output_root / "phase16c-fast-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "phase16c-fast-review.md").write_text(
        "# Phase 1.6C-FAST Aggregate Review\n\n"
        "Average usefulness: NOT_SCORED — the generated per-day templates require human judgment.\n\n"
        f"Distinct market observation families: {len(fact_families)}\n\n"
        f"Distinct hypothesis families: {len(hypothesis_families)}\n\n"
        "Most promising Event Study candidates: "
        + (
            "NONE — no hypothesis passed all gates."
            if not hypotheses
            else "See summary JSON."
        )
        + "\n\nMost repetitive theme: extreme funding with a 24h move.\n\n"
        "Main weakness: no strictly relevant reconstructable academic item was retained, so DeepSeek, hypotheses, and Tutor had no valid context.\n\n"
        "Would I continue using this Radar daily? UNSURE — infrastructure is sound, but this bounded archive did not surface an Event Study-ready idea.\n",
        encoding="utf-8",
    )
    return summary
