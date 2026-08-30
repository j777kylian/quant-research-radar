from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    Hypothesis,
    MarketMetric,
    MarketObservation,
    SourceItem,
)
from quant_research_radar.live import rendered_report_counts, write_live_review
from quant_research_radar.llm import FakeLLMClient
from quant_research_radar.pipeline import (
    academic_relevant,
    analyze,
    calculate_metrics,
    daily_report,
    generate_market_observations,
    ingest_records,
)
from quant_research_radar.sources import HyperliquidSource, SourceRecord


def db() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def academic(title: str, text: str, categories: list[str]) -> SourceRecord:
    return SourceRecord(
        "ACADEMIC",
        "arxiv",
        title,
        title,
        None,
        [],
        datetime.now(UTC),
        text,
        {"categories": categories},
    )


def test_academic_gate_rejects_off_topic_and_retains_quant_research() -> None:
    assert not academic_relevant(
        "Planetary migration", "protoplanetary dust", ["astro-ph.EP"]
    )
    assert not academic_relevant(
        "Quantum chaos", "quantum transport is well-defined", ["quant-ph"]
    )
    assert not academic_relevant(
        "Spherical collapse", "the derivative of radial velocity", ["astro-ph.CO"]
    )
    assert not academic_relevant("Quantum risk", "quantum states", ["quant-ph"])
    assert not academic_relevant("Eval-awareness", "AI safety evaluations", ["cs.AI"])
    assert not academic_relevant(
        "Quantum research", "market-free physics", ["q-fin.ST"]
    )
    assert not academic_relevant(
        "Spherical collapse", "market-free physics", ["stat.ML"]
    )
    assert not academic_relevant("Quantum finance", "quantum states", ["quant-ph"])
    assert academic_relevant(
        "Asset pricing with volatility", "return predictability", ["q-fin.ST"]
    )
    assert academic_relevant(
        "Crypto market microstructure", "perpetual funding and order flow", ["q-fin.TR"]
    )
    assert not academic_relevant(
        "Crypto market microstructure",
        "perpetual funding and order flow",
        ["q-finance"],
    )
    assert academic_relevant("Crypto funding", "funding rate", ["q-fin"])
    assert academic_relevant("Crypto funding", "funding rate", ["econ"])
    assert academic_relevant("Crypto funding", "funding rate", ["econ.EM"])
    assert academic_relevant("Crypto funding", "funding rate", ["stat"])
    assert academic_relevant("Crypto funding", "funding rate", ["stat.ML"])


def test_metrics_exclude_future_received_support_records() -> None:
    session = db()
    as_of = datetime(2026, 8, 29, 10, 12, tzinfo=UTC)
    valuation = datetime(2026, 8, 29, 9, tzinfo=UTC)
    for hour in range(31):
        observed_at = valuation - timedelta(hours=hour)
        receipt = as_of + timedelta(seconds=1) if hour in {1, 4} else as_of
        session.add_all(
            [
                MarketObservation(
                    asset="BTC",
                    observed_at=observed_at,
                    observation_kind="funding",
                    funding_rate=99.0 if hour == 1 else float(hour),
                    source_name="hyperliquid",
                    retrieved_at=receipt,
                ),
                MarketObservation(
                    asset="BTC",
                    observed_at=observed_at,
                    observation_kind="candle",
                    mark_price=100.0 + hour,
                    source_name="hyperliquid",
                    retrieved_at=receipt,
                ),
            ]
        )
    session.commit()

    calculate_metrics(session, as_of=as_of)
    target = session.scalar(
        select(MarketObservation).where(
            MarketObservation.asset == "BTC",
            MarketObservation.observation_kind == "candle",
            MarketObservation.observed_at == valuation,
        )
    )
    assert target is not None
    metrics = {
        metric.metric_name: metric
        for metric in session.scalars(
            select(MarketMetric).where(MarketMetric.observation_id == target.id)
        )
    }
    assert "return_1h" not in metrics
    assert "return_4h" not in metrics
    assert "rolling_volatility" not in metrics
    assert metrics["funding_percentile"].metric_value != 100.0
    assert (
        metrics["funding_percentile"].calculation_metadata["support_receipt_cutoff"]
        == as_of.isoformat()
    )


def test_market_metrics_create_deterministic_observation_only_when_rule_holds() -> None:
    session = db()
    as_of = datetime(2026, 8, 29, 10, 12, tzinfo=UTC)
    valuation = datetime(2026, 8, 29, 9, tzinfo=UTC)
    records = []
    for asset in ("BTC", "ETH"):
        records.extend(
            HyperliquidSource._history_record(asset, valuation - timedelta(hours=i), i)
            for i in range(31)
        )
        records.extend(
            HyperliquidSource._candle_record(asset, valuation - timedelta(hours=i), i)
            for i in range(25)
        )
    ingest_records(session, records)
    from quant_research_radar.db import MarketMetric, MarketObservation

    candles = session.scalars(
        select(MarketObservation).where(MarketObservation.observation_kind == "candle")
    ).all()
    for observation in session.scalars(select(MarketObservation)).all():
        observation.retrieved_at = as_of
        if observation.asset == "BTC" and observation.observation_kind == "funding":
            observation.funding_rate = (
                1.0 if observation.observed_at.replace(tzinfo=UTC) == valuation else 0.0
            )
        if observation.asset == "BTC" and observation.observation_kind == "candle":
            observation.mark_price = (
                102.0
                if observation.observed_at.replace(tzinfo=UTC) == valuation
                else 100.0
            )
    for candle in candles:
        if (
            candle.asset == "BTC"
            and candle.observed_at.replace(tzinfo=UTC) == valuation
        ):
            candle.retrieved_at = as_of
            session.add_all(
                [
                    MarketMetric(
                        observation_id=candle.id,
                        metric_name="funding_percentile",
                        metric_value=100,
                        calculation_metadata={
                            "pit_cutoff": valuation.isoformat(),
                            "support_receipt_cutoff": as_of.isoformat(),
                        },
                        calculated_at=as_of,
                    ),
                    MarketMetric(
                        observation_id=candle.id,
                        metric_name="return_24h",
                        metric_value=0.02,
                        calculation_metadata={
                            "pit_cutoff": valuation.isoformat(),
                            "support_receipt_cutoff": as_of.isoformat(),
                        },
                        calculated_at=as_of,
                    ),
                    MarketMetric(
                        observation_id=candle.id,
                        metric_name="return_1h",
                        metric_value=0.0,
                    ),
                    MarketMetric(
                        observation_id=candle.id,
                        metric_name="return_4h",
                        metric_value=0.0,
                    ),
                    MarketMetric(
                        observation_id=candle.id,
                        metric_name="rolling_volatility",
                        metric_value=0.01,
                    ),
                ]
            )
    session.commit()
    calculate_metrics(session, as_of=as_of)
    assert generate_market_observations(session, as_of) == 1
    assert generate_market_observations(session, as_of) == 0
    item = session.scalar(
        select(SourceItem).where(SourceItem.external_id.like("market-observation:%"))
    )
    assert (
        item is not None
        and item.raw_metadata["observation_rule"] == "EXTREME_FUNDING_WITH_24H_MOVE"
    )


def test_market_observation_rejects_untrusted_or_non_pit_metrics() -> None:
    session = db()
    as_of = datetime(2026, 8, 29, 10, 12, tzinfo=UTC)
    valuation = datetime(2026, 8, 29, 9, tzinfo=UTC)
    candle = MarketObservation(
        asset="BTC",
        observed_at=valuation,
        observation_kind="candle",
        mark_price=100.0,
        source_name="untrusted-fixture",
        retrieved_at=as_of,
    )
    session.add(candle)
    session.flush()
    session.add_all(
        [
            MarketMetric(
                observation_id=candle.id,
                metric_name="funding_percentile",
                metric_value=100.0,
                calculation_metadata={"pit_cutoff": "2099-01-01T00:00:00+00:00"},
            ),
            MarketMetric(
                observation_id=candle.id,
                metric_name="return_24h",
                metric_value=-0.02,
                calculation_metadata={"pit_cutoff": "2099-01-01T00:00:00+00:00"},
            ),
        ]
    )
    session.commit()

    trusted = MarketObservation(
        asset="ETH",
        observed_at=valuation,
        observation_kind="candle",
        mark_price=100.0,
        source_name="hyperliquid",
        retrieved_at=as_of,
    )
    session.add(trusted)
    session.flush()
    for metric_name, metric_value in (
        ("funding_percentile", 100.0),
        ("return_24h", -0.02),
    ):
        session.add(
            MarketMetric(
                observation_id=trusted.id,
                metric_name=metric_name,
                metric_value=metric_value,
                calculation_metadata={
                    "pit_cutoff": valuation.isoformat(),
                    "support_receipt_cutoff": as_of.isoformat(),
                },
                calculated_at=as_of,
            )
        )
    session.commit()
    assert generate_market_observations(session, as_of) == 0


def test_daily_report_excludes_future_received_market_evidence(tmp_path: Path) -> None:
    session = db()
    as_of = datetime(2026, 8, 29, 10, 12, tzinfo=UTC)
    session.add(
        SourceItem(
            source_type="MARKET",
            source_name="quant-radar-metric-rule",
            external_id="market-observation:BTC:2026-08-29T09:00:00+00:00:future",
            canonical_url=None,
            title="Future-received evidence",
            authors=[],
            published_at=datetime(2026, 8, 29, 9, tzinfo=UTC),
            retrieved_at=as_of + timedelta(hours=1),
            raw_text="",
            raw_metadata={"as_of": "2026-08-29T09:00:00+00:00"},
            content_sha256="f" * 64,
        )
    )
    session.commit()
    assert (
        "Future-received evidence"
        not in daily_report(session, str(tmp_path), as_of=as_of).read_text()
    )


def test_accepted_source_claim_requires_independent_critic_provenance() -> None:
    class InsufficientCritic(FakeLLMClient):
        def critique(self, hypothesis: str):
            result = super().critique(hypothesis)
            return result.model_copy(update={"provenance_sufficient": False})

    session = db()
    ingest_records(
        session,
        [
            academic(
                "Funding and returns", "crypto funding predicts returns", ["q-fin.ST"]
            )
        ],
    )
    assert analyze(session, InsufficientCritic()) == 0
    assert not session.scalars(select(Hypothesis)).all()


def test_source_claim_does_not_become_hypothesis_without_quant_gate() -> None:
    session = db()
    ingest_records(session, [academic("Quantum chaos", "quantum states", ["quant-ph"])])
    assert analyze(session, FakeLLMClient()) == 0
    assert not session.scalars(select(Hypothesis)).all()


def test_report_counts_only_retained_quant_context_and_concepts(tmp_path: Path) -> None:
    session = db()
    ingest_records(
        session,
        [
            academic("Quantum chaos", "quantum states", ["quant-ph"]),
            academic(
                "Funding and returns", "crypto funding predicts returns", ["q-fin.ST"]
            ),
        ],
    )
    assert analyze(session, FakeLLMClient()) == 1
    report = daily_report(session, str(tmp_path)).read_text()
    assert "Quantum chaos" not in report
    assert "Falsification criterion" in report
    assert "Hypothesis Candidates" in report


def test_human_review_template_includes_both_cycles(tmp_path: Path) -> None:
    text = write_live_review(tmp_path).read_text()
    assert "## Cycle 1" in text and "## Cycle 2" in text
    assert "Proceed to 5–7 day observation: YES / NO / UNSURE" in text


def test_rendered_report_counts_match_labels() -> None:
    report = "**FACT:** market\n- **CLAIM:** paper\n- **HYPOTHESIS:** candidate\n"
    assert rendered_report_counts(report) == {
        "market_facts": 1,
        "academic_claim_lines": 1,
        "hypotheses": 1,
    }
