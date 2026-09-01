from quant_research_radar.benchmarks import (
    ANALYST_CASES,
    routing_rationale,
    score_analyst_output,
    summarize_scorecard,
)
from quant_research_radar.llm import AnalystOutput


def test_benchmark_corpus_is_versioned_and_has_adversarial_case() -> None:
    assert len(ANALYST_CASES) >= 3
    assert any(case.adversarial for case in ANALYST_CASES)
    assert {case.expected_relevant for case in ANALYST_CASES} == {True, False}


def test_analyst_score_rewards_faithful_testable_output_and_penalizes_hallucination() -> (
    None
):
    good = AnalystOutput(
        core_question="Does extreme funding alter subsequent return distributions?",
        reported_finding="The abstract reports an association; no causal claim is made.",
        actual_evidence="The abstract reports an association; no causal claim is made.",
        causal_status="CORRELATIONAL",
        analysis_confidence="ABSTRACT_ONLY",
        limitations=["abstract-only"],
        mechanism="unknown",
        market="crypto perpetuals",
        universe="BTC perpetuals",
        horizon="24h",
        required_data=["funding", "completed candles"],
        possible_hypothesis="Extreme funding is associated with a different subsequent return distribution.",
        practical_reproducibility="requires PIT data",
        unknowns=["sample size unavailable"],
    )
    bad = good.model_copy(
        update={"reported_finding": "Sample size was 9999 and effect size was 50%"}
    )

    assert score_analyst_output(good, expected_relevant=True)["hallucination"] == 0
    assert score_analyst_output(bad, expected_relevant=True)["hallucination"] == 1


def test_scorecard_keeps_task_dimensions_separate_and_declines_unjustified_reroute() -> (
    None
):
    scorecard = summarize_scorecard(
        {
            "flash": [
                {"faithfulness": 1, "hallucination": 0, "schema_compliance": 1},
                {"faithfulness": 0, "hallucination": 0, "schema_compliance": 1},
            ],
            "pro": [
                {"faithfulness": 1, "hallucination": 0, "schema_compliance": 1},
                {"faithfulness": 0, "hallucination": 0, "schema_compliance": 1},
            ],
        }
    )
    assert scorecard["flash"]["faithfulness"] == 0.5
    assert (
        routing_rationale(scorecard) == "KEEP_EXISTING_ROUTING_INSUFFICIENT_DIFFERENCE"
    )
