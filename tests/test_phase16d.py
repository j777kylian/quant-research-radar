from datetime import UTC, datetime

import pytest

from quant_research_radar.intelligence import (
    Channel,
    Evidence,
    FusionInput,
    HypothesisDraft,
    Maturity,
    analyze_academic,
    analyze_market,
    analyze_social,
    fuse_hypotheses,
)


def market_evidence() -> Evidence:
    return Evidence(
        evidence_id="market:sol:2026-08-28T22:00:00+00:00",
        channel=Channel.MARKET,
        title="SOL funding is extreme while the 24h return is negative",
        body="funding percentile=100; return_24h=-0.0463",
        provenance_id="hyperliquid:sol:2026-08-28T22:00:00+00:00",
        independence_key="hyperliquid:sol:2026-08-28T22:00:00+00:00",
        observed_at=datetime(2026, 8, 28, 22, tzinfo=UTC),
        metadata={"asset": "SOL", "funding_percentile": 100.0, "return_24h": -0.0463},
    )


def test_market_analyzer_creates_structured_h1_without_other_channels() -> None:
    draft = analyze_market([market_evidence()])[0]

    assert draft.origin == Channel.MARKET
    assert draft.maturity == Maturity.H1
    assert draft.condition == "SOL funding percentile >= 90"
    assert draft.outcome == "subsequent 4h, 24h return distribution"
    assert draft.universe == "SOL perpetual"
    assert draft.horizon == "4h and 24h"
    assert draft.evidence_ids == ["market:sol:2026-08-28T22:00:00+00:00"]
    assert draft.supporting_channels == [Channel.MARKET]


def test_market_analyzer_rejects_cross_channel_input() -> None:
    academic = market_evidence().model_copy(update={"channel": Channel.ACADEMIC})

    with pytest.raises(ValueError, match="MARKET analyzer"):
        analyze_market([academic])


def test_first_pass_research_analyzers_refuse_foreign_channel_evidence() -> None:
    market = market_evidence()
    academic = market.model_copy(update={"channel": Channel.ACADEMIC})
    social = market.model_copy(update={"channel": Channel.SOCIAL})

    assert len(analyze_academic([academic])) == 1
    assert len(analyze_social([social])) == 1
    with pytest.raises(ValueError, match="ACADEMIC analyzer"):
        analyze_academic([market])
    with pytest.raises(ValueError, match="SOCIAL analyzer"):
        analyze_social([academic])


def test_fusion_only_upgrades_to_h3_for_two_independent_channels() -> None:
    market = HypothesisDraft.from_market(market_evidence())
    academic = market.model_copy(
        update={
            "origin": Channel.ACADEMIC,
            "evidence_ids": ["doi:10.1000/funding"],
            "supporting_channels": [Channel.ACADEMIC],
            "maturity": Maturity.H2,
            "mechanism": "Funding proxies positioning imbalance.",
            "independence_keys": ["doi:10.1000/funding"],
        }
    )

    unified = fuse_hypotheses(
        [
            FusionInput(
                draft=market, evidence_independence_keys=market.independence_keys
            ),
            FusionInput(
                draft=academic, evidence_independence_keys=academic.independence_keys
            ),
        ]
    )
    assert len(unified) == 1
    assert unified[0].maturity == Maturity.H3
    assert unified[0].supporting_channels == [Channel.ACADEMIC, Channel.MARKET]


def test_fusion_refuses_unrelated_research_as_cross_channel_convergence() -> None:
    academic = HypothesisDraft.from_research(
        Evidence(
            evidence_id="academic:volatility",
            channel=Channel.ACADEMIC,
            title="Realized volatility in crypto markets",
            body="Research measures realized volatility.",
            provenance_id="doi:volatility",
            independence_key="doi:volatility",
            observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        Channel.ACADEMIC,
    )
    social = HypothesisDraft.from_research(
        Evidence(
            evidence_id="social:order-book",
            channel=Channel.SOCIAL,
            title="Limit order book dynamics",
            body="Practitioner note about order books.",
            provenance_id="rss:order-book",
            independence_key="rss:order-book",
            observed_at=datetime(2026, 8, 2, tzinfo=UTC),
        ),
        Channel.SOCIAL,
    )

    unified = fuse_hypotheses(
        [
            FusionInput(
                draft=academic, evidence_independence_keys=academic.independence_keys
            ),
            FusionInput(
                draft=social, evidence_independence_keys=social.independence_keys
            ),
        ]
    )

    assert len(unified) == 2
    assert {draft.maturity for draft in unified} == {Maturity.H1}


def test_fusion_does_not_upgrade_duplicate_versions_or_reposts() -> None:
    market = HypothesisDraft.from_market(market_evidence())
    repost = market.model_copy(
        update={
            "origin": Channel.SOCIAL,
            "evidence_ids": ["social:repost:1"],
            "supporting_channels": [Channel.SOCIAL],
            "independence_keys": [market.independence_keys[0]],
        }
    )

    unified = fuse_hypotheses(
        [
            FusionInput(
                draft=market, evidence_independence_keys=market.independence_keys
            ),
            FusionInput(
                draft=repost, evidence_independence_keys=repost.independence_keys
            ),
        ]
    )
    assert len(unified) == 1
    assert unified[0].maturity == Maturity.H1
    assert unified[0].independent_evidence_count == 1
