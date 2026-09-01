from quant_research_radar.retrieval import expand_query


def test_domain_query_expansion_is_bounded_and_deterministic() -> None:
    assert expand_query("crowded perp longs") == [
        "crowded perp longs",
        "funding",
        "perpetual",
        "positioning",
        "crowding",
    ]
    assert expand_query("mean reversion") == [
        "mean reversion",
        "reversal",
        "subsequent returns",
    ]


def test_query_expansion_does_not_add_unrelated_terms() -> None:
    assert expand_query("unrelated astronomy") == ["unrelated astronomy"]
