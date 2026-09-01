from quant_research_radar.retrieval import expand_query


def test_domain_query_expansion_normalizes_paraphrases() -> None:
    assert expand_query("Crowded   Perp LONGS") == [
        "crowded perp longs",
        "funding",
        "perpetual",
        "positioning",
        "crowding",
    ]
