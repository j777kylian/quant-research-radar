from quant_research_radar.sources import (
    CrossrefSource,
    InstitutionalHtmlSource,
    InstitutionalRssSource,
    NberSource,
    source_registry,
)


class _Response:
    def __init__(self, payload: dict | None = None, text: str = "") -> None:
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("XML response")
        return self._payload


class _Client:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def get(self, *_args: object, **_kwargs: object) -> _Response:
        return self.response


def test_crossref_and_nber_preserve_working_paper_taxonomy() -> None:
    crossref = CrossrefSource(
        client=_Client(
            _Response(
                {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1/example",
                                "title": ["Funding and perpetual futures"],
                                "published": {"date-parts": [[2026, 8, 1]]},
                                "author": [{"given": "Ada", "family": "Researcher"}],
                                "abstract": "A market microstructure study.",
                            }
                        ]
                    }
                }
            )
        )
    )
    nber = NberSource(
        client=_Client(
            _Response(
                text="""<rss><channel><item><title>Liquidity and return predictability</title><link>https://www.nber.org/papers/w123</link><guid>w123</guid><pubDate>Fri, 01 Aug 2026 00:00:00 +0000</pubDate><description>Market microstructure evidence.</description></item></channel></rss>"""
            )
        )
    )

    crossref_record = crossref.collect(1)[0]
    nber_record = nber.collect(1)[0]

    assert (
        crossref_record.raw_metadata["publication_status"] == "PEER_REVIEWED_OR_UNKNOWN"
    )
    assert crossref_record.raw_metadata["doi"] == "10.1/example"
    assert nber_record.source_type == "ACADEMIC"
    assert nber_record.raw_metadata["publication_status"] == "WORKING_PAPER"
    assert nber_record.raw_metadata["access_mode"] == "METADATA_ONLY"


def test_institutional_html_metadata_does_not_invent_publication_time() -> None:
    adapter = InstitutionalHtmlSource(
        name="man-institute",
        endpoint="https://example.test/insights",
        client=_Client(
            _Response(
                text='<a href="https://example.test/research" aria-label="Article Market microstructure research">Research</a>'
            )
        ),
    )

    record = adapter.collect(1)[0]

    assert record.source_name == "man-institute"
    assert record.published_at is None
    assert record.raw_metadata["access_mode"] == "METADATA_ONLY"
    adapter = InstitutionalRssSource(
        name="aqr",
        endpoint="https://example.test/feed",
        client=_Client(
            _Response(
                text="""<rss><channel><item><title>Volatility regime research</title><link>https://example.test/research</link><guid>aqr-1</guid><pubDate>Fri, 01 Aug 2026 00:00:00 +0000</pubDate><description>Institutional market research.</description></item></channel></rss>"""
            )
        ),
    )

    record = adapter.collect(1)[0]
    registry = {row["source_name"]: row for row in source_registry()}

    assert record.source_name == "aqr"
    assert record.source_type == "PRACTITIONER"
    assert record.raw_metadata["independence_key"] == "https://example.test/research"
    assert registry["nber"]["publication_status"] == "WORKING_PAPER"
    assert registry["crossref"]["source_class"] == "ACADEMIC"
    assert registry["aqr"]["source_class"] == "PRACTITIONER"
