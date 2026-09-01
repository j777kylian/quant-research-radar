from datetime import UTC, datetime

from quant_research_radar.sources import (
    OpenAlexSource,
    PractitionerRssSource,
    source_registry,
)


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        return Response(self.payload)


def test_source_registry_preserves_content_class_and_publication_status() -> None:
    registry = {entry["source_name"]: entry for entry in source_registry()}
    assert registry["arxiv"]["source_class"] == "ACADEMIC"
    assert registry["arxiv"]["publication_status"] == "PREPRINT"
    assert registry["ssrn"]["publication_status"] == "WORKING_PAPER_OR_PREPRINT"


def test_openalex_targeted_record_preserves_doi_oa_and_topic_provenance() -> None:
    client = Client(
        {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1000/funding",
                    "display_name": "Funding rates and return predictability",
                    "publication_date": "2026-08-01",
                    "authorships": [{"author": {"display_name": "A Researcher"}}],
                    "open_access": {
                        "is_oa": True,
                        "oa_url": "https://repo.example/paper",
                    },
                    "primary_location": {
                        "landing_page_url": "https://doi.org/10.1000/funding"
                    },
                    "topics": [{"display_name": "Market microstructure"}],
                    "abstract_inverted_index": {
                        "Funding": [0],
                        "predicts": [1],
                        "returns": [2],
                    },
                }
            ],
            "meta": {"count": 1, "next_cursor": None},
        }
    )
    adapter = OpenAlexSource(
        client=client, now=lambda: datetime(2026, 8, 31, tzinfo=UTC)
    )

    records = adapter.collect(10)

    assert len(records) == 1
    record = records[0]
    assert record.source_type == "ACADEMIC"
    assert record.external_id == "openalex:W1"
    assert record.canonical_url == "https://repo.example/paper"
    assert record.raw_metadata["doi"] == "10.1000/funding"
    assert record.raw_metadata["access_mode"] == "OA_FULLTEXT"
    assert record.raw_metadata["topics"] == ["Market microstructure"]
    assert client.calls[0][1]["search"] == "perpetual funding market microstructure"


def test_practitioner_adapter_preserves_primary_identity_and_never_claims_independence() -> (
    None
):
    client = Client(
        {
            "items": [
                {
                    "id": "alpha:1",
                    "title": "Funding rates and crypto market liquidity",
                    "url": "https://example.org/funding",
                    "published_at": "2026-08-20T00:00:00+00:00",
                    "summary": "A public practitioner research note.",
                    "primary_url": "https://example.org/original",
                }
            ]
        }
    )
    adapter = PractitionerRssSource(client=client)

    records = adapter.collect(5)

    assert records[0].source_type == "PRACTITIONER"
    assert records[0].raw_metadata["independence_key"] == "https://example.org/original"
    assert records[0].raw_metadata["access_mode"] == "PUBLIC_WEB"


class XmlResponse:
    status_code = 200

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self):
        raise ValueError("RSS is XML, not JSON")


class XmlClient:
    def get(self, _url: str) -> XmlResponse:
        return XmlResponse(
            """<rss><channel><item><guid>post-1</guid><title>Crypto funding and liquidity</title><link>https://example.org/post-1</link><pubDate>Mon, 25 Aug 2026 12:00:00 +0000</pubDate><description>Public practitioner research.</description></item></channel></rss>"""
        )


def test_practitioner_rss_xml_path_preserves_primary_identity() -> None:
    records = PractitionerRssSource(client=XmlClient()).collect(5)

    assert records[0].external_id == "post-1"
    assert records[0].raw_metadata["independence_key"] == "https://example.org/post-1"
    assert records[0].raw_metadata["source_payload"].startswith("<item>")


def test_source_registry_marks_unavailable_scraping_targets_without_using_them() -> (
    None
):
    registry = {entry["source_name"]: entry for entry in source_registry()}

    assert registry["openalex"]["adapter_status"] == "READY"
    assert registry["alpha-architect"]["source_class"] == "PRACTITIONER"
    assert registry["ssrn"]["adapter_status"] == "UNAVAILABLE"
    assert registry["x"]["adapter_status"] == "UNAVAILABLE"
