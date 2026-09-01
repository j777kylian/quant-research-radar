from datetime import UTC, datetime

from quant_research_radar.sources import SourceRecord, collect_isolated


class Good:
    name = "good"

    def collect(self, limit: int, offline: bool = False):
        return [
            SourceRecord(
                "PRACTITIONER",
                "good",
                "1",
                "Funding and perpetual microstructure",
                None,
                [],
                datetime(2026, 8, 30, tzinfo=UTC),
                "funding perpetual",
                {},
            )
        ][:limit]


class Bad:
    name = "bad"

    def collect(self, limit: int, offline: bool = False):
        raise RuntimeError("provider-controlled secret-like detail")


def test_isolated_collection_retains_healthy_sources_and_sanitizes_failure() -> None:
    records, status = collect_isolated([Good(), Bad()], 3)

    assert [record.source_name for record in records] == ["good"]
    assert status == {"good": "READY", "bad": "DEGRADED"}
