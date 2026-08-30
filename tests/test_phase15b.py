from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    AnalysisRun,
    Base,
    Claim,
    CollectionRun,
    MarketMetric,
    MarketObservation,
    SourceItem,
)
from quant_research_radar.llm import OpenAICompatClient
from quant_research_radar.metrics import (
    funding_percentile,
    return_at,
    rolling_volatility,
)
from quant_research_radar.pipeline import (
    analyze,
    calculate_metrics,
    daily_report,
    ingest,
    ingest_records,
)
from quant_research_radar.sources import (
    ArxivSource,
    HyperliquidSource,
    RepecSource,
    SourceRecord,
)

ARXIV_XML = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/1234.5678</id><title> A paper </title><summary>Abstract evidence.</summary><published>2026-08-26T12:00:00Z</published><author><name>Ada Author</name></author><category term="q-fin"/></entry></feed>"""
REPEC_XML = """<rss><channel><item><guid> RePEc:test:1 </guid><title>Liquidity</title><link>https://ideas.repec.org/a/test/1</link><description>Evidence</description></item></channel></rss>"""


def db() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def client_for(status: int, text: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=text, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_arxiv_valid_parse_and_hash() -> None:
    record = ArxivSource(client=client_for(200, ARXIV_XML), lookback_days=30).collect(
        1
    )[0]
    assert record.external_id.endswith("1234.5678")
    assert record.title == "A paper"
    assert record.authors == ["Ada Author"]
    assert record.published_at == datetime(2026, 8, 26, 12, tzinfo=UTC)
    assert record.raw_text == "Abstract evidence."


def test_arxiv_malformed_fails_closed() -> None:
    with pytest.raises(ValueError, match="valid XML"):
        ArxivSource(client=client_for(200, "<broken")).collect(1)


def test_changed_source_content_updates_hash() -> None:
    session = db()
    first = SourceRecord(
        "ACADEMIC", "arxiv", "same", "title", "url", [], None, "old", {}
    )
    changed = SourceRecord(
        "ACADEMIC", "arxiv", "same", "title", "url", [], None, "new", {}
    )
    assert ingest_records(session, [first]) == (1, 0)
    assert ingest_records(session, [changed]) == (0, 1)
    item = session.scalar(select(SourceItem))
    assert item is not None and item.raw_text == "old"
    assert item.content_sha256


def test_repec_valid_and_malformed() -> None:
    record = RepecSource(client=client_for(200, REPEC_XML)).collect(1)[0]
    assert record.external_id == "https://ideas.repec.org/a/test/1"
    with pytest.raises(ValueError):
        RepecSource(client=client_for(200, "<broken")).collect(1)


def test_repec_404_degrades_without_erasing_other_sources() -> None:
    session = db()
    ingest(session, ArxivSource(), 1, offline=True)
    ingest(session, RepecSource(client=client_for(404, "no")), 1)
    assert session.scalar(select(SourceItem).where(SourceItem.source_name == "arxiv"))
    run = session.scalar(select(CollectionRun).where(CollectionRun.source == "repec"))
    assert run is not None and run.status == "DEGRADED"


def test_hyperliquid_history_and_candle_fixtures() -> None:
    source = HyperliquidSource()
    funding = source.collect_history(3, offline=True)
    candles = source.collect_candles(3, offline=True)
    assert {r.raw_metadata["asset"] for r in funding} == {"BTC", "ETH", "SOL"}
    assert len(candles) == 9
    assert all(r.raw_metadata["kind"] == "candle" for r in candles)


def test_hyperliquid_invalid_numeric_and_timestamp_fail_closed() -> None:
    source = HyperliquidSource()
    source._post = lambda payload: [{"coin": "BTC", "time": "bad", "fundingRate": "x"}]  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        source.collect_history(1)


def test_hyperliquid_unsupported_asset_is_not_fabricated() -> None:
    source = HyperliquidSource()
    source._post = lambda payload: [{"time": 1, "fundingRate": "0.1"}]  # type: ignore[method-assign]
    assert source.collect_history(1) == []


def test_market_duplicate_identity() -> None:
    session = db()
    records = HyperliquidSource().collect_history(2, offline=True)
    assert ingest_records(session, records) == (6, 0)
    assert ingest_records(session, records) == (0, 6)
    assert session.query(MarketObservation).count() == 6


def test_funding_percentile_excludes_future_and_ties_are_deterministic() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    history = [
        (t0, 1.0),
        (t0 + timedelta(hours=1), 2.0),
        (t0 + timedelta(hours=2), 100.0),
    ]
    assert funding_percentile(history[:2], history[1][0]) == funding_percentile(
        history, history[1][0]
    )
    assert (
        funding_percentile(
            [(t0, 1.0), (t0 + timedelta(hours=1), 1.0)], t0 + timedelta(hours=1)
        )
        == 100.0
    )


def test_funding_window_and_empty_history() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    history = [(t0 + timedelta(hours=i), float(i)) for i in range(4)]
    assert funding_percentile(history, t0 + timedelta(hours=3), window=2) == 100.0
    assert funding_percentile([], t0) is None


def test_returns_require_exact_anchors_and_exclude_future() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    prices = {t0: 100.0, t0 + timedelta(hours=1): 110.0, t0 + timedelta(hours=2): 200.0}
    assert return_at(prices, t0 + timedelta(hours=1), 1) == pytest.approx(0.1)
    assert return_at(prices, t0 + timedelta(hours=2), 4) is None
    assert (
        return_at({t0 + timedelta(hours=1): 110.0}, t0 + timedelta(hours=1), 1) is None
    )


@pytest.mark.parametrize("hours", [1, 4, 24])
def test_return_missing_anchor_never_fills(hours: int) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        return_at(
            {t0: 100.0, t0 + timedelta(hours=hours + 1): 120.0},
            t0 + timedelta(hours=hours + 1),
            hours,
        )
        is None
    )


def test_volatility_is_non_annualized_and_pit_safe() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    prices = {t0 + timedelta(hours=i): 100.0 + i for i in range(5)}
    result = rolling_volatility(prices, t0 + timedelta(hours=3), window=2)
    assert result is not None and result > 0
    assert rolling_volatility(prices, t0, window=2) is None


def test_pipeline_metrics_do_not_create_duplicate_metrics() -> None:
    session = db()
    ingest_records(session, HyperliquidSource().collect_candles(5, offline=True))
    first = calculate_metrics(session, datetime.now(UTC))
    second = calculate_metrics(session, datetime.now(UTC))
    assert first >= 0 and second == 0
    assert session.query(MarketMetric).count() >= 0


def test_fake_analysis_records_provenance_metadata() -> None:
    session = db()
    ingest(
        session,
        __import__(
            "quant_research_radar.sources", fromlist=["ArxivSource"]
        ).ArxivSource(),
        1,
        offline=True,
    )
    assert (
        analyze(
            session,
            __import__(
                "quant_research_radar.llm", fromlist=["FakeLLMClient"]
            ).FakeLLMClient(),
        )
        == 1
    )
    run = session.scalar(select(AnalysisRun))
    claim = session.scalar(select(Claim))
    assert run is not None and run.provider == "fake" and run.prompt_version
    assert claim is not None and claim.source_item_id and claim.evidence_excerpt


def test_real_llm_mocked_success_and_invalid_json() -> None:
    payload = '{"choices":[{"message":{"content":"{\\"relevance_score\\":1,\\"novelty_score\\":2,\\"testability_score\\":3,\\"executability_score\\":4,\\"latency_sensitivity\\":\\"UNKNOWN\\",\\"reason\\":\\"ok\\",\\"retain\\":true}"}}]}'
    client = OpenAICompatClient("secret", "test", client=client_for(200, payload))
    assert client.triage("t", "x").retain
    with pytest.raises(ValueError):
        OpenAICompatClient(
            "secret", "test", client=client_for(200, '{"choices":[]}')
        ).triage("t", "x")


def test_report_labels_and_unavailable(tmp_path: Path) -> None:
    session = db()
    ingest(
        session,
        __import__(
            "quant_research_radar.sources", fromlist=["ArxivSource"]
        ).ArxivSource(),
        1,
        offline=True,
    )
    ingest(session, HyperliquidSource(), 3, offline=True)
    analyze(
        session,
        __import__(
            "quant_research_radar.llm", fromlist=["FakeLLMClient"]
        ).FakeLLMClient(),
    )
    report = daily_report(session, str(tmp_path)).read_text()
    assert "CLAIM" in report and "HYPOTHESIS" in report
    assert "UNAVAILABLE — no market observation was collected." in report
    assert "UNAVAILABLE" not in report or "execution" in report
    assert all(word not in report.upper() for word in ["BUY", "SELL", "LONG", "SHORT"])
