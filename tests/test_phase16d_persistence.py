from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    ChannelHypothesis,
    EvidenceLink,
    EvidenceSource,
    ResearchWork,
    SourceItem,
    WorkLocation,
)


def test_research_work_versions_link_to_one_canonical_work() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    source = EvidenceSource(
        source_name="openalex",
        source_class="ACADEMIC",
        access_mode="METADATA_ONLY",
        reliability_prior="INDEX_METADATA",
        adapter_status="READY",
    )
    work = ResearchWork(
        canonical_identity="doi:10.1000/funding",
        normalized_title="funding and subsequent returns",
    )
    session.add_all([source, work])
    session.flush()
    first = SourceItem(
        source_type="ACADEMIC",
        source_name="openalex",
        external_id="openalex:W1",
        canonical_url="https://doi.org/10.1000/funding",
        title="Funding and subsequent returns",
        authors=["A Researcher"],
        published_at=datetime(2026, 8, 1, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 2, tzinfo=UTC),
        raw_text="abstract",
        raw_metadata={},
        content_sha256="a" * 64,
    )
    second = SourceItem(
        source_type="PREPRINT",
        source_name="arxiv",
        external_id="arxiv:2608.1",
        canonical_url="https://arxiv.org/abs/2608.1",
        title="Funding and subsequent returns",
        authors=["A Researcher"],
        published_at=datetime(2026, 7, 1, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 2, tzinfo=UTC),
        raw_text="preprint abstract",
        raw_metadata={},
        content_sha256="b" * 64,
    )
    session.add_all([first, second])
    session.flush()
    session.add_all(
        [
            WorkLocation(
                work_id=work.id,
                source_item_id=first.id,
                source_id=source.id,
                access_mode="METADATA_ONLY",
            ),
            WorkLocation(
                work_id=work.id,
                source_item_id=second.id,
                source_id=source.id,
                access_mode="OA_PREPRINT",
            ),
        ]
    )
    session.commit()

    assert len(session.scalars(select(WorkLocation)).all()) == 2
    assert len(session.scalars(select(ResearchWork)).all()) == 1


def test_evidence_links_retain_origin_and_challenge_without_merging_channels() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    item = SourceItem(
        source_type="MARKET",
        source_name="hyperliquid",
        external_id="market:sol:1",
        canonical_url=None,
        title="SOL funding observation",
        authors=[],
        published_at=datetime(2026, 8, 1, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
        raw_text="funding percentile=100",
        raw_metadata={},
        content_sha256="c" * 64,
    )
    hypothesis = ChannelHypothesis(
        channel="MARKET",
        statement="Extreme SOL funding changes subsequent return distribution.",
        condition="SOL funding percentile >= 90",
        outcome="subsequent return distribution",
        universe="SOL perpetual",
        horizon="4h and 24h",
        required_data=["funding", "candles"],
        falsification_criterion="No conditional difference from baseline.",
        maturity="H1_STATISTICAL_HYPOTHESIS",
        as_of=datetime(2026, 8, 1, tzinfo=UTC),
    )
    session.add_all([item, hypothesis])
    session.flush()
    session.add_all(
        [
            EvidenceLink(
                channel_hypothesis_id=hypothesis.id,
                source_item_id=item.id,
                relation="ORIGIN",
                channel="MARKET",
                independence_key="hyperliquid:sol:1",
            ),
            EvidenceLink(
                channel_hypothesis_id=hypothesis.id,
                source_item_id=item.id,
                relation="CHALLENGE",
                channel="MARKET",
                independence_key="hyperliquid:sol:1",
            ),
        ]
    )
    session.commit()

    links = session.scalars(select(EvidenceLink)).all()
    assert {link.relation for link in links} == {"ORIGIN", "CHALLENGE"}
    assert {link.channel for link in links} == {"MARKET"}
