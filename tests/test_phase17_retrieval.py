from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from quant_research_radar.db import (
    Base,
    ChannelHypothesis,
    CollectionRun,
    EvidenceLink,
    RawArtifact,
    RawArtifactReceipt,
    SourceItem,
    UnifiedHypothesisMember,
    UnifiedHypothesisRecord,
)
from quant_research_radar.retrieval import (
    expand_query,
    hypothesis_lineage,
    search_hypotheses,
)


def _hypothesis(
    session: Session, asset: str, at: datetime, mode: str = "PRODUCTION_LIVE"
) -> ChannelHypothesis:
    item = SourceItem(
        source_type="ACADEMIC",
        source_name="openalex",
        external_id=f"{asset}:{at.isoformat()}:{mode}",
        canonical_url="https://example.test/work",
        title=f"{asset} funding",
        authors=[],
        published_at=at,
        retrieved_at=at,
        raw_text="funding",
        raw_metadata={},
        content_sha256="a" * 64,
    )
    session.add(item)
    session.flush()
    hypothesis = ChannelHypothesis(
        channel="ACADEMIC",
        statement=f"{asset} extreme funding predicts returns",
        condition=f"{asset} funding",
        outcome="returns",
        universe=asset,
        horizon="24h",
        expected_direction=None,
        required_data=[],
        falsification_criterion="no effect",
        maturity="H1_STATISTICAL_HYPOTHESIS",
        fingerprint="funding-template",
        analysis_mode=mode,
        availability_basis="RECEIPT_TIME",
        as_of=at,
    )
    session.add(hypothesis)
    session.flush()
    return hypothesis


def test_retrieval_scope_as_of_and_exact_archived_lineage() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    at = datetime(2026, 8, 30, tzinfo=UTC)
    first = _hypothesis(session, "SOL", at)
    second = _hypothesis(session, "ETH", at + timedelta(days=1))
    replay = _hypothesis(session, "BTC", at, "ACCELERATED_RECONSTRUCTIVE_REPLAY")
    family = UnifiedHypothesisRecord(
        fingerprint="funding-template",
        statement="asset extreme funding predicts returns",
        maturity="H1_STATISTICAL_HYPOTHESIS",
        supporting_channels=["ACADEMIC"],
        independent_evidence_count=1,
    )
    run = CollectionRun(
        source="phase16d-discover", diagnostics={"retrieval_scope": {"limit": 1}}
    )
    artifact = RawArtifact(
        content_sha256="b" * 64,
        media_type="application/json",
        byte_size=2,
        storage_uri="data/raw/objects/bb/" + "b" * 64,
    )
    session.add_all([family, run, artifact])
    session.flush()
    receipt = RawArtifactReceipt(
        raw_artifact_id=artifact.id,
        provider="openalex",
        canonical_url="https://example.test/work",
        source_native_timestamp=at,
        retrieved_at=at,
        source_item_id=first.id,
        collection_run_id=run.id,
    )
    session.add(receipt)
    session.flush()
    session.add_all(
        [
            UnifiedHypothesisMember(
                unified_hypothesis_id=family.id, channel_hypothesis_id=first.id
            ),
            UnifiedHypothesisMember(
                unified_hypothesis_id=family.id, channel_hypothesis_id=second.id
            ),
            EvidenceLink(
                channel_hypothesis_id=first.id,
                source_item_id=session.scalar(
                    select(SourceItem.id).where(
                        SourceItem.external_id
                        == f"SOL:{at.isoformat()}:PRODUCTION_LIVE"
                    )
                ),
                relation="ORIGIN",
                channel="ACADEMIC",
                independence_key="doi:x",
                raw_artifact_receipt_id=receipt.id,
            ),
        ]
    )
    session.commit()

    assert [row["entity_id"] for row in search_hypotheses(session, "funding")] == [
        str(second.id),
        str(first.id),
    ]
    assert search_hypotheses(session, "funding", scope="REPLAY")[0]["entity_id"] == str(
        replay.id
    )
    assert [
        row["entity_id"] for row in search_hypotheses(session, "funding", as_of=at)
    ] == [str(first.id)]
    lineage = hypothesis_lineage(session, str(first.id))
    assert len(lineage["occurrences"]) == 2
    assert lineage["evidence"][0]["sha256"] == "b" * 64
    assert lineage["evidence"][0]["collection_run_id"] == str(run.id)


def test_retrieval_expands_domain_paraphrases_and_enforces_result_bound() -> None:
    assert expand_query("basis trade") == [
        "basis trade",
        "carry",
        "futures basis",
        "funding",
    ]
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    at = datetime(2026, 8, 30, tzinfo=UTC)
    for offset, asset in enumerate(("SOL", "ETH", "BTC")):
        _hypothesis(session, asset, at + timedelta(hours=offset))
    session.commit()

    assert len(search_hypotheses(session, "crowded perp longs", limit=2)) == 2
    with pytest.raises(ValueError, match="result limit"):
        search_hypotheses(session, "funding", limit=0)


def test_occurrences_keep_same_family_and_distinct_as_of_rows() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    at = datetime(2026, 8, 30, tzinfo=UTC)
    first = _hypothesis(session, "SOL", at)
    second = _hypothesis(session, "SOL", at + timedelta(days=1))
    family = UnifiedHypothesisRecord(
        fingerprint="funding-template",
        statement="SOL extreme funding predicts returns",
        maturity="H1_STATISTICAL_HYPOTHESIS",
        supporting_channels=["ACADEMIC"],
        independent_evidence_count=1,
    )
    session.add(family)
    session.flush()
    session.add_all(
        [
            UnifiedHypothesisMember(
                unified_hypothesis_id=family.id, channel_hypothesis_id=first.id
            ),
            UnifiedHypothesisMember(
                unified_hypothesis_id=family.id, channel_hypothesis_id=second.id
            ),
        ]
    )
    session.commit()
    history = hypothesis_lineage(session, str(first.id))
    assert [entry["as_of"] for entry in history["occurrences"]] == [
        at.isoformat(),
        (at + timedelta(days=1)).isoformat(),
    ]


def test_lineage_scope_and_as_of_hide_cross_scope_and_future_members() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    at = datetime(2026, 8, 30, tzinfo=UTC)
    first = _hypothesis(session, "SOL", at)
    future = _hypothesis(session, "SOL", at + timedelta(days=1))
    replay = _hypothesis(session, "SOL", at, "ACCELERATED_RECONSTRUCTIVE_REPLAY")
    family = UnifiedHypothesisRecord(
        fingerprint="funding-template",
        statement="SOL funding predicts returns",
        maturity="H1_STATISTICAL_HYPOTHESIS",
        supporting_channels=["ACADEMIC"],
        independent_evidence_count=1,
    )
    session.add(family)
    session.flush()
    session.add_all(
        [
            UnifiedHypothesisMember(
                unified_hypothesis_id=family.id, channel_hypothesis_id=row.id
            )
            for row in (first, future, replay)
        ]
    )
    session.commit()

    lineage = hypothesis_lineage(session, str(first.id), scope="PRODUCTION", as_of=at)

    assert [row["id"] for row in lineage["occurrences"]] == [str(first.id)]


def test_occurrence_identity_rejects_missing_as_of() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(
        ChannelHypothesis(
            channel="ACADEMIC",
            statement="x",
            condition="x",
            outcome="x",
            universe="x",
            horizon="1h",
            expected_direction=None,
            required_data=[],
            falsification_criterion="x",
            maturity="H1_STATISTICAL_HYPOTHESIS",
            fingerprint="x",
            analysis_mode="PRODUCTION_LIVE",
            availability_basis="RECEIPT_TIME",
            as_of=None,
        )
    )
    with __import__("pytest").raises(IntegrityError):
        session.commit()


def test_occurrence_identity_is_enforced_by_database() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    at = datetime(2026, 8, 30, tzinfo=UTC)
    _hypothesis(session, "SOL", at)
    session.add(
        ChannelHypothesis(
            channel="ACADEMIC",
            statement="SOL extreme funding predicts returns",
            condition="SOL funding",
            outcome="returns",
            universe="SOL",
            horizon="24h",
            expected_direction=None,
            required_data=[],
            falsification_criterion="no effect",
            maturity="H1_STATISTICAL_HYPOTHESIS",
            fingerprint="funding-template",
            analysis_mode="PRODUCTION_LIVE",
            availability_basis="RECEIPT_TIME",
            as_of=at,
        )
    )
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
    else:
        raise AssertionError("duplicate occurrence identity was accepted")


def test_retrieval_rejects_empty_or_oversized_query() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    for query in ("", "x" * 501):
        try:
            search_hypotheses(session, query)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid lexical query was accepted")
    try:
        search_hypotheses(session, "funding", scope="INVALID")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("invalid scope was accepted")
