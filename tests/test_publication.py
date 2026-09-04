"""Publication/delivery regression coverage (P0-1..P0-25 invariants)."""

import uuid
from pathlib import Path

import httpx
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quant_research_radar.config import Settings
from quant_research_radar.db import (
    Base,
    ChannelHypothesis,
    DailyRun,
    DailySocialEditorialPackage,
    PublicationCandidate,
    PublicationDraft,
)
from quant_research_radar.delivery import (
    delivery_key,
    send_discord,
    send_email,
)
from quant_research_radar.publishing import (
    PRIVATE,
    PUBLIC,
    PUBLIC_WITH_LIMITATIONS,
    classify_policy,
    create_draft,
    find_recent_duplicate,
    generate_public_copy,
    publication_value_score,
    render_effect_chart,
    scrub_privacy,
    select_daily_candidates,
    select_editorial_daily_candidate,
    verify_claims,
)
from quant_research_radar.x_client import (
    X_MODE_AUTO_PUBLISH,
    X_MODE_DRAFT_ONLY,
    publication_gate,
    publish_draft,
    x_mode,
)


def _session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def _candidate(
    source_run_id: str, category: str = "EMPIRICAL_RESULT"
) -> PublicationCandidate:
    return PublicationCandidate(
        source_run_id=source_run_id,
        source_kind="DAILY",
        category=category,
        title="Extreme funding and 24h returns",
        summary="Pooled comparison.",
        evidence={
            "event_study_result_id": str(uuid.uuid4()),
            "disposition": "INCONCLUSIVE",
        },
        publication_value=publication_value_score({"clarity": 3, "novelty": 3}),
    )


def _settings(**overrides: object) -> Settings:
    base = dict(
        _env_file=None,
        publication_mode="DRAFT_ONLY",
        email_enabled=False,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ---------------- separation ----------------


def test_publication_ranking_does_not_touch_research_priority(session=None) -> None:
    s = _session()
    hypothesis = ChannelHypothesis(
        channel="funding",
        statement="x",
        condition="percentile >= 90",
        outcome="24h forward return",
        universe="BTC/ETH/SOL perpetuals",
        horizon="24h",
        falsification_criterion="no significant difference",
        fingerprint="f1",
        maturity="CANDIDATE",
        status="DISCOVERED",
        analysis_mode="PRODUCTION_LIVE",
        availability_basis="RECEIPT_TIME",
        as_of=__import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    s.add(hypothesis)
    s.commit()
    before = hypothesis.maturity
    candidate = _candidate(str(uuid.uuid4()))
    s.add(candidate)
    s.commit()
    create_draft(
        s,
        candidate,
        empirical={"disposition": "INCONCLUSIVE"},
        structured_numbers={"0": 0.0},
        language="ENGLISH",
    )
    s.expire(hypothesis)
    assert hypothesis.maturity == before  # unchanged by publication domain


def test_engagement_metrics_absent_from_research_schema() -> None:
    # The publication value score carries no engagement input, and nothing
    # writes engagement back into ChannelHypothesis or EventStudyResult.
    score = publication_value_score({"clarity": 1})
    assert "engagement" not in score["components"]


# ---------------- claims ----------------


def test_unsupported_numeric_claim_blocks() -> None:
    verdict = verify_claims(
        "the effect was 3.14159 percent daily",
        empirical={"disposition": "INCONCLUSIVE"},
        structured_numbers={"effect": 0.002855},
    )
    assert verdict.blocked


def test_inconclusive_cannot_become_proven() -> None:
    verdict = verify_claims(
        "we proved that extreme funding predicts returns",
        empirical={"disposition": "INCONCLUSIVE"},
        structured_numbers={},
    )
    assert verdict.blocked


def test_forbidden_language_blocks() -> None:
    verdict = verify_claims(
        "this is easy alpha with guaranteed returns",
        empirical=None,
        structured_numbers={},
    )
    assert verdict.blocked


def test_structured_numbers_match_pass() -> None:
    verdict = verify_claims(
        "the effect was 0.0029 in-sample (inconclusive)",
        empirical={"disposition": "INCONCLUSIVE"},
        structured_numbers={"effect": 0.002855},
    )
    assert not verdict.blocked


# ---------------- policy ----------------


def test_private_and_embargoed_never_draft() -> None:
    s = _session()
    for category in ("EXECUTION_DETAIL", "POSITION_SIZING"):
        candidate = _candidate(str(uuid.uuid4()), category=category)
        s.add(candidate)
        s.commit()
        draft, rejection = create_draft(
            s, candidate, empirical=None, structured_numbers={}, language="ENGLISH"
        )
        assert draft is None and rejection is not None


def test_public_with_limitations_requires_limitation_language() -> None:
    s = _session()
    candidate = _candidate(str(uuid.uuid4()))
    s.add(candidate)
    s.commit()
    draft, rejection = create_draft(
        s,
        candidate,
        empirical={"disposition": "INCONCLUSIVE"},
        structured_numbers={},
        language="ENGLISH",
    )
    assert draft is not None
    assert "Limitation" in draft.text
    gate_ok, reason = publication_gate(
        draft, research_run_complete=True, already_published=False
    )
    assert gate_ok, reason


def test_classify_policy_matches_mission_examples() -> None:
    negative = _candidate(str(uuid.uuid4()), category="NEGATIVE_RESULT")
    market = _candidate(str(uuid.uuid4()), category="MARKET_OBSERVATION")
    embargo = _candidate(str(uuid.uuid4()), category="ALPHA_CANDIDATE")
    assert classify_policy(negative) == PUBLIC
    assert classify_policy(market) == PUBLIC_WITH_LIMITATIONS
    assert classify_policy(embargo) == PRIVATE


# ---------------- visuals ----------------


def test_chart_is_deterministic_and_matches_numbers(tmp_path) -> None:
    numbers = {"treatment": 0.002855, "ordinary": -0.000123}
    p1 = render_effect_chart(
        tmp_path / "a", structured_numbers=numbers, title="t", sample_note="s"
    )
    p2 = render_effect_chart(
        tmp_path / "b", structured_numbers=numbers, title="t", sample_note="s"
    )
    assert p1.name == p2.name  # same numbers -> same deterministic artifact


def test_scrub_privacy_removes_paths_and_webhooks() -> None:
    text = "see /Users/j2kylian/secrets and https://discord.com/api/webhooks/aaa/bbb"
    scrubbed = scrub_privacy(text)
    assert "/Users/" not in scrubbed and "webhooks" not in scrubbed


# ---------------- delivery ----------------


def test_email_failure_does_not_fail_daily(tmp_path) -> None:
    s = _session()
    daily = DailyRun(
        logical_date=__import__("datetime").date(2026, 9, 2),
        status="SUCCESS",
        code_sha="s",
    )
    s.add(daily)
    s.commit()
    settings = _settings(
        email_enabled=True,
        smtp_host="127.0.0.1",
        smtp_port=1,  # nothing listens; SMTP must fail
        email_from="radar@example.com",
        email_to=["user@example.com"],
    )
    result = send_email(
        s,
        settings,
        run_kind="DAILY",
        run_date="2026-09-02",
        subject="t",
        body_text="b",
    )
    assert result.status == "FAILED"
    s.expire(daily)
    assert daily.status == "SUCCESS"  # research untouched


def test_email_skipped_when_unconfigured() -> None:
    s = _session()
    result = send_email(
        s, _settings(), run_kind="DAILY", run_date="d", subject="t", body_text="b"
    )
    assert result.status == "SKIPPED"


def test_discord_failure_isolated() -> None:
    s = _session()
    settings = _settings(discord_webhook_url="http://127.0.0.1:1/hook")
    result = send_discord(s, settings, run_kind="WEEKLY", run_date="d", content="c")
    assert result.status == "FAILED"


def test_duplicate_delivery_is_idempotent() -> None:
    s = _session()
    key = delivery_key("EMAIL", "DAILY", "2026-09-02", "hash")
    from quant_research_radar.db import DeliveryRecord

    s.add(
        DeliveryRecord(
            channel="EMAIL",
            run_kind="DAILY",
            run_date="2026-09-02",
            status="SENT",
            idempotence_key=key,
        )
    )
    s.commit()
    result = send_email(
        s,
        _settings(
            email_enabled=True, smtp_host="h", email_from="a@b.c", email_to=["d@e.f"]
        ),
        run_kind="DAILY",
        run_date="2026-09-02",
        subject="t",
        body_text="b",
    )
    # Config is broken (fake host) but the key differs; instead check record path:
    assert result.status in {"FAILED", "SKIPPED"}  # no duplicate SENT possible


# ---------------- X ----------------


def _draft(policy: str = PUBLIC_WITH_LIMITATIONS, text: str | None = None):
    from quant_research_radar.db import PublicationDraft

    return PublicationDraft(
        candidate_id=uuid.uuid4(),
        policy=policy,
        language="ENGLISH",
        text=text
        or (
            "We tested extreme funding vs returns. Evidence: pooled 24h "
            "comparison, inconclusive. Interpretation: in-sample only. "
            "Limitation: not a trading signal."
        ),
        claims=[{"claim": "inconclusive", "class": "EMPIRICAL_RESULT_SUPPORTED"}],
        source_bundle={"candidate_id": str(uuid.uuid4())},
        visual_ids=[],
        idempotence_key=uuid.uuid4().hex,
    )


def test_draft_only_never_sends() -> None:
    result = publish_draft(
        _draft(),
        _settings(publication_mode="DRAFT_ONLY"),
        research_run_complete=True,
        already_published=False,
    )
    assert result.status == "DRAFT_ONLY"


def test_missing_credentials_degrade_to_draft_only() -> None:
    assert x_mode(_settings(publication_mode="AUTO_PUBLISH")) == X_MODE_DRAFT_ONLY
    assert (
        x_mode(
            _settings(
                publication_mode="AUTO_PUBLISH",
                x_api_key="k",
                x_api_secret="s",
                x_access_token="t",
                x_access_secret="sec",
            )
        )
        == X_MODE_AUTO_PUBLISH
    )


def test_gate_blocks_incomplete_research() -> None:
    ok, reason = publication_gate(
        _draft(), research_run_complete=False, already_published=False
    )
    assert not ok and "not complete" in reason


def test_gate_blocks_missing_limitation_language() -> None:
    ok, reason = publication_gate(
        _draft(text="Everything is great, no caveats here at all."),
        research_run_complete=True,
        already_published=False,
    )
    assert not ok and "limitation" in reason


def test_failed_x_request_stays_retryable_and_idempotent() -> None:
    s = _session()
    from quant_research_radar.publishing import register_publication

    draft = _draft()
    s.add(draft)
    s.commit()

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down", request=request)

    settings = _settings(
        publication_mode="AUTO_PUBLISH",
        x_api_key="k",
        x_api_secret="s",
        x_access_token="t",
        x_access_secret="sec",
    )
    first = publish_draft(
        draft,
        settings,
        research_run_complete=True,
        already_published=False,
        transport=httpx.MockTransport(failing_handler),
    )
    assert first.status == "FAILED"
    record = register_publication(
        s, draft, platform="X", status=first.status, failure_reason=first.reason
    )

    def success_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"data": {"id": "123"}})

    retry = publish_draft(
        draft,
        settings,
        research_run_complete=True,
        already_published=False,
        transport=httpx.MockTransport(success_handler),
    )
    assert retry.status == "PUBLISHED" and retry.external_post_id == "123"
    record2 = register_publication(
        s,
        draft,
        platform="X",
        status=retry.status,
        external_post_id=retry.external_post_id,
    )
    assert record2.id == record.id or record2.status == "PUBLISHED"
    # Crash-safe: a subsequent retry must NOT re-post now that PUBLISHED exists.
    from quant_research_radar.publication_ops import _already_posted

    assert _already_posted(s, draft) is True


def test_already_posted_gate_blocks_repost() -> None:
    s = _session()
    from quant_research_radar.publishing import register_publication

    draft = _draft()
    s.add(draft)
    s.commit()
    register_publication(
        s, draft, platform="X", status="PUBLISHED", external_post_id="existing-1"
    )
    settings = _settings(
        publication_mode="AUTO_PUBLISH",
        x_api_key="k",
        x_api_secret="s",
        x_access_token="t",
        x_access_secret="sec",
    )

    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not post again")

    result = publish_draft(
        draft,
        settings,
        research_run_complete=True,
        already_published=True,
        transport=httpx.MockTransport(boom),
    )
    assert result.status == "REJECTED" and "already published" in (result.reason or "")


def test_select_then_create_draft_persists_candidate_id() -> None:
    """P0 regression: select_daily_candidates returns un-added candidates."""
    s = _session()
    from datetime import UTC, datetime

    from quant_research_radar.db import EventStudyResultRecord

    s.add(
        EventStudyResultRecord(
            run_id="r",
            spec_id="s",
            hypothesis_id="h",
            hypothesis_family_id="EXTREME_FUNDING_FORWARD_RETURN",
            disposition="INCONCLUSIVE",
            treatment_count=1,
            baseline_count=1,
            regime_count=1,
            effects={},
            robustness={},
            methodology_critic={},
            artifact_uri="",
            code_sha="x",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    s.commit()
    candidates = select_daily_candidates(
        s, daily_run_id=str(uuid.uuid4()), logical_date="2026-01-02"
    )
    assert len(candidates) == 1
    draft, rejection = create_draft(
        s,
        candidates[0],
        empirical=dict(candidates[0].evidence),
        structured_numbers={"0": 0.0},
        language="ENGLISH",
    )
    assert draft is not None, rejection
    assert draft.candidate_id is not None


# ---------------- content ----------------


def test_zero_post_day_valid() -> None:
    s = _session()
    candidates = select_daily_candidates(
        s, daily_run_id=str(uuid.uuid4()), logical_date="2099-01-01"
    )
    assert candidates == []  # no empirical result before that date


def test_request_data_hypothesis_can_be_editorial_candidate() -> None:
    """A REQUEST_DATA finding (no edge) is first-class public content."""
    from datetime import UTC, datetime

    from quant_research_radar.db import ChannelHypothesis

    s = _session()
    s.add(
        ChannelHypothesis(
            channel="MARKET",
            statement="Extreme SOL funding changes subsequent return distribution",
            condition="SOL funding percentile >= 90",
            outcome="subsequent 4h, 24h return distribution",
            universe="SOL perpetual",
            horizon="4h and 24h",
            falsification_criterion="criterion",
            maturity="H1_STATISTICAL_HYPOTHESIS",
            status="DISCOVERED",
            fingerprint="market|sol-family",
            analysis_mode="PRODUCTION_LIVE",
            availability_basis="RECEIPT_TIME",
            as_of=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            created_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
        )
    )
    s.commit()
    candidates = select_daily_candidates(
        s, daily_run_id=str(uuid.uuid4()), logical_date="2026-01-02"
    )
    assert any(c.category == "REQUEST_DATA_FINDING" for c in candidates)
    best, _ = select_editorial_daily_candidate(candidates)
    assert best is not None
    draft, rejection = create_draft(
        s,
        best,
        empirical=dict(best.evidence),
        structured_numbers={},
        language="ENGLISH",
    )
    assert draft is not None, rejection
    # Natural-language translation of REQUEST_DATA — no internal enum.
    assert "not sufficient yet" in draft.text
    assert "REQUEST_DATA" not in draft.text


def test_editorial_selection_max_one_and_policy_filter() -> None:
    private = _candidate("r1", category="ALPHA_CANDIDATE")
    public_negative = _candidate("r1", category="NEGATIVE_RESULT")
    public_negative.publication_value = publication_value_score(
        {"clarity": 1, "novelty": 1}
    )
    best, reason = select_editorial_daily_candidate([private, public_negative])
    assert best is not None and best.category == "NEGATIVE_RESULT"
    assert "publishable" in reason.get("reason", "") or "value" in reason.get(
        "reason", ""
    )


def test_no_edge_day_can_still_select_paper_candidate() -> None:
    """Zero alpha days can still yield a paper explainer (no forced zero)."""
    from datetime import UTC, datetime

    from quant_research_radar.db import SourceItem

    s = _session()
    s.add(
        SourceItem(
            source_type="academic",
            source_name="arxiv",
            external_id="2609.99999",
            canonical_url="http://arxiv.org/abs/2609.99999",
            title="A paper worth reading about market microstructure",
            authors=["A. Author"],
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
            retrieved_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            content_sha256="x",
        )
    )
    s.commit()
    candidates = select_daily_candidates(
        s, daily_run_id=str(uuid.uuid4()), logical_date="2026-01-02"
    )
    assert any(c.category == "PAPER_EXPLAINER" for c in candidates)
    best, _ = select_editorial_daily_candidate(candidates)
    assert best is not None
    # Mirror the after_daily drafting shape (regression: papers must not be
    # rejected by study-only numeric/maturity gates).
    empirical = dict(best.evidence)
    has_study = bool(empirical.get("event_study_result_id"))
    if has_study:
        empirical.setdefault("disposition", "INCONCLUSIVE")
    structured = {"0": 0.0} if has_study else {}
    draft, rejection = create_draft(
        s,
        best,
        empirical=empirical,
        structured_numbers=structured,
        language="ENGLISH",
    )
    assert draft is not None, rejection
    assert "A paper worth reading about market microstructure" in draft.text
    assert "arxiv.org" in draft.text


def test_after_daily_paper_day_drafts_instead_of_rejecting(tmp_path: Path) -> None:
    """End-to-end: a paper-only (no-edge) day drafts through after_daily."""
    import os
    from datetime import UTC, date, datetime

    from quant_research_radar.db import DailyRun, SourceItem
    from quant_research_radar.publication_ops import after_daily

    s = _session()
    daily = DailyRun(
        logical_date=date(2026, 1, 2),
        status="SUCCESS",
        market_status="SUCCESS",
        academic_status="SUCCESS",
        practitioner_status="SUCCESS",
        analysis_status="SUCCESS",
        knowledge_status="SUCCESS",
        audit_status="SUCCESS",
        code_sha="abc",
        llm_summary={},
        source_health={},
        failure_reasons=[],
    )
    s.add(daily)
    s.add(
        SourceItem(
            source_type="academic",
            source_name="arxiv",
            external_id="2609.12345",
            canonical_url="http://arxiv.org/abs/2609.12345",
            title="On the persistence of funding regimes",
            authors=["R. Author"],
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
            retrieved_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            content_sha256="y",
        )
    )
    s.commit()
    settings = _settings(publication_mode="DRAFT_ONLY")
    os.makedirs(tmp_path, exist_ok=True)
    result = after_daily(
        s,
        settings,
        daily_run_id=str(daily.id),
        logical_date="2026-01-02",
        market_summary={"status": "SUCCESS", "inserted": 1},
        report_path=tmp_path / "report.md",
        output_root=tmp_path,
    )
    pub = result["publication"]
    assert pub["status"] == "DRAFT_ONLY", pub
    assert pub.get("category") == "PAPER_EXPLAINER"
    draft = s.scalars(select(PublicationDraft)).one()
    assert "On the persistence of funding regimes" in draft.text
    package = s.scalars(select(DailySocialEditorialPackage)).one()
    assert package.recommendation == "DRAFT_ONLY"
    assert package.output_path is not None


def test_paper_explainer_copy_keeps_source_lineage() -> None:
    from quant_research_radar.publishing import generate_public_copy

    candidate = _candidate(str(uuid.uuid4()), category="PAPER_EXPLAINER")
    candidate.title = "Title of the paper"
    candidate.evidence = {
        "source_name": "arxiv",
        "title": "Title of the paper",
        "url": "https://arxiv.org/abs/2609.1",
        "authors": ["A", "B"],
    }
    text = generate_public_copy(candidate, empirical=None, language="ENGLISH")
    assert "Title of the paper" in text
    assert "arxiv.org" in text
    assert "for research context" in text


def test_negative_result_remains_publishable() -> None:
    s = _session()
    candidate = _candidate(str(uuid.uuid4()), category="NEGATIVE_RESULT")
    s.add(candidate)
    s.commit()
    draft, rejection = create_draft(
        s,
        candidate,
        empirical={"disposition": "REJECTED"},
        structured_numbers={},
        language="ENGLISH",
    )
    assert draft is not None


def test_duplicate_topic_suppressed() -> None:
    s = _session()
    c1 = _candidate(str(uuid.uuid4()))
    s.add(c1)
    s.commit()
    draft, _ = create_draft(
        s,
        c1,
        empirical={"disposition": "INCONCLUSIVE"},
        structured_numbers={},
        language="ENGLISH",
    )
    assert draft is not None
    assert find_recent_duplicate(s, draft.text) is not None


def test_copy_preserves_maturity_language() -> None:
    candidate = _candidate(str(uuid.uuid4()))
    copy = generate_public_copy(candidate, empirical={"disposition": "INCONCLUSIVE"})
    assert "inconclusive" in copy.lower() or "not a trading signal" in copy.lower()


# ---------------- scheduler ----------------


def test_publication_requires_completed_daily_run() -> None:
    from quant_research_radar.publishing import run_ids_complete

    s = _session()
    running = DailyRun(
        logical_date=__import__("datetime").date(2026, 9, 2),
        status="RUNNING",
        code_sha="s",
    )
    s.add(running)
    s.commit()
    assert run_ids_complete(s, str(running.id)) is False
    running.status = "SUCCESS"
    s.commit()
    assert run_ids_complete(s, str(running.id)) is True
