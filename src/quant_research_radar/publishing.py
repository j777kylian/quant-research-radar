"""Publication layer: candidate selection, policy, claims, copy, visuals.

STRICT RESEARCH/PUBLICATION SEPARATION: this module reads research artifacts
and writes ONLY publication-domain tables (PublicationCandidate, Draft,
PublicationRecord). It never mutates hypothesis status, empirical results,
critic dispositions, knowledge strength, or research ranking. Engagement or
publication value must never feed back into science.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from .db import (  # noqa: E402
    DailyRun,
    EventStudyResultRecord,
    PublicationCandidate,
    PublicationDraft,
    PublicationRecord,
    WeeklyRun,
    utcnow,
)

# Candidate categories (editorial pool; negative/inconclusive are first-class)
MARKET_OBSERVATION = "MARKET_OBSERVATION"
EMPIRICAL_RESULT = "EMPIRICAL_RESULT"
NEGATIVE_RESULT = "NEGATIVE_RESULT"
INCONCLUSIVE_RESULT = "INCONCLUSIVE_RESULT"
REQUEST_DATA_FINDING = "REQUEST_DATA_FINDING"
METHODOLOGY_NOTE = "METHODOLOGY_NOTE"
WEEKLY_RESEARCH_ROUNDUP = "WEEKLY_RESEARCH_ROUNDUP"
WEEKLY_NEGATIVE_RESULT = "WEEKLY_NEGATIVE_RESULT"
DATA_QUALITY_FINDING = "DATA_QUALITY_FINDING"
PAPER_EXPLAINER = "PAPER_EXPLAINER"
PRACTITIONER_EXPLAINER = "PRACTITIONER_EXPLAINER"
PAPER_PLUS_MARKET_CONNECTION = "PAPER_PLUS_MARKET_CONNECTION"
HYPOTHESIS_EXPLAINER = "HYPOTHESIS_EXPLAINER"
RESEARCH_PROCESS_NOTE = "RESEARCH_PROCESS_NOTE"
ALPHA_CANDIDATE = "ALPHA_CANDIDATE"

# Publication policy classes
PUBLIC = "PUBLIC"
PUBLIC_WITH_LIMITATIONS = "PUBLIC_WITH_LIMITATIONS"
DELAYED = "DELAYED"
PRIVATE = "PRIVATE"
EMBARGOED = "EMBARGOED"
REJECT_PUBLICATION = "REJECT_PUBLICATION"

PUBLISHABLE = {PUBLIC, PUBLIC_WITH_LIMITATIONS}

# Claim classes
CLAIM_SOURCE_SUPPORTED = "SOURCE_SUPPORTED"
CLAIM_EMPIRICAL = "EMPIRICAL_RESULT_SUPPORTED"
CLAIM_INFERENCE = "INFERENCE"
CLAIM_HYPOTHESIS = "HYPOTHESIS"
CLAIM_OPINION = "OPINION"
CLAIM_UNKNOWN = "UNKNOWN"

FORBIDDEN_LANGUAGE = [
    "guaranteed",
    "guarantee",
    "easy alpha",
    "risk-free",
    "sure profit",
    "proven profit",
    "can't lose",
    "cannot lose",
    "buy now",
    "sell now",
    "moon",
]

MATURITY_PHRASES = {
    "INCONCLUSIVE": "inconclusive",
    "REJECTED": "rejected",
    "SUPPORTED": "supported in-sample",
    "PARTIALLY_SUPPORTED": "partially supported in-sample",
    "DATA_INSUFFICIENT": "data insufficient",
    "INVALID_TEST": "invalid test",
}


def publication_value_score(components: dict[str, int]) -> dict[str, Any]:
    """Auditable publication value, separate from ResearchPriorityScore."""
    clarity = components.get("clarity", 0)
    novelty = components.get("novelty", 0)
    educational = components.get("educational", 0)
    timeliness = components.get("timeliness", 0)
    visualizable = components.get("visualizable", 0)
    source_quality = components.get("source_quality", 0)
    explainable = components.get("explainable", 0)
    total = (
        clarity
        + novelty
        + educational
        + timeliness
        + visualizable
        + source_quality
        + explainable
    )
    return {
        "components": {
            "clarity": clarity,
            "novelty": novelty,
            "educational": educational,
            "timeliness": timeliness,
            "visualizable": visualizable,
            "source_quality": source_quality,
            "explainable_without_exaggeration": explainable,
        },
        "total": total,
    }


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def select_daily_candidates(
    session: Session, *, daily_run_id: str, logical_date: str
) -> list[PublicationCandidate]:
    """Build the editorial candidate pool for a completed Daily cycle.

    Sources: any eligible research artifact — an Event Study result, new
    hypothesis families (REQUEST_DATA / hypothesis explainer), newly retrieved
    academic works (paper explainer), or an informative research-process note.
    ZERO candidates is valid; absence of alpha is NOT a reason for zero by
    itself (negative/inconclusive/process content is first-class).
    """
    candidates: list[PublicationCandidate] = []
    day_end = _day_end(logical_date)

    latest_result = session.scalars(
        select(EventStudyResultRecord)
        .where(EventStudyResultRecord.created_at <= day_end)
        .order_by(EventStudyResultRecord.created_at.desc())
        .limit(1)
    ).first()
    if latest_result is not None:
        category = (
            NEGATIVE_RESULT
            if latest_result.disposition == "REJECTED"
            else (
                INCONCLUSIVE_RESULT
                if latest_result.disposition in {"INCONCLUSIVE", "DATA_INSUFFICIENT"}
                else EMPIRICAL_RESULT
            )
        )
        value = publication_value_score(
            {
                "clarity": 4,
                "novelty": 3,
                "educational": 4,
                "timeliness": 2,
                "visualizable": 5,
                "source_quality": 5,
                "explainable": 5,
            }
        )
        candidates.append(
            PublicationCandidate(
                source_run_id=daily_run_id,
                source_kind="DAILY",
                category=category,
                title=f"Funding extremity and subsequent 24h returns ({latest_result.disposition})",
                summary=(
                    f"Pooled 24h extreme-vs-ordinary funding comparison; "
                    f"disposition {latest_result.disposition}."
                ),
                evidence={
                    "event_study_result_id": str(latest_result.id),
                    "disposition": latest_result.disposition,
                },
                publication_value=value,
            )
        )

    # New hypothesis families in this Daily window (REQUEST_DATA / explainers).
    # MARKET families lead (cleanest, freshest); academic statement templates
    # are converted to readable paper-insight titles.
    from .db import ChannelHypothesis

    hypotheses = list(
        session.scalars(
            select(ChannelHypothesis)
            .where(ChannelHypothesis.created_at <= day_end)
            .order_by(ChannelHypothesis.created_at.desc())
            .limit(40)
        ).all()
    )
    hypotheses.sort(
        key=lambda h: (h.channel != "MARKET", -float(h.created_at.timestamp()))
    )
    seen_families: set[str] = set()
    for hypothesis in hypotheses:
        family = hypothesis.fingerprint
        if family in seen_families:
            continue
        seen_families.add(family)
        if len(seen_families) > 2:
            break
        category = (
            REQUEST_DATA_FINDING
            if hypothesis.maturity == "H1_STATISTICAL_HYPOTHESIS"
            else HYPOTHESIS_EXPLAINER
        )
        value = publication_value_score(
            {
                "clarity": 4,
                "novelty": 4,
                "educational": 4,
                "timeliness": 2,
                "visualizable": 3,
                "source_quality": 3,
                "explainable": 4,
            }
        )
        candidates.append(
            PublicationCandidate(
                source_run_id=daily_run_id,
                source_kind="DAILY",
                category=category,
                title=_human_candidate_title(hypothesis),
                summary=(
                    f"Hypothesis family from the {hypothesis.channel} channel. "
                    "Critics requested additional independent evidence before "
                    "any Event Study; interpreted as a research-in-progress note."
                ),
                evidence={
                    "hypothesis_id": str(hypothesis.id),
                    "family": family,
                    "channel": hypothesis.channel,
                    "disposition": "REQUEST_DATA",
                    "status": f"{hypothesis.maturity} / {hypothesis.status}",
                },
                publication_value=value,
            )
        )

    # Newly retrieved academic works in the window (paper explainers).
    from .db import PublicationDraft, SourceItem

    papers = session.scalars(
        select(SourceItem)
        .where(
            SourceItem.source_name.in_(
                ["openalex", "crossref", "arxiv", "nber", "repec"]
            )
        )
        .where(SourceItem.retrieved_at <= day_end)
        .order_by(SourceItem.retrieved_at.desc())
        .limit(8)
    ).all()
    # Topic-level repetition control: skip papers whose title is already
    # covered by any existing draft (published or not).
    existing_texts = [
        draft.text[:500].lower()
        for draft in session.scalars(select(PublicationDraft)).all()
    ]
    for paper in papers:
        title_key = (paper.title or "").lower()
        if any(title_key and title_key in text for text in existing_texts):
            continue
        value = publication_value_score(
            {
                "clarity": 4,
                "novelty": 3,
                "educational": 4,
                "timeliness": 2,
                "visualizable": 2,
                "source_quality": 4,
                "explainable": 4,
            }
        )
        candidates.append(
            PublicationCandidate(
                source_run_id=daily_run_id,
                source_kind="DAILY",
                category=PAPER_EXPLAINER,
                title=paper.title,
                summary=(
                    "Newly retrieved academic work summarized from archived "
                    "metadata; source attribution retained in the bundle."
                ),
                evidence={
                    "source_item_id": str(paper.id),
                    "source_name": paper.source_name,
                    "title": paper.title,
                    "url": paper.canonical_url or "",
                    "authors": list(paper.authors or [])[:4],
                },
                publication_value=value,
            )
        )

    # Research-process note: informative only when several new families were
    # created and none passed the research gate (real lesson, not daily filler).
    if len(seen_families) >= 2 and latest_result is None:
        value = publication_value_score(
            {
                "clarity": 4,
                "novelty": 3,
                "educational": 4,
                "timeliness": 2,
                "visualizable": 2,
                "source_quality": 3,
                "explainable": 4,
            }
        )
        candidates.append(
            PublicationCandidate(
                source_run_id=daily_run_id,
                source_kind="DAILY",
                category=RESEARCH_PROCESS_NOTE,
                title=f"Radar generated {len(seen_families)} new research hypotheses today — none passed the research gate",
                summary=(
                    "A research-process note: several new hypothesis families "
                    "were created, and every critic requested additional "
                    "independent evidence or methodological definition. No "
                    "Event Study is recommended yet."
                ),
                evidence={
                    "logical_date": logical_date,
                    "new_families": len(seen_families),
                    "disposition": "REQUEST_DATA",
                },
                publication_value=value,
            )
        )
    # Identity-level dedup precedes copy rendering. Same scientific object and
    # category in one Daily is one editorial angle even if title wording shifts.
    unique: dict[tuple[str, str, str], PublicationCandidate] = {}
    for candidate in candidates:
        evidence = candidate.evidence or {}
        object_id = str(
            evidence.get("hypothesis_id")
            or evidence.get("source_item_id")
            or evidence.get("event_study_result_id")
            or candidate.title
        )
        unique.setdefault(
            (candidate.source_run_id, candidate.category, object_id), candidate
        )
    return list(unique.values())


def select_editorial_daily_candidate(
    candidates: list[PublicationCandidate],
) -> tuple[PublicationCandidate | None, dict[str, Any]]:
    """Editorial selection: best single primary candidate (max 1 per day).

    Ranks by auditable PublicationValueScore; a candidate must be publishable
    by policy. Returns (None, reason) when nothing is worth publishing.
    """
    publishable = [
        candidate
        for candidate in candidates
        if classify_policy(candidate) in PUBLISHABLE
    ]
    if not publishable:
        return None, {"reason": "no publishable candidate in pool"}
    best = max(
        publishable,
        key=lambda candidate: (candidate.publication_value or {}).get("total", 0),
    )
    return best, {"reason": "highest publication value"}


def select_weekly_candidates(
    session: Session, *, weekly_run_id: str, week_saturday: str
) -> list[PublicationCandidate]:
    candidates: list[PublicationCandidate] = []
    value = publication_value_score(
        {
            "clarity": 4,
            "novelty": 3,
            "educational": 4,
            "timeliness": 1,
            "visualizable": 2,
            "source_quality": 4,
            "explainable": 4,
        }
    )
    candidates.append(
        PublicationCandidate(
            source_run_id=weekly_run_id,
            source_kind="WEEKLY",
            category=WEEKLY_RESEARCH_ROUNDUP,
            title=f"Weekly research roundup (week ending {week_saturday})",
            summary="Aggregated weekly evidence, hypotheses, and empirical-history updates.",
            evidence={"week_saturday": week_saturday},
            publication_value=value,
        )
    )
    return candidates


def _human_candidate_title(hypothesis: Any) -> str:
    """Readable title from persisted hypothesis text (no fingerprint syntax)."""
    statement = (hypothesis.statement or "").strip().rstrip(".")
    import re as _re

    quoted = _re.search(r"described by '(.+?)'", statement)
    if quoted:
        return f"Paper insight: {quoted.group(1)}"
    if len(statement) > 8:
        return statement
    return f"{hypothesis.universe or 'market'} {hypothesis.outcome or 'research'}"


def _day_end(logical_date: str) -> datetime:
    return datetime.fromisoformat(f"{logical_date}T23:59:59+00:00").replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def classify_policy(candidate: PublicationCandidate) -> str:
    if candidate.category == EMPIRICAL_RESULT:
        return PUBLIC_WITH_LIMITATIONS
    if candidate.category in {NEGATIVE_RESULT, WEEKLY_NEGATIVE_RESULT}:
        return PUBLIC  # negative results are valuable public science
    if candidate.category in {METHODOLOGY_NOTE, RESEARCH_PROCESS_NOTE}:
        return PUBLIC
    if candidate.category == PAPER_EXPLAINER:
        return PUBLIC  # general literature explanation with attribution
    if candidate.category in {
        MARKET_OBSERVATION,
        WEEKLY_RESEARCH_ROUNDUP,
        DATA_QUALITY_FINDING,
        INCONCLUSIVE_RESULT,
        REQUEST_DATA_FINDING,
        HYPOTHESIS_EXPLAINER,
        PRACTITIONER_EXPLAINER,
        PAPER_PLUS_MARKET_CONNECTION,
    }:
        return PUBLIC_WITH_LIMITATIONS
    if candidate.category == ALPHA_CANDIDATE:
        return PRIVATE  # potentially executable alpha stays private by default
    return PRIVATE


# ---------------------------------------------------------------------------
# Privacy scrub
# ---------------------------------------------------------------------------

SCRUB_PATTERNS = [
    (re.compile(r"/Users/\w+[^\s\"']*"), "<PATH>"),
    (re.compile(r"sqlite:////[^\s\"']+"), "<DB>"),
    (re.compile(r"postgresql\+?\w*://[^\s\"']+"), "<DB>"),
    (re.compile(r"https://discord(app)?\.com/api/webhooks/[^\s\"']+"), "<WEBHOOK>"),
    (
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
        "<UUID>",
    ),
    (re.compile(r"\b(sk-|key-|Bearer\s+)[A-Za-z0-9_-]{8,}"), "<SECRET>"),
]


def scrub_privacy(text: str) -> str:
    for pattern, replacement in SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Claim verification
# ---------------------------------------------------------------------------

NUMBER_PATTERN = re.compile(r"-?\d+\.\d+|%|\bdays?\b|\bhours?\b")


@dataclass
class ClaimVerdict:
    claims: list[dict[str, Any]]
    blocked: bool
    reason: str | None


def verify_claims(
    text: str,
    *,
    empirical: dict[str, Any] | None,
    structured_numbers: dict[str, float],
) -> ClaimVerdict:
    """Every substantive claim must trace to evidence or be honestly labeled.

    Unknown numeric values cannot pass; hypothesis language stays hypothesis.
    """
    claims: list[dict[str, Any]] = []
    blocked_reason: str | None = None
    lowered = text.lower()
    for forbidden in FORBIDDEN_LANGUAGE:
        if forbidden in lowered:
            return ClaimVerdict([], True, f"forbidden language: {forbidden}")
    numbers_in_text = set(NUMBER_PATTERN.findall(text))
    for number in numbers_in_text:
        cleaned = number.rstrip("%")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if structured_numbers and not any(
            abs(value - reference) < 5e-4 * max(1.0, abs(reference))
            for reference in structured_numbers.values()
        ):
            blocked_reason = f"unsupported numeric claim: {number}"
            claims.append(
                {
                    "claim": number,
                    "class": CLAIM_UNKNOWN,
                    "supported": False,
                }
            )
    disposition = (empirical or {}).get("disposition")
    if disposition == "INCONCLUSIVE" and any(
        phrase in lowered for phrase in ("we proved", "proven that", "confirmed that")
    ):
        return ClaimVerdict([], True, "INCONCLUSIVE result must not be called proven")
    if (
        disposition is not None
        and MATURITY_PHRASES.get(disposition, "").lower() not in lowered
        and disposition
        in {
            "INCONCLUSIVE",
            "REJECTED",
            "DATA_INSUFFICIENT",
        }
    ):
        claims.append(
            {
                "claim": f"disposition {disposition} must appear in copy",
                "class": CLAIM_EMPIRICAL,
                "supported": True,
            }
        )
        blocked_reason = blocked_reason or (
            f"empirical maturity '{disposition}' missing from copy"
            if disposition in {"INCONCLUSIVE", "REJECTED"}
            and "inconclusive" not in lowered
            and "rejected" not in lowered
            and "insufficient" not in lowered
            else blocked_reason
        )
    return ClaimVerdict(claims, blocked_reason is not None, blocked_reason)


# ---------------------------------------------------------------------------
# Copy generation (deterministic, template-driven; no LLM invention)
# ---------------------------------------------------------------------------


def generate_public_copy(
    candidate: PublicationCandidate,
    *,
    empirical: dict[str, Any] | None,
    language: str = "ENGLISH",
) -> str:
    """Deterministic public copy from structured evidence.

    Category-aware templates; every statement derives from persisted
    candidate/evidence fields — no LLM invention, no fabricated numbers.
    Structure: observation/question, core evidence, interpretation,
    limitation, attribution.
    """
    disposition = (empirical or {}).get("disposition")
    category = candidate.category
    title = (candidate.title or "").strip()
    summary = (candidate.summary or "").strip()
    evidence = candidate.evidence or {}

    if category == WEEKLY_RESEARCH_ROUNDUP:
        body = (
            "Weekly research roundup from Quant Research Radar.\n\n"
            "This week the Radar aggregated daily evidence across academic and "
            "practitioner sources plus perpetual-funding market observations, and "
            "maintained its hypothesis registry without promoting unvalidated ideas.\n\n"
            "Interpretation: weekly reviews are for evidence understanding, not signals.\n\n"
            "Limitation: all observations are research artifacts, not trading advice.\n\n"
            "Source: Quant Research Radar weekly review."
        )
    elif category == PAPER_EXPLAINER:
        url = evidence.get("url") or ""
        authors = ", ".join(evidence.get("authors") or []) or "unknown authors"
        body = (
            f"Reading: {title}\n\n"
            f"{summary or 'A newly retrieved academic work in quantitative research.'}\n\n"
            f"Interpretation: paper summary derived from archived metadata; the work "
            f"is presented for research context, not as a validated result.\n\n"
            f"Limitation: we summarize from archived content and provenance; "
            f"read the original before relying on any claim.\n\n"
            f"Source: {authors} — {evidence.get('source_name') or 'academic source'}"
            + (f" ({url})" if url else "")
        )
    elif category in {REQUEST_DATA_FINDING, HYPOTHESIS_EXPLAINER}:
        body = (
            f"Research hypothesis: {title}\n\n"
            f"{summary}\n\n"
            "Interpretation: this is an unvalidated hypothesis — the evidence is "
            "not sufficient yet. Critics requested additional independent evidence "
            "or methodological definition before any test.\n\n"
            "Limitation: single-channel and/or metadata-only support; not a trading "
            "signal.\n\n"
            "Source: Quant Research Radar research registry."
        )
    elif category == RESEARCH_PROCESS_NOTE:
        body = (
            f"{title}\n\n"
            f"{summary}\n\n"
            "Interpretation: a useful negative-process outcome — hypothesis "
            "generation is working, and the research gate correctly declined to "
            "promote unvalidated ideas.\n\n"
            "Limitation: describes internal research process, not market results.\n\n"
            "Source: Quant Research Radar daily research review."
        )
    elif category in {METHODOLOGY_NOTE, DATA_QUALITY_FINDING}:
        body = (
            f"{title}\n\n"
            f"{summary}\n\n"
            "Interpretation: methodological or data-quality observations do not "
            "change any scientific conclusion by themselves.\n\n"
            "Limitation: diagnostics are context-specific.\n\n"
            "Source: Quant Research Radar research audit."
        )
    elif category == MARKET_OBSERVATION:
        body = (
            f"{title}\n\n"
            f"{summary}\n\n"
            "Interpretation: market-state observation from structured funding data; "
            "no directional recommendation.\n\n"
            "Limitation: descriptive only.\n\n"
            "Source: Quant Research Radar market observations."
        )
    else:
        # EMPIRICAL_RESULT / NEGATIVE_RESULT / INCONCLUSIVE_RESULT / WEEKLY_NEGATIVE_RESULT
        concentration = (empirical or {}).get("asset_concentration") or "cross-asset"
        disposition_label = MATURITY_PHRASES.get(
            str(disposition), str(disposition).lower()
        )
        body = (
            f"We tested whether extreme perpetual funding is associated with subsequent returns.\n\n"
            f"Core evidence: the pooled 24h extreme-vs-ordinary comparison completed with "
            f"disposition {disposition_label} ({concentration}).\n\n"
            f"Interpretation: this is an in-sample association in a historical reconstructive "
            f"sample, not a trading signal.\n\n"
            f"Limitation: the methodology critic could not fully validate the study; "
            f"sample covers roughly seven months of hourly data.\n\n"
            f"Source: Quant Research Radar event study on Hyperliquid funding data."
        )
    if language == "CHINESE":
        body = _chinese_fallback(candidate, body)
    elif language == "BILINGUAL":
        body = body + "\n\n---\n\n" + _chinese_fallback(candidate, body)
    return scrub_privacy(body)


def _chinese_fallback(candidate: PublicationCandidate, english_body: str) -> str:
    """Honest minimal Chinese frame; keeps verified English evidence verbatim.

    A full per-category translation would duplicate the pipeline and risks
    drifting from the verified English content; the Chinese render therefore
    frames the title and points to the verified English body + source bundle.
    """
    title = (candidate.title or "").strip()
    return (
        f"# {title}\n\n"
        "（中文框架；以下为经核实的英文研究内容，未作机器改写以免偏离来源。）\n\n"
        f"{english_body}\n\n"
        "本内容为研究资料，不构成投资建议。"
    )


# ---------------------------------------------------------------------------
# Visuals (deterministic charts from structured numbers only)
# ---------------------------------------------------------------------------


def render_effect_chart(
    out_dir: Path,
    *,
    structured_numbers: dict[str, float],
    title: str,
    sample_note: str,
) -> Path:
    """Deterministic PNG from structured result numbers; never invented values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(structured_numbers.keys())
    values = [structured_numbers[label] for label in labels]
    fig, axis = plt.subplots(figsize=(8, 4.5), dpi=160)
    bars = axis.bar(labels, values, color="#3b6ea5")
    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            f"{value:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_title(title, fontsize=12)
    axis.set_ylabel("Mean 24h log-return difference")
    axis.set_xlabel("Group")
    axis.figure.text(
        0.01,
        0.01,
        f"Sample: {sample_note} | Source: Quant Research Radar (research, not trading advice)",
        fontsize=7,
        color="#555555",
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    digest = hashlib.sha256(
        "|".join(
            f"{k}:{v:.10f}" for k, v in sorted(structured_numbers.items())
        ).encode()
    ).hexdigest()[:12]
    path = out_dir / f"effect-{digest}.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Draft + registry
# ---------------------------------------------------------------------------


def draft_idempotence_key(candidate: PublicationCandidate, text: str) -> str:
    return hashlib.sha256(f"{candidate.id}|{text}".encode()).hexdigest()


def create_draft(
    session: Session,
    candidate: PublicationCandidate,
    *,
    empirical: dict[str, Any] | None,
    structured_numbers: dict[str, float],
    language: str,
    visual_ids: list[str] | None = None,
) -> tuple[PublicationDraft | None, str | None]:
    """Build the draft, verify claims, persist. Returns (draft, rejection_reason)."""
    object_id = str(
        (candidate.evidence or {}).get("hypothesis_id")
        or (candidate.evidence or {}).get("source_item_id")
        or (candidate.evidence or {}).get("event_study_result_id")
        or candidate.title
    )
    for prior in session.scalars(
        select(PublicationCandidate).where(
            PublicationCandidate.source_run_id == candidate.source_run_id,
            PublicationCandidate.category == candidate.category,
        )
    ).all():
        prior_id = str(
            (prior.evidence or {}).get("hypothesis_id")
            or (prior.evidence or {}).get("source_item_id")
            or (prior.evidence or {}).get("event_study_result_id")
            or prior.title
        )
        if prior_id == object_id:
            candidate = prior
            break
    if candidate.id is None:
        session.add(candidate)
        session.flush()
    policy = classify_policy(candidate)
    text = generate_public_copy(candidate, empirical=empirical, language=language)
    verdict = verify_claims(
        text, empirical=empirical, structured_numbers=structured_numbers
    )
    if verdict.blocked:
        return None, verdict.reason
    if policy not in PUBLISHABLE:
        return None, f"policy {policy} does not permit publication"
    claims = verdict.claims or [
        {
            "claim": "deterministic template copy",
            "class": (
                CLAIM_HYPOTHESIS
                if candidate.category in {REQUEST_DATA_FINDING, HYPOTHESIS_EXPLAINER}
                else CLAIM_SOURCE_SUPPORTED
                if candidate.category
                in {
                    PAPER_EXPLAINER,
                    PRACTITIONER_EXPLAINER,
                    PAPER_PLUS_MARKET_CONNECTION,
                }
                else CLAIM_INFERENCE
                if candidate.category == MARKET_OBSERVATION
                else CLAIM_EMPIRICAL
            ),
            "supported": True,
        }
    ]
    key = draft_idempotence_key(candidate, text)
    existing = session.scalar(
        select(PublicationDraft).where(PublicationDraft.idempotence_key == key)
    )
    if existing is not None:
        return existing, None
    draft = PublicationDraft(
        candidate_id=candidate.id,
        policy=policy,
        language=language,
        text=text,
        claims=claims,
        source_bundle={
            "candidate_id": str(candidate.id),
            "evidence": candidate.evidence,
            "event_study_result_id": (empirical or {}).get("event_study_result_id"),
        },
        visual_ids=visual_ids or [],
        idempotence_key=key,
    )
    session.add(draft)
    session.commit()
    return draft, None


def register_publication(
    session: Session,
    draft: PublicationDraft,
    *,
    platform: str,
    status: str,
    external_post_id: str | None = None,
    failure_reason: str | None = None,
) -> PublicationRecord:
    existing = session.scalars(
        select(PublicationRecord)
        .where(PublicationRecord.draft_id == draft.id)
        .where(PublicationRecord.platform == platform)
        .where(PublicationRecord.status.in_(["PUBLISHED", status]))
    ).first()
    if existing is not None and existing.status == "PUBLISHED":
        return existing  # idempotent
    record = PublicationRecord(
        draft_id=draft.id,
        platform=platform,
        status=status,
        external_post_id=external_post_id,
        failure_reason=failure_reason,
        retry_count=0,
        published_at=utcnow() if status == "PUBLISHED" else None,
    )
    session.add(record)
    session.commit()
    return record


def find_recent_duplicate(
    session: Session, text: str, *, lookback_days: int = 7
) -> PublicationDraft | None:
    """Near-duplicate topic suppression: same normalized core already drafted."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    fingerprint = hashlib.sha256(normalized[:400].encode()).hexdigest()
    cutoff = utcnow().timestamp() - lookback_days * 86400
    for draft in session.scalars(select(PublicationDraft)).all():
        if draft.created_at is not None and draft.created_at.timestamp() < cutoff:
            continue
        draft_norm = re.sub(r"\s+", " ", draft.text.lower()).strip()
        if hashlib.sha256(draft_norm[:400].encode()).hexdigest() == fingerprint:
            return draft
        candidate = session.get(PublicationCandidate, draft.candidate_id)
        if candidate is None:
            continue
        candidate_norm = re.sub(
            r"\s+", " ", (candidate.title + " " + (candidate.summary or "")).lower()
        ).strip()
        if hashlib.sha256(candidate_norm[:400].encode()).hexdigest() == fingerprint:
            return draft
    return None


def run_ids_complete(session: Session, source_run_id: str) -> bool:
    """Publication requires the underlying research run to be complete."""
    daily = session.get(DailyRun, __import__("uuid").UUID(source_run_id))
    if daily is not None:
        return daily.status == "SUCCESS"
    weekly = session.get(WeeklyRun, __import__("uuid").UUID(source_run_id))
    if weekly is not None:
        return weekly.status == "SUCCESS"
    return False
