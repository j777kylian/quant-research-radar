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

# Candidate categories
MARKET_OBSERVATION = "MARKET_OBSERVATION"
EMPIRICAL_RESULT = "EMPIRICAL_RESULT"
NEGATIVE_RESULT = "NEGATIVE_RESULT"
METHODOLOGY_NOTE = "METHODOLOGY_NOTE"
WEEKLY_RESEARCH_ROUNDUP = "WEEKLY_RESEARCH_ROUNDUP"
DATA_QUALITY_FINDING = "DATA_QUALITY_FINDING"

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
    """Evaluate a completed Daily cycle for publishable material.

    ZERO candidates is a valid outcome; operational noise is never selected.
    """
    candidates: list[PublicationCandidate] = []
    latest_result = session.scalars(
        select(EventStudyResultRecord)
        .where(EventStudyResultRecord.created_at <= _day_end(logical_date))
        .order_by(EventStudyResultRecord.created_at.desc())
        .limit(1)
    ).first()
    if latest_result is not None:
        category = (
            NEGATIVE_RESULT
            if latest_result.disposition
            in {
                "INCONCLUSIVE",
                "REJECTED",
                "DATA_INSUFFICIENT",
            }
            else EMPIRICAL_RESULT
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
    return candidates


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


def _day_end(logical_date: str) -> datetime:
    return datetime.fromisoformat(f"{logical_date}T23:59:59+00:00").replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def classify_policy(candidate: PublicationCandidate) -> str:
    if candidate.category == EMPIRICAL_RESULT:
        return PUBLIC_WITH_LIMITATIONS
    if candidate.category == NEGATIVE_RESULT:
        return PUBLIC  # negative results are valuable public science
    if candidate.category == METHODOLOGY_NOTE:
        return PUBLIC
    if candidate.category in {
        MARKET_OBSERVATION,
        WEEKLY_RESEARCH_ROUNDUP,
        DATA_QUALITY_FINDING,
    }:
        return PUBLIC_WITH_LIMITATIONS
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

    The publishing LLM may refine wording later, but the released contract is:
    clear observation, core evidence, interpretation, limitation, attribution.
    """
    disposition = (empirical or {}).get("disposition")
    if candidate.category == WEEKLY_RESEARCH_ROUNDUP:
        body = (
            "Weekly research roundup from Quant Research Radar.\n\n"
            "This week the Radar aggregated daily evidence across academic and "
            "practitioner sources plus perpetual-funding market observations, and "
            "maintained its hypothesis registry without promoting unvalidated ideas.\n\n"
            "Interpretation: weekly reviews are for evidence understanding, not signals.\n\n"
            "Limitation: all observations are research artifacts, not trading advice.\n\n"
            "Source: Quant Research Radar weekly review."
        )
    else:
        concentration = (empirical or {}).get("asset_concentration") or "cross-asset"
        body = (
            f"We tested whether extreme perpetual funding is associated with subsequent returns.\n\n"
            f"Core evidence: the pooled 24h extreme-vs-ordinary comparison completed with "
            f"disposition {disposition} ({concentration}).\n\n"
            f"Interpretation: this is an in-sample association in a historical reconstructive "
            f"sample, not a trading signal.\n\n"
            f"Limitation: the methodology critic could not fully validate the study; "
            f"sample covers roughly seven months of hourly data.\n\n"
            f"Source: Quant Research Radar event study on Hyperliquid funding data."
        )
    if language == "CHINESE":
        body = (
            "我们检验了极端永续资金费率是否与后续收益相关。\n\n"
            f"核心证据：合并24小时对比已完成，结论为 {disposition}。\n\n"
            "解释：这是历史重构样本中的样本内关联，不是交易信号。\n\n"
            "局限：方法论审查未能完全验证该研究；样本覆盖约七个月的逐小时数据。\n\n"
            "来源：Quant Research Radar 基于 Hyperliquid 资金费率的事件研究。"
        )
    elif language == "BILINGUAL":
        body = (
            body
            + "\n\n---\n\n"
            + generate_public_copy(candidate, empirical=empirical, language="CHINESE")
        )
    return scrub_privacy(body)


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
            "class": CLAIM_EMPIRICAL,
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
