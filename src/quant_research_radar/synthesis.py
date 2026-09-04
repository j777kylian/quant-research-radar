"""Deterministic topic synthesis from finalized persisted state only."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import DailyRun, SourceItem, TopicBrief
from .reporting import collect_daily_snapshot

TOPIC_BRIEF_VERSION = "1"
DEPTH_FULL_TEXT = "FULL_TEXT"
DEPTH_ABSTRACT = "ABSTRACT"
DEPTH_STRUCTURED_EXCERPT = "STRUCTURED_EXCERPT"
DEPTH_METADATA_ONLY = "METADATA_ONLY"


def evidence_depth(item: SourceItem) -> str:
    """Classify persisted content conservatively; never infer results from title."""
    text = (item.raw_text or "").strip()
    metadata = item.raw_metadata or {}
    if text:
        return DEPTH_FULL_TEXT
    if metadata.get("abstract") or metadata.get("abstract_text"):
        return DEPTH_ABSTRACT
    if metadata.get("excerpt") or metadata.get("claims"):
        return DEPTH_STRUCTURED_EXCERPT
    return DEPTH_METADATA_ONLY


def _hash_packet(packet: dict[str, Any]) -> str:
    payload = json.dumps(
        packet, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _bounded_text(item: SourceItem, depth: str) -> str:
    if depth == DEPTH_FULL_TEXT:
        return (item.raw_text or "").strip()[:1200]
    if depth == DEPTH_ABSTRACT:
        return str(
            (item.raw_metadata or {}).get("abstract")
            or (item.raw_metadata or {}).get("abstract_text")
        )[:1200]
    if depth == DEPTH_STRUCTURED_EXCERPT:
        return str((item.raw_metadata or {}).get("excerpt") or "")[:1200]
    return "Only metadata is available; methodology and results cannot be reliably summarized."


def build_daily_topic_packet(session: Session, daily: DailyRun) -> dict[str, Any]:
    snapshot = collect_daily_snapshot(session, daily.id) or {}
    inputs = (snapshot.get("human") or {}).get("inputs") or {}
    source_items = session.scalars(
        select(SourceItem)
        .where(SourceItem.retrieved_at <= (daily.ended_at or datetime.now(UTC)))
        .order_by(SourceItem.retrieved_at.desc())
        .limit(20)
    ).all()
    selected = []
    for item in source_items:
        if item.source_name.lower() not in {
            "openalex",
            "crossref",
            "arxiv",
            "nber",
            "repec",
            "alpha-architect",
            "man-institute",
            "aqr",
        }:
            continue
        depth = evidence_depth(item)
        selected.append(
            {
                "source_item_id": str(item.id),
                "title": item.title,
                "authors": list(item.authors or []),
                "source": item.source_name,
                "date": item.published_at.isoformat() if item.published_at else None,
                "url": item.canonical_url,
                "evidence_depth": depth,
                "bounded_content": _bounded_text(item, depth),
            }
        )
    return {
        "logical_date": daily.logical_date.isoformat(),
        "run_id": str(daily.id),
        "status": daily.status,
        "inputs": inputs,
        "source_items": selected,
        "findings": (snapshot.get("human") or {}).get("findings", []),
        "critic_reasons": (snapshot.get("human") or {}).get("critic_reasons", {}),
        "research": snapshot.get("research", {}),
        "knowledge": snapshot.get("knowledge", {}),
        "market": snapshot.get("market", {}),
        "provenance": {"code_sha": daily.code_sha},
    }


def synthesize_daily_topics(session: Session, daily_run_id: Any) -> list[TopicBrief]:
    daily = session.get(DailyRun, daily_run_id)
    if daily is None or daily.status == "RUNNING":
        return []
    packet = build_daily_topic_packet(session, daily)
    packet_hash = _hash_packet(packet)
    findings = packet.get("findings") or []
    topics: list[TopicBrief] = []
    for index, finding in enumerate(findings[:4], 1):
        topic_id = str(
            finding.get("fingerprint") or finding.get("title") or f"daily-{index}"
        )
        title = str(finding.get("title") or f"Research topic {index}")
        endpoints = finding.get("horizon_endpoints") or []
        brief = {
            "background": "Prior knowledge and today's persisted evidence motivate this question; no unsupported market theory is added.",
            "research_question": finding.get("question") or title,
            "what_changed_today": finding.get("novelty", "UNKNOWN"),
            "evidence": {
                "channels": [finding.get("channel")] if finding.get("channel") else [],
                "finding": finding,
            },
            "methods": "Unavailable unless supported by persisted source content or empirical artifacts.",
            "results": "No new empirical result is claimed by this TopicBrief.",
            "interpretation": "Research context only; association is not causality or a trading signal.",
            "criticisms": packet.get("critic_reasons", {}),
            "limitations": finding.get("limitation")
            or "Evidence and methodology remain incomplete.",
            "knowledge_relationship": finding.get("novelty", "UNKNOWN"),
            "horizon_endpoints": endpoints,
            "user_fit": endpoints,
            "scientific_status": finding.get("status"),
            "recommendation": finding.get("next") or "ACCUMULATE_EVIDENCE",
            "next_action": finding.get("next"),
            "sources": [],
            "evidence_depth": "STRUCTURED_EXCERPT",
            "public_eligibility": True,
            "research_priority": None,
        }
        existing = session.scalar(
            select(TopicBrief).where(
                TopicBrief.source_run_id == str(daily.id),
                TopicBrief.topic_id == topic_id,
                TopicBrief.topic_version == TOPIC_BRIEF_VERSION,
            )
        )
        if existing:
            topics.append(existing)
            continue
        topic = TopicBrief(
            topic_id=topic_id,
            topic_version=TOPIC_BRIEF_VERSION,
            logical_date=daily.logical_date,
            source_run_id=str(daily.id),
            source_kind="DAILY",
            human_title=title,
            packet=packet,
            brief=brief,
            input_packet_hash=packet_hash,
            model_metadata={"role": "DETERMINISTIC_TOPIC_SYNTHESIS", "model": None},
        )
        session.add(topic)
        session.flush()
        topics.append(topic)
    session.commit()
    return topics
