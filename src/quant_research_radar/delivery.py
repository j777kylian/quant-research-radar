"""Private delivery: Email (SMTP) and Discord webhook, idempotent and isolated.

Delivery failure NEVER fails the underlying research run: callers persist a
DeliveryRecord (SENT / FAILED / SKIPPED) and the research status is untouched.
Duplicate scheduler ticks are absorbed by the unique idempotence key.
"""

from __future__ import annotations

import hashlib
import json
import os
import smtplib
from dataclasses import dataclass
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .db import DeliveryRecord, utcnow


def delivery_key(channel: str, run_kind: str, run_date: str, content_hash: str) -> str:
    value = f"{channel}|{run_kind}|{run_date}|{content_hash}"
    return hashlib.sha256(value.encode()).hexdigest()


def _already_delivered(session: Session, key: str) -> DeliveryRecord | None:
    return session.scalar(
        select(DeliveryRecord).where(DeliveryRecord.idempotence_key == key)
    )


def _record_and_return(
    session: Session,
    *,
    channel: str,
    run_kind: str,
    run_date: str,
    key: str,
    status: str,
    failure_reason: str | None,
) -> DeliveryRecord:
    existing = _already_delivered(session, key)
    if existing is not None:
        if status == "SENT" and existing.status != "SENT":
            existing.status = "SENT"
            existing.sent_at = utcnow()
            existing.failure_reason = None
            session.commit()
        return existing
    record = DeliveryRecord(
        channel=channel,
        run_kind=run_kind,
        run_date=run_date,
        status=status,
        idempotence_key=key,
        failure_reason=failure_reason,
        retry_count=0 if status == "SENT" else 1,
        sent_at=utcnow() if status == "SENT" else None,
    )
    session.add(record)
    session.commit()
    return record


@dataclass
class DeliveryResult:
    channel: str
    status: str
    reason: str | None = None


# --------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------


def email_ready(settings: Settings) -> bool:
    return (
        settings.email_enabled
        and bool(settings.smtp_host)
        and bool(settings.email_from)
        and bool(settings.email_to)
    )


def send_email(
    session: Session,
    settings: Settings,
    *,
    run_kind: str,
    run_date: str,
    subject: str,
    body_text: str,
) -> DeliveryResult:
    if not email_ready(settings):
        return DeliveryResult("EMAIL", "SKIPPED", "EMAIL_NEEDS_CONFIGURATION")
    key = delivery_key(
        "EMAIL", run_kind, run_date, hashlib.sha256(body_text.encode()).hexdigest()
    )
    existing = _already_delivered(session, key)
    if existing is not None and existing.status == "SENT":
        return DeliveryResult("EMAIL", "SENT", None)  # idempotent no-op
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = settings.email_from or ""
    message["To"] = ", ".join(settings.email_to or [])
    message.attach(MIMEText(body_text, "plain", "utf-8"))
    try:
        assert settings.smtp_host is not None
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
            server.starttls()
            if settings.smtp_username and settings.smtp_password:
                server.login(settings.smtp_username, settings.smtp_password)
            server.sendmail(
                settings.email_from or "", settings.email_to or [], message.as_string()
            )
    except (smtplib.SMTPException, OSError) as error:
        _record_and_return(
            session,
            channel="EMAIL",
            run_kind=run_kind,
            run_date=run_date,
            key=key,
            status="FAILED",
            failure_reason=str(error)[:500],
        )
        return DeliveryResult("EMAIL", "FAILED", str(error)[:200])
    _record_and_return(
        session,
        channel="EMAIL",
        run_kind=run_kind,
        run_date=run_date,
        key=key,
        status="SENT",
        failure_reason=None,
    )
    return DeliveryResult("EMAIL", "SENT", None)


# --------------------------------------------------------------------------
# Discord
# --------------------------------------------------------------------------


def discord_ready(settings: Settings) -> bool:
    return bool(settings.discord_webhook_url)


def send_discord(
    session: Session,
    settings: Settings,
    *,
    run_kind: str,
    run_date: str,
    content: str,
) -> DeliveryResult:
    if not discord_ready(settings):
        return DeliveryResult("DISCORD", "SKIPPED", "DISCORD_NEEDS_CONFIGURATION")
    key = delivery_key(
        "DISCORD", run_kind, run_date, hashlib.sha256(content.encode()).hexdigest()
    )
    existing = _already_delivered(session, key)
    if existing is not None and existing.status == "SENT":
        return DeliveryResult("DISCORD", "SENT", None)
    payload: dict[str, Any] = {"content": content[:1900]}
    try:
        response = httpx.post(
            settings.discord_webhook_url or "", json=payload, timeout=30
        )
        response.raise_for_status()
    except (httpx.HTTPError, OSError) as error:
        _record_and_return(
            session,
            channel="DISCORD",
            run_kind=run_kind,
            run_date=run_date,
            key=key,
            status="FAILED",
            failure_reason=str(error)[:500],
        )
        return DeliveryResult("DISCORD", "FAILED", str(error)[:200])
    _record_and_return(
        session,
        channel="DISCORD",
        run_kind=run_kind,
        run_date=run_date,
        key=key,
        status="SENT",
        failure_reason=None,
    )
    return DeliveryResult("DISCORD", "SENT", None)


# --------------------------------------------------------------------------
# Digest builders (concise; no secrets, no filesystem paths)
# --------------------------------------------------------------------------


def daily_digest_text(summary: dict[str, Any], report_path: str | None) -> str:
    market = summary.get("market", {})
    lines = [
        f"Quant Research Radar — Daily {summary.get('logical_date', '')}",
        f"Status: {summary.get('status', 'UNKNOWN')}",
        f"Market: {market.get('status', 'n/a')} (inserted={market.get('inserted', 0)})",
        f"Analysis: {summary.get('intelligence_technical_status', 'n/a')}",
    ]
    if summary.get("failure_reasons"):
        lines.append(f"Failures: {'; '.join(summary['failure_reasons'][:3])}")
    lines.append(
        "No high-priority low-frequency research candidate today."
        if not summary.get("high_fit_count")
        else f"High-fit low-frequency candidates: {summary['high_fit_count']}"
    )
    if report_path:
        lines.append(f"Full report: {os.path.basename(report_path)}")
    return "\n".join(lines)


def weekly_digest_text(summary: dict[str, Any], report_path: str | None) -> str:
    lines = [
        f"Quant Research Radar — Weekly review {summary.get('week_saturday', '')}",
        f"Status: {summary.get('status', 'UNKNOWN')}",
        f"Daily cycles included: {len(summary.get('included_daily_dates', []))}",
        f"Hypotheses tracked: {summary.get('hypothesis_count', 0)}",
    ]
    priorities = summary.get("priorities", [])[:5]
    if priorities:
        lines.append("Top priorities:")
        for item in priorities:
            lines.append(
                f"  - {item.get('fit', '?')}: {str(item.get('statement', ''))[:100]}"
            )
    if summary.get("failure_reasons"):
        lines.append(f"Failures: {'; '.join(summary['failure_reasons'][:3])}")
    if report_path:
        lines.append(f"Full report: {os.path.basename(report_path)}")
    return "\n".join(lines)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
