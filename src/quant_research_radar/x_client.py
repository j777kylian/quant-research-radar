"""X publishing adapter: official-API client with safe modes and a hard gate.

Modes: DISABLED | DRAFT_ONLY (default; no external request) | AUTO_PUBLISH.
Missing credentials degrade X to DRAFT_ONLY/UNAVAILABLE and never block Email,
Discord, drafts, visuals, or the publication registry.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .db import PublicationDraft, PublicationRecord, utcnow
from .publishing import (
    PUBLIC,
    PUBLIC_WITH_LIMITATIONS,
)

X_MODE_DISABLED = "DISABLED"
X_MODE_DRAFT_ONLY = "DRAFT_ONLY"
X_MODE_AUTO_PUBLISH = "AUTO_PUBLISH"

TWEET_URL = "https://api.twitter.com/2/tweets"
MEDIA_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"


@dataclass
class XResult:
    status: str  # PUBLISHED | FAILED | DRAFT_ONLY | REJECTED
    external_post_id: str | None = None
    reason: str | None = None


def x_mode(settings: Any) -> str:
    """Effective X mode; missing credentials degrade AUTO_PUBLISH to DRAFT_ONLY."""
    configured = getattr(settings, "publication_mode", X_MODE_DRAFT_ONLY)
    if configured == X_MODE_DISABLED:
        return X_MODE_DISABLED
    has_credentials = all(
        getattr(settings, field, None)
        for field in ("x_api_key", "x_api_secret", "x_access_token", "x_access_secret")
    )
    if configured == X_MODE_AUTO_PUBLISH and not has_credentials:
        return X_MODE_DRAFT_ONLY
    return configured


def _percent_encode(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _oauth1_header(
    *,
    method: str,
    url: str,
    credentials: dict[str, str],
    extra_params: dict[str, str] | None = None,
) -> str:
    """Minimal OAuth 1.0a HMAC-SHA256 signature (no third-party dependency)."""
    oauth_params = {
        "oauth_consumer_key": credentials["api_key"],
        "oauth_nonce": hmac.new(uuid.uuid4().bytes, digestmod="sha256").hexdigest(),
        "oauth_signature_method": "HMAC-SHA256",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": credentials["access_token"],
        "oauth_version": "1.0",
    }
    all_params = {**oauth_params, **(extra_params or {})}
    param_string = "&".join(
        f"{_percent_encode(k)}={_percent_encode(v)}"
        for k, v in sorted(all_params.items())
    )
    base_string = "&".join(
        [method.upper(), _percent_encode(url), _percent_encode(param_string)]
    )
    signing_key = "&".join(
        [
            _percent_encode(credentials["api_secret"]),
            _percent_encode(credentials["access_secret"]),
        ]
    )
    signature = hmac.new(
        signing_key.encode(), base_string.encode(), hashlib.sha256
    ).digest()
    oauth_params["oauth_signature"] = urllib.parse.quote(
        signature.hex(), safe=""
    )  # hex keeps it simple and deterministic
    header = ", ".join(
        f'{_percent_encode(k)}="{_percent_encode(v)}"'
        for k, v in sorted(oauth_params.items())
    )
    return f"OAuth {header}"


# ---------------------------------------------------------------------------
# Final publication gate
# ---------------------------------------------------------------------------


def publication_gate(
    draft: PublicationDraft,
    *,
    research_run_complete: bool,
    already_published: bool,
    visual_values_trace: bool = True,
) -> tuple[bool, str | None]:
    """All applicable checks must pass before AUTO_PUBLISH. Fail closed."""
    if not research_run_complete:
        return False, "underlying research run is not complete"
    if already_published:
        return False, "idempotence key already published"
    if draft.policy not in {PUBLIC, PUBLIC_WITH_LIMITATIONS}:
        return False, f"policy {draft.policy} does not allow posting"
    if draft.policy == PUBLIC_WITH_LIMITATIONS:
        lowered = draft.text.lower()
        if (
            "limitation" not in lowered
            and "局限" not in lowered
            and "not a trading signal" not in lowered
        ):
            return False, "PUBLIC_WITH_LIMITATIONS copy lacks limitation language"
    if not draft.claims and draft.policy == PUBLIC_WITH_LIMITATIONS:
        return False, "claim verification record missing"
    if not visual_values_trace:
        return False, "visual values do not trace to structured data"
    return True, None


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def publish_draft(
    draft: PublicationDraft,
    settings: Any,
    *,
    research_run_complete: bool,
    already_published: bool,
    media_path: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> XResult:
    """Post via official API when AUTO_PUBLISH and every gate passes.

    DRAFT_ONLY/DISABLED never perform an external request.
    """
    mode = x_mode(settings)
    if mode != X_MODE_AUTO_PUBLISH:
        return XResult("DRAFT_ONLY", None, f"publication mode is {mode}")
    allowed, reason = publication_gate(
        draft,
        research_run_complete=research_run_complete,
        already_published=already_published,
    )
    if not allowed:
        return XResult("REJECTED", None, reason)
    credentials = {
        "api_key": settings.x_api_key,
        "api_secret": settings.x_api_secret,
        "access_token": settings.x_access_token,
        "access_secret": settings.x_access_secret,
    }
    media_id: str | None = None
    if media_path is not None:
        try:
            with httpx.Client(transport=transport, timeout=60) as client:
                media_bytes = os.path.basename(media_path).encode()
                upload = client.post(
                    MEDIA_UPLOAD_URL,
                    data={"command": "UPLOAD"},
                    headers={
                        "Authorization": _oauth1_header(
                            method="POST", url=MEDIA_UPLOAD_URL, credentials=credentials
                        )
                    },
                    files={
                        "media": (
                            os.path.basename(media_path),
                            media_bytes,
                            "image/png",
                        )
                    },
                )
                upload.raise_for_status()
                media_id = upload.json().get("media_id_string")
        except (httpx.HTTPError, OSError) as error:
            return XResult("FAILED", None, f"media upload failed: {error}")
    payload: dict[str, Any] = {"text": draft.text[:280]}
    if media_id:
        payload["media"] = {"media_ids": [media_id]}
    try:
        with httpx.Client(transport=transport, timeout=60) as client:
            response = client.post(
                TWEET_URL,
                json=payload,
                headers={
                    "Authorization": _oauth1_header(
                        method="POST",
                        url=TWEET_URL,
                        credentials=credentials,
                    )
                },
            )
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, OSError) as error:
        return XResult("FAILED", None, f"post failed (retryable): {error}")
    post_id = data.get("data", {}).get("id")
    return XResult("PUBLISHED", post_id, None)


def publication_record_for(
    existing: PublicationRecord | None,
    result: XResult,
) -> str:
    if existing is not None and existing.status == "PUBLISHED":
        return "PUBLISHED"
    return result.status


def now() -> str:
    return utcnow().isoformat()
