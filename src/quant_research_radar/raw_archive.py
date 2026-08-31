from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session


@dataclass(frozen=True)
class RawObject:
    content_sha256: str
    media_type: str
    byte_size: int
    retrieved_at: datetime
    storage_uri: str


class RawArchive:
    """Bounded content-addressed store for public source payloads only."""

    MAX_OBJECT_BYTES = 1_000_000
    MEDIA_TYPES = frozenset({"application/json", "application/xml", "text/plain"})

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _extension(media_type: str) -> str:
        return {
            "application/json": "json",
            "application/xml": "xml",
            "application/atom+xml": "xml",
            "text/xml": "xml",
            "text/html": "html",
            "application/pdf": "pdf",
        }.get(media_type, "bin")

    def store(
        self, content: bytes, *, media_type: str, retrieved_at: datetime
    ) -> RawObject:
        if media_type not in self.MEDIA_TYPES:
            raise ValueError(f"unsupported raw archive media type: {media_type}")
        if len(content) > self.MAX_OBJECT_BYTES:
            raise ValueError("raw archive object exceeds bounded retrieval limit")
        digest = hashlib.sha256(content).hexdigest()
        relative = Path("objects") / digest[:2] / digest
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        return RawObject(
            content_sha256=digest,
            media_type=media_type,
            byte_size=len(content),
            retrieved_at=retrieved_at,
            storage_uri=(Path("data/raw") / relative).as_posix(),
        )


def archive_receipt(
    session: Session,
    archive: RawArchive,
    *,
    content: bytes,
    media_type: str,
    provider: str,
    canonical_url: str | None,
    retrieved_at: datetime,
    source_native_timestamp: datetime | None,
    source_item_id: object | None = None,
    market_observation_id: object | None = None,
    analysis_mode: str = "PRODUCTION_LIVE",
) -> RawObject:
    """Store bytes once and append a retrieval receipt without overwriting prior evidence."""
    from sqlalchemy import select

    from .db import RawArtifact, RawArtifactReceipt

    stored = archive.store(content, media_type=media_type, retrieved_at=retrieved_at)
    artifact = session.scalar(
        select(RawArtifact).where(RawArtifact.content_sha256 == stored.content_sha256)
    )
    if artifact is None:
        artifact = RawArtifact(
            content_sha256=stored.content_sha256,
            media_type=stored.media_type,
            byte_size=stored.byte_size,
            storage_uri=stored.storage_uri,
        )
        session.add(artifact)
        session.flush()
    session.add(
        RawArtifactReceipt(
            raw_artifact_id=artifact.id,
            provider=provider,
            canonical_url=canonical_url,
            source_native_timestamp=source_native_timestamp,
            retrieved_at=retrieved_at,
            source_item_id=source_item_id,
            market_observation_id=market_observation_id,
            analysis_mode=analysis_mode,
        )
    )
    session.flush()
    return stored
