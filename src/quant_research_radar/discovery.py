from __future__ import annotations

import json
import re
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (
    EvidenceSource,
    ResearchWork,
    SourceItem,
    WorkLocation,
    content_hash,
    normalize_utc,
)
from .raw_archive import RawArchive, archive_receipt
from .sources import SourceRecord, source_registry


def _normalized_title(value: str) -> str:
    return re.sub(r"\W+", " ", value.lower()).strip()


def _identity(record: SourceRecord) -> tuple[str, str | None, str | None]:
    metadata = record.raw_metadata
    doi = str(metadata.get("doi") or "").lower() or None
    arxiv_id = str(metadata.get("arxiv_id") or "").lower() or None
    if record.source_name == "arxiv" and not arxiv_id:
        arxiv_id = record.external_id.rsplit("/", 1)[-1]
    identity = f"doi:{doi}" if doi else f"title:{_normalized_title(record.title)}"
    return identity, doi, arxiv_id


def _source(session: Session, record: SourceRecord) -> EvidenceSource:
    profile = next(
        (
            candidate
            for candidate in source_registry()
            if candidate["source_name"] == record.source_name
        ),
        {},
    )
    source = session.scalar(
        select(EvidenceSource).where(EvidenceSource.source_name == record.source_name)
    )
    if source is None:
        source = EvidenceSource(
            source_name=record.source_name,
            source_class=str(
                record.raw_metadata.get("content_class")
                or profile.get("source_class")
                or record.source_type
            ),
            venue=str(record.raw_metadata.get("venue") or record.source_name),
            peer_review_status=str(
                record.raw_metadata.get("publication_status")
                or profile.get("publication_status")
                or "UNKNOWN"
            ),
            access_mode=str(
                record.raw_metadata.get("access_mode")
                or profile.get("access_mode")
                or "METADATA_ONLY"
            ),
            reliability_prior=str(
                record.raw_metadata.get("reliability_prior")
                or profile.get("reliability_prior")
                or "PUBLIC"
            ),
            domain_tags=list(
                record.raw_metadata.get("topics")
                or record.raw_metadata.get("categories")
                or []
            ),
            adapter_status="READY",
        )
        session.add(source)
        session.flush()
    return source


def ingest_records(
    session: Session,
    records: list[SourceRecord],
    *,
    retrieved_at: datetime,
    archive: RawArchive | None = None,
    collection_run_id: object | None = None,
) -> dict[str, int]:
    """Persist source records and canonical work/version locations without inflating studies."""
    retrieved_at = normalize_utc(retrieved_at)
    items = 0
    works: set[str] = set()
    locations = 0
    archive_failures = 0
    for record in records:
        source_item = session.scalar(
            select(SourceItem).where(
                SourceItem.source_type == record.source_type,
                SourceItem.external_id == record.external_id,
            )
        )
        if source_item is None:
            source_item = SourceItem(
                source_type=record.source_type,
                source_name=record.source_name,
                external_id=record.external_id,
                canonical_url=record.canonical_url,
                title=record.title,
                authors=record.authors,
                published_at=record.published_at,
                retrieved_at=retrieved_at,
                raw_text=record.raw_text,
                raw_metadata=record.raw_metadata,
                content_sha256=content_hash(record.raw_text, record.raw_metadata),
            )
            session.add(source_item)
            session.flush()
            items += 1
        if archive is not None:
            payload = record.raw_metadata.get("source_payload")
            if payload is None:
                content = record.raw_text.encode("utf-8")
                media_type = "text/plain"
            elif isinstance(payload, str) and payload.lstrip().startswith("<"):
                content = payload.encode("utf-8")
                media_type = "application/xml"
            else:
                content = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode()
                media_type = "application/json"
            try:
                archive_receipt(
                    session,
                    archive,
                    content=content,
                    media_type=media_type,
                    provider=record.source_name,
                    canonical_url=record.canonical_url,
                    retrieved_at=retrieved_at,
                    source_native_timestamp=record.published_at,
                    source_item_id=source_item.id,
                    collection_run_id=collection_run_id,
                )
            except (OSError, TypeError, ValueError):
                archive_failures += 1
                source_item.raw_metadata = source_item.raw_metadata | {
                    "archive_status": "ARCHIVE_FAILED"
                }
        identity, doi, arxiv_id = _identity(record)
        work = session.scalar(
            select(ResearchWork).where(ResearchWork.canonical_identity == identity)
        )
        if work is None:
            work = ResearchWork(
                canonical_identity=identity,
                normalized_title=_normalized_title(record.title),
                doi=doi,
                arxiv_id=arxiv_id,
            )
            session.add(work)
            session.flush()
        works.add(str(work.id))
        source = _source(session, record)
        location = session.scalar(
            select(WorkLocation).where(
                WorkLocation.work_id == work.id,
                WorkLocation.source_item_id == source_item.id,
            )
        )
        if location is None:
            session.add(
                WorkLocation(
                    work_id=work.id,
                    source_item_id=source_item.id,
                    source_id=source.id,
                    access_mode=str(
                        record.raw_metadata.get("access_mode") or "METADATA_ONLY"
                    ),
                    version_label=str(record.raw_metadata.get("version") or "") or None,
                    is_primary=record.source_name == "openalex",
                    discovered_at=retrieved_at,
                )
            )
            locations += 1
    session.commit()
    return {
        "discovered": len(records),
        "source_items": items,
        "canonical_works": len(works),
        "locations": locations,
        "archive_failures": archive_failures,
    }
