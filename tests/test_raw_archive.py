from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quant_research_radar.db import Base, RawArtifact, RawArtifactReceipt
from quant_research_radar.raw_archive import RawArchive, archive_receipt


def test_content_addressed_archive_deduplicates_bytes_and_preserves_changed_version(
    tmp_path,
) -> None:
    archive = RawArchive(tmp_path / "raw")
    retrieved_at = datetime(2026, 8, 31, tzinfo=UTC)

    first = archive.store(
        b'{"v":1}', media_type="application/json", retrieved_at=retrieved_at
    )
    duplicate = archive.store(
        b'{"v":1}', media_type="application/json", retrieved_at=retrieved_at
    )
    changed = archive.store(
        b'{"v":2}', media_type="application/json", retrieved_at=retrieved_at
    )

    cross_media = archive.store(
        b'{"v":1}', media_type="text/plain", retrieved_at=retrieved_at
    )
    assert (
        first.content_sha256 == duplicate.content_sha256 == cross_media.content_sha256
    )
    assert first.storage_uri == duplicate.storage_uri == cross_media.storage_uri
    assert first.byte_size == 7
    assert first.storage_uri.startswith("data/raw/objects/")
    assert (
        tmp_path / "raw" / first.storage_uri.removeprefix("data/raw/")
    ).read_bytes() == b'{"v":1}'
    assert changed.content_sha256 != first.content_sha256


def test_archive_rejects_unbounded_or_unsupported_payloads(tmp_path) -> None:
    archive = RawArchive(tmp_path / "raw")
    at = datetime(2026, 8, 31, tzinfo=UTC)
    for content, media_type, expected in (
        (b"x", "application/pdf", "media type"),
        (b"x" * (RawArchive.MAX_OBJECT_BYTES + 1), "text/plain", "bounded"),
    ):
        try:
            archive.store(content, media_type=media_type, retrieved_at=at)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError("invalid payload was archived")


def test_archive_receipts_reference_one_content_object_and_preserve_provenance(
    tmp_path,
) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    archive = RawArchive(tmp_path / "raw")
    at = datetime(2026, 8, 31, tzinfo=UTC)

    archive_receipt(
        session,
        archive,
        content=b"same",
        media_type="application/json",
        provider="openalex",
        canonical_url="https://api.openalex.org/works/W1",
        retrieved_at=at,
        source_native_timestamp=at,
    )
    archive_receipt(
        session,
        archive,
        content=b"same",
        media_type="application/json",
        provider="openalex",
        canonical_url="https://api.openalex.org/works/W1",
        retrieved_at=at,
        source_native_timestamp=at,
    )

    assert session.query(RawArtifact).count() == 1
    assert session.query(RawArtifactReceipt).count() == 2
