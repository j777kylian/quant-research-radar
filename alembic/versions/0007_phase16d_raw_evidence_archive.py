"""Add immutable Phase 1.6D raw evidence archive metadata."""

import sqlalchemy as sa
from alembic import op

revision = "0007_phase16d_raw_evidence_archive"
down_revision = "0006_phase16d_candidate_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "raw_artifacts" in existing:
        return
    op.create_table(
        "raw_artifacts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("content_sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("storage_uri", sa.String(1000), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_raw_artifacts_content_sha256", "raw_artifacts", ["content_sha256"]
    )
    op.create_table(
        "raw_artifact_receipts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "raw_artifact_id",
            sa.Uuid(),
            sa.ForeignKey("raw_artifacts.id"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("canonical_url", sa.String(1000)),
        sa.Column("source_native_timestamp", sa.DateTime(timezone=True)),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("parser_version", sa.String(50), nullable=False, server_default="1"),
        sa.Column(
            "archive_status", sa.String(30), nullable=False, server_default="ARCHIVED"
        ),
        sa.Column(
            "analysis_mode",
            sa.String(50),
            nullable=False,
            server_default="PRODUCTION_LIVE",
        ),
        sa.Column("source_item_id", sa.Uuid(), sa.ForeignKey("source_items.id")),
        sa.Column(
            "market_observation_id", sa.Uuid(), sa.ForeignKey("market_observations.id")
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in (
        "raw_artifact_id",
        "provider",
        "source_item_id",
        "market_observation_id",
    ):
        op.create_index(
            f"ix_raw_artifact_receipts_{column}", "raw_artifact_receipts", [column]
        )


def downgrade() -> None:
    op.drop_table("raw_artifact_receipts")
    op.drop_table("raw_artifacts")
