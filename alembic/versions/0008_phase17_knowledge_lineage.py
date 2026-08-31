"""Bind Phase 1.7 exact archival lineage and occurrence retrieval indexes."""

import sqlalchemy as sa
from alembic import op

revision = "0008_phase17_knowledge_lineage"
down_revision = "0007_phase16d_raw_evidence_archive"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    # 0001 uses current Base.metadata.create_all(); fresh DBs can already hold
    # mapped columns before this migration is recorded.
    if "raw_artifact_receipts" in tables:
        columns = {
            column["name"] for column in inspector.get_columns("raw_artifact_receipts")
        }
        if "collection_run_id" not in columns:
            op.add_column(
                "raw_artifact_receipts",
                sa.Column("collection_run_id", sa.Uuid(), nullable=True),
            )
        indexes = {
            index["name"] for index in inspector.get_indexes("raw_artifact_receipts")
        }
        if "ix_raw_artifact_receipts_collection_run_id" not in indexes:
            op.create_index(
                "ix_raw_artifact_receipts_collection_run_id",
                "raw_artifact_receipts",
                ["collection_run_id"],
            )
    if "evidence_links" in tables:
        columns = {column["name"] for column in inspector.get_columns("evidence_links")}
        if "raw_artifact_receipt_id" not in columns:
            op.add_column(
                "evidence_links",
                sa.Column("raw_artifact_receipt_id", sa.Uuid(), nullable=True),
            )
        indexes = {index["name"] for index in inspector.get_indexes("evidence_links")}
        if "ix_evidence_links_raw_artifact_receipt_id" not in indexes:
            op.create_index(
                "ix_evidence_links_raw_artifact_receipt_id",
                "evidence_links",
                ["raw_artifact_receipt_id"],
            )
    if "channel_hypotheses" in tables:
        indexes = {
            index["name"]: index
            for index in inspector.get_indexes("channel_hypotheses")
        }
        occurrence = indexes.get("ix_channel_hypotheses_occurrence")
        if occurrence and not occurrence["unique"]:
            op.drop_index(
                "ix_channel_hypotheses_occurrence", table_name="channel_hypotheses"
            )
        if not occurrence or not occurrence["unique"]:
            op.create_index(
                "ix_channel_hypotheses_occurrence",
                "channel_hypotheses",
                [
                    "channel",
                    "fingerprint",
                    "analysis_mode",
                    "availability_basis",
                    "as_of",
                ],
                unique=True,
            )


def downgrade() -> None:
    # Published migrations are not used as a data-repair mechanism.
    pass
