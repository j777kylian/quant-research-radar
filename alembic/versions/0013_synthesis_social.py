"""Add versioned TopicBriefs and social editorial packages."""

import sqlalchemy as sa
from alembic import op

revision = "0013_synthesis_social"
down_revision = "0012_publication_delivery"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "topic_briefs" not in tables:
        op.create_table(
            "topic_briefs",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("topic_id", sa.String(200), nullable=False, index=True),
            sa.Column("topic_version", sa.String(30), nullable=False),
            sa.Column("logical_date", sa.Date(), nullable=False, index=True),
            sa.Column("source_run_id", sa.String(64), nullable=False, index=True),
            sa.Column("source_kind", sa.String(20), nullable=False),
            sa.Column("human_title", sa.String(500), nullable=False),
            sa.Column("packet", sa.JSON(), nullable=False),
            sa.Column("brief", sa.JSON(), nullable=False),
            sa.Column("input_packet_hash", sa.String(64), nullable=False),
            sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("model_metadata", sa.JSON(), nullable=False),
            sa.UniqueConstraint(
                "source_run_id",
                "topic_id",
                "topic_version",
                name="uq_topic_brief_version",
            ),
        )
    if "daily_social_packages" not in tables:
        op.create_table(
            "daily_social_packages",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("logical_date", sa.Date(), nullable=False, index=True),
            sa.Column("source_run_id", sa.String(64), nullable=False, index=True),
            sa.Column("topic_brief_ids", sa.JSON(), nullable=False),
            sa.Column("candidates", sa.JSON(), nullable=False),
            sa.Column("selected_candidate_id", sa.String(64)),
            sa.Column("recommendation", sa.String(30), nullable=False),
            sa.Column("selection_reason", sa.Text(), nullable=False),
            sa.Column("content_format", sa.String(30), nullable=False),
            sa.Column("draft_text", sa.Text()),
            sa.Column("source_bundle", sa.JSON(), nullable=False),
            sa.Column("output_path", sa.String(1000)),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("logical_date", name="uq_social_package_date"),
        )
    if "weekly_social_packages" not in tables:
        op.create_table(
            "weekly_social_packages",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("week_saturday", sa.Date(), nullable=False, index=True),
            sa.Column("source_run_id", sa.String(64), nullable=False, index=True),
            sa.Column("candidates", sa.JSON(), nullable=False),
            sa.Column("recommendation", sa.String(30), nullable=False),
            sa.Column("selection_reason", sa.Text(), nullable=False),
            sa.Column("content_format", sa.String(30), nullable=False),
            sa.Column("draft_text", sa.Text()),
            sa.Column("output_path", sa.String(1000)),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("week_saturday", name="uq_weekly_social_package_date"),
        )


def downgrade() -> None:
    for table in ("weekly_social_packages", "daily_social_packages", "topic_briefs"):
        if table in set(sa.inspect(op.get_bind()).get_table_names()):
            op.drop_table(table)
