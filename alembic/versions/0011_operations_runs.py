"""Add Daily/Weekly autonomous operations run records."""

import sqlalchemy as sa
from alembic import op

revision = "0011_operations_runs"
down_revision = "0010_phase20_event_study"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "daily_runs" not in existing:
        op.create_table(
            "daily_runs",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("logical_date", sa.Date(), nullable=False),
            sa.Column("status", sa.String(20), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True)),
            sa.Column("code_sha", sa.String(64), nullable=False),
            sa.Column("market_status", sa.String(20), nullable=False),
            sa.Column("academic_status", sa.String(20), nullable=False),
            sa.Column("practitioner_status", sa.String(20), nullable=False),
            sa.Column("analysis_status", sa.String(20), nullable=False),
            sa.Column("knowledge_status", sa.String(20), nullable=False),
            sa.Column("audit_status", sa.String(20), nullable=False),
            sa.Column("report_path", sa.String(1000)),
            sa.Column("failure_reasons", sa.JSON(), nullable=False),
            sa.Column("source_health", sa.JSON(), nullable=False),
            sa.Column("llm_summary", sa.JSON(), nullable=False),
        )
        op.create_index("ix_daily_runs_logical_date", "daily_runs", ["logical_date"])
    if "weekly_runs" not in existing:
        op.create_table(
            "weekly_runs",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("week_saturday", sa.Date(), nullable=False),
            sa.Column("status", sa.String(20), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True)),
            sa.Column("code_sha", sa.String(64), nullable=False),
            sa.Column("included_daily_dates", sa.JSON(), nullable=False),
            sa.Column("report_path", sa.String(1000)),
            sa.Column("failure_reasons", sa.JSON(), nullable=False),
            sa.Column("priorities", sa.JSON(), nullable=False),
            sa.Column("low_frequency_fit", sa.JSON(), nullable=False),
        )
        op.create_index(
            "ix_weekly_runs_week_saturday", "weekly_runs", ["week_saturday"]
        )


def downgrade() -> None:
    op.drop_table("weekly_runs")
    op.drop_table("daily_runs")
