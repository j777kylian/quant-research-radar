"""Persist Phase 1.6D candidate availability provenance."""

import sqlalchemy as sa
from alembic import op

revision = "0006_phase16d_candidate_provenance"
down_revision = "0005_phase16d_research_intelligence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("channel_hypotheses")
    }
    if "analysis_mode" in columns:
        return
    with op.batch_alter_table("channel_hypotheses") as batch:
        batch.add_column(
            sa.Column(
                "analysis_mode",
                sa.String(50),
                nullable=False,
                server_default="PRODUCTION_LIVE",
            )
        )
        batch.add_column(
            sa.Column(
                "availability_basis",
                sa.String(60),
                nullable=False,
                server_default="RECEIPT_TIME",
            )
        )
        batch.add_column(sa.Column("as_of", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("channel_hypotheses") as batch:
        batch.drop_column("as_of")
        batch.drop_column("availability_basis")
        batch.drop_column("analysis_mode")
