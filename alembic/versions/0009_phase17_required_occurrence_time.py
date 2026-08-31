"""Require complete occurrence identities for Phase 1.7."""

import sqlalchemy as sa
from alembic import op

revision = "0009_phase17_required_occurrence_time"
down_revision = "0008_phase17_knowledge_lineage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    missing = bind.execute(
        sa.text("SELECT COUNT(*) FROM channel_hypotheses WHERE as_of IS NULL")
    ).scalar_one()
    if missing:
        raise RuntimeError("cannot upgrade: channel_hypotheses contains NULL as_of")
    with op.batch_alter_table("channel_hypotheses") as batch:
        batch.alter_column(
            "as_of", existing_type=sa.DateTime(timezone=True), nullable=False
        )


def downgrade() -> None:
    with op.batch_alter_table("channel_hypotheses") as batch:
        batch.alter_column(
            "as_of", existing_type=sa.DateTime(timezone=True), nullable=True
        )
