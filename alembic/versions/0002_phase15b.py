"""Add Phase 1.5B provenance and observability fields."""

import sqlalchemy as sa
from alembic import op

revision = "0002_phase15b"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade() -> None:
    if "evidence_excerpt" not in _columns("claims"):
        op.add_column(
            "claims",
            sa.Column("evidence_excerpt", sa.Text(), nullable=False, server_default=""),
        )
    if "observation_kind" not in _columns("market_observations"):
        op.add_column(
            "market_observations",
            sa.Column(
                "observation_kind",
                sa.String(30),
                nullable=False,
                server_default="snapshot",
            ),
        )
    if "status" not in _columns("collection_runs"):
        op.add_column(
            "collection_runs",
            sa.Column(
                "status", sa.String(20), nullable=False, server_default="SUCCESS"
            ),
        )
    if "role" not in _columns("analysis_runs"):
        op.add_column(
            "analysis_runs",
            sa.Column("role", sa.String(20), nullable=False, server_default="ANALYST"),
        )
    if "prompt_version" not in _columns("analysis_runs"):
        op.add_column(
            "analysis_runs",
            sa.Column(
                "prompt_version", sa.String(30), nullable=False, server_default="1"
            ),
        )


def downgrade() -> None:
    for table, column in (
        ("analysis_runs", "prompt_version"),
        ("analysis_runs", "role"),
        ("collection_runs", "status"),
        ("market_observations", "observation_kind"),
        ("claims", "evidence_excerpt"),
    ):
        if column in _columns(table):
            op.drop_column(table, column)
