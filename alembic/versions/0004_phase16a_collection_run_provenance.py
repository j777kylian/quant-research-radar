"""Add Phase 1.6A CollectionRun provenance fields."""

import sqlalchemy as sa
from alembic import op

revision = "0004_phase16a_collection_run_provenance"
down_revision = "0003_phase15c"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("phase16a_run_id", sa.String(100)),
    ("requested_start", sa.DateTime(timezone=True)),
    ("requested_end", sa.DateTime(timezone=True)),
    ("code_sha", sa.String(64)),
    ("diagnostics", sa.JSON()),
)


def _columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {column["name"] for column in inspector.get_columns("collection_runs")}


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.alter_column(
            "alembic_version",
            "version_num",
            existing_type=sa.String(32),
            type_=sa.String(128),
        )
    existing = _columns()
    for name, column_type in _COLUMNS:
        if name not in existing:
            op.add_column(
                "collection_runs", sa.Column(name, column_type, nullable=True)
            )


def downgrade() -> None:
    existing = _columns()
    for name, _column_type in reversed(_COLUMNS):
        if name in existing:
            op.drop_column("collection_runs", name)
