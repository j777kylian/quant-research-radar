"""Add Phase 1.5C LLM routing observability fields."""

import sqlalchemy as sa
from alembic import op

revision = "0003_phase15c"
down_revision = "0002_phase15b"
branch_labels = None
depends_on = None


_COLUMNS = (
    ("requested_model_tier", sa.String(20), True, None),
    ("actual_model_name", sa.String(100), True, None),
    ("thinking_enabled", sa.Boolean(), True, None),
    ("reasoning_effort", sa.String(20), True, None),
    ("fallback_used", sa.Boolean(), False, False),
    ("status", sa.String(20), False, "RUNNING"),
    ("failure_reason", sa.Text(), True, None),
)


def _columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {column["name"] for column in inspector.get_columns("analysis_runs")}


def upgrade() -> None:
    existing = _columns()
    for name, column_type, nullable, default in _COLUMNS:
        if name not in existing:
            kwargs = {"nullable": nullable}
            if default is not None:
                kwargs["server_default"] = (
                    sa.text(repr(default)) if isinstance(default, str) else sa.text("0")
                )
            op.add_column("analysis_runs", sa.Column(name, column_type, **kwargs))


def downgrade() -> None:
    existing = _columns()
    for name, _column_type, _nullable, _default in reversed(_COLUMNS):
        if name in existing:
            op.drop_column("analysis_runs", name)
