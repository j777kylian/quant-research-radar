"""Add publication and delivery domain (read-only toward research conclusions)."""

from collections.abc import Callable

import sqlalchemy as sa
from alembic import op

revision = "0012_publication_delivery"
down_revision = "0011_operations_runs"
branch_labels = None
depends_on = None

TABLES = (
    "publication_candidates",
    "publication_drafts",
    "publications",
    "delivery_records",
)


def _create_if_missing(table: str, create: Callable[[], None]) -> None:
    if table not in set(sa.inspect(op.get_bind()).get_table_names()):
        create()


def upgrade() -> None:
    def candidates() -> None:
        op.create_table(
            "publication_candidates",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("source_run_id", sa.String(64), index=True),
            sa.Column("source_kind", sa.String(20)),
            sa.Column("category", sa.String(40)),
            sa.Column("title", sa.String(500)),
            sa.Column("summary", sa.Text()),
            sa.Column("evidence", sa.JSON()),
            sa.Column("publication_value", sa.JSON()),
            sa.Column("created_at", sa.DateTime(timezone=True)),
            sa.UniqueConstraint(
                "source_run_id", "category", "title", name="uq_candidate_identity"
            ),
        )

    def drafts() -> None:
        op.create_table(
            "publication_drafts",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("candidate_id", sa.Uuid(), index=True),
            sa.Column("policy", sa.String(30)),
            sa.Column("language", sa.String(20)),
            sa.Column("text", sa.Text()),
            sa.Column("claims", sa.JSON()),
            sa.Column("source_bundle", sa.JSON()),
            sa.Column("visual_ids", sa.JSON()),
            sa.Column("idempotence_key", sa.String(64), unique=True, index=True),
            sa.Column("created_at", sa.DateTime(timezone=True)),
        )

    def publications() -> None:
        op.create_table(
            "publications",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("draft_id", sa.Uuid(), index=True),
            sa.Column("platform", sa.String(20)),
            sa.Column("status", sa.String(30)),
            sa.Column("external_post_id", sa.String(100)),
            sa.Column("failure_reason", sa.Text()),
            sa.Column("retry_count", sa.Integer()),
            sa.Column("published_at", sa.DateTime(timezone=True)),
            sa.Column("created_at", sa.DateTime(timezone=True)),
        )

    def deliveries() -> None:
        op.create_table(
            "delivery_records",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("channel", sa.String(20)),
            sa.Column("run_kind", sa.String(20)),
            sa.Column("run_date", sa.String(20)),
            sa.Column("status", sa.String(20)),
            sa.Column("idempotence_key", sa.String(64), unique=True, index=True),
            sa.Column("failure_reason", sa.Text()),
            sa.Column("retry_count", sa.Integer()),
            sa.Column("sent_at", sa.DateTime(timezone=True)),
            sa.Column("created_at", sa.DateTime(timezone=True)),
        )

    _create_if_missing("publication_candidates", candidates)
    _create_if_missing("publication_drafts", drafts)
    _create_if_missing("publications", publications)
    _create_if_missing("delivery_records", deliveries)


def downgrade() -> None:
    for table in reversed(TABLES):
        if table in set(sa.inspect(op.get_bind()).get_table_names()):
            op.drop_table(table)
