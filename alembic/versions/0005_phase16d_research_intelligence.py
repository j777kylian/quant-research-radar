"""Add Phase 1.6D research-intelligence relational tables."""

import sqlalchemy as sa
from alembic import op

revision = "0005_phase16d_research_intelligence"
down_revision = "0004_phase16a_collection_run_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    # 0001 intentionally uses Base.metadata.create_all(), so fresh databases
    # already contain current mapped tables before this revision is recorded.
    if "evidence_sources" in existing:
        return
    op.create_table(
        "evidence_sources",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("source_name", sa.String(100), nullable=False, unique=True),
        sa.Column("source_class", sa.String(30), nullable=False),
        sa.Column("venue", sa.String(300)),
        sa.Column(
            "peer_review_status",
            sa.String(40),
            nullable=False,
            server_default="UNKNOWN",
        ),
        sa.Column("domain_tags", sa.JSON(), nullable=False),
        sa.Column("access_mode", sa.String(30), nullable=False),
        sa.Column("reliability_prior", sa.String(40), nullable=False),
        sa.Column(
            "provenance_class", sa.String(40), nullable=False, server_default="PUBLIC"
        ),
        sa.Column(
            "adapter_status", sa.String(20), nullable=False, server_default="READY"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "research_works",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("canonical_identity", sa.String(500), nullable=False, unique=True),
        sa.Column("normalized_title", sa.String(1000), nullable=False),
        sa.Column("doi", sa.String(300), unique=True),
        sa.Column("arxiv_id", sa.String(100)),
        sa.Column("ssrn_id", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "work_locations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "work_id", sa.Uuid(), sa.ForeignKey("research_works.id"), nullable=False
        ),
        sa.Column(
            "source_item_id",
            sa.Uuid(),
            sa.ForeignKey("source_items.id"),
            nullable=False,
        ),
        sa.Column(
            "source_id", sa.Uuid(), sa.ForeignKey("evidence_sources.id"), nullable=False
        ),
        sa.Column("access_mode", sa.String(30), nullable=False),
        sa.Column("version_label", sa.String(100)),
        sa.Column(
            "is_primary", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("work_id", "source_item_id"),
    )
    op.create_table(
        "channel_hypotheses",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("mechanism", sa.Text()),
        sa.Column("condition", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("universe", sa.String(500), nullable=False),
        sa.Column("horizon", sa.String(100), nullable=False),
        sa.Column("expected_direction", sa.String(100)),
        sa.Column("required_data", sa.JSON(), nullable=False),
        sa.Column("falsification_criterion", sa.Text(), nullable=False),
        sa.Column("maturity", sa.String(40), nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="DISCOVERED"),
        sa.Column("fingerprint", sa.String(1000), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "evidence_links",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "channel_hypothesis_id",
            sa.Uuid(),
            sa.ForeignKey("channel_hypotheses.id"),
            nullable=False,
        ),
        sa.Column(
            "source_item_id",
            sa.Uuid(),
            sa.ForeignKey("source_items.id"),
            nullable=False,
        ),
        sa.Column("relation", sa.String(20), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("independence_key", sa.String(500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("channel_hypothesis_id", "source_item_id", "relation"),
    )
    op.create_table(
        "unified_hypotheses",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("fingerprint", sa.String(1000), nullable=False, unique=True),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("maturity", sa.String(40), nullable=False),
        sa.Column("supporting_channels", sa.JSON(), nullable=False),
        sa.Column(
            "independent_evidence_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "unified_hypothesis_members",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "unified_hypothesis_id",
            sa.Uuid(),
            sa.ForeignKey("unified_hypotheses.id"),
            nullable=False,
        ),
        sa.Column(
            "channel_hypothesis_id",
            sa.Uuid(),
            sa.ForeignKey("channel_hypotheses.id"),
            nullable=False,
        ),
        sa.UniqueConstraint("unified_hypothesis_id", "channel_hypothesis_id"),
    )
    op.create_table(
        "user_feedback",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "unified_hypothesis_id",
            sa.Uuid(),
            sa.ForeignKey("unified_hypotheses.id"),
            nullable=False,
        ),
        sa.Column("preference", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for table, column in (
        ("work_locations", "work_id"),
        ("work_locations", "source_item_id"),
        ("work_locations", "source_id"),
        ("channel_hypotheses", "channel"),
        ("channel_hypotheses", "fingerprint"),
        ("evidence_links", "channel_hypothesis_id"),
        ("evidence_links", "source_item_id"),
        ("evidence_links", "independence_key"),
        ("unified_hypothesis_members", "unified_hypothesis_id"),
        ("unified_hypothesis_members", "channel_hypothesis_id"),
        ("user_feedback", "unified_hypothesis_id"),
    ):
        op.create_index(f"ix_{table}_{column}", table, [column])


def downgrade() -> None:
    for table in (
        "user_feedback",
        "unified_hypothesis_members",
        "unified_hypotheses",
        "evidence_links",
        "channel_hypotheses",
        "work_locations",
        "research_works",
        "evidence_sources",
    ):
        op.drop_table(table)
