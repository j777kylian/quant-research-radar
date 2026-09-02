"""Persist Phase 2.0 Event Study immutable specs, runs, and results."""

import sqlalchemy as sa
from alembic import op

revision = "0010_phase20_event_study"
down_revision = "0009_phase17_required_occurrence_time"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    # 0001 intentionally calls current Base.metadata.create_all for fresh SQLite DBs.
    # On that legacy path these tables already exist; old DBs need the explicit DDL below.
    if "event_study_specs" in existing:
        return
    op.create_table(
        "event_study_specs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("hypothesis_id", sa.String(100), nullable=False),
        sa.Column("hypothesis_family_id", sa.String(200), nullable=False),
        sa.Column("spec_version", sa.String(30), nullable=False),
        sa.Column("spec_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("immutable_spec", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_event_study_specs_hypothesis_id", "event_study_specs", ["hypothesis_id"]
    )
    op.create_index(
        "ix_event_study_specs_family", "event_study_specs", ["hypothesis_family_id"]
    )
    op.create_table(
        "event_study_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "spec_id",
            sa.String(64),
            sa.ForeignKey("event_study_specs.id"),
            nullable=False,
        ),
        sa.Column("hypothesis_id", sa.String(100), nullable=False),
        sa.Column("analysis_mode", sa.String(60), nullable=False),
        sa.Column("availability_basis", sa.String(80), nullable=False),
        sa.Column("real_receipt_pit", sa.String(30), nullable=False),
        sa.Column("data_lineage", sa.JSON(), nullable=False),
        sa.Column("code_sha", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_event_study_runs_spec_id", "event_study_runs", ["spec_id"])
    op.create_index(
        "ix_event_study_runs_hypothesis_id", "event_study_runs", ["hypothesis_id"]
    )
    op.create_table(
        "event_study_results",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("event_study_runs.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "spec_id",
            sa.String(64),
            sa.ForeignKey("event_study_specs.id"),
            nullable=False,
        ),
        sa.Column("hypothesis_id", sa.String(100), nullable=False),
        sa.Column("hypothesis_family_id", sa.String(200), nullable=False),
        sa.Column("disposition", sa.String(30), nullable=False),
        sa.Column("treatment_count", sa.Integer(), nullable=False),
        sa.Column("baseline_count", sa.Integer(), nullable=False),
        sa.Column("regime_count", sa.Integer(), nullable=False),
        sa.Column("effects", sa.JSON(), nullable=False),
        sa.Column("robustness", sa.JSON(), nullable=False),
        sa.Column("methodology_critic", sa.JSON(), nullable=False),
        sa.Column("artifact_uri", sa.String(1000), nullable=False),
        sa.Column("code_sha", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for name, columns in (
        ("ix_event_study_results_run", ["run_id"]),
        ("ix_event_study_results_spec", ["spec_id"]),
        ("ix_event_study_results_hypothesis", ["hypothesis_id"]),
        ("ix_event_study_results_family", ["hypothesis_family_id"]),
        ("ix_event_study_results_disposition", ["disposition"]),
    ):
        op.create_index(name, "event_study_results", columns)


def downgrade() -> None:
    op.drop_table("event_study_results")
    op.drop_table("event_study_runs")
    op.drop_table("event_study_specs")
