from datetime import UTC, datetime

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from quant_research_radar.cli import migrate_database
from quant_research_radar.db import CollectionRun


def test_legacy_database_is_stamped_then_upgraded(tmp_path):
    path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE source_items (id CHAR(32) PRIMARY KEY NOT NULL, source_type VARCHAR(40) NOT NULL, source_name VARCHAR(100) NOT NULL, external_id VARCHAR(255) NOT NULL, canonical_url VARCHAR(1000), title VARCHAR(1000) NOT NULL, authors JSON NOT NULL, published_at DATETIME, retrieved_at DATETIME NOT NULL, raw_text TEXT NOT NULL, raw_metadata JSON NOT NULL, content_sha256 VARCHAR(64) NOT NULL, ingestion_version VARCHAR(32) NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE collection_runs (id CHAR(32) PRIMARY KEY NOT NULL, "
            "source VARCHAR(50) NOT NULL, started_at DATETIME NOT NULL, "
            "ended_at DATETIME, requested INTEGER NOT NULL, retrieved INTEGER NOT NULL, "
            "inserted INTEGER NOT NULL, updated INTEGER NOT NULL, "
            "skipped_duplicates INTEGER NOT NULL, failed INTEGER NOT NULL, "
            "status VARCHAR(20) NOT NULL, error_reason TEXT)"
        )
        connection.execute(
            text(
                "INSERT INTO collection_runs VALUES "
                "('old', 'hyperliquid', '2026-01-01', NULL, 1, 1, 1, 0, 0, 0, 'SUCCESS', NULL)"
            )
        )
    url = f"sqlite:///{path}"

    migrate_database(url)
    migrate_database(url)

    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT count(*) FROM collection_runs")).scalar()
            == 1
        )
    columns = {
        column["name"] for column in inspect(engine).get_columns("collection_runs")
    }
    assert {
        "phase16a_run_id",
        "requested_start",
        "requested_end",
        "code_sha",
        "diagnostics",
    } <= columns
    assert inspect(engine).has_table("alembic_version")
    with Session(engine) as session:
        run = CollectionRun(
            source="hyperliquid",
            phase16a_run_id="new",
            requested_start=datetime.now(UTC),
            requested_end=datetime.now(UTC),
            code_sha="sha",
            diagnostics={},
            status="SUCCESS",
        )
        session.add(run)
        session.commit()
        assert session.get(CollectionRun, run.id) is not None


def test_fresh_database_reaches_head(tmp_path):
    path = tmp_path / "fresh.db"
    url = f"sqlite:///{path}"
    migrate_database(url)
    migrate_database(url)
    engine = create_engine(url)
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
            == "0006_phase16d_candidate_provenance"
        )
    assert "requested_start" in {
        column["name"] for column in inspect(engine).get_columns("collection_runs")
    }


def test_phase16d_migration_adds_research_intelligence_tables(tmp_path):
    path = tmp_path / "phase16d.db"
    url = f"sqlite:///{path}"

    migrate_database(url)
    engine = create_engine(url)

    assert {
        "evidence_sources",
        "research_works",
        "work_locations",
        "channel_hypotheses",
        "evidence_links",
        "unified_hypotheses",
        "unified_hypothesis_members",
        "user_feedback",
    } <= set(inspect(engine).get_table_names())
    assert {"analysis_mode", "availability_basis", "as_of"} <= {
        column["name"] for column in inspect(engine).get_columns("channel_hypotheses")
    }
    with engine.connect() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
            == "0006_phase16d_candidate_provenance"
        )
