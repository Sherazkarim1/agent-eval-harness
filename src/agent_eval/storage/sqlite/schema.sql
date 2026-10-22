-- Schema for the SQLite evaluation store.
--
-- Design notes:
--   * `runs` holds the header plus a JSON snapshot of the aggregate. The
--     aggregate is read as a whole far more often than it is queried by field,
--     so a JSON column keeps the schema stable as metrics are added.
--   * Queryable/filterable dimensions (dataset, version, mode, config hash,
--     timestamps) are promoted to real columns and indexed.
--   * `case_results` stores per-case payload as JSON for the same reason: its
--     shape is owned by the domain models, not by SQL.
--   * Foreign keys are declared and enforced so a case row can never outlive
--     its run.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    dataset_name    TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    execution_mode  TEXT NOT NULL,
    config_hash     TEXT,
    agent_name      TEXT NOT NULL,
    agent_version   TEXT NOT NULL,
    agent_framework TEXT NOT NULL,
    judge_name      TEXT NOT NULL,
    judge_version   TEXT NOT NULL,
    judge_model     TEXT,
    judge_provider  TEXT,
    started_at      TEXT NOT NULL,
    duration_ms     REAL NOT NULL DEFAULT 0.0,
    git_commit      TEXT,
    branch          TEXT,
    aggregate_json  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_dataset
    ON runs (dataset_name, dataset_version, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_mode
    ON runs (execution_mode, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_config_hash
    ON runs (dataset_name, config_hash);

CREATE TABLE IF NOT EXISTS case_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    case_id        TEXT NOT NULL,
    sequence       INTEGER NOT NULL,
    status         TEXT NOT NULL,
    passed         INTEGER NOT NULL,
    latency_ms     REAL NOT NULL DEFAULT 0.0,
    cost_usd       REAL NOT NULL DEFAULT 0.0,
    started_at     TEXT NOT NULL,
    payload_json   TEXT NOT NULL,
    UNIQUE (run_id, case_id)
);

CREATE INDEX IF NOT EXISTS idx_case_results_run
    ON case_results (run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_case_results_failed
    ON case_results (run_id, passed);

CREATE TABLE IF NOT EXISTS baselines (
    name             TEXT PRIMARY KEY,
    run_id           TEXT,
    dataset_name     TEXT NOT NULL,
    dataset_version  TEXT NOT NULL,
    agent_version    TEXT,
    config_hash      TEXT,
    execution_mode   TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    aggregate_json   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_baselines_created
    ON baselines (created_at DESC);

CREATE TABLE IF NOT EXISTS datasets (
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    case_count  INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (name, version)
);
