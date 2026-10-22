"""Schema versioning for the SQLite store.

The schema is applied from ``schema.sql`` and its version recorded in
``schema_version``. Migrations are ordered and idempotent: applying them to a
fresh database and to an existing one both converge on the same state.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).with_name("schema.sql")

#: Bumped whenever ``schema.sql`` changes in a way that needs a migration.
TARGET_VERSION = 1


def current_version(connection: sqlite3.Connection) -> int:
    """Return the applied schema version, or ``0`` for an empty database."""
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("schema_version",),
    ).fetchone()
    if row is None:
        return 0
    row = connection.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def apply_migrations(connection: sqlite3.Connection) -> int:
    """Create or upgrade the schema, returning the resulting version.

    Args:
        connection: An open SQLite connection.

    Returns:
        The schema version in effect after this call.
    """
    version = current_version(connection)
    if version >= TARGET_VERSION:
        return version

    logger.debug("Applying schema v%s (was v%s)", TARGET_VERSION, version)
    connection.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    connection.execute(
        "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
        (TARGET_VERSION, datetime.now().astimezone().isoformat()),
    )
    connection.commit()
    return TARGET_VERSION
