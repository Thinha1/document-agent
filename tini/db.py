"""One SQLite file (state.db) holds everything Tini remembers and does.

This mirrors the Hermes approach on the whiteboard: SQLite + FTS5, no server.
Open it yourself anytime:  sqlite3 .tini/state.db '.tables'
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
-- Semantic memory: durable facts about you, your people, your projects.
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,         -- who/what the fact is about, e.g. 'alex'
    content TEXT NOT NULL,         -- the fact itself
    source TEXT DEFAULT 'user',    -- 'user' (told directly) or 'consolidation'
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    subject, content, content=facts, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, subject, content) VALUES (new.id, new.subject, new.content);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, subject, content) VALUES ('delete', old.id, old.subject, old.content);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, subject, content) VALUES ('delete', old.id, old.subject, old.content);
    INSERT INTO facts_fts(rowid, subject, content) VALUES (new.id, new.subject, new.content);
END;

-- Episodic memory: dated things that happened (past chats, distilled).
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    happened_at TEXT NOT NULL,     -- ISO 8601 date of the episode
    summary TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    summary, content=episodes, content_rowid=id
);
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, summary) VALUES (new.id, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, summary) VALUES ('delete', old.id, old.summary);
END;

-- Raw chat log ("save the messages" box). Consolidation reads from here.
-- session_id tags each row with which conversation it belongs to, so the
-- dashboard can offer "New chat" and switch between past sessions (like a
-- chat app). Everything shares this one table — sessions are just a label.
CREATE TABLE IF NOT EXISTS chat_log (
    id INTEGER PRIMARY KEY,
    role TEXT NOT NULL,            -- 'user' | 'assistant'
    content TEXT NOT NULL,
    consolidated INTEGER DEFAULT 0,
    session_id TEXT DEFAULT 'default',
    created_at TEXT DEFAULT (datetime('now'))
);

-- Internal knowledge base. Files remain outside SQLite; this stores their
-- provenance plus query-sized text chunks. workspace_id is present from day
-- one so a future Postgres adapter can enforce the same isolation with RLS.
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL DEFAULT 'default',
    source_uri TEXT NOT NULL,
    title TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'indexing',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(workspace_id, source_uri)
);
CREATE INDEX IF NOT EXISTS documents_workspace_idx
    ON documents(workspace_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL DEFAULT 'default',
    ordinal INTEGER NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS document_chunks_document_idx
    ON document_chunks(document_id, ordinal);
CREATE INDEX IF NOT EXISTS document_chunks_workspace_idx
    ON document_chunks(workspace_id);
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    title, text, content=document_chunks, content_rowid=rowid
);
CREATE TRIGGER IF NOT EXISTS document_chunks_ai AFTER INSERT ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(rowid, title, text) VALUES (new.rowid, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS document_chunks_ad AFTER DELETE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, title, text)
    VALUES ('delete', old.rowid, old.title, old.text);
END;
CREATE TRIGGER IF NOT EXISTS document_chunks_au AFTER UPDATE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, title, text)
    VALUES ('delete', old.rowid, old.title, old.text);
    INSERT INTO document_chunks_fts(rowid, title, text) VALUES (new.rowid, new.title, new.text);
END;
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive, idempotent column upgrades for databases created before a
    column existed. SQLite has no 'ADD COLUMN IF NOT EXISTS', so we check."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chat_log)").fetchall()}
    if "session_id" not in cols:
        conn.execute("ALTER TABLE chat_log ADD COLUMN session_id TEXT DEFAULT 'default'")
        conn.commit()
    if "source" not in cols:
        # which gateway a message came in through (cli / voice / telegram / dashboard)
        conn.execute("ALTER TABLE chat_log ADD COLUMN source TEXT DEFAULT 'cli'")
        conn.commit()
    if "meta" not in cols:
        # per-turn telemetry as JSON on the assistant row (gate decision,
        # latency, iterations, tools) — so reopening a thread still shows how
        # each answer was produced, not just the plain text.
        conn.execute("ALTER TABLE chat_log ADD COLUMN meta TEXT")
        conn.commit()


def connect(home: Path, check_same_thread: bool = True) -> sqlite3.Connection:
    # check_same_thread=False lets the dashboard's threaded HTTP server reuse
    # one agent connection across worker threads (guarded by a lock). busy_timeout
    # avoids "database is locked" when the dashboard reads while a chat writes.
    conn = sqlite3.connect(home / "state.db", check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn
