"""The document catalog: what each indexed document IS, kept apart from its chunks.

Chunks answer "what does the text say"; the catalog answers "what kind of
document is this, which year, from whom, is it still current, and did a person
confirm it". It lives in TINI_HOME/state.db (table `document_catalog`), keyed by
(workspace, source_uri), so it works the same under SQLite and Qdrant and
survives Re-index.

Two rules keep it cheap and trustworthy:

* Editing the catalog is a metadata write. It never touches chunks, vectors or
  OCR, so correcting a type or a year costs nothing and needs no re-ingest.
* A field a person has edited is LOCKED: a later re-ingest (even of a changed
  file) refreshes the profile and the auto fields, but never overwrites the
  locked ones. Unchanged content (same hash) is not re-classified at all.

Classification here is rule-based (title, filename, opening text). It reports a
confidence and the text that triggered it; anything below REVIEW_BELOW is
flagged for a human instead of being trusted.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DOC_TYPES = {
    "report": "Báo cáo",
    "governance": "Quản trị công ty",
    "policy": "Quy định & chính sách nội bộ",
    "supplier": "Yêu cầu nhà cung cấp",
    "other": "Khác / chưa xác định",
}
STATUSES = {
    "current": "hiện hành",
    "superseded": "đã bị thay thế",
    "needs_check": "cần xác nhận",
    "excluded": "loại bỏ",
}
REVIEW_BELOW = 0.8
EDITABLE = ("doc_type", "subtype", "issuer", "period", "status", "summary")
_LIMITS = {"subtype": 80, "issuer": 80, "period": 40, "summary": 1000}

# (pattern, doc_type, subtype, confidence). Matched against the title, the
# filename and the first ~3,000 characters (NFKC-normalised, case-folded).
_RULES: list[tuple[str, str, str, float]] = [
    (r"有価証券報告書", "report", "tài chính pháp định", 0.95),
    (r"統合報告書|integrated report", "report", "thường niên tích hợp", 0.9),
    (
        r"corporate governance report|コーポレート・?ガバナンス(に関する)?報告書",
        "governance", "báo cáo", 0.9,
    ),
    (
        r"basic polic(?:y|ies) on corporate governance|コーポレート・?ガバナンス.{0,6}基本方針",
        "governance", "chính sách", 0.85,
    ),
    (r"code of conduct|行動規範|behavior guideline", "policy", "quy tắc ứng xử", 0.85),
    (r"human rights policy|人権方針", "policy", "chính sách tập đoàn", 0.85),
    (
        r"supplier.{0,40}(?:guideline|sustainab)|sustainab.{0,40}guideline|サプライヤー",
        "supplier", "hướng dẫn nhà cung cấp", 0.85,
    ),
    (r"quality assurance manual|qa manual|品質保証", "supplier", "sổ tay chất lượng", 0.85),
]


def _norm(text: str) -> str:
    """NFKC (full-width digits/letters -> ASCII), case-folded, whitespace
    collapsed — PDFs break lines in the middle of a title."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold())


def _fiscal_year(head: str) -> tuple[str, str]:
    """(period, evidence) from the opening text, or ("", "")."""
    m = re.search(r"事業年度[^\n]{0,20}?自\s*(\d{4})年\s*(\d{1,2})月", head)
    if m:
        return f"FY{m.group(1)}", m.group(0)[:60]
    m = re.search(r"(\d{4})年\s*(\d{1,2})月期", head)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        return f"FY{year - 1 if month <= 3 else year}", m.group(0)
    m = re.search(r"last update[:\s]+[a-z]+ \d{1,2},? (\d{4})", head)
    if m:
        return m.group(1), m.group(0)
    m = re.search(r"rev\.?\s*[a-z0-9]+\s+(?:[a-z]+\s+)?(\d{4})", head)
    if m:
        return m.group(1), m.group(0)
    return "", ""


def _issuer(head: str) -> str:
    if re.search(r"denso ten america", head):
        return "DENSO TEN America"
    if re.search(r"denso ten|デンソーテン", head):
        return "DENSO TEN"
    if re.search(r"denso|デンソー", head):
        return "DENSO"
    return ""


def classify(title: str, text: str) -> dict[str, Any]:
    """Type, subtype, issuer and period with a confidence and the evidence."""
    head = _norm(text[:3000])
    name = _norm(re.sub(r"[_.\-]+", " ", title))
    if name in {"readme", "readme md"}:
        return {"doc_type": "other", "subtype": "tệp mô tả", "issuer": "", "period": "",
                "status": "excluded", "confidence": 0.9, "evidence": "tên tệp README"}
    # A document names itself near the top ("統合報告書2025", "Corporate
    # Governance Report"), while other document types only get MENTIONED later
    # ("see the Securities Report"). So the title/filename wins outright, and
    # in the opening text the EARLIEST match wins, not the most confident one
    # (picking by confidence mislabelled the Integrated Report as a securities
    # filing and the governance report as an integrated report). A match that
    # only appears after the first 400 characters is trusted less.
    candidates = []
    for pattern, doc_type, subtype, confidence in _RULES:
        m = re.search(pattern, name)
        if m:
            candidates.append((0, m.start(), min(confidence + 0.05, 0.99), doc_type, subtype,
                               f"tiêu đề: “{m.group(0)}”"))
        m = re.search(pattern, head)
        if m:
            late = m.start() > 400
            candidates.append((1, m.start(), confidence - (0.1 if late else 0.0), doc_type, subtype,
                               f"trang đầu: “{m.group(0)}”"))
    best = min(candidates, key=lambda c: (c[0], c[1])) if candidates else None
    period, period_evidence = _fiscal_year(head)
    result = {"issuer": _issuer(name + " " + head), "period": period, "status": "current"}
    if best is None:
        return {**result, "doc_type": "other", "subtype": "chưa xác định",
                "confidence": 0.2, "evidence": "không khớp quy tắc nào"}
    evidence = best[5] + (f"; kỳ: “{period_evidence}”" if period_evidence else "")
    return {**result, "doc_type": best[3], "subtype": best[4], "confidence": round(best[2], 2),
            "evidence": evidence}


def profile_document(text: str, metadata: dict[str, Any], media_type: str, size: int) -> dict:
    """How the document is BUILT (drives how it should be processed), as opposed
    to what it is about. Cheap and deterministic: no model call."""
    letters = [c for c in text if c.isalpha()]
    cjk = sum("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in letters)
    vi = sum(c in "ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ" for c in text.casefold())
    total = max(len(letters), 1)
    languages = {"ja": round(cjk / total, 2), "vi": round(vi / total, 2)}
    languages["en"] = round(max(0.0, 1 - languages["ja"] - languages["vi"]), 2)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    numeric = sum(len(re.findall(r"\d[\d,.]*", ln)) >= 3 for ln in lines)
    density = round(numeric / max(len(lines), 1), 2)
    return {
        "extension": metadata.get("extension", ""),
        "media_type": media_type,
        "size_bytes": size,
        "pages": metadata.get("page_count"),
        "characters": len(text),
        "scanned": bool(metadata.get("ocr_applied") or metadata.get("ocr_skipped")),
        # Vietnamese is mostly plain Latin letters, so "largest share" would call it
        # English; its tone-marked letters are the tell (English text has ~none).
        "language": "ja" if languages["ja"] > 0.2 else "vi" if languages["vi"] > 0.05 else "en",
        "languages": languages,
        "table_density": density,
        # 0.10 separates the one real table-heavy filing in the DENSO set (0.13)
        # from every narrative document (<= 0.06); calibrated on 9 files, so treat
        # it as a starting point, not a measured boundary.
        "table_heavy": density > 0.10,
    }


# --- storage ------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_catalog (
    workspace_id TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    doc_type TEXT NOT NULL DEFAULT 'other',
    subtype TEXT NOT NULL DEFAULT '',
    issuer TEXT NOT NULL DEFAULT '',
    period TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'current',
    summary TEXT NOT NULL DEFAULT '',
    profile TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0,
    evidence TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    locked TEXT NOT NULL DEFAULT '[]',
    needs_review INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (workspace_id, source_uri)
)"""
_AUTO = ("doc_type", "subtype", "issuer", "period", "status")


def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row(row: sqlite3.Row) -> dict[str, Any]:
    entry = dict(row)
    entry["profile"] = json.loads(entry["profile"] or "{}")
    entry["locked"] = json.loads(entry["locked"] or "[]")
    entry["needs_review"] = bool(entry["needs_review"])
    return entry


def record_ingest(conn: sqlite3.Connection, entry: dict[str, Any]) -> dict[str, Any]:
    """Store what an ingest learned about a document, respecting person edits."""
    _ensure(conn)
    conn.row_factory = sqlite3.Row
    workspace, uri = entry["workspace"], entry["source_uri"]
    existing = conn.execute(
        "SELECT * FROM document_catalog WHERE workspace_id=? AND source_uri=?", (workspace, uri)
    ).fetchone()
    auto = classify(entry["title"], entry.get("text", ""))
    profile = entry.get("profile", {})
    with conn:
        if existing is None:
            conn.execute(
                """INSERT INTO document_catalog (workspace_id, source_uri, title, doc_type, subtype,
                   issuer, period, status, profile, confidence, evidence, content_hash,
                   needs_review, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (workspace, uri, entry["title"], auto["doc_type"], auto["subtype"], auto["issuer"],
                 auto["period"], auto["status"], json.dumps(profile, ensure_ascii=False),
                 auto["confidence"], auto["evidence"], entry.get("content_hash", ""),
                 int(auto["confidence"] < REVIEW_BELOW), _now()),
            )
        elif existing["content_hash"] != entry.get("content_hash", ""):
            locked = set(json.loads(existing["locked"]))
            changed = any(auto[f] != existing[f] and f not in locked for f in _AUTO)
            values = {f: (existing[f] if f in locked else auto[f]) for f in _AUTO}
            conn.execute(
                """UPDATE document_catalog SET title=?, doc_type=?, subtype=?, issuer=?, period=?,
                   status=?, profile=?, confidence=?, evidence=?, content_hash=?,
                   needs_review=?, updated_at=? WHERE workspace_id=? AND source_uri=?""",
                (entry["title"], values["doc_type"], values["subtype"], values["issuer"],
                 values["period"], values["status"], json.dumps(profile, ensure_ascii=False),
                 auto["confidence"], auto["evidence"], entry.get("content_hash", ""),
                 int(bool(existing["needs_review"]) or changed), _now(), workspace, uri),
            )
    return get(conn, workspace, uri)


def get(conn: sqlite3.Connection, workspace: str, source_uri: str) -> dict[str, Any] | None:
    _ensure(conn)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM document_catalog WHERE workspace_id=? AND source_uri=?", (workspace, source_uri)
    ).fetchone()
    return _row(row) if row else None


def list_entries(conn: sqlite3.Connection, workspace: str) -> dict[str, dict[str, Any]]:
    _ensure(conn)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM document_catalog WHERE workspace_id=?", (workspace,)).fetchall()
    return {row["source_uri"]: _row(row) for row in rows}


class CatalogError(ValueError):
    """An edit that is not allowed (unknown type/status, over-long text, no such document)."""


def update(conn: sqlite3.Connection, workspace: str, source_uri: str, fields: dict[str, Any]) -> dict:
    """A person's edit: validated, locked against auto-overwrite, reviewed."""
    entry = get(conn, workspace, source_uri)
    if entry is None:
        raise CatalogError("This document has no catalog entry yet — re-index it once first.")
    unknown = set(fields) - set(EDITABLE)
    if unknown:
        raise CatalogError(f"Cannot edit: {', '.join(sorted(unknown))}")
    clean: dict[str, str] = {}
    for name, value in fields.items():
        value = str(value or "").strip()
        if name == "doc_type" and value not in DOC_TYPES:
            raise CatalogError(f"doc_type must be one of: {', '.join(DOC_TYPES)}")
        if name == "status" and value not in STATUSES:
            raise CatalogError(f"status must be one of: {', '.join(STATUSES)}")
        if len(value) > _LIMITS.get(name, 200):
            raise CatalogError(f"{name} is longer than {_LIMITS.get(name, 200)} characters")
        clean[name] = value
    locked = sorted(set(entry["locked"]) | {n for n, v in clean.items() if v != entry[n]})
    sets = ", ".join(f"{n}=?" for n in clean)
    with conn:
        conn.execute(
            f"UPDATE document_catalog SET {sets + ', ' if sets else ''}locked=?, needs_review=0, "
            "updated_at=? WHERE workspace_id=? AND source_uri=?",
            (*clean.values(), json.dumps(locked), _now(), workspace, source_uri),
        )
    return get(conn, workspace, source_uri)


def delete(conn: sqlite3.Connection, workspace: str, source_uri: str) -> None:
    _ensure(conn)
    with conn:
        conn.execute(
            "DELETE FROM document_catalog WHERE workspace_id=? AND source_uri=?", (workspace, source_uri)
        )


def make_recorder(home: Path):
    """The pipeline's `catalog_log` callback: opens state.db, records one ingest."""
    from tini.db import connect

    def record(entry: dict[str, Any]) -> None:
        conn = connect(home)
        try:
            record_ingest(conn, entry)
        finally:
            conn.close()

    return record
