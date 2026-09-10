"""Safe, local document parsing with optional PDF and DOCX adapters."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


PLAIN_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".yaml", ".yml",
}
SUPPORTED_SUFFIXES = PLAIN_TEXT_SUFFIXES | {".html", ".htm", ".pdf", ".docx"}


class DocumentLoadError(ValueError):
    """A source is unsupported, unsafe, too large, or cannot be parsed."""


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    source_uri: str
    title: str
    text: str
    media_type: str
    size_bytes: int
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


class _TextOnlyHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        return "\n".join(line.strip() for line in "".join(self.parts).splitlines() if line.strip())


def discover_documents(path: Path, recursive: bool = True) -> list[Path]:
    """Return supported, regular files in deterministic order.

    Directory traversal never follows symlinks. An explicitly supplied symlink
    is rejected as well, so a watched ingestion folder cannot escape its root.
    """
    path = path.expanduser()
    if not path.exists():
        raise DocumentLoadError(f"source does not exist: {path}")
    if path.is_symlink():
        raise DocumentLoadError(f"symbolic links are not ingested: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise DocumentLoadError(f"unsupported document type: {path.suffix or '(none)'}")
        return [path.resolve()]
    if not path.is_dir():
        raise DocumentLoadError(f"source is not a regular file or directory: {path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    files = [
        item.resolve()
        for item in iterator
        if item.is_file()
        and not item.is_symlink()
        and not item.name.startswith("~$")
        and item.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda item: str(item).lower())


def load_document(path: Path, max_file_bytes: int) -> LoadedDocument:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise DocumentLoadError(f"source is not a regular file: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentLoadError(f"unsupported document type: {suffix or '(none)'}")
    size = path.stat().st_size
    if size > max_file_bytes:
        raise DocumentLoadError(
            f"document is {size} bytes; limit is {max_file_bytes} bytes: {path.name}"
        )

    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    metadata: dict[str, Any] = {"extension": suffix}
    title = path.stem

    if suffix in PLAIN_TEXT_SUFFIXES:
        text = _decode_utf8(raw, path)
    elif suffix in {".html", ".htm"}:
        parser = _TextOnlyHTMLParser()
        parser.feed(_decode_utf8(raw, path))
        text = parser.text()
    elif suffix == ".pdf":
        text, pdf_meta = _load_pdf(path)
        metadata.update(pdf_meta)
    else:
        text, docx_meta = _load_docx(path)
        title = docx_meta.pop("title", "") or title
        metadata.update(docx_meta)

    text = _normalise_text(text)
    if not text:
        raise DocumentLoadError(f"document contains no extractable text: {path.name}")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return LoadedDocument(
        path=path,
        source_uri=path.as_uri(),
        title=title,
        text=text,
        media_type=media_type,
        size_bytes=size,
        content_hash=content_hash,
        metadata=metadata,
    )


def _decode_utf8(raw: bytes, path: Path) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"text document must be UTF-8: {path.name}") from exc


def _normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _load_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError(
            "PDF ingestion requires the documents extra: pip install -e '.[documents]'"
        ) from exc
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise DocumentLoadError(f"encrypted PDF is not supported: {path.name}")
        pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"could not parse PDF {path.name}: {exc}") from exc
    return "\n\n".join(pages), {"page_count": len(pages)}


def _load_docx(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from docx import Document
    except ImportError as exc:
        raise DocumentLoadError(
            "DOCX ingestion requires the documents extra: pip install -e '.[documents]'"
        ) from exc
    try:
        document = Document(path)
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        title = document.core_properties.title or ""
    except Exception as exc:
        raise DocumentLoadError(f"could not parse DOCX {path.name}: {exc}") from exc
    return "\n\n".join(parts), {"title": title}
