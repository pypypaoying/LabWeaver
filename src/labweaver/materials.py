"""Read explicit project materials and retrieve literal, traceable passages.

Limits: 10 files; 5 MiB/file; 20 MiB total; 100 pages/PDF; 500,000
extracted characters/source; 2,000,000 extracted characters total; 4,000
chunks total. Chunks are at most 800 characters, with 100-character overlaps
inside long paragraphs/pages. PDF extraction is not OCR or a security sandbox.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import re
import tempfile
import threading
from typing import Iterator

import jieba
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


MAX_DOCUMENTS = 10
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_SOURCE_CHARACTERS = 500_000
MAX_TOTAL_CHARACTERS = 2_000_000
MAX_CHUNKS = 4_000
CHUNK_CHARACTERS = 800
CHUNK_OVERLAP = 100
MAX_QUERY_CHARACTERS = 300

_TOKEN_BLOCK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+(?:\.[0-9]+)?|[\u3400-\u4dbf\u4e00-\u9fff]+")
_TOKENIZER_LOCK = threading.Lock()
_TOKENIZER = None
_TOKENIZER_CACHE = None


class MaterialError(ValueError):
    """A controlled material/query error that never includes parser input."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _tokens(text: str) -> list[str]:
    global _TOKENIZER, _TOKENIZER_CACHE
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            # Do not rely on another user's existing jieba.cache permissions.
            # Windows app-container temporary directories can permit mkdir but
            # reject files, making tempfile.mkstemp retry for minutes. Keep the
            # cache under the caller's writable cwd and probe a file first.
            # The local dictionary is bundled with jieba; no downloads occur.
            try:
                cache_parent = Path.cwd() / ".cache" / "jieba"
                cache_parent.mkdir(parents=True, exist_ok=True)
                cache = tempfile.TemporaryDirectory(prefix="labweaver-", dir=cache_parent)
                probe = Path(cache.name) / ".write-probe"
                with probe.open("xb") as handle:
                    handle.write(b"cache-writable")
                probe.unlink()
                tokenizer = jieba.Tokenizer()
                tokenizer.tmp_dir = cache.name
                tokenizer.initialize()
            except Exception:
                raise MaterialError("material_tokenizer_failed", "The local material tokenizer could not initialize.") from None
            _TOKENIZER_CACHE = cache
            _TOKENIZER = tokenizer
    result = []
    for match in _TOKEN_BLOCK.finditer(text):
        block = match.group()
        if block[0].isascii():
            result.append(block.lower())
        else:
            result.extend(token for token in _TOKENIZER.cut(block, HMM=False) if token.strip())
    return result


def _paragraph_ranges(text: str) -> Iterator[tuple[int, int]]:
    start = None
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            if start is None:
                start = offset
        elif start is not None:
            yield start, offset
            start = None
        offset += len(line)
    if start is not None:
        yield start, len(text)


def _split_text(text: str) -> Iterator[tuple[str, int, int]]:
    for start, stop in _paragraph_ranges(text):
        cursor = start
        while cursor < stop:
            end = min(cursor + CHUNK_CHARACTERS, stop)
            piece = text[cursor:end]
            if piece.strip():
                yield piece, cursor, end
            if end == stop:
                break
            cursor = end - CHUNK_OVERLAP


def _line_starts(text: str) -> list[int]:
    starts = [0]
    offset = 0
    for line in text.splitlines(keepends=True):
        offset += len(line)
        starts.append(offset)
    return starts


def _pdf_pages(raw: bytes) -> tuple[list[str], list[int]]:
    try:
        reader = PdfReader(io.BytesIO(raw), strict=True)
        if reader.is_encrypted:
            raise MaterialError("pdf_encrypted", "Encrypted PDF materials are not supported.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise MaterialError("pdf_page_limit", "The PDF exceeds the 100-page material limit.")
        pages = []
        empty_pages = []
        characters = 0
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not isinstance(text, str):
                raise MaterialError("pdf_invalid", "PDF text extraction did not return text.")
            characters += len(text)
            if characters > MAX_SOURCE_CHARACTERS:
                raise MaterialError("material_text_limit", "A material exceeds the extracted-text character limit.")
            if not text.strip():
                empty_pages.append(number)
            pages.append(text)
    except MaterialError:
        raise
    except Exception:
        raise MaterialError("pdf_invalid", "The selected PDF could not be parsed or its text extracted.") from None
    if not pages or all(not text.strip() for text in pages):
        raise MaterialError("pdf_no_text", "The PDF has no extractable text; scanned/image-only PDFs require OCR.")
    return pages, empty_pages


class MaterialIndex:
    """An in-memory BM25 index; source and result metadata are JSON-safe copies."""

    def __init__(self, sources: list[dict], chunks: list[dict]):
        self._sources = sources
        self._chunks = chunks
        self._chunk_lookup = {chunk["id"]: chunk for chunk in chunks}
        corpus = [_tokens(chunk["text"]) for chunk in chunks]
        self._token_sets = [set(tokens) for tokens in corpus]
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None

    @property
    def sources(self) -> list[dict]:
        return deepcopy(self._sources)

    def chunk_by_id(self, chunk_id: str) -> dict | None:
        chunk = self._chunk_lookup.get(chunk_id)
        return deepcopy(chunk) if chunk is not None else None

    def search(self, query: str) -> dict:
        if not isinstance(query, str):
            raise MaterialError("query_invalid", "A material query must be text.")
        if not query.strip():
            raise MaterialError("query_empty", "A material query must not be empty.")
        if len(query) > MAX_QUERY_CHARACTERS:
            raise MaterialError("query_too_long", "A material query exceeds the 300-character limit.")
        query_tokens = _tokens(query)
        query_terms = set(query_tokens)
        eligible = [position for position, terms in enumerate(self._token_sets) if terms & query_terms]
        if self._bm25 is None or not eligible:
            return {"status": "completed", "query": query, "matches": []}
        scores = self._bm25.get_scores(query_tokens)
        # A matched term may have zero/negative BM25 IDF in a tiny corpus. Do
        # not fabricate zero-overlap hits or suppress genuine overlap for that.
        ranked = sorted(eligible, key=lambda position: (-float(scores[position]), position))[:3]
        return {"status": "completed", "query": query,
                "matches": [deepcopy(self._chunks[position]) for position in ranked]}


def build_material_index(material_paths: list[str | Path]) -> MaterialIndex:
    """Read only listed TXT/MD/text PDFs and build a bounded, local index.

    Document IDs follow input order; chunk IDs follow source order. Text files
    preserve literal decoded UTF-8 text (apart from a leading BOM). PDF text is
    the extractor's page text; pages without text are listed in source metadata
    if other pages are usable. No directories are scanned and no OCR is run.
    """
    if not isinstance(material_paths, list):
        raise MaterialError("material_paths_invalid", "Materials must be a list of explicitly selected file paths.")
    if len(material_paths) > MAX_DOCUMENTS:
        raise MaterialError("material_count_limit", "Select at most 10 material files.")
    sources = []
    chunks = []
    total_bytes = 0
    total_characters = 0
    for number, selected in enumerate(material_paths, start=1):
        if not isinstance(selected, (str, Path)) or (isinstance(selected, str) and not selected.strip()):
            raise MaterialError("material_path_invalid", "A selected material must have a nonempty file path.")
        path = Path(selected)
        file_format = path.suffix.lower().lstrip(".")
        if file_format not in {"txt", "md", "pdf"}:
            raise MaterialError("material_format_unsupported", "Only UTF-8 TXT, Markdown, and text PDF materials are supported.")
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
        except (OSError, ValueError):
            raise MaterialError("material_unreadable", "The selected material file could not be read.") from None
        if len(raw) > MAX_FILE_BYTES:
            raise MaterialError("material_file_limit", "A material exceeds the 5 MiB file limit.")
        total_bytes += len(raw)
        if total_bytes > MAX_TOTAL_BYTES:
            raise MaterialError("material_total_limit", "The selected materials exceed the 20 MiB total limit.")
        source_id = f"D{number}"
        source = {"id": source_id, "name": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                  "format": file_format}
        if file_format == "pdf":
            pages, empty_pages = _pdf_pages(raw)
            source["page_count"] = len(pages)
            source["pages_without_text"] = empty_pages
            parts = [(text, {"page": page}) for page, text in enumerate(pages, start=1)]
        else:
            try:
                text = raw.decode("utf-8-sig", errors="strict")
            except UnicodeError:
                raise MaterialError("material_encoding", "TXT/Markdown materials must be encoded as UTF-8.") from None
            if not text.strip():
                raise MaterialError("material_empty", "The selected text material contains no usable text.")
            if len(text) > MAX_SOURCE_CHARACTERS:
                raise MaterialError("material_text_limit", "A material exceeds the extracted-text character limit.")
            parts = [(text, None)]
        characters = sum(len(text) for text, _ in parts)
        total_characters += characters
        if total_characters > MAX_TOTAL_CHARACTERS:
            raise MaterialError("material_total_text_limit", "Materials exceed the total extracted-text character limit.")
        source["extracted_characters"] = characters
        source_chunk_count = 0
        for text, page_location in parts:
            starts = _line_starts(text) if page_location is None else None
            for piece, start, end in _split_text(text):
                if len(chunks) >= MAX_CHUNKS:
                    raise MaterialError("material_chunk_limit", "Materials exceed the 4,000-chunk limit.")
                source_chunk_count += 1
                location = page_location if page_location is not None else {
                    "line_start": bisect_right(starts, start),
                    "line_end": bisect_right(starts, end - 1),
                }
                chunks.append({"id": f"{source_id}-C{source_chunk_count}", "source_id": source_id,
                               "name": path.name, "sha256": source["sha256"], "text": piece,
                               "location": dict(location)})
        source["chunk_count"] = source_chunk_count
        sources.append(source)
    return MaterialIndex(sources, chunks)
