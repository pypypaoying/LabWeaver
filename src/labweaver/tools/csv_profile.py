"""Create a bounded, read-only CSV profile without imposing a domain schema."""

from __future__ import annotations

import csv
import codecs
from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, NoReturn

from charset_normalizer import from_bytes


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_DATA_ROWS = 100_000
MAX_COLUMNS = 200
MAX_FIELD_CHARS = 64 * 1024
MAX_FIELD_BYTES = 64 * 1024
MAX_SAMPLE_ROWS = 5

_CSV_LOCK = threading.RLock()
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_LEADING_ZERO = re.compile(r"[+-]?0[0-9]+\Z")
_LONG_INTEGER = re.compile(r"[+-]?[0-9]{16,}\Z")
_DELIMITERS = (",", ";", "\t", "|")
_DETECTION_RECORDS = 50


@dataclass(frozen=True)
class CsvSnapshot:
    """One strictly parsed read of the source, shared by profile and statistics.

    Cell strings and row/column order are unchanged. Metadata dictionaries are
    owned by this snapshot; callers should copy them before adding report data.
    """

    source: dict[str, Any]
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    profile: dict[str, Any]


class CsvReadError(ValueError):
    """A bounded CSV read failed, or needs an explicit format choice."""

    def __init__(
        self, code: str, message: str, *, source: dict | None = None,
        candidates: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.source = source
        self.candidates = candidates

    def as_result(self) -> dict[str, Any]:
        result = _error(self.code, self.message, source=self.source)
        if self.candidates:
            result["error"]["candidates"] = self.candidates
        return result


def _decode_csv(raw: bytes, encoding: str) -> tuple[str, str, str]:
    """Decode strictly; heuristic ambiguity is exposed instead of repaired."""
    if encoding != "auto":
        try:
            return raw.decode(encoding, errors="strict"), codecs.lookup(encoding).name, "explicit"
        except LookupError as exc:
            raise CsvReadError("invalid_encoding", "The requested text encoding is not available.") from exc
        except UnicodeDecodeError as exc:
            raise CsvReadError("decode_error", "The CSV could not be decoded with the selected encoding.") from exc
    # UTF-32 BOMs begin with a UTF-16 BOM, so test the longer signatures first.
    for signatures, codec in (
        ((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE), "utf-32"),
        ((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE), "utf-16"),
        ((codecs.BOM_UTF8,), "utf-8-sig"),
    ):
        if any(raw.startswith(signature) for signature in signatures):
            try:
                return raw.decode(codec, errors="strict"), codec, "bom"
            except UnicodeDecodeError as exc:
                raise CsvReadError("decode_error", "The CSV has a BOM but its encoded contents are invalid.") from exc
    try:
        return raw.decode("utf-8", errors="strict"), "utf-8", "strict_utf8"
    except UnicodeDecodeError:
        pass

    # Charset-normalizer scores describe text coherence/chaos, not a probability
    # of correctness. Compare distinct full decodings and retain an ambiguity
    # whenever the leading plausible decodings are too close to distinguish.
    decoded: list[tuple[str, str, float, float]] = []
    seen_text: set[str] = set()
    for match in from_bytes(raw):
        try:
            text = raw.decode(match.encoding, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
        if "\x00" in text or text in seen_text:
            continue
        seen_text.add(text)
        decoded.append((match.encoding, text, match.chaos, match.coherence))
    if not decoded:
        raise CsvReadError("decode_error", "No supported encoding can reliably decode the CSV. Set encoding explicitly.")
    first = decoded[0]
    runner_up = decoded[1] if len(decoded) > 1 else None
    clear_margin = runner_up is None or runner_up[2] - first[2] >= 0.05 - 1e-9
    clear_coherence = runner_up is not None and first[3] >= 0.20 and first[3] - runner_up[3] >= 0.10
    if first[2] <= 0.10 and (clear_margin or clear_coherence):
        return first[1], first[0], "charset_normalizer"
    candidates = [
        {"encoding": codec, "preview": text[:300], "chaos": chaos, "coherence": coherence}
        for codec, text, chaos, coherence in decoded[:5]
    ]
    raise CsvReadError(
        "ambiguous_encoding", "The CSV encoding is ambiguous. Choose an encoding after checking these previews.",
        candidates=candidates,
    )


def _choose_delimiter(text: str, delimiter: str) -> tuple[str, str]:
    """Choose a separator from quoted records, before the full strict parse.

    A separator present in the header remains a candidate even when a later
    sample record is malformed. This prevents treating a broken multicolumn CSV
    as a valid single column after a parse failure.
    """
    if delimiter != "auto":
        return delimiter, "explicit"
    candidates: list[dict] = []
    with _CSV_LOCK:
        previous_field_limit = csv.field_size_limit()
        csv.field_size_limit(MAX_FIELD_CHARS)
        try:
            for separator in _DELIMITERS:
                reader = csv.reader(io.StringIO(text, newline=""), delimiter=separator, strict=True)
                header = None
                preview = []
                matches = 0
                records = 0
                malformed = False
                try:
                    for row in reader:
                        if not row:
                            continue
                        if header is None:
                            header = row
                            if len(row) <= 1:
                                break
                            preview.append(row)
                            continue
                        records += 1
                        if len(preview) < 3:
                            preview.append(row)
                        matches += len(row) == len(header)
                        if records >= _DETECTION_RECORDS:
                            break
                except csv.Error:
                    malformed = True
                if header is not None and len(header) > 1:
                    candidates.append({"delimiter": separator, "preview": preview,
                                       "matches": matches, "records": records, "malformed": malformed})
        finally:
            csv.field_size_limit(previous_field_limit)
    if not candidates:
        return ",", "single_column"
    if len(candidates) == 1:
        return candidates[0]["delimiter"], "quoted_record_sample"
    # Width consistency distinguishes a separator from punctuation in a field.
    # Equal consistency is genuinely ambiguous and is not decided by popularity.
    def score(candidate: dict) -> tuple[float, int]:
        records = candidate["records"] + int(candidate["malformed"])
        return (candidate["matches"] / records if records else 1.0, candidate["matches"])
    candidates.sort(key=score, reverse=True)
    if score(candidates[0]) > score(candidates[1]):
        return candidates[0]["delimiter"], "quoted_record_sample"
    raise CsvReadError(
        "ambiguous_delimiter", "More than one CSV separator fits the quoted records. Choose a delimiter explicitly.",
        candidates=[{"delimiter": item["delimiter"], "preview": item["preview"]} for item in candidates],
    )


def _error(code: str, message: str, *, source: dict | None = None) -> dict[str, Any]:
    # Partial counters are deliberately excluded: an error is not a full profile.
    result = {"status": "error", "error": {"code": code, "message": message}}
    if source is not None:
        result["source"] = source
    return result


def _add_to_exact_sum(accumulator: dict[str, Any], value: float) -> None:
    """Sum finite binary floats exactly without storing the column's values."""
    numerator, denominator = value.as_integer_ratio()
    previous_denominator = accumulator["sum_denominator"]
    # Float denominators are powers of two, so the larger is a common multiple.
    if denominator > previous_denominator:
        accumulator["sum_numerator"] *= denominator // previous_denominator
        accumulator["sum_denominator"] = denominator
    else:
        numerator *= previous_denominator // denominator
    accumulator["sum_numerator"] += numerator


def _parse_numeric(value: str) -> float | None:
    if not _NUMBER.fullmatch(value):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    # A syntactically nonzero decimal can round all the way to zero in float.
    # Keep that value as out-of-range text rather than claiming it was a zero.
    mantissa = value.lower().split("e", 1)[0]
    if number == 0.0 and any(digit in "123456789" for digit in mantissa):
        return None
    return number


def load_csv_snapshot(
    path: str | os.PathLike[str],
    *,
    encoding: str = "auto",
    delimiter: str = "auto",
    sample_rows: int = 5,
) -> CsvSnapshot:
    """Read, identify the format, and strictly parse one selected local CSV.

    The first nonblank CSV record is the header. Empty/duplicate header names are
    preserved and columns are addressed by their one-based position. A missing
    value is a whitespace-only cell, not a literal NA, NULL, or 0. Numeric ranges
    and means describe all nonmissing cells only when the column is numeric.
    Leading-zero and long integer tokens are treated as identifier-like text.
    Automatic encoding/separator inference can ask for an explicit choice when
    ambiguous. Once selected, full parsing never changes format or repairs rows.
    Each decoded field is limited to 64 KiB when encoded as UTF-8, with the
    stdlib character limit providing an additional early rejection. Limits apply
    to the full file: errors never return partial statistics.
    """
    if not isinstance(encoding, str) or not encoding:
        raise CsvReadError("invalid_argument", "encoding must be a nonempty codec name or auto.")
    if (
        not isinstance(delimiter, str)
        or (delimiter != "auto" and (len(delimiter) != 1 or delimiter in {"\r", "\n", "\x00", '"'}))
    ):
        raise CsvReadError("invalid_argument", "delimiter must be auto or one non-quote separator character.")
    if isinstance(sample_rows, bool) or not isinstance(sample_rows, int) or sample_rows < 0:
        raise CsvReadError("invalid_argument", "sample_rows must be a nonnegative integer.")
    try:
        source_path = Path(path)
    except (TypeError, ValueError):
        raise CsvReadError("invalid_argument", "path must identify a local file.")

    warnings: list[str] = []
    sample_limit = min(sample_rows, MAX_SAMPLE_ROWS)
    if sample_rows > MAX_SAMPLE_ROWS:
        warnings.append(f"sample_rows was capped at {MAX_SAMPLE_ROWS}.")
    try:
        with source_path.open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
    except FileNotFoundError:
        raise CsvReadError("file_not_found", "The selected CSV file does not exist.")
    except (OSError, ValueError):
        raise CsvReadError("read_error", "The selected CSV file could not be read.")
    if len(raw) > MAX_FILE_BYTES:
        raise CsvReadError("file_too_large", f"The CSV exceeds the {MAX_FILE_BYTES}-byte limit.")
    source = {"name": source_path.name, "sha256": hashlib.sha256(raw).hexdigest()}

    def file_error(code: str, message: str) -> NoReturn:
        raise CsvReadError(code, message, source=source)

    try:
        text, actual_encoding, encoding_method = _decode_csv(raw, encoding)
        actual_delimiter, delimiter_method = _choose_delimiter(text, delimiter)
    except CsvReadError as exc:
        exc.source = source
        raise
    if "\x00" in text:
        return file_error("invalid_csv", "The decoded CSV contains a NUL character.")

    header: list[str] | None = None
    accumulators: list[dict[str, Any]] = []
    samples: list[list[str]] = []
    parsed_rows: list[tuple[str, ...]] = []
    row_count = 0
    blank_records = 0

    # field_size_limit is process-wide in the stdlib. Restore it even on errors.
    with _CSV_LOCK:
        previous_field_limit = csv.field_size_limit()
        csv.field_size_limit(MAX_FIELD_CHARS)
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=actual_delimiter, strict=True)
        try:
            for row in reader:
                if not row:
                    blank_records += 1
                    continue
                try:
                    oversized = any(len(value.encode("utf-8")) > MAX_FIELD_BYTES for value in row)
                except UnicodeEncodeError:
                    return file_error("invalid_csv", "The decoded CSV contains invalid Unicode characters.")
                if oversized:
                    return file_error("field_too_large", f"A CSV field exceeds the {MAX_FIELD_BYTES}-byte UTF-8 limit.")
                if header is None:
                    if len(row) > MAX_COLUMNS:
                        return file_error("too_many_columns", f"The CSV exceeds the {MAX_COLUMNS}-column limit.")
                    header = row
                    accumulators = [
                        {"missing": 0, "numeric": 0, "text": 0, "identifier": False,
                         "nonfinite": False, "min": None, "max": None,
                         "sum_numerator": 0, "sum_denominator": 1}
                        for _ in header
                    ]
                    continue
                if len(row) != len(header):
                    return file_error(
                        "row_width_mismatch",
                        f"Data record {row_count + 1} has {len(row)} cells; expected {len(header)}.",
                    )
                if row_count >= MAX_DATA_ROWS:
                    return file_error("too_many_rows", f"The CSV exceeds the {MAX_DATA_ROWS}-data-record limit.")
                row_count += 1
                parsed_rows.append(tuple(row))
                if len(samples) < sample_limit:
                    samples.append(row.copy())
                for value, accumulator in zip(row, accumulators):
                    stripped = value.strip()
                    if not stripped:
                        accumulator["missing"] += 1
                        continue
                    if _LEADING_ZERO.fullmatch(stripped) or _LONG_INTEGER.fullmatch(stripped):
                        accumulator["identifier"] = True
                        accumulator["text"] += 1
                        continue
                    number = _parse_numeric(stripped)
                    if number is None:
                        accumulator["text"] += 1
                        if _NUMBER.fullmatch(stripped) or stripped.lower() in {
                            "nan", "+nan", "-nan", "inf", "+inf", "-inf",
                            "infinity", "+infinity", "-infinity",
                        }:
                            accumulator["nonfinite"] = True
                        continue
                    accumulator["numeric"] += 1
                    count = accumulator["numeric"]
                    accumulator["min"] = number if count == 1 else min(accumulator["min"], number)
                    accumulator["max"] = number if count == 1 else max(accumulator["max"], number)
                    _add_to_exact_sum(accumulator, number)
        except csv.Error as exc:
            if "field larger than field limit" in str(exc):
                return file_error("field_too_large", f"A CSV field exceeds the {MAX_FIELD_CHARS}-character limit.")
            return file_error("invalid_csv", f"CSV parsing failed near physical line {reader.line_num}.")
        finally:
            csv.field_size_limit(previous_field_limit)

    if header is None:
        return file_error("empty_input", "The CSV contains no header or data records.")
    if row_count == 0:
        warnings.append("The CSV contains a header but no data records.")
    if blank_records:
        warnings.append(f"Skipped {blank_records} blank CSV records.")
    duplicate_names = {name for name in header if header.count(name) > 1}
    if duplicate_names:
        warnings.append("Duplicate header names are preserved; address columns by position.")
    if any(not name.strip() for name in header):
        warnings.append("Empty header names are preserved; address columns by position.")

    columns: list[dict[str, Any]] = []
    for position, (name, accumulator) in enumerate(zip(header, accumulators), start=1):
        numeric = accumulator["numeric"]
        textual = accumulator["text"]
        summary = None
        if not numeric and not textual:
            inferred_type = "empty"
        elif accumulator["identifier"]:
            inferred_type = "text"
            warnings.append(f"Column {position} contains identifier-like digit strings; kept as text.")
        elif numeric and not textual:
            inferred_type = "numeric"
            summary = {
                "min": accumulator["min"],
                "max": accumulator["max"],
                "mean": accumulator["sum_numerator"] / (accumulator["sum_denominator"] * numeric),
            }
        elif numeric and textual:
            inferred_type = "mixed"
        else:
            inferred_type = "text"
        if accumulator["nonfinite"]:
            warnings.append(f"Column {position} contains nonfinite or out-of-range numeric text; no numeric summary is produced.")
        columns.append({
            "position": position,
            "name": name,
            "inferred_type": inferred_type,
            "missing_count": accumulator["missing"],
            "missing_ratio": accumulator["missing"] / row_count if row_count else None,
            "numeric_summary": summary,
        })

    profile = {
        "status": "completed",
        "source": source,
        "parsing": {"encoding": actual_encoding, "delimiter": actual_delimiter,
                    "encoding_method": encoding_method, "delimiter_method": delimiter_method},
        "row_count": row_count,
        "column_count": len(header),
        "columns": columns,
        "sample_rows": samples,
        "warnings": warnings,
    }
    return CsvSnapshot(source=source, headers=tuple(header), rows=tuple(parsed_rows), profile=profile)


def profile_csv(
    path: str | os.PathLike[str], *, encoding: str = "auto", delimiter: str = "auto", sample_rows: int = 5,
) -> dict[str, Any]:
    """Return a JSON-safe full profile or a precise error, without source writes.

    This standalone function creates a snapshot internally. An Agent session
    should retain ``load_csv_snapshot``'s result and reuse it for every tool.
    """
    try:
        return load_csv_snapshot(path, encoding=encoding, delimiter=delimiter, sample_rows=sample_rows).profile
    except CsvReadError as exc:
        return exc.as_result()
