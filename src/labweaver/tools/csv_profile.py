"""Create a bounded, read-only CSV profile without imposing a domain schema."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
from pathlib import Path
import re
import threading
from typing import Any


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


def profile_csv(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    sample_rows: int = 5,
) -> dict[str, Any]:
    """Profile an explicitly selected local CSV; never modify its contents.

    The first nonblank CSV record is the header. Empty/duplicate header names are
    preserved and columns are addressed by their one-based position. A missing
    value is a whitespace-only cell, not a literal NA, NULL, or 0. Numeric ranges
    and means describe all nonmissing cells only when the column is numeric.
    Leading-zero and long integer tokens are treated as identifier-like text.
    Explicit encodings/delimiters are supported; this function never guesses
    them or silently repairs ragged rows. Samples preserve parsed cell strings.
    Each decoded field is limited to 64 KiB when encoded as UTF-8, with the
    stdlib character limit providing an additional early rejection. Limits apply
    to the full file: errors never return partial statistics.
    """
    if not isinstance(encoding, str) or not encoding:
        return _error("invalid_argument", "encoding must be a nonempty codec name.")
    if (
        not isinstance(delimiter, str)
        or len(delimiter) != 1
        or delimiter in {"\r", "\n", "\x00", '"'}
    ):
        return _error("invalid_argument", "delimiter must be one non-quote separator character.")
    if isinstance(sample_rows, bool) or not isinstance(sample_rows, int) or sample_rows < 0:
        return _error("invalid_argument", "sample_rows must be a nonnegative integer.")
    try:
        source_path = Path(path)
    except (TypeError, ValueError):
        return _error("invalid_argument", "path must identify a local file.")

    warnings: list[str] = []
    sample_limit = min(sample_rows, MAX_SAMPLE_ROWS)
    if sample_rows > MAX_SAMPLE_ROWS:
        warnings.append(f"sample_rows was capped at {MAX_SAMPLE_ROWS}.")
    try:
        with source_path.open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
    except FileNotFoundError:
        return _error("file_not_found", "The selected CSV file does not exist.")
    except (OSError, ValueError):
        return _error("read_error", "The selected CSV file could not be read.")
    if len(raw) > MAX_FILE_BYTES:
        return _error("file_too_large", f"The CSV exceeds the {MAX_FILE_BYTES}-byte limit.")
    source = {"name": source_path.name, "sha256": hashlib.sha256(raw).hexdigest()}

    def file_error(code: str, message: str) -> dict[str, Any]:
        return _error(code, message, source=source)

    try:
        text = raw.decode(encoding, errors="strict")
    except LookupError:
        return file_error("invalid_encoding", "The requested text encoding is not available.")
    except UnicodeDecodeError:
        return file_error("decode_error", "The CSV could not be decoded with the selected encoding.")
    if "\x00" in text:
        return file_error("invalid_csv", "The decoded CSV contains a NUL character.")

    header: list[str] | None = None
    accumulators: list[dict[str, Any]] = []
    samples: list[list[str]] = []
    row_count = 0
    blank_records = 0

    # field_size_limit is process-wide in the stdlib. Restore it even on errors.
    with _CSV_LOCK:
        previous_field_limit = csv.field_size_limit()
        csv.field_size_limit(MAX_FIELD_CHARS)
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
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

    return {
        "status": "completed",
        "source": source,
        "row_count": row_count,
        "column_count": len(header),
        "columns": columns,
        "sample_rows": samples,
        "warnings": warnings,
    }
