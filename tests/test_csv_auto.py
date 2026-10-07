"""Format inference must preserve source bytes and expose real ambiguity."""

import codecs
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from labweaver.tools.csv_profile import CsvReadError, load_csv_snapshot, profile_csv


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-32"])
@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_auto_bom_and_delimiters(tmp_path, encoding, delimiter):
    path = tmp_path / "input.csv"
    raw = f'编号{delimiter}分数\n"001"{delimiter}3\n"002"{delimiter}5\n'.encode(encoding)
    path.write_bytes(raw)
    snapshot = load_csv_snapshot(path)
    assert snapshot.headers == ("编号", "分数")
    assert snapshot.rows == (("001", "3"), ("002", "5"))
    assert snapshot.profile["parsing"]["delimiter"] == delimiter
    assert snapshot.profile["parsing"]["encoding_method"] == ("strict_utf8" if encoding == "utf-8" else "bom")
    assert snapshot.profile["columns"][1]["numeric_summary"]["mean"] == 4
    assert snapshot.source["sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw
    json.dumps(snapshot.profile, allow_nan=False)


@pytest.mark.parametrize("codec,bom", [("utf-16-le", codecs.BOM_UTF16_LE), ("utf-16-be", codecs.BOM_UTF16_BE),
                                      ("utf-32-le", codecs.BOM_UTF32_LE), ("utf-32-be", codecs.BOM_UTF32_BE)])
def test_endian_boms_strip_header_bom_and_utf32_precedes_utf16(tmp_path, codec, bom):
    path = tmp_path / "input.csv"
    path.write_bytes(bom + "name,value\n中文,4\n".encode(codec))
    snapshot = load_csv_snapshot(path)
    assert snapshot.headers == ("name", "value")
    assert snapshot.rows == (("中文", "4"),)
    assert snapshot.profile["parsing"]["encoding"] == codec[:6]


def test_common_gb18030_can_be_detected_without_replacement(tmp_path):
    path = tmp_path / "gb.csv"
    raw = "姓名,分数\n张三,3\n李四,5\n".encode("gb18030")
    path.write_bytes(raw)
    snapshot = load_csv_snapshot(path)
    assert snapshot.headers == ("姓名", "分数")
    assert snapshot.rows == (("张三", "3"), ("李四", "5"))
    assert snapshot.profile["parsing"]["encoding"] == "gb18030"
    assert snapshot.profile["parsing"]["encoding_method"] == "charset_normalizer"
    assert path.read_bytes() == raw


def test_short_legacy_encoding_ambiguity_returns_strict_previews(tmp_path):
    path = tmp_path / "ambiguous.csv"
    raw = b"name\n\xff\n"
    path.write_bytes(raw)
    result = profile_csv(path)
    assert result["status"] == "error"
    assert result["error"]["code"] == "ambiguous_encoding"
    candidates = result["error"]["candidates"]
    assert len(candidates) >= 2
    for candidate in candidates:
        assert raw.decode(candidate["encoding"], errors="strict").startswith(candidate["preview"])
        assert "\ufffd" not in candidate["preview"]
    assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw
    assert profile_csv(path, encoding="cp1252")["sample_rows"] == [["ÿ"]]


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_quotes_and_embedded_newline_do_not_confuse_format_detection(tmp_path, delimiter):
    path = tmp_path / "quoted.csv"
    # All separators occur inside a quoted field, only the actual one outside it.
    path.write_bytes(f'name{delimiter}number\n"comma, semicolon; tab\t pipe|\nsecond line"{delimiter}2\n'.encode("utf-8"))
    snapshot = load_csv_snapshot(path)
    assert snapshot.profile["parsing"]["delimiter"] == delimiter
    assert len(snapshot.rows) == 1
    assert snapshot.rows[0][0] == "comma, semicolon; tab\t pipe|\nsecond line"


def test_true_single_column_even_when_quoted_header_contains_commas(tmp_path):
    path = tmp_path / "one.csv"
    path.write_bytes(b'"single, column"\n"hello, world"\n"line one\nline two"\n')
    snapshot = load_csv_snapshot(path)
    assert snapshot.headers == ("single, column",)
    assert snapshot.rows == (("hello, world",), ("line one\nline two",))
    assert snapshot.profile["parsing"]["delimiter_method"] == "single_column"


def test_delimiter_ambiguity_does_not_choose_popular_comma(tmp_path):
    path = tmp_path / "ambiguous.csv"
    path.write_text("a,b;c\n1,2;3\n4,5;6\n", encoding="utf-8")
    result = profile_csv(path)
    assert result["error"]["code"] == "ambiguous_delimiter"
    assert {item["delimiter"] for item in result["error"]["candidates"]} == {",", ";"}
    assert profile_csv(path, delimiter=";")["sample_rows"] == [["1,2", "3"], ["4,5", "6"]]


def test_sample_consistency_distinguishes_punctuation_from_real_separator(tmp_path):
    path = tmp_path / "consistent.csv"
    path.write_text("label;extra,value\nx,1\ny,2\n", encoding="utf-8")
    snapshot = load_csv_snapshot(path)
    assert snapshot.profile["parsing"]["delimiter"] == ","
    assert snapshot.headers == ("label;extra", "value")


def test_quote_rules_can_disambiguate_two_header_separators(tmp_path):
    path = tmp_path / "quote-rules.csv"
    # The header alone fits both separators. Quoted data fit semicolon: comma
    # would illegally leave a semicolon immediately after a closing quote.
    path.write_bytes(b'a,b;c\n"x";y,2\n"z";q,4\n')
    snapshot = load_csv_snapshot(path)
    assert snapshot.headers == ("a,b", "c")
    assert snapshot.rows == (("x", "y,2"), ("z", "q,4"))
    assert snapshot.profile["parsing"]["delimiter"] == ";"


def test_equal_sample_width_evidence_is_ambiguous_even_if_one_parse_is_malformed(tmp_path):
    path = tmp_path / "ambiguous-malformed.csv"
    # Neither candidate establishes a consistent shape before the broken quote.
    # An explicit choice is needed; neither interpretation gets a full profile.
    raw = b'a,b;c\n"unclosed,2;3\n'
    path.write_bytes(raw)
    result = profile_csv(path)
    assert result["error"]["code"] == "ambiguous_delimiter"
    for delimiter in (",", ";"):
        assert profile_csv(path, delimiter=delimiter)["error"]["code"] == "invalid_csv"
    assert path.read_bytes() == raw


@pytest.mark.parametrize("separator", [",", ";", "\t", "|"])
def test_early_and_late_malformed_rows_never_rescue_to_one_column(tmp_path, separator):
    path = tmp_path / "bad.csv"
    for correct_prefix in (0, 60):
        raw = (f"a{separator}b\n" + f"1{separator}2\n" * correct_prefix + "3\n").encode()
        path.write_bytes(raw)
        result = profile_csv(path)
        assert result["error"]["code"] == "row_width_mismatch"
        assert "row_count" not in result
        assert path.read_bytes() == raw


def test_unclosed_quote_is_not_reinterpreted_with_another_separator(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text('a;b\n1;2\n"unfinished;3\n', encoding="utf-8")
    assert profile_csv(path)["error"]["code"] == "invalid_csv"


def test_unclosed_quote_after_detection_window_is_still_a_full_parse_error(tmp_path):
    path = tmp_path / "late-bad.csv"
    raw = ("a|b\n" + "1|2\n" * 60 + '"unfinished|3\n').encode()
    path.write_bytes(raw)
    result = profile_csv(path)
    assert result["error"]["code"] == "invalid_csv"
    assert "row_count" not in result
    assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw


def test_header_only_auto_profile_retains_multi_column_shape(tmp_path):
    path = tmp_path / "header.csv"
    path.write_bytes(b"name;score;score\n")
    snapshot = load_csv_snapshot(path)
    assert snapshot.headers == ("name", "score", "score")
    assert snapshot.rows == ()
    assert snapshot.profile["row_count"] == 0
    assert snapshot.profile["parsing"]["delimiter"] == ";"
    assert snapshot.profile["warnings"]


def test_explicit_format_is_recorded_and_does_not_get_auto_corrected(tmp_path):
    path = tmp_path / "explicit.csv"
    path.write_bytes("姓名;分数\n张三;4\n".encode("gb18030"))
    profile = profile_csv(path, encoding="gb18030", delimiter=";")
    assert profile["parsing"] == {"encoding": "gb18030", "delimiter": ";",
                                  "encoding_method": "explicit", "delimiter_method": "explicit"}
    assert profile_csv(path, encoding="utf-8", delimiter=";")["error"]["code"] == "decode_error"
    # Explicitly overriding with comma is the user's choice, not auto inference.
    assert profile_csv(path, encoding="gb18030", delimiter=",")["column_count"] == 1


def test_bom_invalid_bytes_never_fall_back_to_legacy_encoding(tmp_path):
    path = tmp_path / "bad-bom.csv"
    path.write_bytes(codecs.BOM_UTF8 + b"a,b\n\xff,2\n")
    assert profile_csv(path)["error"]["code"] == "decode_error"


def test_detector_sample_never_overrides_full_strict_decoding(tmp_path):
    path = tmp_path / "late-bad-encoding.csv"
    raw = b"name,value\n" + b"A,2\n" * 800 + b"\xff,3\n"
    path.write_bytes(raw)
    # A detector may have sampled only a valid prefix. Its codec must still
    # strictly decode every byte, including the invalid byte after the sample.
    sampled_match = SimpleNamespace(encoding="utf-8", chaos=0.0, coherence=1.0)
    with patch("labweaver.tools.csv_profile.from_bytes", return_value=[sampled_match]):
        result = profile_csv(path)
    assert result["error"]["code"] == "decode_error"
    assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw


def test_snapshot_reads_source_once_and_survives_source_changes(tmp_path):
    path = tmp_path / "snapshot.csv"
    raw = b"group,value\nA,2\nB,3\n"
    path.write_bytes(raw)
    original_open = Path.open
    calls = []

    def record_open(this, *args, **kwargs):
        calls.append((this, args))
        return original_open(this, *args, **kwargs)

    with patch.object(Path, "open", record_open):
        snapshot = load_csv_snapshot(path)
    assert len(calls) == 1
    path.write_bytes(b"group,value\nchanged,999\n")
    assert snapshot.rows == (("A", "2"), ("B", "3"))
    assert snapshot.source["sha256"] == hashlib.sha256(raw).hexdigest()


def test_loader_error_has_source_and_profile_wrapper_is_json_safe(tmp_path):
    path = tmp_path / "bad.csv"
    raw = b"a,b\n1\n"
    path.write_bytes(raw)
    with pytest.raises(CsvReadError) as caught:
        load_csv_snapshot(path)
    assert caught.value.code == "row_width_mismatch"
    assert caught.value.source["sha256"] == hashlib.sha256(raw).hexdigest()
    json.dumps(caught.value.as_result(), allow_nan=False)
