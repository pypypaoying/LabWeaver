"""Behavioral tests for bounded, read-only profiles across unrelated schemas."""

import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from labweaver.tools import profile_csv


profile_module = importlib.import_module("labweaver.tools.csv_profile")


class CsvProfileTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "input.csv"

    def profile_text(self, text, *, write_encoding="utf-8", **options):
        self.path.write_bytes(text.encode(write_encoding))
        result = profile_csv(self.path, **options)
        # Every result, including errors, must be strict JSON-safe.
        json.loads(json.dumps(result, allow_nan=False))
        return result

    def assert_profile_error(self, result, code):
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], code)
        self.assertTrue(result["error"]["message"])
        self.assertNotIn("row_count", result)
        self.assertNotIn("columns", result)

    def test_unrelated_schemas_need_no_named_identifier(self):
        for text in ("name,score\nAlice,1\nBob,3\n", "temperature,loss\n20,0.2\n30,0.4\n"):
            with self.subTest(text=text):
                result = self.profile_text(text)
                self.assertEqual(result["status"], "completed")
                self.assertEqual((result["row_count"], result["column_count"]), (2, 2))
                self.assertEqual(result["columns"][1]["inferred_type"], "numeric")

    def test_missingness_preserves_na_null_zero_and_raw_samples(self):
        result = self.profile_text("label,value\nNA,0\nNULL, \n  ,2\n")
        self.assertEqual(result["columns"][0]["missing_count"], 1)
        self.assertEqual(result["columns"][1]["missing_count"], 1)
        self.assertAlmostEqual(result["columns"][1]["missing_ratio"], 1 / 3)
        self.assertEqual(result["columns"][1]["numeric_summary"], {"min": 0.0, "max": 2.0, "mean": 1.0})
        self.assertEqual(result["sample_rows"], [["NA", "0"], ["NULL", " "], ["  ", "2"]])

    def test_identifier_digits_remain_text(self):
        result = self.profile_text("reference,account\n001,12345678901234567890\n002,12345678901234567891\n")
        self.assertTrue(all(item["inferred_type"] == "text" for item in result["columns"]))
        self.assertTrue(all(item["numeric_summary"] is None for item in result["columns"]))
        self.assertEqual(result["sample_rows"][0][0], "001")
        self.assertTrue(result["warnings"])

    def test_numeric_stats_and_mixed_column(self):
        result = self.profile_text("value,mixed\n-2,1\n.5,unknown\n3e0,2\n")
        summary = result["columns"][0]["numeric_summary"]
        self.assertEqual((summary["min"], summary["max"]), (-2.0, 3.0))
        self.assertAlmostEqual(summary["mean"], 0.5)
        self.assertEqual(result["columns"][1]["inferred_type"], "mixed")
        self.assertIsNone(result["columns"][1]["numeric_summary"])

    def test_delimiters_quotes_and_multiline_cells(self):
        for delimiter in (",", ";", "\t", "|"):
            with self.subTest(delimiter=repr(delimiter)):
                text = f'name{delimiter}value\n"hello{delimiter}world\nline two"{delimiter}2\n'
                result = self.profile_text(text, delimiter=delimiter)
                self.assertEqual(result["row_count"], 1)
                self.assertEqual(result["sample_rows"][0][0], f"hello{delimiter}world\nline two")
                self.assertEqual(result["columns"][1]["numeric_summary"]["mean"], 2)

    def test_explicit_encodings_and_utf8_bom(self):
        for encoding in ("utf-8-sig", "gb18030", "utf-16"):
            with self.subTest(encoding=encoding):
                result = self.profile_text("姓名,分数\n张三,3\n", write_encoding=encoding, encoding=encoding)
                self.assertEqual(result["columns"][0]["name"], "姓名")
                self.assertEqual(result["sample_rows"][0][0], "张三")

    def test_duplicate_empty_headers_preserve_every_column(self):
        result = self.profile_text("score,score,\n1,2,003\n")
        self.assertEqual([item["name"] for item in result["columns"]], ["score", "score", ""])
        self.assertEqual([item["position"] for item in result["columns"]], [1, 2, 3])
        self.assertEqual(result["columns"][0]["numeric_summary"]["mean"], 1)
        self.assertEqual(result["columns"][1]["numeric_summary"]["mean"], 2)
        self.assertTrue(result["warnings"])

    def test_empty_file_and_header_only_are_distinct(self):
        for text in ("", "\n\r\n"):
            self.assert_profile_error(self.profile_text(text), "empty_input")
        result = self.profile_text("name,score\n")
        self.assertEqual(result["row_count"], 0)
        self.assertEqual(result["columns"][0]["inferred_type"], "empty")
        self.assertIsNone(result["columns"][0]["missing_ratio"])
        self.assertIsNone(result["columns"][0]["numeric_summary"])

    def test_blank_records_skip_but_all_missing_record_counts(self):
        result = self.profile_text("\na,b\n\n,\n1,2\n\n")
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["columns"][0]["missing_count"], 1)
        self.assertEqual(result["sample_rows"][0], ["", ""])

    def test_ragged_rows_and_unclosed_quote_fail(self):
        for text in ("a,b\n1\n", "a,b\n1,2,3\n"):
            self.assert_profile_error(self.profile_text(text), "row_width_mismatch")
        self.assert_profile_error(self.profile_text('a,b\n"unclosed,2\n'), "invalid_csv")

    def test_decode_unknown_codec_and_nul_errors(self):
        self.path.write_bytes(b"a\n\xff\n")
        self.assert_profile_error(profile_csv(self.path, encoding="utf-8"), "decode_error")
        self.assert_profile_error(profile_csv(self.path, encoding="not-a-codec"), "invalid_encoding")
        self.assert_profile_error(self.profile_text("a\n\x00\n"), "invalid_csv")

    def test_nonfinite_numeric_text_never_enters_json_numbers(self):
        result = self.profile_text("value\nNaN\nInfinity\n1e309\n2\n")
        self.assertEqual(result["columns"][0]["inferred_type"], "mixed")
        self.assertIsNone(result["columns"][0]["numeric_summary"])
        self.assertTrue(result["warnings"])

    def test_extreme_numeric_mean_remains_finite(self):
        for values, expected in ((["1.7976931348623157e308"] * 3, 1.7976931348623157e308),
                                 (["-1.7e308", "1.7e308"], 0.0)):
            with self.subTest(values=values):
                result = self.profile_text("value\n" + "\n".join(values) + "\n")
                summary = result["columns"][0]["numeric_summary"]
                self.assertTrue(all(math.isfinite(value) for value in summary.values()))
                self.assertEqual(summary["mean"], expected)

    def test_numeric_mean_preserves_small_residual_after_cancellation(self):
        values = [1e16, 1.0, -1e16]
        result = self.profile_text("value\n1e16\n1\n-1e16\n")
        self.assertEqual(result["columns"][0]["numeric_summary"]["mean"], statistics.mean(values))
        self.assertEqual(result["columns"][0]["numeric_summary"]["mean"], 1 / 3)

    def test_mean_matches_independent_reference_with_mixed_binary_denominators(self):
        values = [1e300, 0.1, -1e300, 0.2, 5e-324]
        result = self.profile_text("value\n1e300\n0.1\n-1e300\n0.2\n5e-324\n")
        self.assertEqual(result["columns"][0]["numeric_summary"]["mean"], statistics.mean(values))

    def test_nonzero_underflow_is_text_but_true_zero_remains_numeric(self):
        for value in ("1e-400", "-1e-400", ".0001e-999"):
            with self.subTest(value=value):
                result = self.profile_text("value\n" + value + "\n")
                self.assertEqual(result["columns"][0]["inferred_type"], "text")
                self.assertIsNone(result["columns"][0]["numeric_summary"])
                self.assertTrue(result["warnings"])
                self.assertEqual(result["sample_rows"], [[value]])
        zero = self.profile_text("value\n0e-400\n-0.000e-999\n")
        self.assertEqual(zero["columns"][0]["inferred_type"], "numeric")
        self.assertEqual(zero["columns"][0]["numeric_summary"]["mean"], 0.0)

    def test_sampling_is_capped_but_statistics_are_full(self):
        result = self.profile_text("n\n" + "\n".join(str(value) for value in range(8)), sample_rows=100)
        self.assertEqual(result["row_count"], 8)
        self.assertEqual(len(result["sample_rows"]), 5)
        self.assertEqual(result["columns"][0]["numeric_summary"]["max"], 7)
        self.assertTrue(result["warnings"])
        result = self.profile_text("n\n1\n", sample_rows=0)
        self.assertEqual(result["sample_rows"], [])

    def test_every_limit_fails_without_partial_profile(self):
        cases = (
            ("MAX_FILE_BYTES", 3, "name\nAlice\n", "file_too_large"),
            ("MAX_DATA_ROWS", 1, "n\n1\n2\n", "too_many_rows"),
            ("MAX_COLUMNS", 1, "a,b\n1,2\n", "too_many_columns"),
            ("MAX_FIELD_CHARS", 3, "a\nlong\n", "field_too_large"),
        )
        for constant, limit, text, code in cases:
            with self.subTest(constant=constant), patch.object(profile_module, constant, limit):
                self.assert_profile_error(self.profile_text(text), code)

    def test_multibyte_field_exceeds_byte_limit_before_character_limit(self):
        value = "中" * (profile_module.MAX_FIELD_BYTES // 3 + 1)
        self.assertLess(len(value), profile_module.MAX_FIELD_CHARS)
        self.assertGreater(len(value.encode("utf-8")), profile_module.MAX_FIELD_BYTES)
        self.assert_profile_error(self.profile_text("name\n" + value + "\n"), "field_too_large")

    def test_field_at_exact_utf8_byte_limit_is_accepted(self):
        value = "a" * profile_module.MAX_FIELD_BYTES
        result = self.profile_text("name\n" + value + "\n", sample_rows=0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["row_count"], 1)

    def test_invalid_arguments_and_unavailable_file(self):
        self.assert_profile_error(profile_csv(self.path), "file_not_found")
        for options in ({"sample_rows": -1}, {"sample_rows": True}, {"sample_rows": 1.5},
                        {"delimiter": ""}, {"delimiter": "xx"}, {"delimiter": "\n"},
                        {"encoding": ""}):
            with self.subTest(options=options):
                self.assert_profile_error(profile_csv(self.path, **options), "invalid_argument")

    def test_profile_leaves_source_unchanged_and_names_do_not_expose_paths(self):
        raw = b"id,value\r\n001,2\r\n002,3\r\n"
        self.path.write_bytes(raw)
        result = profile_csv(self.path)
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertEqual(result["source"], {"name": "input.csv", "sha256": hashlib.sha256(raw).hexdigest()})
        self.path.write_bytes(b"a,b\n1\n")
        before = self.path.read_bytes()
        profile_csv(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_fully_read_failure_preserves_source_fingerprint_without_partial_statistics(self):
        for raw, code in ((b"a\n\xff\n", "decode_error"),
                          (b"a,b\n1\n", "row_width_mismatch"),
                          (b"", "empty_input")):
            with self.subTest(code=code):
                self.path.write_bytes(raw)
                result = profile_csv(self.path, encoding="utf-8")
                self.assert_profile_error(result, code)
                self.assertEqual(result["source"], {
                    "name": "input.csv", "sha256": hashlib.sha256(raw).hexdigest(),
                })
                self.assertEqual(self.path.read_bytes(), raw)

    def test_oversized_or_unreadable_file_never_gets_a_complete_fingerprint(self):
        self.assertNotIn("source", profile_csv(self.path))
        with patch.object(profile_module, "MAX_FILE_BYTES", 3):
            result = self.profile_text("name\nAlice\n")
        self.assert_profile_error(result, "file_too_large")
        self.assertNotIn("source", result)

    def test_csv_global_field_limit_is_restored_after_failure(self):
        original_limit = csv.field_size_limit()
        with patch.object(profile_module, "MAX_FIELD_CHARS", 3):
            self.profile_text("a\nlong\n")
        self.assertEqual(csv.field_size_limit(), original_limit)


if __name__ == "__main__":
    unittest.main()
