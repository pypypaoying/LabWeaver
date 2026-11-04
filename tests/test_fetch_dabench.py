"""Pinned public benchmark fetches use verified bytes; tests never use network."""

import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.request import Request

import pytest


@pytest.fixture
def fetcher():
    path = Path(__file__).resolve().parents[1] / "evaluation/fetch_dabench.py"
    spec = importlib.util.spec_from_file_location("fetch_dabench_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def small_snapshot(fetcher, monkeypatch):
    content = {
        "da-dev-questions.jsonl": b'{"id":0,"question":"count rows"}\n',
        "da-dev-labels.jsonl": b'{"id":0,"common_answers":[["rows",1]]}\n',
        "da-dev-tables/public sample.csv": b"value\n1\n",
    }
    monkeypatch.setattr(fetcher, "DATA_FILES", tuple(
        (name, hashlib.sha256(raw).hexdigest(), len(raw)) for name, raw in content.items()
    ))
    scorer = b"# Inert source. Never import or exec downloaded Python.\n"
    monkeypatch.setattr(fetcher, "SCORER_FILES", ((
        "eval_closed_form.py", "examples/DA-Agent/eval_closed_form.py",
        hashlib.sha256(scorer).hexdigest(), len(scorer)),))
    content["examples/DA-Agent/eval_closed_form.py"] = scorer
    calls = []
    def public_bytes(url, expected_size):
        calls.append(url)
        assert url.startswith(("https://huggingface.co/datasets/infiagent/DABench/resolve/",
                               "https://raw.githubusercontent.com/InfiAgent/InfiAgent/"))
        assert "main" not in url
        for name, raw in content.items():
            if url.endswith(fetcher.quote(name, safe="/")):
                assert len(raw) == expected_size
                return raw
        raise AssertionError("Unexpected URL")
    monkeypatch.setattr(fetcher, "_public_bytes", public_bytes)
    return fetcher, content, calls


def test_first_fetch_verifies_static_resources_and_writes_manifests(small_snapshot, tmp_path):
    module, content, calls = small_snapshot
    result = module.fetch_dataset(tmp_path)
    assert result["downloaded"] == len(calls) == 4
    assert result["reused"] == 0 and result["csv_files"] == 1
    manifest = json.loads((tmp_path / "source.json").read_text())
    assert manifest["revision"] == module.DATASET_REVISION
    for entry in manifest["files"]:
        assert (tmp_path / entry["path"]).read_bytes() == content[entry["path"]]
    assert (tmp_path / "upstream-eval_closed_form.py").read_bytes() == content["examples/DA-Agent/eval_closed_form.py"]


def test_offline_reuse_checks_all_hashes_without_changing_manifest_bytes(small_snapshot, tmp_path):
    module, _, calls = small_snapshot
    module.fetch_dataset(tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in ("source.json", "scorer-source.json")}
    # Preserve even semantically equivalent manifest whitespace: it might be
    # part of a frozen downstream run contract.
    (tmp_path / "source.json").write_text(json.dumps(json.loads(before["source.json"])))
    changed = (tmp_path / "source.json").read_bytes()
    calls.clear()
    result = module.fetch_dataset(tmp_path, offline=True)
    assert result["reused"] == 4 and result["downloaded"] == 0 and calls == []
    assert (tmp_path / "source.json").read_bytes() == changed
    assert (tmp_path / "scorer-source.json").read_bytes() == before["scorer-source.json"]


def test_missing_offline_resource_fails_without_network(small_snapshot, tmp_path):
    module, _, calls = small_snapshot
    with pytest.raises(ValueError, match="Offline resource missing"):
        module.fetch_dataset(tmp_path, offline=True)
    assert calls == []
    assert not (tmp_path / "source.json").exists()


def test_corrupted_cache_is_not_silently_replaced(small_snapshot, tmp_path):
    module, _, calls = small_snapshot
    module.fetch_dataset(tmp_path)
    path = tmp_path / "da-dev-tables/public sample.csv"
    path.write_bytes(b"value\n9\n")
    calls.clear()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.fetch_dataset(tmp_path)
    assert calls == []
    assert path.read_bytes() == b"value\n9\n"


def test_changed_manifest_fails_before_network(small_snapshot, tmp_path):
    module, _, calls = small_snapshot
    module.fetch_dataset(tmp_path)
    manifest = json.loads((tmp_path / "source.json").read_text())
    manifest["revision"] = "different"
    (tmp_path / "source.json").write_text(json.dumps(manifest))
    calls.clear()
    with pytest.raises(ValueError, match="Source manifest differs"):
        module.fetch_dataset(tmp_path)
    assert calls == []


def test_downloaded_wrong_bytes_are_never_registered(small_snapshot, monkeypatch, tmp_path):
    module, _, _ = small_snapshot
    monkeypatch.setattr(module, "_public_bytes", lambda url, expected_size: b"x" * expected_size)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        module.fetch_dataset(tmp_path, workers=1)
    assert not (tmp_path / "source.json").exists()
    assert not (tmp_path / "da-dev-questions.jsonl").exists()


@pytest.mark.parametrize("relative", [
    "../escape.csv", "/absolute.csv", "C:/absolute.csv", "back\\slash.csv",
    "normal/../../escape.csv", "one//two.csv", "one/./two.csv", ".", "", None,
    "nul\x00.csv",
])
def test_download_paths_reject_escape_and_ambiguous_spellings(fetcher, tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe"):
        fetcher._target(tmp_path, relative)


def test_symlink_files_and_directories_are_rejected(fetcher, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")
    with pytest.raises(ValueError, match="links or reparse"):
        fetcher._checked_root(link)
    with pytest.raises(ValueError, match="links or reparse"):
        fetcher._target(tmp_path, "link/one.csv")


@pytest.mark.parametrize("url", [
    "http://huggingface.co/file", "https://evil.test/file",
    "https://huggingface.co.evil.test/file", "https://user:pass@huggingface.co/file",
    "https://huggingface.co:444/file", "https://huggingface.co/file#fragment",
    "https://huggingface.co/file?token=secret",
])
def test_only_public_official_static_https_urls_are_allowed(fetcher, url):
    with pytest.raises(ValueError, match="HTTPS"):
        fetcher._validate_url(url)


def test_redirects_cannot_escape_host_or_https(fetcher):
    handler = fetcher._OfficialRedirects()
    request = Request("https://huggingface.co/datasets/a/resolve/fixed/file.csv")
    with pytest.raises(ValueError, match="HTTPS"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://evil.test/file")
    with pytest.raises(ValueError, match="HTTPS"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://huggingface.co/file")
    official = handler.redirect_request(request, None, 302, "Found", {}, "https://huggingface.co/api/resolve-cache/file?etag=public")
    assert official.full_url.endswith("?etag=public")


def test_fixed_catalog_contains_68_tables_and_known_primary_hashes(fetcher):
    assert len(fetcher.DATA_FILES) == 70
    assert sum(name.startswith("da-dev-tables/") for name, _, _ in fetcher.DATA_FILES) == 68
    assert len({name for name, _, _ in fetcher.DATA_FILES}) == 70
    primary = dict((name, sha) for name, sha, _ in fetcher.DATA_FILES)
    assert primary["da-dev-questions.jsonl"] == "430306c5813bf4f2b0e38f1b6ccee9693c36d5920b3322bd7c8ae46f475ce443"
    assert primary["da-dev-labels.jsonl"] == "83b8fb8133c1794fcb6b42435e9cce124d22f68b71e2eedb37c973e40b5a18c2"
    assert fetcher.SCORER_FILES[0][2] == fetcher.SCORER_SHA256


@pytest.mark.parametrize("workers", [0, 5, True, 1.5])
def test_worker_count_is_bounded(fetcher, tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        fetcher.fetch_dataset(tmp_path, workers=workers)
