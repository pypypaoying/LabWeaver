"""Packaged skills are bounded, fingerprinted guidance, never a filesystem tool."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

import labweaver.skills as skills


def resource_copy(tmp_path):
    root = Path(skills.__file__).parent
    target = tmp_path / "packaged-skills"
    shutil.copytree(root, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def test_catalog_does_not_load_skill_bodies(monkeypatch):
    reads = []
    original = skills._resource_text

    def capture(parts, limit):
        reads.append(parts)
        return original(parts, limit)

    monkeypatch.setattr(skills, "_resource_text", capture)
    library = skills.SkillLibrary()
    catalog = library.catalog()
    assert reads == [("manifest.json",)]
    assert [item["id"] for item in catalog] == list(skills.SKILL_IDS)
    assert all(set(item) == {"id", "description", "sha256"} for item in catalog)
    assert sum(len(item["description"]) for item in catalog) < 1200


def test_packaged_bodies_match_reviewed_metadata_and_do_not_share_mutable_state():
    library = skills.SkillLibrary()
    total = 0
    catalog = library.catalog()
    for item in catalog:
        loaded = library.read(item["id"])
        raw = loaded["content"].encode("utf-8")
        total += len(raw)
        assert loaded["status"] == "completed"
        assert loaded["sha256"] == item["sha256"] == hashlib.sha256(raw).hexdigest()
        assert len(raw) <= skills.MAX_SKILL_BYTES
        assert f"name: {item['id']}\n" in loaded["content"]
        assert "Modified by AnaliyMaster" in loaded["content"]
        assert loaded["source"]["license"] == "Apache-2.0"
        loaded["source"]["commit"] = "changed"
        assert library.read(item["id"])["source"]["commit"] != "changed"
    assert total <= skills.MAX_TOTAL_BYTES
    catalog[0]["description"] = "changed"
    assert library.catalog()[0]["description"] != "changed"


@pytest.mark.parametrize("skill_id", [None, True, [], 1, "", "../.env", "data-exploration/../analysis-validation", "C:/Users/.env", "analysis-validation\x00", "THIRD_PARTY_LICENSE.txt"])
def test_unknown_or_path_like_ids_cannot_trigger_any_resource_read(skill_id, monkeypatch):
    library = skills.SkillLibrary()

    def blocked(*args):
        raise AssertionError("Invalid skill ID reached resource loading")

    monkeypatch.setattr(skills.resources, "files", blocked)
    with pytest.raises(ValueError, match="Unknown skill ID"):
        library.read(skill_id)


def test_modified_body_is_rejected_and_windows_line_endings_are_stable(tmp_path, monkeypatch):
    copy = resource_copy(tmp_path)
    monkeypatch.setattr(skills.resources, "files", lambda name: copy)
    library = skills.SkillLibrary()
    path = copy / "data-exploration/SKILL.md"
    expected = library.read("data-exploration")
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert library.read("data-exploration") == expected
    with path.open("ab") as stream:
        stream.write(b"\nUnreviewed content")
    with pytest.raises(ValueError, match="reviewed fingerprint"):
        library.read("data-exploration")


def test_oversized_resource_is_bounded_before_content_is_returned(tmp_path, monkeypatch):
    copy = resource_copy(tmp_path)
    monkeypatch.setattr(skills.resources, "files", lambda name: copy)
    library = skills.SkillLibrary()
    (copy / "data-exploration/SKILL.md").write_bytes(b"x" * (skills.MAX_SKILL_BYTES * 2 + 1))
    with pytest.raises(ValueError, match="size limit"):
        library.read("data-exploration")


@pytest.mark.parametrize("mutation", [
    lambda m: m["skills"][0].update(id="../.env"),
    lambda m: m["skills"][0].update(description="x" * 301),
    lambda m: m["skills"][0].update(sha256="not-a-fingerprint"),
    lambda m: m["skills"][0].update(byte_size=skills.MAX_SKILL_BYTES + 1),
    lambda m: [row.update(byte_size=skills.MAX_SKILL_BYTES) for row in m["skills"]],
])
def test_unreviewed_or_oversized_manifest_is_rejected(tmp_path, monkeypatch, mutation):
    copy = resource_copy(tmp_path)
    manifest = copy / "manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(value)
    manifest.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(skills.resources, "files", lambda name: copy)
    with pytest.raises(ValueError):
        skills.SkillLibrary()


def test_skill_files_are_loaded_from_a_zipped_installed_package(tmp_path):
    """Exercise Traversable resources without a source checkout/cwd fallback."""
    root = Path(skills.__file__).parent
    archive = tmp_path / "packaged-skills.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("labweaver/__init__.py", "")
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                bundle.write(path, "labweaver/skills/" + path.relative_to(root).as_posix())
    # An unrelated cwd with a same-named decoy must not influence discovery.
    decoy = tmp_path / "skills/data-exploration"
    decoy.mkdir(parents=True)
    (decoy / "SKILL.md").write_text("UNREVIEWED_CWD_SKILL", encoding="utf-8")
    code = """import json,sys
sys.path.insert(0,sys.argv[1])
from labweaver.skills import SkillLibrary
import labweaver.skills as module
library=SkillLibrary()
print(json.dumps({'origin':module.__file__,'items':[library.read(x['id']) for x in library.catalog()]},ensure_ascii=False))
"""
    result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code, str(archive)], cwd=tmp_path,
                            check=True, capture_output=True, text=True, encoding="utf-8", timeout=20)
    data = json.loads(result.stdout)
    assert str(archive) in data["origin"]
    assert len(data["items"]) == 4
    assert all("UNREVIEWED_CWD_SKILL" not in item["content"] for item in data["items"])


def test_symlinked_skill_file_cannot_read_outside_package(tmp_path, monkeypatch):
    copy = resource_copy(tmp_path)
    path = copy / "data-exploration/SKILL.md"
    path.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_PACKAGE", encoding="utf-8")
    try:
        path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"OS does not permit this test symlink: {exc.errno}")
    monkeypatch.setattr(skills.resources, "files", lambda name: copy)
    with pytest.raises(ValueError, match="symbolic links"):
        skills.SkillLibrary().read("data-exploration")


def test_license_and_attribution_are_distributed_but_not_exposed_as_skills():
    root = Path(skills.__file__).parent
    license_text = (root / "THIRD_PARTY_LICENSE.txt").read_text(encoding="utf-8")
    attribution = (root / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "Apache License" in license_text and "END OF TERMS AND CONDITIONS" in license_text
    assert "2574e29377c7173d0cecffef29a6ecbd7f8ccb5d" in attribution
    assert "unrelated non-license material" in attribution
    for item in skills.SkillLibrary().catalog():
        assert "license" not in item["id"].lower()
