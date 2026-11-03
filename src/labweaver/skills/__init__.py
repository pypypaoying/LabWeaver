"""Reviewed analysis guidance loaded only from this installed package.

No skill can provide executable tools, runtime paths, or extra permissions.
Fingerprints use canonical UTF-8/LF so Windows checkouts and wheels agree.
"""

from __future__ import annotations

import copy
import hashlib
from importlib import resources
import json
from pathlib import Path
from types import MappingProxyType


SKILL_IDS = (
    "data-exploration",
    "statistical-analysis",
    "data-visualization",
    "analysis-validation",
)
MAX_SKILL_BYTES = 12 * 1024
MAX_TOTAL_BYTES = 32 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024


def _resource_text(parts, max_bytes):
    root = resources.files(__name__)
    item = root
    for part in parts:
        item = item.joinpath(part)
        if isinstance(item, Path) and item.is_symlink():
            raise ValueError("Packaged skill resources must not be symbolic links")
    # Only fixed package-relative components reach this function. Traversable
    # also supports zipped installations; no working-directory fallback exists.
    with item.open("rb") as stream:
        raw = stream.read(max_bytes * 2 + 1)
    if len(raw) > max_bytes * 2:
        raise ValueError("Packaged skill resource exceeds its size limit")
    try:
        text = raw.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise ValueError("Packaged skill resource must be UTF-8") from exc
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("Packaged skill resource exceeds its size limit")
    return text


class SkillLibrary:
    """Fixed, versioned resources; summaries do not eagerly load skill bodies."""

    def __init__(self):
        try:
            manifest = json.loads(_resource_text(("manifest.json",), _MAX_MANIFEST_BYTES))
            rows = manifest["skills"]
            if manifest["version"] != 1 or not isinstance(rows, list):
                raise ValueError("Unsupported skill manifest")
            if [row["id"] for row in rows] != list(SKILL_IDS):
                raise ValueError("Skill manifest must contain the reviewed fixed IDs")
            for row in rows:
                if not isinstance(row["name"], str) or not 1 <= len(row["name"]) <= 80:
                    raise ValueError("Invalid packaged skill name")
                if not isinstance(row["description"], str) or not 1 <= len(row["description"]) <= 300:
                    raise ValueError("Invalid packaged skill description")
                digest = row["sha256"]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Invalid packaged skill fingerprint")
                if type(row["byte_size"]) is not int or not 1 <= row["byte_size"] <= MAX_SKILL_BYTES:
                    raise ValueError("Packaged skill size exceeds its limit")
                source = row["source"]
                if (not isinstance(source, dict)
                        or source.get("repository") != "anthropics/knowledge-work-plugins"
                        or source.get("license") != "Apache-2.0"
                        or source.get("commit") != "2574e29377c7173d0cecffef29a6ecbd7f8ccb5d"):
                    raise ValueError("Unreviewed packaged skill source")
            if sum(row["byte_size"] for row in rows) > MAX_TOTAL_BYTES:
                raise ValueError("Packaged skill library exceeds its total limit")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid packaged skill manifest") from exc
        self._records = MappingProxyType({row["id"]: copy.deepcopy(row) for row in rows})

    def catalog(self):
        """Return small independent summaries; never include complete bodies."""
        return [{"id": row["id"], "description": row["description"], "sha256": row["sha256"]}
                for row in self._records.values()]

    def read(self, skill_id):
        """Read one whitelisted ID and verify its exact reviewed content."""
        if not isinstance(skill_id, str) or skill_id not in self._records:
            raise ValueError("Unknown skill ID; select an ID from the packaged catalog")
        row = self._records[skill_id]
        text = _resource_text((skill_id, "SKILL.md"), MAX_SKILL_BYTES)
        raw = text.encode("utf-8")
        if len(raw) != row["byte_size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Packaged skill content differs from its reviewed fingerprint")
        return {"status": "completed", "skill_id": skill_id, "name": row["name"],
                "sha256": row["sha256"], "content": text, "source": copy.deepcopy(row["source"])}


__all__ = ["SkillLibrary", "SKILL_IDS", "MAX_SKILL_BYTES", "MAX_TOTAL_BYTES"]
