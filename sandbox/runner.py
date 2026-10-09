"""Container-side supervisor: generated stdout cannot impersonate protocol output."""

import base64
import json
from pathlib import Path
import stat
import subprocess
import time

MAX_BYTES = 50 * 1024 * 1024
LOG_BYTES = 16 * 1024


def regular_file(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Only regular artifact files are allowed")
    return path.read_bytes()


def main():
    start = time.monotonic()
    with Path("/work/stdout").open("wb") as out, Path("/work/stderr").open("wb") as err:
        process = subprocess.Popen(
            ["python", "/input/analysis.py"], stdout=out, stderr=err
        )
        code = process.wait()

    def preview(name):
        path = Path("/work") / name
        with path.open("rb") as stream:
            data = stream.read(LOG_BYTES)
        return {
            "text": data.decode("utf-8", errors="replace"),
            "truncated": path.stat().st_size > LOG_BYTES,
        }

    result = {
        "protocol": 1,
        "exit_code": code,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "stdout": preview("stdout"),
        "stderr": preview("stderr"),
        "artifacts": [],
    }
    if code == 0:
        try:
            root = Path("/work/artifacts")
            total = 0
            if root.exists() and (root.is_symlink() or not root.is_dir()):
                raise ValueError("Invalid artifact root")
            for entry in sorted(root.iterdir()) if root.exists() else []:
                if (
                    entry.is_symlink()
                    or not entry.is_dir()
                    or len(result["artifacts"]) >= 50
                ):
                    raise ValueError("Invalid artifact directory or count")
                metadata_bytes = regular_file(entry / "metadata.json")
                total += len(metadata_bytes)
                if len(metadata_bytes) > 32 * 1024:
                    raise ValueError("Artifact metadata exceeds 32 KiB")
                metadata = json.loads(metadata_bytes)
                files = {}
                for path in sorted(entry.iterdir()):
                    if path.name == "metadata.json":
                        continue
                    size = path.lstat().st_size
                    total += size
                    if total > MAX_BYTES:
                        raise ValueError("Total artifacts exceed 50 MiB")
                    files[path.name] = base64.b64encode(regular_file(path)).decode(
                        "ascii"
                    )
                result["artifacts"].append({"metadata": metadata, "files": files})
        except (OSError, ValueError, TypeError) as exc:
            result.update(artifacts=[], protocol_error=str(exc))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
