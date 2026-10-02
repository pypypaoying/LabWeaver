"""Write JSON-safe run artifacts without replacing existing reports."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def save_run(report: dict, output_dir: str | Path = "runs") -> Path:
    document = {
        "run_id": uuid4().hex,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **report,
    }
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{document['run_id']}.json"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
    return target

