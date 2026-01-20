from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def generate_run_id(prefix: str, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{seed}"


def compute_config_hash(config: Dict[str, Any], config_path: Path | None = None) -> str:
    payload: Dict[str, Any] = dict(config)
    if config_path is not None and config_path.exists():
        payload["config_path"] = str(config_path)
        payload["config_file_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_git_commit(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"
