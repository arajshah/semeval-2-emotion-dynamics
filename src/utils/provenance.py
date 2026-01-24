from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict
import hashlib
import json

from src.utils.git_utils import get_git_commit
from src.utils.hash_utils import config_hash
from src.utils.run_id import validate_run_id


def write_run_metadata(
    *,
    repo_root: Path,
    run_id: str,
    task: str,
    task_tag: str,
    seed: int | None,
    regime: str,
    split_path: str,
    config: dict,
    artifacts: Dict[str, str],
) -> Path:
    validate_run_id(run_id)
    runs_dir = repo_root / "reports" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "task": task,
        "task_tag": task_tag,
        "seed": seed,
        "regime": regime,
        "split_path": split_path,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(repo_root),
        "config_hash": config_hash(config),
        "config": config,
        "artifacts": artifacts,
    }

    out_path = runs_dir / f"{run_id}.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        existing.update(metadata)
        metadata = existing
    out_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def merge_run_metadata(*, repo_root: Path, run_id: str, updates: Dict[str, object]) -> Path:
    validate_run_id(run_id)
    runs_dir = repo_root / "reports" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"{run_id}.json"
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    payload.update(updates)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_subtask1_artifact_paths(run_id: str) -> Dict[str, str]:
    validate_run_id(run_id)
    return {
        "subtask1_val_preds": f"reports/preds/subtask1_val_preds__{run_id}.parquet",
        "subtask1_val_user_agg": f"reports/preds/subtask1_val_user_agg__{run_id}.parquet",
    }
