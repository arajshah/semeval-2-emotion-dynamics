from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import hashlib
import json
import os
import platform
import subprocess

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

    payload = _ensure_manifest(payload, repo_root, run_id)
    deep_update(payload, updates)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)  # type: ignore[arg-type]
        else:
            dst[key] = value
    return dst


def artifact_ref(path: Path, repo_root: Path | None = None) -> Dict[str, Any]:
    ref: Dict[str, Any] = {"path": str(path)}
    try:
        if repo_root is not None:
            try:
                ref["path"] = str(Path(path).resolve().relative_to(repo_root.resolve()))
            except Exception:
                ref["path"] = str(path)
        exists = Path(path).exists()
        ref["exists"] = bool(exists)
        if exists:
            ref["sha256"] = sha256_file(Path(path))
            ref["bytes"] = Path(path).stat().st_size
    except Exception:
        ref["exists"] = False
        ref["sha256"] = "unknown"
        ref["bytes"] = None
    return ref


def get_git_snapshot(repo_root: Path) -> Dict[str, Any]:
    snapshot = {"commit": "unknown", "branch": "unknown", "is_dirty": None}
    try:
        snapshot["commit"] = get_git_commit(repo_root)
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if proc.stdout:
            snapshot["branch"] = proc.stdout.strip()
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        snapshot["is_dirty"] = bool(proc.stdout.strip())
    except Exception:
        pass
    return snapshot


def get_env_snapshot(device: str | None = None) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch  # type: ignore

        snapshot["torch_version"] = torch.__version__
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        snapshot["torch_version"] = "unknown"
        snapshot["cuda_available"] = None
    if device is not None:
        snapshot["device"] = device
    return snapshot


def _ensure_manifest(payload: Dict[str, Any], repo_root: Path, run_id: str) -> Dict[str, Any]:
    payload.setdefault("schema_version", "run_manifest_v1")
    payload.setdefault("run_id", run_id)
    payload.setdefault("created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    payload.setdefault("updated_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    payload.setdefault("git", get_git_snapshot(repo_root))
    payload.setdefault("env", get_env_snapshot())
    payload.setdefault("inputs", {})
    payload.setdefault("artifacts", {})
    payload.setdefault("counts", {})
    payload.setdefault("metrics", {})
    return payload


def default_subtask1_artifact_paths(run_id: str) -> Dict[str, str]:
    validate_run_id(run_id)
    return {
        "subtask1_val_preds": f"reports/preds/subtask1_val_preds__{run_id}.parquet",
        "subtask1_val_user_agg": f"reports/preds/subtask1_val_user_agg__{run_id}.parquet",
    }
