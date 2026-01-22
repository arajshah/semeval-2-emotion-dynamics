from __future__ import annotations

import argparse
import re
from datetime import datetime
from typing import Optional


_TASK_TAG_SAFE = re.compile(r"[^a-zA-Z0-9_]+")
_RUN_ID_ALLOWED = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _sanitize_task_tag(task_tag: str) -> str:
    return _TASK_TAG_SAFE.sub("_", task_tag).strip("_")


def generate_run_id(
    task_tag: str,
    seed: int | None = None,
    *,
    now: datetime | None = None,
) -> str:
    safe_tag = _sanitize_task_tag(task_tag)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_id = f"{safe_tag}_{timestamp}"
    if seed is not None:
        run_id = f"{run_id}_seed{seed}"
    return run_id


def validate_run_id(run_id: str) -> None:
    if not run_id:
        raise ValueError("run_id must be non-empty.")
    if "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must not contain path separators.")
    if any(ch.isspace() for ch in run_id):
        raise ValueError("run_id must not contain whitespace.")
    if not _RUN_ID_ALLOWED.fullmatch(run_id):
        raise ValueError("run_id contains invalid characters.")


def resolve_run_id(
    provided_run_id: Optional[str],
    task_tag: str,
    seed: int | None = None,
) -> str:
    if provided_run_id is not None:
        validate_run_id(provided_run_id)
        return provided_run_id
    return generate_run_id(task_tag, seed=seed)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a run_id.")
    parser.add_argument("--task_tag", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_id = generate_run_id(args.task_tag, seed=args.seed)
    print(run_id)


if __name__ == "__main__":
    _main()
