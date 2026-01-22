from __future__ import annotations

import re
import subprocess
from pathlib import Path


_GIT_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")


def get_git_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    output = proc.stdout.strip()
    return output if _GIT_SHA.fullmatch(output) else "unknown"
