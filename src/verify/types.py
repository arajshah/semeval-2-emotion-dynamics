from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Ctx:
    repo_root: Path
    mode: str
    tasks: List[str]
    seed: int


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""
    hint: str = ""
    path: str = ""


def any_failed(results: List[CheckResult]) -> bool:
    return any(not result.passed for result in results)
