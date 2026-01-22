from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.verify import shared
from src.verify.shared import CheckResult, VerifyContext
from src.verify import subtask1, subtask2a, subtask2b


def _find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        if (parent / "README.md").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verification gate (Phase VG-0).")
    parser.add_argument("--mode", choices=["smoke", "strict"], default="smoke")
    parser.add_argument(
        "--tasks",
        choices=["all", "subtask1", "subtask2a", "subtask2b"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    repo_root = _find_repo_root()
    tasks = ["subtask1", "subtask2a", "subtask2b"] if args.tasks == "all" else [args.tasks]
    ctx = VerifyContext(
        repo_root=repo_root,
        mode=args.mode,
        tasks=tasks,
        seed=args.seed,
        run_id=args.run_id,
    )

    results: List[CheckResult] = []
    args_message = f"mode={ctx.mode}, tasks={ctx.tasks}, seed={ctx.seed}, run_id={ctx.run_id}"
    results.append(shared.pass_result("args_parsed", args_message))
    results.extend(shared.run_checks(ctx))

    for task in tasks:
        if task == "subtask1":
            results.extend(subtask1.run_checks(ctx))
        elif task == "subtask2a":
            results.extend(subtask2a.run_checks(ctx))
        elif task == "subtask2b":
            results.extend(subtask2b.run_checks(ctx))

    if ctx.mode == "strict":
        results.extend(shared.run_strict_shared_checks(ctx))

    header = f"Verification Gate (mode={ctx.mode}, tasks={','.join(tasks)}, seed={ctx.seed})"
    shared.print_results(header, results)
    raise SystemExit(shared.exit_code(results))


if __name__ == "__main__":
    main()
