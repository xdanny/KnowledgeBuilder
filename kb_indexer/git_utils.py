from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Change:
    status: str
    old_path: str | None
    new_path: str | None


def git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *cmd],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolve_sha(repo_root: Path, rev: str = "HEAD") -> str:
    return git(["rev-parse", rev], repo_root)


def commit_exists(repo_root: Path, sha: str) -> bool:
    if not sha:
        return False
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def list_all_files(repo_root: Path) -> list[str]:
    output = git(["ls-files"], repo_root)
    return [line for line in output.splitlines() if line]


def parse_git_changes(repo_root: Path, base_sha: str, head_sha: str) -> list[Change]:
    output = git(["diff", "--name-status", "--find-renames", base_sha, head_sha], repo_root)
    changes: list[Change] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status_code = parts[0]
        status = status_code[0]
        if status == "R" and len(parts) >= 3:
            changes.append(Change(status="R", old_path=parts[1], new_path=parts[2]))
        elif status == "D" and len(parts) >= 2:
            changes.append(Change(status="D", old_path=parts[1], new_path=None))
        elif status in {"A", "M"} and len(parts) >= 2:
            changes.append(Change(status=status, old_path=None, new_path=parts[1]))
    return changes
