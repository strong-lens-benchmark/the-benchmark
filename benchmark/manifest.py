from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORG_ROOT = PROJECT_ROOT
REPOS = [
    "benchmark",
    "analosis",
    "lenstronomy",
    "JAXtronomy",
    "herculens",
    "TinyLensGpu",
    "PyAutoLens",
    "blackjax",
    "caskade",
]


@dataclass(frozen=True)
class RepoState:
    name: str
    path: str
    commit: str | None
    branch: str | None
    remote: str | None
    dirty: bool
    status_short: str


def repo_state(path: Path) -> RepoState:
    name = path.name
    commit = git(path, "rev-parse", "HEAD")
    branch = git(path, "branch", "--show-current")
    remote = git(path, "remote", "get-url", "origin")
    status = git(path, "status", "--short") or ""
    return RepoState(
        name=name,
        path=str(path),
        commit=commit,
        branch=branch,
        remote=remote,
        dirty=bool(status.strip()),
        status_short=status,
    )


def collect_manifest() -> dict:
    repos = []
    for repo in REPOS:
        path = ORG_ROOT / repo
        if path.exists():
            repos.append(asdict(repo_state(path)))
    lockfile = ORG_ROOT / "benchmark" / "uv.lock"
    return {
        "repos": repos,
        "uv_lock_sha256": sha256(lockfile) if lockfile.exists() else None,
    }


def write_manifest(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collect_manifest(), indent=2, sort_keys=True) + "\n")


def git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Path to the JSON manifest to write.")
    args = parser.parse_args()
    write_manifest(args.output)
