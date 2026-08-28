"""Config loading: Neo4j connection details and the target-repo registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("config/repos.yaml")


@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "cgxpassword"
    database: str = "neo4j"


@dataclass
class RepoConfig:
    id: str
    path: Path
    package_root: Path
    created_at: str | None = None
    last_commit_at: str | None = None
    source: str | None = None


@dataclass
class Config:
    neo4j: Neo4jConfig
    repos: dict[str, RepoConfig]

    def repo(self, repo_id: str) -> RepoConfig:
        try:
            return self.repos[repo_id]
        except KeyError:
            known = ", ".join(sorted(self.repos)) or "<none>"
            raise SystemExit(f"unknown repo id {repo_id!r}; known ids: {known}") from None


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    neo = Neo4jConfig(**(raw.get("neo4j") or {}))
    repos: dict[str, RepoConfig] = {}
    for entry in raw.get("repos") or []:
        repos[entry["id"]] = RepoConfig(
            id=entry["id"],
            path=Path(entry["path"]),
            package_root=Path(entry.get("package_root") or Path(entry["path"]).parent),
            created_at=entry.get("created_at"),
            last_commit_at=entry.get("last_commit_at"),
            source=entry.get("source"),
        )
    return Config(neo4j=neo, repos=repos)
