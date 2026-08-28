"""Thin Neo4j driver wrapper."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from neo4j import GraphDatabase

from ..config import Neo4jConfig

SCHEMA_CYPHER = Path(__file__).parent / "cypher" / "schema.cypher"


class GraphClient:
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self._driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run(self, cypher: str, **params: Any) -> list[dict]:
        with self._driver.session(database=self.cfg.database) as session:
            return [record.data() for record in session.run(cypher, **params)]

    def apply_schema(self) -> int:
        statements = [
            s.strip()
            for s in SCHEMA_CYPHER.read_text().split(";")
            if s.strip() and not s.strip().startswith("//")
        ]
        for stmt in statements:
            self.run(stmt)
        return len(statements)

    def verify(self) -> None:
        self._driver.verify_connectivity()


@contextmanager
def client(cfg: Neo4jConfig) -> Iterator[GraphClient]:
    c = GraphClient(cfg)
    try:
        yield c
    finally:
        c.close()
