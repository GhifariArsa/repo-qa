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

    def run_with_timeout(self, cypher: str, timeout: float, **params: Any) -> list[dict] | None:
        """Run `cypher`, returning None if it exceeds `timeout` seconds.

        Variable-length traversals on a large graph have wildly pair-dependent
        cost: the same query is milliseconds for one node pair and minutes for
        another in a dense region. A generator that cannot skip such a pair
        stalls the whole run, so callers that sample candidates use this and
        move on to the next candidate instead.
        """
        try:
            with self._driver.session(database=self.cfg.database) as session:
                tx = session.begin_transaction(timeout=timeout)
                try:
                    return [record.data() for record in tx.run(cypher, **params)]
                finally:
                    tx.close()
        except Exception:
            return None

    def apply_schema(self) -> int:
        """Apply every constraint and index in schema.cypher.

        Comment lines are stripped from each statement rather than used to
        filter statements out: splitting on ';' leaves the preceding comment
        block attached to the *next* statement, so discarding chunks that start
        with '//' silently dropped every statement that happened to be
        documented -- which was three of them, including the `codenode_id`
        index that edge ingest depends on.
        """
        statements = []
        for chunk in SCHEMA_CYPHER.read_text().split(";"):
            body = "\n".join(
                line for line in chunk.splitlines() if not line.strip().startswith("//")
            ).strip()
            if body:
                statements.append(body)
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
