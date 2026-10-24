import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def minigraph():
    from cgx.extract.pipeline import extract_repo

    graph, meta = extract_repo("minirepo", ROOT / "fixtures/minirepo", ROOT / "fixtures")
    return graph, meta


@pytest.fixture(scope="session")
def neo4j_client():
    """Live Neo4j, or skip. Integration tests only."""
    from cgx.config import load_config
    from cgx.graph.client import GraphClient

    cfg = load_config(ROOT / "config/repos.yaml")
    try:
        client = GraphClient(cfg.neo4j)
        client.verify()
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"Neo4j unavailable: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="session")
def loaded(neo4j_client):
    """Run integration queries in a disposable repo namespace, never minirepo."""
    from uuid import uuid4
    from cgx.extract.pipeline import extract_repo
    from cgx.graph.ingest import ingest, clear_repo

    repo_id = f"cgx_test_{uuid4().hex}"
    graph, _ = extract_repo(repo_id, ROOT / "fixtures/minirepo", ROOT / "fixtures")
    neo4j_client.apply_schema()

    class ScopedClient:
        def run(self, cypher, **params):
            if params.get("repo") == "minirepo":
                params["repo"] = repo_id
            return neo4j_client.run(cypher, **params)

        def run_with_timeout(self, cypher, timeout, **params):
            if params.get("repo") == "minirepo":
                params["repo"] = repo_id
            return neo4j_client.run_with_timeout(cypher, timeout, **params)

    try:
        ingest(neo4j_client, graph.to_dict(), replace=False)
        yield ScopedClient()
    finally:
        clear_repo(neo4j_client, repo_id)
