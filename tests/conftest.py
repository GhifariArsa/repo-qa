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
