"""cgx command line: extract | ingest | generate | verify | select | evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import load_config
from .logconfig import setup_logging
from .extract.pipeline import extract_repo, write_graph
from .graph.client import client as graph_client
from .graph.ingest import ingest as ingest_graph, load_graph_json
from .qgen.base import read_questions, write_questions
from .qgen.generate import generate as generate_questions

app = typer.Typer(add_completion=False, help=__doc__)

DATA = Path("data")


def _graph_path(repo_id: str) -> Path:
    return DATA / "graphs" / f"{repo_id}.json"


def _questions_path(repo_id: str) -> Path:
    return DATA / "questions" / f"{repo_id}.jsonl"


@app.command()
def extract(
    repo: str = typer.Argument(..., help="repo id from config/repos.yaml"),
    config: Path = typer.Option(Path("config/repos.yaml"), "--config"),
    out: Path = typer.Option(None, "--out"),
    include_tests: bool = typer.Option(
        False, "--include-tests", help="also extract test directories (excluded by default)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="per-file detail"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="warnings and errors only"),
):
    """Static-analyse a repo into a CodexGraph-schema graph JSON."""
    setup_logging(verbose=verbose, quiet=quiet)
    cfg = load_config(config)
    rc = cfg.repo(repo)
    graph, meta = extract_repo(rc.id, rc.path, rc.package_root, include_tests=include_tests)
    path = write_graph(graph, out or _graph_path(rc.id))
    typer.echo(json.dumps(meta["stats"], indent=2))
    if meta["rejected_calls"]:
        typer.echo(
            f"\n{len(meta['rejected_calls'])} pyan edge(s) rejected by the call-site "
            f"cross-check (field-access artifacts, not calls)."
        )
    typer.echo(f"\nwrote {path}")


@app.command()
def ingest(
    repo: str = typer.Argument(...),
    config: Path = typer.Option(Path("config/repos.yaml"), "--config"),
    graph: Path = typer.Option(None, "--graph"),
    keep: bool = typer.Option(False, "--keep", help="do not clear existing nodes for this repo"),
):
    """Apply the Cypher schema and load a graph JSON into Neo4j."""
    cfg = load_config(config)
    data = load_graph_json(graph or _graph_path(repo))
    with graph_client(cfg.neo4j) as c:
        c.verify()
        n = c.apply_schema()
        counts = ingest_graph(c, data, replace=not keep)
    typer.echo(f"applied {n} schema statements; ingested {counts}")


@app.command()
def generate(
    repo: str = typer.Argument(...),
    config: Path = typer.Option(Path("config/repos.yaml"), "--config"),
    per_template: int = typer.Option(5, "--per-template", "-n"),
    seed: int = typer.Option(0, "--seed"),
    include_tier5: bool = typer.Option(False, "--include-tier5"),
    out: Path = typer.Option(None, "--out"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    """Generate tier-labelled Q&A pairs from the graph."""
    setup_logging(verbose=verbose, quiet=quiet)
    cfg = load_config(config)
    tiers = (1, 2, 3, 4, 5) if include_tier5 else (1, 2, 3, 4)
    with graph_client(cfg.neo4j) as c:
        records, stats = generate_questions(c, repo, tiers, per_template, seed)
    path = write_questions(records, out or _questions_path(repo))
    typer.echo(json.dumps(stats, indent=2))
    typer.echo(f"\nwrote {len(records)} questions to {path}")


@app.command()
def naturalize(
    repo: str = typer.Argument(...),
    config: Path = typer.Option(Path("config/repos.yaml"), "--config"),
    questions: Path = typer.Option(None, "--questions"),
    out: Path = typer.Option(None, "--out"),
    model: str = typer.Option("anthropic:claude-opus-5", "--model",
                              help="<provider>:<model>, e.g. anthropic:claude-opus-5 or openai:gpt-4.1"),
    personas: int = typer.Option(1, "--personas", help="how many persona rewrites to try per question"),
    no_judge: bool = typer.Option(False, "--no-judge", help="skip the LLM same-answer check"),
    limit: int = typer.Option(None, "--limit"),
    batch: bool = typer.Option(False, "--batch", help="use the Batches API (50%% cheaper)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print prompts, make no API calls"),
):
    """Rewrite templated questions as natural developer questions (answers untouched)."""
    from .qgen.naturalize import (
        NaturalizationConfig, Naturalizer, build_rewrite_prompt, collect_batch,
        load_vocabulary, submit_batch,
    )
    from .qgen.personas import personas_for_tier

    cfg = load_config(config)
    qpath = questions or _questions_path(repo)
    records = read_questions(qpath)
    if limit:
        records = records[:limit]
    ncfg = NaturalizationConfig(model=model, judge=not no_judge, per_question=personas)

    with graph_client(cfg.neo4j) as c:
        vocab = load_vocabulary(c, repo)

    if dry_run:
        for rec in records:
            ps = personas_for_tier(rec.tier)[:personas]
            typer.echo("=" * 78)
            typer.echo(f"{rec.id}   (tier {rec.tier}, personas: {', '.join(p.id for p in ps)})")
            typer.echo(build_rewrite_prompt(rec, vocab, ps))
        typer.echo(f"\n{len(records)} prompt(s) shown. No API calls made.")
        return

    naturalizer = Naturalizer(ncfg)
    if batch:
        batch_id = submit_batch(naturalizer.client, records, vocab, ncfg)
        typer.echo(f"submitted batch {batch_id}; poll with `cgx naturalize-collect {repo} {batch_id}`")
        return

    for i, rec in enumerate(records, 1):
        naturalizer.naturalize(rec, vocab)
        typer.echo(f"[{i}/{len(records)}] {rec.naturalization.get('status')}  {rec.id}")

    path = write_questions(records, out or qpath)
    _naturalization_summary(records)
    typer.echo(f"\nwrote {path}")


@app.command("naturalize-collect")
def naturalize_collect(
    repo: str = typer.Argument(...),
    batch_id: str = typer.Argument(...),
    config: Path = typer.Option(Path("config/repos.yaml"), "--config"),
    questions: Path = typer.Option(None, "--questions"),
    model: str = typer.Option("anthropic:claude-opus-5", "--model"),
    no_judge: bool = typer.Option(False, "--no-judge"),
):
    """Apply a finished naturalization batch to the question set."""
    from .qgen.naturalize import NaturalizationConfig, Naturalizer, collect_batch, load_vocabulary

    cfg = load_config(config)
    qpath = questions or _questions_path(repo)
    records = read_questions(qpath)
    ncfg = NaturalizationConfig(model=model, judge=not no_judge)
    naturalizer = Naturalizer(ncfg)
    with graph_client(cfg.neo4j) as c:
        vocab = load_vocabulary(c, repo)
    collect_batch(naturalizer.client, batch_id, records, vocab, ncfg, naturalizer)
    write_questions(records, qpath)
    _naturalization_summary(records)
    typer.echo(f"\nwrote {qpath}")


def _naturalization_summary(records) -> None:
    from collections import Counter

    status = Counter(r.naturalization.get("status", "not_run") for r in records)
    persona = Counter(r.naturalization.get("persona") for r in records if r.naturalization.get("persona"))
    rejected = Counter(
        a.get("rejected_by", "error")
        for r in records
        for a in r.naturalization.get("rejected_attempts", [])
    )
    typer.echo("\nnaturalization: " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())))
    if persona:
        typer.echo("personas used:  " + ", ".join(f"{k}={v}" for k, v in sorted(persona.items())))
    if rejected:
        typer.echo("rejected by:    " + ", ".join(f"{k}={v}" for k, v in sorted(rejected.items())))


@app.command()
def verify(
    repo: str = typer.Argument(...),
    tier: int = typer.Option(None, "--tier", help="restrict to one tier"),
    sample: int = typer.Option(10, "--sample", "-n"),
    seed: int = typer.Option(0, "--seed"),
    status: str = typer.Option("pending", "--status", help="which records to review"),
    questions: Path = typer.Option(None, "--questions"),
):
    """Review sampled questions and mark them verified or rejected."""
    from .verify.cli import run_review

    run_review(questions or _questions_path(repo), tier, sample, seed, status)


@app.command("verify-report")
def verify_report(
    repo: str = typer.Argument(...),
    questions: Path = typer.Option(None, "--questions"),
):
    """Show verification coverage per tier."""
    from .verify.cli import report

    report(questions or _questions_path(repo))


@app.command("recheck")
def recheck(
    repo: str = typer.Argument(...),
    config: Path = typer.Option(Path("config/repos.yaml"), "--config"),
    questions: Path = typer.Option(None, "--questions"),
):
    """Re-run every stored Cypher and confirm it still yields the stored answer."""
    cfg = load_config(config)
    records = read_questions(questions or _questions_path(repo))
    from .verify.cli import recheck_answers

    with graph_client(cfg.neo4j) as c:
        mismatches = recheck_answers(c, records)
    if mismatches:
        typer.echo(f"{len(mismatches)} MISMATCH(es):")
        for m in mismatches:
            typer.echo(f"  {m}")
        raise typer.Exit(1)
    typer.echo(f"all {len(records)} stored answers reproduce from their Cypher")


@app.command()
def select(
    out: Path = typer.Option(Path("data/candidates.json"), "--out"),
    min_created: str = typer.Option("2025-06-01", "--min-created"),
    min_stars: int = typer.Option(50, "--min-stars"),
    limit: int = typer.Option(30, "--limit"),
    query: str = typer.Option("language:python", "--query"),
):
    """Find contamination-resistant candidate repos (created after a cutoff)."""
    from .select.repos import find_candidates

    find_candidates(out=out, min_created=min_created, min_stars=min_stars, limit=limit, query=query)


@app.command()
def evaluate(
    repo: str = typer.Argument(...),
    questions: Path = typer.Option(None, "--questions"),
    out: Path = typer.Option(None, "--out"),
    model: str = typer.Option("stub", "--model"),
    limit: int = typer.Option(None, "--limit"),
):
    """Score a model's answers per tier (Phase 5; `stub` runs the scorer only)."""
    from .eval.harness import run_evaluation

    run_evaluation(
        questions_path=questions or _questions_path(repo),
        out_path=out or DATA / "eval" / f"{repo}.json",
        model=model,
        limit=limit,
    )


if __name__ == "__main__":
    app()
