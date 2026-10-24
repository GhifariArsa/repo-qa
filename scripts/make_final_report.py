"""Build the final-experiment HTML report from the rescored results.

    uv run python scripts/make_final_report.py

Reads data/eval/final/rescored.jsonl (scripts/rescore.py), the per-model
records in data/eval/final/models/, and the GPU log; writes
docs/reports/2026-10-21-final-experiment.html. Styling is taken from the
pilot report so the two read as one series.
"""

from __future__ import annotations

import collections
import csv
import glob
import html
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data/eval/final"
OUT = ROOT / "docs/reports/2026-10-21-final-experiment.html"
PILOT = ROOT / "docs/reports/2026-09-28-pilot-report.html"

FAMILIES = [("definition_organisation", "Definition"), ("dependency_interaction", "Dependency"),
            ("inheritance_specialisation", "Inheritance"), ("state_access", "State access"),
            ("structural_comparison", "Comparison")]
TEMPLATE_LABEL = {
    "t3_reach_set_size": "How many functions can X eventually reach?",
    "t2_transitive_callers": "What ends up calling Y, at any depth?",
    "comparison_reachable_shared": "What can both X and Y reach?",
    "t2_files_touched": "Which files does X's call chain pass through?",
    "t2_shortest_chain": "Shortest call chain from X to Y",
    "comparison_reachable_difference": "What can X reach that Y cannot?",
    "inheritance_overriders": "Which subclasses override method M?",
    "t5_closure_globals_read": "Which globals does X read through its calls?",
    "t3_all_subclasses": "Every subclass of C, at any depth",
    "organisation_subclass_files": "Which files define subclasses of C?",
    "t5_closure_writes_field": "Can X end up assigning field F?",
    "t2_reachable_within_2": "What can X reach within two calls?",
    "t2_hop_count": "Fewest calls from X to Y",
    "t2_reaches_global_read": "Can X end up reading global G?",
    "t3_common_ancestor": "Nearest common base class of A and B",
    "inheritance_redeclared_methods": "Which inherited methods does C override?",
}

esc = html.escape
mean = lambda xs: st.mean(xs) if xs else float("nan")  # noqa: E731


def load():
    rows = [json.loads(line) for line in (RES / "rescored.jsonl").open()]
    info = {}
    for f in glob.glob(str(RES / "models/*.json")):
        d = json.load(open(f))
        info[re.sub(r"[^A-Za-z0-9.-]+", "_", d["model"])] = d
    gpu = collections.defaultdict(int)
    with (RES / "logs/gpu.tsv").open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["mem_used_mib"].isdigit():
                gpu[row["model"]] = max(gpu[row["model"]], int(row["mem_used_mib"]))
    summary = json.loads((RES / "rescored_summary.json").read_text())
    return rows, info, gpu, summary


def model_name(slug: str, info: dict) -> str:
    return info.get(slug, {}).get("model", slug)


def pct(x: float) -> str:
    return f"{x:.2f}"


def heat(x: float) -> str:
    return f'style="background: color-mix(in srgb, var(--repo) {round(x * 70)}%, transparent)"'


def build() -> str:
    rows, info, gpu, summary = load()
    repo = [r for r in rows if r["condition"] == "repo"]
    closed = [r for r in rows if r["condition"] == "closed"]
    models = sorted({r["model"] for r in repo}, key=lambda m: -mean([r["score_v2"] for r in repo if r["model"] == m]))
    by = lambda rs, m: [r for r in rs if r["model"] == m]  # noqa: E731

    stats = {}
    for m in models:
        rr, cc = by(repo, m), by(closed, m)
        stats[m] = {
            "repo": mean([r["score_v2"] for r in rr]), "repo_v1": mean([r["score_v1"] for r in rr]),
            "hard": mean([r["score_v2"] for r in rr if r["hard"]]),
            "lookup": mean([r["score_v2"] for r in rr if not r["hard"]]),
            "closed": mean([r["score_v2"] for r in cc]),
            "noans": sum(not r["answered"] for r in rr), "forced": sum(bool(r["forced"]) for r in rr),
            "steps": mean([r["steps"] for r in rr]), "secs": st.median([r["seconds"] for r in rr]),
            "tin": mean([r["prompt_tokens"] for r in rr]), "tout": mean([r["completion_tokens"] for r in rr]),
            "invalid": sum(r["invalid_tool_calls"] or 0 for r in rr),
            "fam": {f: mean([r["score_v2"] for r in rr if r["family"] == f]) for f, _ in FAMILIES},
        }
    n_q = len({r["id"] for r in repo})
    n_hard = len({r["id"] for r in repo if r["hard"]})
    best, worst = models[0], models[-1]
    closed_max = max(s["closed"] for s in stats.values())
    closed_min = min(s["closed"] for s in stats.values())

    # --- leaderboard chart ------------------------------------------------
    chart = []
    for m in models:
        s = stats[m]
        chart.append(
            f'<div class="row"><span class="who">{esc(model_name(m, info))}</span>'
            f'<div class="bar"><span class="r" style="width:{s["repo"] * 100:.0f}%"></span>'
            f'<span class="c" style="width:{s["closed"] * 100:.0f}%"></span></div>'
            f'<span class="vals">{pct(s["repo"])} / {pct(s["closed"])}</span></div>')

    # --- leaderboard table ------------------------------------------------
    lb = []
    for i, m in enumerate(models, 1):
        s, d = stats[m], info.get(m, {})
        slots = d.get("parallel_slots")
        lb.append(
            f"<tr><td class='n'>{i}</td><td><strong>{esc(model_name(m, info))}</strong>"
            f"<div class='sub'>{esc(str(d.get('family', '')))} · {esc(str(d.get('parameter_size', '')))} · "
            f"{esc(str(d.get('quantization', '')))}</div></td>"
            f"<td class='n'><strong>{pct(s['repo'])}</strong></td><td class='n'>{pct(s['hard'])}</td>"
            f"<td class='n'>{pct(s['lookup'])}</td><td class='n'>{pct(s['closed'])}</td>"
            f"<td class='n'>{s['noans']}</td><td class='n'>{s['steps']:.1f}</td>"
            f"<td class='n'>{s['tin'] / 1000:.0f}k / {s['tout'] / 1000:.1f}k</td>"
            f"<td class='n'>{s['secs']:.0f}</td><td class='n'>{slots if slots else '—'}</td></tr>")

    # --- family heatmap ---------------------------------------------------
    fam_rows = []
    for m in models:
        cells = "".join(f"<td class='n heat' {heat(stats[m]['fam'][f])}>{pct(stats[m]['fam'][f])}</td>"
                        for f, _ in FAMILIES)
        fam_rows.append(f"<tr><td>{esc(model_name(m, info))}</td>{cells}</tr>")
    fam_n = {f: len({r["id"] for r in repo if r["family"] == f}) for f, _ in FAMILIES}

    # --- template difficulty (top half of the field) ----------------------
    top = models[: max(1, len(models) // 2)]
    tpl = collections.defaultdict(list)
    for r in repo:
        if r["hard"] and r["model"] in top:
            tpl[r["template_id"]].append(r["score_v2"])
    tpl_rows = []
    for t, v in sorted(tpl.items(), key=lambda kv: mean(kv[1])):
        n = len({r["id"] for r in repo if r["template_id"] == t})
        tpl_rows.append(f"<tr><td>{esc(TEMPLATE_LABEL.get(t) or str(t))}<div class='sub mono'>{esc(t)}</div></td>"
                        f"<td class='n'>{n}</td><td class='n heat' {heat(mean(v))}>{pct(mean(v))}</td></tr>")

    # --- repositories -----------------------------------------------------
    repo_rows = []
    for rp in sorted({r["repo"] for r in repo}, key=lambda x: mean([r["score_v2"] for r in repo if r["repo"] == x and r["model"] in top])):
        rr = [r for r in repo if r["repo"] == rp]
        repo_rows.append(f"<tr><td>{esc(rp)}{' <span class=chip-inline>control</span>' if rp == 'django' else ''}</td>"
                         f"<td class='n'>{len({r['id'] for r in rr})}</td>"
                         f"<td class='n'>{pct(mean([r['score_v2'] for r in rr if r['model'] in top]))}</td>"
                         f"<td class='n'>{pct(mean([r['score_v2'] for r in rr]))}</td></tr>")

    # --- model hardware table ---------------------------------------------
    hw_rows = []
    for slug in models + [s for s in info if s not in models]:
        d = info[slug]
        mem = gpu.get(d["model"])
        ok = d["tool_smoke"]["ok"]
        hw_rows.append(
            f"<tr><td>{esc(d['model'])}</td><td>{esc(str(d.get('family')))}</td>"
            f"<td class='n'>{esc(str(d.get('parameter_size')))}</td><td class='n'>{esc(str(d.get('quantization')))}</td>"
            f"<td class='n'>{d.get('parallel_slots') or '—'}</td>"
            f"<td class='n'>{d['throughput'].get('generation_tokens_per_s') or '—'}</td>"
            f"<td class='n'>{f'{mem / 1024:.1f}' if mem else '—'}</td>"
            f"<td>{'yes' if ok else '<span class=bad-c>no</span>'}</td></tr>")

    changed = summary["gold_changed"]
    v1v2 = "".join(f"<tr><td>{esc(model_name(m, info))}</td><td class='n'>{pct(stats[m]['repo_v1'])}</td>"
                   f"<td class='n'>{pct(stats[m]['repo'])}</td>"
                   f"<td class='n'>{stats[m]['repo'] - stats[m]['repo_v1']:+.3f}</td></tr>" for m in models)

    css = PILOT.read_text().split("<style>", 1)[1].split("</style>", 1)[0]
    extra = """
  .lb .row, .lb .scale { grid-template-columns: 190px minmax(0, 1fr) 92px; }
  .sub { font-size: 12px; color: var(--muted); }
  td.heat { font-weight: 500; }
  .bad-c { color: var(--bad); font-weight: 600; }
  .chip-inline { font: 600 10px/1 var(--cond); letter-spacing: .06em; text-transform: uppercase; color: var(--warn); border: 1px solid currentColor; border-radius: 999px; padding: 3px 6px; margin-left: 6px; vertical-align: 1px; }
  ul.tight { margin: 0; padding-left: 20px; display: grid; gap: 8px; max-width: 74ch; }
  @media (max-width: 520px) { .lb .row, .lb .scale { grid-template-columns: 112px minmax(0, 1fr) 76px; } }
"""
    nm = lambda m: esc(model_name(m, info))  # noqa: E731
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Repo-QA Final Experiment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>{css}{extra}</style>
<style>body{{margin:0}}</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="eyebrow">Graph-derived repository QA benchmark &middot; final experiment, 17&ndash;21 October 2026</div>
    <h1>Repo-QA Final Experiment</h1>
    <p class="lede">{len(models)} open models answered the same {n_q} questions about 10 Python repositories, once with read-only access to the code and once without. Every gold answer comes from a Cypher query over a static code graph. With the code, scores range from {pct(stats[worst]['repo'])} to {pct(stats[best]['repo'])}. Without it, every model lands between {pct(closed_min)} and {pct(closed_max)}.</p>
  </header>

  <section aria-labelledby="h-figures">
    <h2 id="h-figures" class="eyebrow" style="font-size:12px">Headline</h2>
    <div class="figures">
      <div class="fig"><div class="num repo-c">{pct(stats[best]['repo'])}</div><div class="lbl">Best accuracy with repo access: {nm(best)}. Hard questions {pct(stats[best]['hard'])}.</div></div>
      <div class="fig"><div class="num closed-c">{pct(closed_min)}&ndash;{pct(closed_max)}</div><div class="lbl">Closed-book accuracy, every model. The questions cannot be answered from memory.</div></div>
      <div class="fig"><div class="num">{n_q} <span class="sep">/</span> {round(100 * n_hard / n_q)}%</div><div class="lbl">Questions, and the share that are hard: repo-wide or transitive searches across files.</div></div>
      <div class="fig"><div class="num">{len(models)} + 2</div><div class="lbl">Models scored, plus two that could not make a structured tool call and were excluded.</div></div>
    </div>
  </section>

  <section aria-labelledby="h-board">
    <h2 id="h-board">Leaderboard</h2>
    <p>Accuracy with repository access, against the same model with none. List answers score by F1, so partial credit shows. Scores use the corrected gold answers and the v2 answer-form rules; the effect of both is shown under <a href="#h-corr">Corrections</a>.</p>
    <div class="legend">
      <span class="key"><span class="swatch" style="background:var(--repo)"></span>Repo access</span>
      <span class="key"><span class="swatch" style="background:var(--closed)"></span>Closed-book</span>
    </div>
    <div class="chart lb" role="img" aria-label="Accuracy with and without repository access for each model">
      {''.join(chart)}
      <div class="scale" aria-hidden="true"><span></span><div class="ticks"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1</span></div><span></span></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th class="n">#</th><th>Model</th><th class="n">Repo</th><th class="n">Hard</th><th class="n">Lookup</th><th class="n">Closed</th><th class="n">No answer</th><th class="n">Steps</th><th class="n">Tokens in / out</th><th class="n">Median s</th><th class="n">Slots</th></tr></thead>
        <tbody>{''.join(lb)}</tbody>
      </table>
    </div>
    <p class="note">Per question with repo access: mean steps, mean input and output tokens, median wall-clock seconds. <em>No answer</em> counts questions where the model ran out of its 30 steps without submitting; they score 0. <em>Slots</em> is how many questions ran at once on the GPU; per-question time is comparable only between models with the same slots.</p>
  </section>

  <section aria-labelledby="h-fam">
    <h2 id="h-fam">By question family</h2>
    <p>Repo-access accuracy. The two ends of the field agree on what is hard: comparing what two functions can reach, and following call chains, where models stop searching before the set is complete.</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th>{''.join(f"<th class='n'>{lbl}<div class='sub'>n={fam_n[f]}</div></th>" for f, lbl in FAMILIES)}</tr></thead>
        <tbody>{''.join(fam_rows)}</tbody>
      </table>
    </div>
  </section>

  <section aria-labelledby="h-tpl">
    <h2 id="h-tpl">Which hard questions are hardest</h2>
    <p>Mean repo-access accuracy of the top {len(top)} models on each hard question type. Counting and listing everything reachable are the hardest. Single-answer traversals (a common base class, a hop count) are mostly solved.</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Question type</th><th class="n">n</th><th class="n">Top-{len(top)} mean</th></tr></thead>
        <tbody>{''.join(tpl_rows)}</tbody>
      </table>
    </div>
  </section>

  <section aria-labelledby="h-repos">
    <h2 id="h-repos">By repository</h2>
    <div class="two">
      <div class="table-wrap">
        <table>
          <thead><tr><th>Repository</th><th class="n">n</th><th class="n">Top-{len(top)}</th><th class="n">All</th></tr></thead>
          <tbody>{''.join(repo_rows)}</tbody>
        </table>
      </div>
      <div style="display:grid;gap:12px;align-content:start">
        <p>Nine repositories were created after July 2026, past every model's training cutoff. Django is the contamination control: old, famous and certainly in every model's training data.</p>
        <p><strong>Django is the hardest repository, not the easiest.</strong> If memorised code helped, the control would score highest. Its size makes the searches long, and familiarity does not substitute for reading it.</p>
      </div>
    </div>
  </section>

  <section aria-labelledby="h-excl">
    <h2 id="h-excl">Excluded models</h2>
    <div class="issues">
      <div class="issue"><span class="chip high">Excluded</span><p><strong>deepseek-r1:32b</strong> answered in prose instead of calling a tool. Every condition, closed-book included, submits through a tool, so it cannot be scored. Ollama lists tool support for it.</p></div>
      <div class="issue"><span class="chip high">Excluded</span><p><strong>qwen2.5-coder:32b</strong> wrote its tool call as text, <code>{{"name": "submit_answer", "arguments": {{"answer": 42}}}}</code>, which Ollama does not parse into a call. Accepting text calls for one model would change the harness after eleven others ran, so it is excluded rather than special-cased.</p></div>
      <div class="issue"><span class="chip med">Scored</span><p><strong>nemotron3:33b</strong> is scored but its result is mostly tool-format failure. It wrote XML tool calls that Ollama often could not parse: {stats.get('nemotron3_33b', {}).get('invalid', 0)} malformed calls, and no answer on {stats.get('nemotron3_33b', {}).get('noans', 0)} of {n_q} questions. Its {pct(stats.get('nemotron3_33b', {}).get('repo', 0))} measures the model and the server's parser together.</p></div>
    </div>
  </section>

  <section aria-labelledby="h-method">
    <h2 id="h-method">How it was run</h2>
    <ul class="tight">
      <li><strong>Questions.</strong> 477 generated from code graphs of 10 repositories (83% hard), rewritten into natural developer wording by an agentic naturaliser with repository access that never sees the answer, then checked by a resolver and a judge; 434 rewrites accepted, the rest keep template wording. A fixed sample of {n_q}, proportional by repository, family and template, was evaluated (<code>data/questions_v4_eval/manifest.json</code>).</li>
      <li><strong>Harness.</strong> The same read-only tool loop for every model: <code>list_dir</code>, <code>read_file</code>, <code>grep</code>, and a typed <code>submit_answer</code>. Budget of 30 steps; on the last step only the submit tool is offered. Closed-book runs get no repository tools.</li>
      <li><strong>Hardware.</strong> One NVIDIA A100 40 GB, Ollama 0.40.5, every model at 4-bit, 64K context (qwen3:8b capped at its native 40K). Each model got as many parallel slots, up to 4, as kept it entirely on the GPU; architectures Ollama cannot batch ran one question at a time.</li>
      <li><strong>Scoring.</strong> Exact match for yes/no, counts and names; F1 for lists; any valid call chain of the right length for shortest-path questions. The first finished attempt at each question counts; only infrastructure outages are retried.</li>
    </ul>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Architecture</th><th class="n">Params</th><th class="n">Quant</th><th class="n">Slots</th><th class="n">Gen tok/s</th><th class="n">Peak GPU GB</th><th>Tool calls</th></tr></thead>
        <tbody>{''.join(hw_rows)}</tbody>
      </table>
    </div>
    <p class="note">Generation speed from a fixed 300-token probe on a warm model. Peak GPU memory from <code>nvidia-smi</code> once a minute; <code>ollama ps</code> under-reports models served with a speculative draft (gemma4 shows 5.5 GB). Blank where the logger was not yet running.</p>
  </section>

  <section aria-labelledby="h-corr">
    <h2 id="h-corr">Corrections and caveats</h2>
    <p>Everything below was found during or after the run. Predictions were stored, so every correction is applied by re-scoring, without re-running a model; both versions are reported.</p>
    <div class="two">
      <div class="table-wrap">
        <table>
          <thead><tr><th>Model, repo access</th><th class="n">v1</th><th class="n">v2</th><th class="n">Change</th></tr></thead>
          <tbody>{v1v2}</tbody>
        </table>
      </div>
      <ul class="tight">
        <li><strong>Gold answers ({len(changed)} changed).</strong> A call through a renamed import, <code>from .joiner import cli as remote_cli</code>, was resolved by pyan but rejected by the call-site check, which saw <code>remote_cli</code>, not <code>cli</code>. Fixed and re-extracted: 206 call edges restored across the 10 graphs.</li>
        <li><strong>Answer forms (v2).</strong> Accepted as correct: <code>Class.method</code> where the question asks for the class; file paths with a note appended; the importable name in a <code>src/</code> layout; and <code>C.__init__</code> for a class with no explicit <code>__init__</code>.</li>
        <li><strong>Retries.</strong> A resume used to re-run questions a model had failed to answer, keeping the latest attempt. Now only outages are retried and the first finished attempt counts. This moved nemotron3 from 0.49 to 0.44 and gpt-oss by &minus;0.006.</li>
        <li><strong>Malformed tool calls</strong> that the server rejects (HTTP 500) are now counted against the model inside the run, as invalid JSON already was. The 15 questions run under the old handling were re-run.</li>
      </ul>
    </div>
    <ul class="tight">
      <li><strong>Rewriting model.</strong> qwen3.8:27b rewrote the questions and is also scored. It never saw the answers, but a home advantage in wording cannot be ruled out.</li>
      <li><strong>Timing.</strong> qwen3.8's first six repositories ran four questions against one slot, so their per-question times include queueing. Scores, steps and tokens are unaffected.</li>
      <li><strong>Context.</strong> 8 of 2,750 repo-access runs passed 60K tokens and may have lost early context; they are almost all failures anyway, and none changes a ranking.</li>
      <li><strong>Sample size.</strong> {n_q} questions give each model a standard error of about 0.02&ndash;0.03. Adjacent models closer than about 0.05 (gpt-oss and muse-glimmer, for one) are not separated.</li>
    </ul>
  </section>

  <section aria-labelledby="h-next">
    <h2 id="h-next">Next steps</h2>
    <ol class="next">
      <li><strong>Manually verify a sample per family</strong> against the source code, to put a number on gold-answer accuracy beyond the fixes above.</li>
      <li><strong>Frontier models through the API</strong>, on the same {n_q} questions: about $20&ndash;90 per model at list prices, less with prompt caching.</li>
      <li><strong>Larger open models</strong> (gpt-oss:120b, nemotron-3.5-super) need an 80 GB GPU.</li>
      <li><strong>Commit</strong> the code, question sets and results, which are not yet in git.</li>
    </ol>
  </section>

  <footer>
    <div>Results: <span class="mono">data/eval/final/</span> &middot; rescored rows: <span class="mono">rescored.jsonl</span> &middot; questions: <span class="mono">data/questions_v4_eval/</span> &middot; this page: <span class="mono">scripts/make_final_report.py</span></div>
  </footer>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
