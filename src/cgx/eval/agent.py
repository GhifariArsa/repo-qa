"""The model under test: one read-only tool loop, identical for every model.

Vendor harnesses (Claude Code, Codex CLI) bring their own tools and system
prompts, so a comparison through them measures the harness as much as the
model. This loop is the same scaffold for every model reached through
OpenRouter's OpenAI-compatible API:

  * condition `repo`   -- tools `list_dir`, `read_file`, `grep`, scoped to the
                          repository checkout, plus `submit_answer`.
  * condition `closed` -- `submit_answer` only. Same prompt, same model, no way
                          to look. The gap between the two is how much the
                          question actually needs the repository.

There is deliberately no shell: with bash a model can import the package and
introspect it at runtime, which is a different task from reading code.

Answers come back through `submit_answer`, whose schema is typed per question
(boolean, integer, list of strings, string), so scoring never parses prose.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..qgen.base import QuestionRecord

MAX_STEPS = 30
#: Past this, the next turn must answer. A run still wandering after five
#: minutes is lost, not slow, and it is scored on what it submits.
MAX_SECONDS = 300
#: Local models share one GPU, so wall-clock measures the hardware and the
#: queue as much as the model: five minutes cut a 27B model off on 11 of 50
#: pilot questions that a faster model finished in the same number of steps.
#: Locally the step cap is the budget; this only catches a hung run.
MAX_SECONDS_LOCAL = 1800
MAX_READ_LINES = 400
MAX_TOOL_CHARS = 24_000
MAX_GREP_HITS = 80
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}

SYSTEM_PROMPT = """\
You answer questions about the structure of one Python repository.

Conventions the answers follow:
- A fully qualified name is the module's dotted path followed by the \
qualified name inside it, e.g. `pkg.sub.module.Class.method`. Module paths are \
relative to the package root given below.
- "Calls directly" means a call written in the function's own body that \
resolves to a function or method defined in this repository. Calls into the \
standard library or third-party packages do not count. Calling a class, e.g. \
`User(...)`, counts as calling that class's `__init__`. A call through `self` or \
`super()` counts as calling the method that actually runs, which may be \
inherited from a base class.
- "In-repository" excludes test directories (`tests`, `test`, `testing`).
- File paths are relative to the repository root.

{access}

When you have the answer, call `submit_answer` exactly once. If the question \
asks for a list and the answer is empty, submit an empty list.\
"""

ACCESS_REPO = (
    "You can inspect the repository with `list_dir`, `read_file` and `grep`. "
    "The package root is `{package_root}`. Look before you answer: do not guess."
)
ACCESS_CLOSED = (
    "You cannot open the repository in this setting. Answer from what you "
    "already know, and give your best answer rather than declining. "
    "The package root is `{package_root}`."
)

_ANSWER_SCHEMA = {
    "bool": {"type": "boolean"},
    "count": {"type": "integer"},
    "list": {"type": "array", "items": {"type": "string"}},
    "path": {"type": "array", "items": {"type": "string"},
             "description": "Every callable on the chain, in order, first to last."},
    "string": {"type": "string"},
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


REPO_TOOLS = [
    _tool("list_dir", "List the files and directories at a path in the repository.",
          {"path": {"type": "string", "description": "Relative to the repository root; '.' for the root."}},
          ["path"]),
    _tool("read_file",
          f"Read lines from a file in the repository, at most {MAX_READ_LINES} per call. "
          "Lines are prefixed with their line numbers.",
          {"path": {"type": "string"},
           "start_line": {"type": "integer", "description": "1-based, inclusive. Default 1."},
           "end_line": {"type": "integer", "description": "1-based, inclusive."}},
          ["path"]),
    _tool("grep",
          f"Search file contents with a Python regular expression. Returns up to {MAX_GREP_HITS} "
          "matches as path:line: text.",
          {"pattern": {"type": "string"},
           "path": {"type": "string", "description": "Directory or file to search. Default '.'."},
           "glob": {"type": "string", "description": "Filename glob, e.g. '*.py'. Default '*.py'."}},
          ["pattern"]),
]


def submit_tool(answer_type: str) -> dict:
    return _tool("submit_answer", "Submit your final answer. Call this exactly once.",
                 {"answer": _ANSWER_SCHEMA[answer_type]}, ["answer"])


class RepoSandbox:
    """Read-only file access confined to one checkout."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def _resolve(self, rel: str | None) -> Path:
        path = (self.root / (rel or ".")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"{rel!r} is outside the repository")
        if any(part in SKIP_DIRS for part in path.relative_to(self.root).parts):
            raise ValueError(f"{rel!r} is not available")
        return path

    def list_dir(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.is_dir():
            raise ValueError(f"{path!r} is not a directory")
        entries = sorted(p for p in target.iterdir() if p.name not in SKIP_DIRS)
        return "\n".join(f"{p.name}/" if p.is_dir() else p.name for p in entries) or "(empty)"

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ValueError(f"{path!r} is not a file")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line or 1)
        end = min(len(lines), end_line or len(lines), start + MAX_READ_LINES - 1)
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        if end < len(lines):
            body += f"\n... ({len(lines)} lines total; continue with start_line={end + 1})"
        return body or "(empty file)"

    def grep(self, pattern: str, path: str = ".", glob: str = "*.py") -> str:
        regex = re.compile(pattern)
        target = self._resolve(path)
        files = [target] if target.is_file() else sorted(target.rglob(glob or "*.py"))
        hits = []
        for f in files:
            if not f.is_file() or any(part in SKIP_DIRS for part in f.relative_to(self.root).parts):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{f.relative_to(self.root)}:{i}: {line.strip()[:200]}")
                    if len(hits) >= MAX_GREP_HITS:
                        return "\n".join(hits) + f"\n... (stopped at {MAX_GREP_HITS} matches)"
        return "\n".join(hits) or "(no matches)"

    def call(self, name: str, args: dict) -> str:
        try:
            out = getattr(self, name)(**args)
        except (ValueError, TypeError, re.error, OSError) as exc:
            return f"error: {exc}"
        return out if len(out) <= MAX_TOOL_CHARS else out[:MAX_TOOL_CHARS] + "\n... (truncated)"


@dataclass
class AgentResult:
    prediction: Any = None
    answered: bool = False
    steps: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    error: str | None = None
    tools_used: dict[str, int] = field(default_factory=dict)
    forced: bool = False            # answered only because the step or time budget ran out
    submitted: dict | None = None   # the final tool's arguments
    text_submission: bool = False   # submitted by writing the call as text; see parse_text_submission
    peak_prompt_tokens: int = 0     # largest single-request context: how close the run came to the window
    reasoning_tokens: int = 0       # when the provider reports them separately
    invalid_tool_calls: int = 0     # bad JSON, unknown tool, or a final call missing its key
    no_tool_turns: int = 0          # turns that called nothing and did not submit
    files_read: list[str] = field(default_factory=list)  # distinct paths opened with read_file
    model_seconds: float = 0.0      # time spent waiting on the model, excluding tool execution


#: How Ollama reports a tool call it could not parse: JSON for gpt-oss-style
#: templates, XML for nemotron's. Both come back as HTTP 500.
_PARSE_ERROR = re.compile(r"error parsing tool call|XML syntax error|failed to parse tool", re.I)


def is_tool_call_parse_error(exc: Exception) -> bool:
    """The model emitted a malformed tool call, which the server refused."""
    return bool(_PARSE_ERROR.search(str(exc)))


def parse_text_submission(content: str, final_tool: str, required_key: str) -> dict | None:
    """A final-tool call written as plain text instead of made as a call.

    Some models, notably small ones served by Ollama, decide their answer and
    then write `submit_answer ["none"]` as text, turn after turn, never as a
    structured call -- 24 times in a row in one trace. The intent is
    unambiguous, so it is accepted, but only when the WHOLE message is the tool
    name followed by valid JSON: prose that merely mentions the tool does not
    count. Every acceptance is recorded (`text_submission`) so it stays visible
    in the results rather than silently smoothing over a model's failure.
    """
    text = content.strip().strip("`").strip()
    match = re.fullmatch(rf"{re.escape(final_tool)}\s*[:(]?\s*(.+?)\)?", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict) and required_key in value:
        return value
    return {required_key: value}


def run_tool_loop(client, model: str, *, system: str, user: str, tools: list[dict],
                  final_tool: str, required_key: str, sandbox: RepoSandbox | None,
                  max_steps: int = MAX_STEPS, max_seconds: float = MAX_SECONDS) -> AgentResult:
    """Drive a model through read-only tools until it calls `final_tool`.

    Shared by the model under test (`run_agent`) and the agentic naturaliser,
    so both see the same sandbox, budgets and failure handling. The submitted
    arguments land in `AgentResult.submitted`.
    """
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    result, started = AgentResult(), time.monotonic()
    try:
        for step in range(max_steps):
            result.steps = step + 1
            final_turn = step == max_steps - 1 or time.monotonic() - started > max_seconds
            # On the last step only the final tool is OFFERED, not merely
            # requested: Ollama ignores `tool_choice`, and a model that cannot
            # see `grep` cannot call it. `tool_choice` still helps where honoured.
            offered = [t for t in tools if t["function"]["name"] == final_tool] if final_turn else tools
            asked = time.monotonic()
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, tools=offered,
                    tool_choice={"type": "function", "function": {"name": final_tool}}
                    if final_turn else "auto",
                    extra_body={"usage": {"include": True}},
                )
            except Exception as exc:
                # Ollama answers a tool call it cannot parse -- gpt-oss writing
                # its reasoning into the arguments -- with HTTP 500. That is the
                # model's malformed call, not an outage: count it, spend the
                # step, say so, and let the model go on, exactly as for a call
                # whose arguments are invalid JSON. Treating it as an
                # infrastructure error re-ran the whole question from scratch.
                if not is_tool_call_parse_error(exc):
                    raise
                result.model_seconds += time.monotonic() - asked
                result.invalid_tool_calls += 1
                messages.append({"role": "user", "content":
                                 "Your last tool call was malformed and could not be parsed. "
                                 "Call a tool with valid JSON arguments."})
                continue
            result.model_seconds += time.monotonic() - asked
            usage = getattr(resp, "usage", None)
            if usage is not None:
                result.prompt_tokens += usage.prompt_tokens or 0
                result.peak_prompt_tokens = max(result.peak_prompt_tokens, usage.prompt_tokens or 0)
                details = getattr(usage, "completion_tokens_details", None)
                result.reasoning_tokens += int(getattr(details, "reasoning_tokens", 0) or 0)
                result.completion_tokens += usage.completion_tokens or 0
                result.cost_usd += float(getattr(usage, "cost", 0) or 0)
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            turn: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if calls:
                turn["tool_calls"] = [c.model_dump() for c in calls]
            messages.append(turn)
            if not calls:
                parsed = parse_text_submission(msg.content or "", final_tool, required_key)
                if parsed is not None:
                    result.submitted, result.answered = parsed, True
                    result.forced, result.text_submission = final_turn, True
                    return result
                result.no_tool_turns += 1
                messages.append({"role": "user", "content":
                                 f"Use the tools to investigate, then call `{final_tool}`."})
                continue
            for call in calls:
                name = call.function.name
                result.tools_used[name] = result.tools_used.get(name, 0) + 1
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = None
                if name == final_tool and isinstance(args, dict) and required_key in args:
                    result.submitted, result.answered = args, True
                    result.forced = final_turn
                    return result
                if args is None:
                    result.invalid_tool_calls += 1
                    content = "error: arguments were not valid JSON"
                elif sandbox is not None and name in {"list_dir", "read_file", "grep"}:
                    result.tool_calls += 1
                    if name == "read_file" and isinstance(args.get("path"), str) \
                            and args["path"] not in result.files_read:
                        result.files_read.append(args["path"])
                    content = sandbox.call(name, args)
                else:
                    result.invalid_tool_calls += 1
                    content = f"error: no tool named {name!r}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        result.error = f"no {final_tool} submitted"
    except Exception as exc:  # network, provider, or model-side failure: record, don't crash the run
        result.error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        result.seconds = round(time.monotonic() - started, 2)
    return result


def run_agent(client, model: str, rec: QuestionRecord, *, condition: str, question: str,
              sandbox: RepoSandbox | None, package_root: str,
              max_seconds: float = MAX_SECONDS) -> AgentResult:
    """Drive one question to a submitted answer, or to MAX_STEPS."""
    access = (ACCESS_REPO if condition == "repo" else ACCESS_CLOSED).format(package_root=package_root)
    res = run_tool_loop(
        client, model, system=SYSTEM_PROMPT.format(access=access), user=question,
        tools=(REPO_TOOLS if condition == "repo" else []) + [submit_tool(rec.answer_type)],
        final_tool="submit_answer", required_key="answer", sandbox=sandbox, max_seconds=max_seconds,
    )
    if res.answered:
        res.prediction = res.submitted["answer"]
    elif res.error == "no submit_answer submitted":
        res.error = "no answer submitted"   # the scorer treats this as a wrong answer, not a crash
    return res
