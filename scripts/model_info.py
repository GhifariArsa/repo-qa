"""Record what a local Ollama model is, and how fast it runs on this GPU.

    python scripts/model_info.py <model> <out.json>

Captures the model card (/api/show: family, parameter count, quantisation,
context length, capabilities), its footprint once loaded (/api/ps: VRAM), a
fixed-prompt throughput probe (/api/generate timings: prompt and generation
tokens per second), and a one-call tool-use smoke test. Exits non-zero when
the model cannot make a structured tool call, so the experiment skips it.
"""

import json
import sys
import time
import urllib.request

BASE = "http://localhost:11434"
PROBE = ("Explain, in about 150 words, what a call graph is and why static analysis "
         "of Python can miss some calls.")
TOOL = {"type": "function", "function": {
    "name": "submit_answer", "description": "Submit the final answer.",
    "parameters": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}}


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def main(model: str, out: str) -> int:
    show = post("/api/show", {"model": model})
    info = show.get("model_info", {})
    arch = info.get("general.architecture", "")
    record = {
        "model": model,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ollama_version": get("/api/version")["version"],
        "family": show.get("details", {}).get("family"),
        "parameter_size": show.get("details", {}).get("parameter_size"),
        "parameter_count": info.get("general.parameter_count"),
        "quantization": show.get("details", {}).get("quantization_level"),
        "format": show.get("details", {}).get("format"),
        "native_context_length": info.get(f"{arch}.context_length"),
        "expert_count": info.get(f"{arch}.expert_count"),
        "experts_used": info.get(f"{arch}.expert_used_count"),
        "capabilities": show.get("capabilities"),
        "digest": next((m.get("digest") for m in get("/api/tags")["models"] if m["name"] == model), None),
        "size_on_disk_bytes": next((m.get("size") for m in get("/api/tags")["models"] if m["name"] == model), None),
    }

    started = time.monotonic()
    gen = post("/api/generate", {"model": model, "prompt": PROBE, "stream": False,
                                 "options": {"num_predict": 300, "temperature": 0}})
    record["load_and_probe_seconds"] = round(time.monotonic() - started, 1)
    gen = post("/api/generate", {"model": model, "prompt": PROBE, "stream": False,
                                 "options": {"num_predict": 300, "temperature": 0}})  # warm run
    ns = 1e9
    record["throughput"] = {
        "prompt_tokens": gen.get("prompt_eval_count"),
        "prompt_tokens_per_s": round(gen["prompt_eval_count"] / (gen["prompt_eval_duration"] / ns), 1)
        if gen.get("prompt_eval_duration") else None,
        "generated_tokens": gen.get("eval_count"),
        "generation_tokens_per_s": round(gen["eval_count"] / (gen["eval_duration"] / ns), 1)
        if gen.get("eval_duration") else None,
    }
    loaded = next((m for m in get("/api/ps")["models"] if m["name"] == model), {})
    record["vram_bytes"] = loaded.get("size_vram")
    record["loaded_context_length"] = loaded.get("context_length")

    chat = post("/v1/chat/completions", {
        "model": model, "tools": [TOOL],
        "messages": [{"role": "user", "content": "What is 17 + 25? Submit it with submit_answer."}]})
    calls = chat["choices"][0]["message"].get("tool_calls") or []
    record["tool_smoke"] = {"ok": bool(calls) and calls[0]["function"]["name"] == "submit_answer",
                            "arguments": calls[0]["function"]["arguments"] if calls else None,
                            "content": (chat["choices"][0]["message"].get("content") or "")[:200]}
    with open(out, "w") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps({k: record[k] for k in ("model", "parameter_size", "quantization", "vram_bytes",
                                             "throughput", "tool_smoke")}))
    return 0 if record["tool_smoke"]["ok"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
