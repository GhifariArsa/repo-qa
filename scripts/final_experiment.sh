#!/usr/bin/env bash
# Final experiment: rewrite the v4 questions, then run every model in MODELS
# through the agent harness with and without repository access.
#
#   on the GPU VM:   tmux new -d -s final scripts/final_experiment.sh
#   from a laptop:   VM=ubuntu@<host> KEY=~/.ssh/id_rsa caffeinate -dims scripts/final_experiment.sh
#
# Re-runnable: finished rewrites are kept, and agent-eval resumes a run from
# the rows already written. A model that cannot make a structured tool call is
# recorded and skipped. Everything is logged under data/eval/final/.
set -uo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$PATH"
VM=${VM:-}          # empty: running on the VM itself, Ollama is local
KEY=${KEY:-~/.ssh/id_rsa}
REPOS=(reef artemis openworker axisagentic rpent printfilm laya edge0 odysseus django)
NATURALISER=${NATURALISER:-qwen3.8:27b}
# Priority order: if the run is cut short, the earlier models are complete.
MODELS=(${MODELS:-qwen3.8:27b gpt-oss:20b gemma4:31b deepseek-r1:32b qwen3-coder:30b qwen3.5:27b nemotron3:33b ornith:35b muse-glimmer:30b nemotron-3.5-lightning:30b qwen2.5-coder:32b qwen3:8b llama3.1:8b})
CONDITIONS=(${CONDITIONS:-repo closed})
QIN=data/questions_v4
QNAT=data/questions_v4_nat
QEVAL=${QEVAL:-data/questions_v4_eval}   # fixed proportional sample of QNAT (scripts/sample_questions.py)
EVAL_N=${EVAL_N:-250}
JOBS=${JOBS:-4}
OUT=data/eval/final
mkdir -p "$QNAT" "$OUT/models" "$OUT/logs"
LOG="$OUT/logs/run.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

ensure_tunnel() {
  curl -s -m 5 localhost:11434/api/version >/dev/null && return 0
  if [ -z "$VM" ]; then log "local ollama down; restarting"; sudo systemctl restart ollama; sleep 10
    curl -s -m 5 localhost:11434/api/version >/dev/null; return; fi
  log "tunnel down; reopening"
  pkill -f "11434:127.0.0.1:11434" 2>/dev/null
  ssh -i "$KEY" -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes \
      -f -N -L 11434:127.0.0.1:11434 "$VM"
  for _ in $(seq 1 10); do curl -s -m 5 localhost:11434/api/version >/dev/null && return 0; sleep 3; done
  log "tunnel still down"; return 1
}

# Largest slot count (4, 2, 1) at which the model stays 100% on the GPU; sets
# Ollama to it and echoes it. Architectures Ollama cannot batch (qwen35 hybrid
# attention) report 1 whatever is asked. Concurrency then matches real slots,
# so per-question time measures the model rather than a queue in front of it.
tune_parallel() {
  local m=$1 conf=/etc/systemd/system/ollama.service.d/override.conf
  for P in 4 2 1; do
    local since; since=$(remote "date '+%F %T'")
    remote "sudo sed -i 's/OLLAMA_NUM_PARALLEL=[0-9]*/OLLAMA_NUM_PARALLEL=$P/' $conf && sudo systemctl daemon-reload && sudo systemctl restart ollama" >/dev/null 2>&1
    sleep 8
    curl -s -m 900 localhost:11434/api/generate -d "{\"model\":\"$m\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" >/dev/null
    local proc; proc=$(remote "ollama ps" | awk -v m="$m" '$1==m')
    if [[ "$proc" == *"100% GPU"* ]]; then
      # Read the journal fully before matching: with pipefail, `grep -q`
      # closing the pipe early makes journalctl die of SIGPIPE and the match
      # read as a failure.
      local journal; journal=$(remote "sudo journalctl -u ollama --since '$since' --no-pager")
      if grep -qE "does not currently support parallel|n_slots = 1," <<<"$journal"; then
        echo 1; return; fi
      echo "$P"; return
    fi
    log "$m at parallel $P: $proc" >&2
  done
  echo 1
}

remote() { if [ -z "$VM" ]; then bash -c "$*"; else ssh -i "$KEY" -o ServerAliveInterval=30 "$VM" "$@"; fi; }

# ---- 0. hardware and environment, once -------------------------------------
[ -s "$OUT/environment.txt" ] || { remote "nvidia-smi; ollama --version; systemctl cat ollama | grep Environment; nproc; free -g" \
  >"$OUT/environment.txt" 2>&1; git rev-parse HEAD >>"$OUT/environment.txt" 2>/dev/null; }

# ---- 1. rewrite every question (one model as writer, resolver and judge) ----
ensure_tunnel || exit 1
remote "ollama pull $NATURALISER" >>"$OUT/logs/pull.log" 2>&1
# Pre-pull every model in the background so no model waits on a download later.
( for m in "${MODELS[@]}"; do remote "ollama pull $m" >>"$OUT/logs/prepull.log" 2>&1 \
    && echo "$(date '+%F %T') pulled $m" >>"$OUT/logs/prepull.log" \
    || echo "$(date '+%F %T') PULL FAILED $m" >>"$OUT/logs/prepull.log"; done ) &
for r in "${REPOS[@]}"; do
  [ -s "$QNAT/$r.jsonl" ] && continue
  for attempt in 1 2 3; do
    ensure_tunnel || sleep 60
    log "naturalise $r (attempt $attempt)"
    if uv run cgx naturalize-agent "$r" --all --questions "$QIN/$r.jsonl" --out "$QNAT/$r.jsonl.tmp" \
         --writer "ollama:$NATURALISER" --resolver "ollama:$NATURALISER" --judge "ollama:$NATURALISER" \
         -j 2 >"$OUT/logs/naturalise_$r.log" 2>&1; then
      mv "$QNAT/$r.jsonl.tmp" "$QNAT/$r.jsonl"; log "naturalise $r: $(tail -1 "$OUT/logs/naturalise_$r.log")"; break
    fi
    log "naturalise $r failed: $(tail -2 "$OUT/logs/naturalise_$r.log" | tr '\n' ' ')"
  done
done

# ---- 2. every model, every repo, both conditions ----------------------------
[ -s "$QEVAL/manifest.json" ] || uv run python scripts/sample_questions.py "$QNAT" "$QEVAL" "$EVAL_N" 0 | tee -a "$LOG"
for m in "${MODELS[@]}"; do
  slug=$(python3 -c "import re,sys;print(re.sub(r'[^A-Za-z0-9.-]+','_',sys.argv[1]))" "$m")
  ensure_tunnel || { sleep 120; ensure_tunnel || continue; }
  log "pull $m"
  if ! remote "ollama pull $m" >>"$OUT/logs/pull.log" 2>&1; then log "SKIP $m: pull failed"; continue; fi
  if ! uv run python scripts/model_info.py "$m" "$OUT/models/$slug.json" >>"$OUT/logs/model_info.log" 2>&1; then
    log "SKIP $m: no structured tool call (see $OUT/models/$slug.json)"; continue
  fi
  log "model info $m: $(tail -1 "$OUT/logs/model_info.log")"
  JOBS=$(tune_parallel "$m")
  log "parallel $m: $JOBS slot(s) on GPU; $(remote "ollama ps" | awk -v m="$m" '$1==m')"
  python3 - "$OUT/models/$slug.json" "$JOBS" <<'PY'
import json, sys
p, jobs = sys.argv[1], int(sys.argv[2])
d = json.load(open(p)); d["parallel_slots"] = jobs; json.dump(d, open(p, "w"), indent=2)
PY
  for c in "${CONDITIONS[@]}"; do
    for r in "${REPOS[@]}"; do
      [ -s "$QEVAL/$r.jsonl" ] || { log "no sampled questions for $r; skipping"; continue; }
      for attempt in 1 2 3; do
        ensure_tunnel || sleep 60
        uv run cgx agent-eval "$r" --model "ollama:$m" --condition "$c" --phrasing natural \
          --questions "$QEVAL/$r.jsonl" --out-dir "$OUT" -j "$JOBS" >>"$OUT/logs/${slug}_${c}_$r.log" 2>&1
        # agent-eval resumes, so rerunning only retries rows that errored.
        summary="$OUT/$r/${slug}__${c}__natural.summary.json"
        errs=$(python3 -c "import json;print(json.load(open('$summary'))['errors'])" 2>/dev/null || echo 99)
        [ "$errs" -eq 0 ] && break
        log "$m $c $r: $errs infrastructure error(s); retrying"
      done
      log "done $m $c $r: $(python3 -c "import json;s=json.load(open('$summary'));print({k:s.get(k) for k in ('n','overall','hard','lookup','errors','forced_answers','mean_seconds','mean_steps')})" 2>/dev/null)"
    done
  done
  # Free the GPU between models (disk is kept; models are reused if the run restarts).
  remote "ollama stop $m" >/dev/null 2>&1
done
log "ALL DONE"
