#!/usr/bin/env bash
# One line a minute: time, loaded model, GPU memory used (MiB), GPU utilisation.
# nvidia-smi is the source for memory: `ollama ps` under-reports models served
# with a speculative draft (gemma4 shows the 5.5 GB draft, not the 31B model).
out=/mnt/cgx/data/eval/final/logs/gpu.tsv
[ -s "$out" ] || printf "time\tmodel\tmem_used_mib\tutil_pct\n" > "$out"
while true; do
  m=$(ollama ps 2>/dev/null | awk "NR==2{print \$1}")
  s=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d " " | tr "," "\t")
  printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "${m:--}" "$s" >> "$out"
  sleep 60
done
