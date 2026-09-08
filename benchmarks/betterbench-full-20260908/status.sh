#!/usr/bin/env bash
# Poll the running BetterBench job. Usage: ./status.sh
OUT="$(cd "$(dirname "$0")" && pwd)"
PID=$(pgrep -f "betterbench-venv/bin/python3" | head -1)
if [ -z "$PID" ]; then
  echo "job: NOT RUNNING"
else
  echo "job: RUNNING  (pid $PID, elapsed $(ps -p "$PID" -o etime= | tr -d ' '))"
fi
m=$(curl -s --max-time 15 http://192.168.1.17:1234/metrics 2>/dev/null)
if [ -z "$m" ]; then
  echo "engine: metrics not reachable via llama-swap (it may not proxy /metrics)"
else
  run=$(echo "$m" | awk '/^vllm:num_requests_running/{print $2}')
  wait=$(echo "$m" | awk '/^vllm:num_requests_waiting/{print $2}')
  pt=$(echo "$m" | awk '/^vllm:prompt_tokens_total/{print $2}')
  echo "engine: running=$run waiting=$wait  prompt_tokens_total=$pt"
fi
if [ -s "$OUT/results.json" ]; then
  echo "results.json: present ($(stat -c %s "$OUT/results.json") bytes) — job likely finished"
else
  echo "results.json: not yet written (written at the end of the run)"
fi
