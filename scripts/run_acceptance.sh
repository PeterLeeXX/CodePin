#!/usr/bin/env bash
set -euo pipefail
: "${RUN_ROOT:?Set a new directory for acceptance artifacts}"
: "${MODEL:?Set the prepared native checkpoint path}"
: "${CODEPIN_TEST_DEPLOYMENT_FILE:?Set the active vLLM deployment file}"
: "${TASKS:?Set a small JSONL or parquet task set}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/logs"
trap 'code=$?; echo "$code" > "$RUN_ROOT/exit_code"' EXIT

"$PYTHON_BIN" scripts/preflight_qwen3_5.py --model "$MODEL" \
  > "$RUN_ROOT/logs/preflight.log" 2>&1
"$PYTHON_BIN" -m pytest tests -q --basetemp "$RUN_ROOT/pytest" \
  --junitxml "$RUN_ROOT/tests.xml" > "$RUN_ROOT/logs/tests.log" 2>&1
"$PYTHON_BIN" scripts/acceptance.py --tasks "$TASKS" --limit 3 \
  --tokenizer "$MODEL" --output "$RUN_ROOT/pipeline" --judge \
  > "$RUN_ROOT/logs/pipeline.log" 2>&1
echo 'CodePin acceptance passed; no training was started.'
