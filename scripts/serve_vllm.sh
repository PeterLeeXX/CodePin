#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the prepared native serving checkpoint}"
: "${DEPLOYMENT_FILE:?Set DEPLOYMENT_FILE to a file outside the target repositories}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
PREFIX_CACHE="${PREFIX_CACHE:-true}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

case "$PREFIX_CACHE" in
  true) CACHE_ARGS=(--enable-prefix-caching --mamba-cache-mode align) ;;
  false) CACHE_ARGS=(--no-enable-prefix-caching --mamba-cache-mode none) ;;
  *) echo 'PREFIX_CACHE must be true or false' >&2; exit 2 ;;
esac

# Every server launch changes the deployment identity, invalidating MCP results.
# The server owns the checkpoint for its lifetime; no weight hot-swap endpoint.
export MODEL DEPLOYMENT_FILE
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os, uuid
from pathlib import Path
model = Path(os.environ['MODEL'])
manifest = model / 'codepin_manifest.json'
if not manifest.is_file():
    raise SystemExit('Run scripts/prepare_serving_model.py first')
path = Path(os.environ['DEPLOYMENT_FILE'])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps({
    'deployment_id': uuid.uuid4().hex,
    'model': str(model.resolve()),
    'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(),
}))
temporary.replace(path)
PY

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name codepin --host "$HOST" --port "$PORT" \
  --language-model-only --dtype bfloat16 --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  "${CACHE_ARGS[@]}" "$@"
