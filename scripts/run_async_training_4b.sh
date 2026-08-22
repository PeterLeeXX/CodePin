#!/usr/bin/env bash
set -euo pipefail
echo "Compatibility launcher: CodePin now targets Qwen3.5-0.8B with SkyRL v0.3." >&2
exec "$(dirname "$0")/run_rl_qwen3_5_0_8b.sh" "$@"
