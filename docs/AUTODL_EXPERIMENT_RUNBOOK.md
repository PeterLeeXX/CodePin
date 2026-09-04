# AutoDL experiment runbook

## 2026-09-02: SSH gateway unavailable

- Event: preflight for the formal Qwen3.5-0.8B DAPO run
- Hosts checked: `connect.westx.seetacloud.com:32219` and the saved WSL alias
  `region-46.seetacloud.com:22086`
- Evidence: TCP connection succeeds, then SSH exits with
  `kex_exchange_identification: Connection closed by remote host` before the
  SSH banner/authentication exchange.
- Also checked historical saved AutoDL endpoints; they were unavailable or
  refused host verification.
- Action: no training process was started, no credentials were changed, and no
  shutdown action was taken. Resume after the AutoDL instance/gateway is
  restored and `ssh autodl` succeeds.

## 2026-09-03: SFT serving and engineering acceptance

- Scope: MCP serving, native vLLM batching/cache, data preparation, rewards,
  evaluation, and SkyRL rollout/readback. No trainer or optimizer was started.
- Runtime: Ubuntu 22.04, Python 3.12.14, torch 2.11.0+cu128, vLLM 0.23.0,
  Transformers 5.8.0, and the existing pinned SkyRL/OpenHands commits.
- GPU: the instance reports one RTX 4090 with 49,140 MiB VRAM. Use observed
  capacity when reproducing the 16,384-token / 8-sequence serving configuration.
- Network: direct GitHub TLS and release downloads failed. An SSH reverse
  tunnel to the local network proxy and pinned Git mirrors allowed the declared
  dependencies to install unchanged. No dependency downgrade or runtime patch.
- Checkpoint: `LeeXugar/CodePin-SFT-Qwen3.5-0.8B` at
  `4480baaf6a1b1f9fc0b3fb54c8480ed036c6f33a`. The native wrapper preparation
  preserves every weight byte and removes the old attention monkey patch.
- Fixes found by real acceptance: MCP requires an explicit typed result for
  `structuredContent`; sample target lists can be null; SkyRL all-assistant
  SFT tokenization rejects an explicit `enable_thinking` kwarg; unconstrained
  JSON whitespace can consume a small model's entire judge token budget.
  The release's generation EOS (`endoftext`) also differs from the chat
  tokenizer EOS (`im_end`); native serving metadata must align the two so
  JSON-constrained requests do not stop before closing the object.
- Artifacts: `runs/20260903-acceptance/` on the instance. Every failed attempt
  has its own log/output directory; the final report links the successful run.
  Raw model traces and infrastructure logs stay out of the public Git commit.
- Commands and supported boundaries: see `SERVING_AND_EVALUATION.md`.
- Final acceptance: `verification-final3/exit_code=0`; 53 tests passed with no
  skips, 3 real task trajectories evaluated, 1 SFT / 3 RL rows read by SkyRL,
  normal/truncated rollouts checked, and 3 judge responses parsed. See
  `ACCEPTANCE_20260903.md` for the evidence and quality limits.

## 2026-09-04: real Agent serving performance

- Scope: the existing SFT model and pinned SkyRL/vLLM environment; no training.
  The previous three-task acceptance is not used as a performance baseline.
- Observed resource limits: RTX 6000D, 85,651 MiB, compute capability 12.0;
  Xeon Platinum 8470Q, 208 visible logical CPUs but a 22-core cgroup quota;
  110 GiB host-memory limit and no swap.
- Nsight Systems 2025.1.1.0 is already installed in the image. CUDA, NVTX and
  OS Runtime collection works. CPU perf sampling/context-switch access is
  unavailable. The targeted Nsight Compute attempt returns
  `ERR_NVGPUCTRPERM`; its hardware-counter results must not be claimed.
- Artifacts: `runs/20260904-vllm-agent-performance/`. Each timing run captures
  its source archive, parameters, per-task outcomes, Prometheus deltas and
  resource samples. Failed attempts remain separate from accepted runs.
- Workload identity uses pinned commits, patches and source-content manifests.
  Git metadata changes across fresh clones; that provenance comparison does
  not weaken the serving result-cache invalidation key.
- Protocol, performance comparisons, limits and reproduction commands:
  [PERFORMANCE_TUNING_20260904.md](PERFORMANCE_TUNING_20260904.md).
- Adding the locked profiling dependencies exposed a packaging problem after
  acceptance: setuptools namespace discovery followed a retained test symlink
  back to its parent under `runs/`. The `include = ["src", "src.*"]` filter
  alone does not prune unrelated trees. Excluding `runs*`, `outputs*`, and
  `tmp*` from native package discovery made the editable build finish in
  555 ms without deleting test evidence or modifying setuptools.
- Slow package downloads were replaced with the exact Linux wheels from
  `uv.lock`, downloaded locally and SHA-256 checked on both hosts. Install
  them by package name/version using `uv pip install --no-deps --no-index
  --find-links <wheel-directory> ...`, then run `uv sync --locked --offline
  --group profiling`. Installing direct wheel paths records a different
  source identity and makes the locked sync request the original URL again;
  that failed attempt and its logs are retained. Existing dependency
  versions were audited after the successful locked sync.
- A Windows `git archive` export inherited `core.autocrlf` and changed the
  original launcher to CRLF; Bash rejected it before model startup. Export
  the baseline with `git -c core.autocrlf=false -c core.eol=lf archive`, then
  compare every extracted file with its Git blob, not only the archive hash.
  The replacement `baseline-native-v34.tar` passed all 58 blob checks in a
  fresh directory; the failed export and interrupted measurements remain in
  the run materials and are excluded from confirmation. Python-generated
  POSIX launchers on Windows also need explicit LF output (`write_bytes` or
  `newline="\n"`); the current confirmation wrapper was verified on both hosts.
