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
