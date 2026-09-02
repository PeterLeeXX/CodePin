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
