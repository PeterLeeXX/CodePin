# Dataset preflight: 2026-09-02

This is a read-only audit of the pinned Hugging Face dataset revision used for
the pending AutoDL DAPO run. The parquet files were downloaded outside the
repository to `D:\CodePin-preflight-20260902\data`.

- Dataset: `LeeXugar/SWE-smith-code-search`
- Revision: `70a929610a8d56136083c7d0f4b72dacd3032abf`
- Train rows: `39,191`
- Validation rows: `100`
- Train SHA-256: `c9a38e502c3d412fdfeb43d94820368db80a88c827d9eb36cd0d7d2f88e34a45`
- Validation SHA-256: `e760f58ab69bd8ecf59c2b5010729f98b8e2b25b635a170ac969f58f359a217e`

## Schema and integrity

Both splits contain the nine expected fields: `instance_id`, `file_changes`,
`repo`, `base_commit`, `problem_statement`, `patch`, `target`, `prompt`, and
`use_patch`. There are no empty problem statements, malformed prompts, or
duplicate instance IDs. `use_patch` is `true` for every row, and `prompt[0]`
is a user message whose content exactly matches `problem_statement`.

## Length audit

Values are character counts except for list counts. Percentiles are p50/p90/p95/p99.

| Split | Field | Min | p50 | p90 | p95 | p99 | Max | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | problem/prompt chars | 35 | 1,097 | 1,571 | 1,771 | 2,359 | 39,434 | 1,168.85 |
| train | patch chars | 199 | 1,142 | 3,549 | 5,065 | 10,336 | 153,275 | 1,799.87 |
| train | file changes / targets | 1 | 1 | 1 | 2 | 3 | 15 | 1.11 |
| train | target entities | 0 | 1 | 3 | 4 | 7 | 262 | 1.56 |
| train | target modules | 0 | 1 | 2 | 3 | 5 | 52 | 1.31 |
| validation | problem/prompt chars | 132 | 1,086 | 1,555 | 1,654 | 1,904 | 2,058 | 1,127.37 |
| validation | patch chars | 387 | 1,057 | 3,504 | 3,922 | 7,686 | 22,237 | 1,723.38 |
| validation | file changes / targets | 1 | 1 | 1 | 2 | 4 | 31 | 1.42 |
| validation | target entities | 0 | 1 | 2 | 4 | 6 | 26 | 1.56 |
| validation | target modules | 1 | 1 | 2 | 3 | 6 | 29 | 1.52 |

Using the released SFT tokenizer (`Qwen2Tokenizer`, vocabulary size `248,044`,
model max length `262,144`), the problem/prompt token distribution is:

| Split | Min | p50 | p90 | p95 | p99 | Max | Mean | >4096 | >8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 13 | 258 | 384 | 439 | 614 | 12,695 | 280.36 | 8 | 1 |
| validation | 37 | 252 | 380 | 399 | 457 | 478 | 265.63 | 0 | 0 |

The dataset has no generated response field; response lengths will be measured
from actual DAPO rollouts after the remote training run is available.
