# CodePin 工程与部署验收（2026-09-03）

验收通过，最终运行目录为 `runs/20260903-acceptance/verification-final3`，
`exit_code=0`。本次未启动正式训练、未更新模型权重，也未运行完整 SWE-bench。
功能分支：`codex/mcp-serving-data-evaluation`；基于 `b2cc195`。

## 完成范围

- 复用现有 OpenHands 工具和 SkyRL rollout，提供 MCP `localize_code` /
  `localize_batch`、结构化结果、真实符号校验与有界源代码上下文。
- 提供原生 vLLM 部署配置、并发批处理、Prefix Cache，以及按仓库内容、
  部署身份和请求配置失效的 TTL/LRU 结果缓存；训练 rollout 禁用结果缓存。
- 完成任务清洗、模型轨迹生成/校验、任务和动作序列去重、质量过滤、难度分层、
  按仓库分割和 SFT/RL parquet 导出。
- 保留原 file/module/entity F1，并报告 file/class/function F1；引入重复搜索、
  重叠读取、输出数量和截断成本。异常轨迹零奖励、零 Loss Mask，保留分项日志。
- 提供定位与工具效率评测、可选 Judge 请求，以及 Coding Agent/SWE-bench
  结果文件的关联统计，明确区分缺失与未匹配记录。

## 环境与模型

Ubuntu 22.04 / Python 3.12.14 / 单 GPU（设备报告 RTX 4090，49,140 MiB）。
使用项目固定的 torch 2.11.0+cu128、Transformers 5.8.0、vLLM 0.23.0、
SkyRL 0.3.0 和 OpenHands 1.7.1；新增 MCP 依赖，完整解析结果保存于 `uv.lock`。
SkyRL Git revision 为 `f5bc3b78dfddfb352870d5d7430cd226e5785838`。

现有模型为 `LeeXugar/CodePin-SFT-Qwen3.5-0.8B`，revision
`4480baaf6a1b1f9fc0b3fb54c8480ed036c6f33a`。
权重 SHA-256：
`d97009d9e41838a2eb8ce0e6275ac41cf80f1186915628f31662d3ba0b67e48e`。

服务使用标准 Qwen3.5 wrapper 和 `--language-model-only`，原样硬链接权重。
输出配置将 generation EOS 从 `endoftext` 对齐到 tokenizer 的 `im_end`，
消除 JSON 提前停止；旧 attention monkey patch 已删除。
最终实例上的配置目录为 `/root/autodl-tmp/models/codepin-native-v2`。

服务参数：bf16、max model length 16,384、batch token 8,192、max sequences 8、
GPU memory utilization 0.70、chunked prefill、Mamba `align` Prefix Cache、
XGrammar 紧凑 JSON。Mamba `align` 在该 vLLM 版本中仍标为实验性。

## 验收证据

| 验收项 | 结果 |
| --- | --- |
| 环境 preflight | 通过，实际导入 CUDA、flash-attn、causal-conv1d、FLA、vLLM、SkyRL |
| 全部 pytest | **53 passed / 0 failed / 0 errors / 0 skipped**，35.91 秒 |
| MCP 真实集成 | stdio 初始化、工具发现、真实模型定位、缓存命中/内容变更失效、双任务批处理、越界拒绝均通过 |
| vLLM 原生批处理 | 单个 completions 请求传入两个 prompt，返回两个实际生成结果 |
| Prefix Cache | 最终部署累计查询 103,368 token、命中 72,352 token；只证明功能生效 |
| 数据闭环 | 3 个真实 SWE-Smith 任务，清洗保留 3，生成并评测 3，SFT 接受 1、拒绝 2，RL 导出 3 |
| SkyRL SFT/RL 读回 | 原生 tokenizer/collator 生成 `TrainingInputBatch`，SFT 170 个监督 token；PromptDataset 读取 3 行 RL |
| 正常 SkyRL rollout | `stop`，reward 2.986668，Loss Mask 中 170 个监督 token，工具观察被屏蔽 |
| 截断 SkyRL rollout | 真实 8-token / 1-turn 请求，`length`，reward 0，Loss Mask 全零 |
| 可选 Judge 链路 | 实际 HTTP/JSON Schema 请求 3/3 完成，0 请求或解析错误 |
| 下游结果接入 | Coding Agent 布尔结果与 SWE-bench 原生聚合 JSON 的解析、关联、重复/缺失处理测试通过 |
| 静态检查 | ruff 与 git diff whitespace 检查通过 |

三个任务取自现有 validation 样例的第 3、4、9 行，并固定实际仓库 commit。
`python-docx` 的 `NamespacePrefixedTag.local_part` 定位正确，文件、类、函数
F1 均为 1；Tweepy 和 glom 轨迹提交了不存在的方法，工具明确报错并被过滤。
三个任务包含失败轨迹后的 file/class/function 平均 F1 均为 1/3；平均工具成本
0.15241，平均总奖励 0.995556。样本全部为 easy，不代表总体定位质量或性能提升。

Judge 使用同一个定位 SFT 模型检验调用与解析，其解释仍会出现搜索计划、
分数也不可靠；这些分数不作为质量验收或效果结论。实际质量评测应配置独立 Judge。
SWE-bench 验收覆盖真实格式接入代码，未将测试输入或定位命中冒充完整基准 resolved。

## 复现与归档

完整准备、启动、任务选取和验收命令见 [SERVING_AND_EVALUATION.md](SERVING_AND_EVALUATION.md)。
最终统一验收命令如下（先按使用说明准备任务和启动服务）：

```bash
MODEL=/root/autodl-tmp/models/codepin-native-v2 \
TASKS=/root/autodl-tmp/codepin/runs/20260903-acceptance/tasks-pinned.jsonl \
CODEPIN_TEST_DEPLOYMENT_FILE=/root/autodl-tmp/codepin/runs/20260903-acceptance/deployment.json \
RUN_ROOT=/root/autodl-tmp/codepin/runs/20260903-acceptance/verification-new \
bash scripts/run_acceptance.sh
```

原始日志、JUnit XML、模型 manifest、任务、真实轨迹、导出 parquet、评测 JSON、
SkyRL 读回和 rollout 证据一并拉回本地 `tmp/verification/20260903-acceptance/`。
先前失败尝试保留在各自目录中；它们不计入通过验收。原始日志和轨迹不进入公开 Git 提交。
网络与模型配置故障处理记录见 [AUTODL_EXPERIMENT_RUNBOOK.md](AUTODL_EXPERIMENT_RUNBOOK.md)。
