# CodePin 服务与验收

本流程沿用 `pyproject.toml` 固定的 SkyRL 0.3.0、vLLM 0.23.0 和
OpenHands 提交，在 Linux x86_64 / Python 3.12 / bf16 NVIDIA GPU 上运行。
所有命令从仓库根目录执行；不启动正式训练。

## 环境与现有模型

```bash
uv sync
export HF_HOME=/root/autodl-tmp/cache/huggingface
uv run hf download LeeXugar/CodePin-SFT-Qwen3.5-0.8B \
  config.json model.safetensors tokenizer.json tokenizer_config.json \
  chat_template.jinja generation_config.json \
  --revision 4480baaf6a1b1f9fc0b3fb54c8480ed036c6f33a \
  --local-dir /root/autodl-tmp/models/codepin-sft
uv run python scripts/prepare_serving_model.py \
  --source /root/autodl-tmp/models/codepin-sft \
  --output /root/autodl-tmp/models/codepin-native
uv run python scripts/preflight_qwen3_5.py \
  --model /root/autodl-tmp/models/codepin-native
```

发布模型的配置是 `qwen3_5_text`，权重名却保留
`model.language_model.*`。准备脚本创建标准 Qwen3.5 wrapper 配置，供 vLLM
内置 `Qwen3_5ForConditionalGeneration` 使用，启动时设置
`--language-model-only`。权重以硬链接原样复用；输入、输出必须在同一文件系统。
输出目录必须不存在。`codepin_manifest.json` 记录原配置和文件 SHA-256。
准备脚本同时将服务配置的 EOS 对齐到 tokenizer 声明的 `<|im_end|>`，
并保留原 generation config。发布文件原先使用 `<|endoftext|>`，会使部分
受 JSON Schema 约束的请求提前停止；修正仅涉及输出目录中的配置文件。
原有会修改 attention 类方法的 vLLM 插件已移除。

## vLLM 与批处理

```bash
MODEL=/root/autodl-tmp/models/codepin-native \
DEPLOYMENT_FILE=/root/autodl-tmp/codepin-deployment.json \
MAX_MODEL_LEN=16384 MAX_BATCHED_TOKENS=8192 MAX_NUM_SEQS=8 \
PREFIX_CACHE=true bash scripts/serve_vllm.sh
```

默认监听 `127.0.0.1:8000`，模型名为 `codepin`，dtype 为 bf16。
`MAX_BATCHED_TOKENS` 控制每次调度的 token 预算；`MAX_NUM_SEQS` 控制引擎并发序列。
chunked prefill 已开启。Qwen3.5 的混合 attention 使用原生
`--enable-prefix-caching --mamba-cache-mode align`；`PREFIX_CACHE=false`
显式关闭两者。启动错误直接报告，不自动降级或更换内核。
可在脚本末尾附加其他 vLLM 原生参数。
vLLM 0.23.0 将 Mamba 的 `align` 缓存标为实验性功能；本项目通过真实请求验证
其功能，未据此声称吞吐或定位质量提升。
结构化输出使用原生 XGrammar，并禁用 JSON 字段间的任意空白，避免小模型
在 schema 约束内重复生成换行直至耗尽 token 预算。

并发 Chat Completions 会进入 vLLM 原生 continuous batching；原生多 prompt
请求也可直接调用：

```bash
curl http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"codepin","prompt":["def add(a,b):\n","def sub(a,b):\n"],"max_tokens":8}'
```

Prefix Cache 缓存模型中间状态，与下述完整定位结果缓存分开。

## MCP 委派

```bash
uv run python -m src.mcp_server \
  --repository-root /root/autodl-tmp/workspaces \
  --base-url http://127.0.0.1:8000/v1 --model openai/codepin \
  --concurrency 4 --max-turns 8 --max-tokens 2048 \
  --cache-size 64 --cache-ttl 300 \
  --deployment-file /root/autodl-tmp/codepin-deployment.json
```

默认 stdio 传输；Coding Agent 将以上 Python 命令配置成 MCP Server 即可。
HTTP 模式增加 `--transport streamable-http --port 8001`，地址为
`http://127.0.0.1:8001/mcp`。远程访问使用 SSH 转发本地监听端口。

`localize_code` 参数：

```json
{
  "request": {
    "repository": "my-repo",
    "issue": "Describe the bug to localize",
    "max_context_chars": 12000,
    "max_context_lines": 160
  }
}
```

`repository` 必须位于服务配置的目录内；工具解析真实路径并拒绝越界符号链接。
返回 `status`、`locations`、`context`、`metrics`、`errors`、`snapshot` 和
`cache_hit`。`locations` 包含 `file`、`class_name`、`function_name`；符号会通过
现有 Python AST 分析器校验，找不到定义时不会提交成功结果。
`context` 中每段带实际行号，所有段的 `text` 总字符数和行数遵守请求预算，
重复行不会重复返回。预算不包含 JSON 元数据；截断通过 `truncated` 表示。

`localize_batch` 接收 `{"requests": [request1, request2]}`，最多 32 个任务，
保持输入顺序和逐任务错误。服务 concurrency 与 vLLM 序列预算分别限制客户端
任务并发和引擎执行并发。

结果缓存默认关闭；启用时必须指定当前启动脚本生成的 deployment 文件。
缓存键包含完整仓库内容摘要（包括未跟踪文件、忽略文件和符号链接）、问题、
模型配置、提示词、工具代码、上下文预算及部署身份。每次启动更换部署身份，
修改工作区、更新提示词或重新部署都会失效。缓存还有 TTL、LRU 容量限制，
失败结果不缓存。运行中工作区或部署身份变化会使当前结果失效。
模型部署期间禁止原地替换权重，更新模型应重启服务。
对大型工作区，完整内容摘要会增加 I/O；将目标仓库与模型、日志、环境分开存放。

SkyRL rollout 直接调用共享的无缓存执行入口；`generator.result_cache=true`
会明确报错。数据轨迹生成同样不经过结果缓存。

## 数据与奖励

原始 SWE-Smith 的补丁应用、AST 标注继续使用
`python -m src.build_swe_smith_code_search`。处理已标注任务与模型轨迹：

```bash
uv run python -m src.data_pipeline clean \
  --tasks data/sample/validation.parquet --output outputs/clean
uv run python -m src.data_pipeline generate \
  --tasks outputs/clean/tasks.jsonl --output outputs/trajectories \
  --concurrency 2 --max-turns 8 --max-tokens 2048
uv run python -m src.data_pipeline export \
  --tasks outputs/clean/tasks.jsonl --trajectories outputs/trajectories \
  --output outputs/export --min-quality 0.5 --max-cost 3 \
  --validation-fraction 0.1
```

输出目录必须不存在。清洗检查必填字段、仓库/路径、不可变 commit、标签和
mutation patch，一致化 prompt；按内容和 instance ID 去重。难度规则为：
单文件且最多一个实体为 easy；最多三个文件且最多五个实体为 medium；其余 hard。
按 repository 哈希分割，防止同一快照的多个变异及同任务轨迹跨 split 泄漏。
小样本可能产生空 split，报告会如实记录。

轨迹校验要求工具参数可解析、Action/Observation 和 chat tool ID 成对、
恰好一个成功且独占最后一轮的 finish、token 前缀连续、无异常。重复工具动作序列
去重，并根据真实定位 F1 和工具成本过滤。输出含 `sft/{train,validation}.parquet`
与 `rl/{train,validation}.parquet`，以及拒绝原因、难度和分割计数。
SFT 包含 SkyRL 原生 messages/tools 和实际 TokenEvent 生成的 loss mask；
RL 保留原有 task/prompt/target 字段。SFT 训练配置仍使用
`train_on_what=all_assistant_messages`；用户、系统、工具观察不参与监督。

保留原始 file/module/entity F1 总分 `[0,3]`，额外显式报告
file/class/function F1。class F1 从原有 module 标签扣除同名顶层函数实体得到。
工具成本包含调用次数、相同搜索参数的重复次数、实际返回行的重叠比例、
输出字符数、超过每次 8000 字符的输出和截断次数。无“新内容”正奖励，
因此读取无关新文件不能增加 reward。默认总奖励为：

```text
max(0, localization_f1_sum - 0.2 * tool_efficiency_cost)
```

`generator.efficiency_weight` 可配置 `[0,1]`。异常、耗尽轮数、训练长度截断和
token 不连续轨迹的 Loss Mask 全零，总奖励为零；逐项指标保存在轨迹 JSON 和
SkyRL 分项日志中。

## 评测与复现

```bash
uv run python -m src.evaluate \
  --tasks outputs/clean/tasks.jsonl --trajectories outputs/trajectories \
  --output outputs/evaluation.json
# 可选；API key 通过 JUDGE_API_KEY 环境变量注入
uv run python -m src.evaluate \
  --tasks outputs/clean/tasks.jsonl --trajectories outputs/trajectories \
  --output outputs/evaluation-judge.json \
  --judge-model YOUR_JUDGE --judge-base-url http://127.0.0.1:8000/v1
```

报告含定位质量、工具调用/重复/重叠/输出成本、耗时、生成 token 数和难度分组。
Judge 使用 JSON Schema 输出相关性、充分性及解释；解析/网络失败单独记录，
不记作零分，也不代替确定性 F1。评测重复 ID 和未知 ID 会报错，缺失结果显式列出。
解释限制为 300 字符。现有定位 SFT 模型可用于验证 Judge 请求和解析链路，
不作为已校准的质量裁判；实际评测可通过 `--judge-model` 选择独立 Judge。

通过 `--downstream results.jsonl` 接入真实 Coding Agent 结果，每行至少有
`instance_id` 与布尔 `resolved`，可附加 `source`。也支持 SWE-bench harness
聚合 JSON 的 `resolved_ids` / `unresolved_ids`。报告区分已评测、缺失和未匹配任务，
不会把定位命中推算成 SWE-bench resolved。

先准备验收任务。本次使用现有 validation 样例的第 3、4、9 行（从零计数），
并记录实际使用的仓库 commit，固定样例原本为空的 `base_commit`：

```bash
mkdir -p outputs
uv run python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq
rows = pq.read_table('data/sample/validation.parquet').to_pylist()
selected = [rows[i] for i in (3, 4, 9)]
commits = (
    'ec4173316bc3b461cbb9712ac737a59584cbef58',
    '682fd3095bf6c405e5acf65ff0e3bbeb52b82eab',
    'cefbc05bb4890356f3782b5ae0b33aef27a47536',
)
for row, commit in zip(selected, commits, strict=True):
    row['base_commit'] = commit
Path('outputs/acceptance-tasks.jsonl').write_text(
    ''.join(json.dumps(row) + '\n' for row in selected), encoding='utf-8'
)
PY
```

启动上述 vLLM 服务，再执行全部单测和真实 MCP 集成测试：

```bash
CODEPIN_TEST_DEPLOYMENT_FILE=/root/autodl-tmp/codepin-deployment.json \
  uv run pytest tests -q
uv run python scripts/acceptance.py \
  --tasks outputs/acceptance-tasks.jsonl --limit 3 \
  --tokenizer /root/autodl-tmp/models/codepin-native \
  --output outputs/live-acceptance --judge
```

也可用 `scripts/run_acceptance.sh` 统一保存
preflight、完整 pytest、JUnit XML 和闭环日志；设置 `MODEL`、`TASKS`、
`CODEPIN_TEST_DEPLOYMENT_FILE` 与不存在的 `RUN_ROOT`。

集成测试没有 mock 模型或 skip 分支；后端不可用即失败。
闭环脚本真实克隆、应用 mutation、推理、校验、打分、过滤、导出，并调用现有
SkyRL tokenizer/collator 读回。仅导出不成功、无合格 SFT 行或 Judge 失败都会报错。
SFT 读回沿用现有训练入口的默认 tokenizer 参数；固定 SkyRL 版本的
`ALL_ASSISTANT_MESSAGES` 不接受额外的 `enable_thinking` 参数，不能把推理请求
参数直接传入该接口。
还会通过原生 `RemoteInferenceClient` 和 Ray 执行正常与过短生成预算的 SkyRL
rollout，检查正常监督、异常零奖励和零 Loss Mask；不会创建训练器或更新模型权重。
