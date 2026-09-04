# 主分支功能提取与验收（2026-09-04）

本次按用户的新授权，将已经实现并重新验收的生产功能提取到 `main`。提取前主分支为 `b2cc195`；工程功能分支 `codex/mcp-serving-data-evaluation` 领先 1 个提交，调优分支 `codex/vllm-agent-performance-tuning` 领先 8 个提交，后者包含前者。

## 分支内容与取舍

| 原提交 | 内容 | 本次提取 |
|---|---|---|
| `9e56b61` | 原生模型部署、MCP 定位服务、结果缓存、数据清洗与 SFT/RL 导出、轨迹奖励和评测 | 保留这一完整功能提交及必要依赖锁文件，避免拆断调用链 |
| `97ada3b` | 真实负载压测、回放、Nsight、性能埋点，以及两个生产问题的修复 | 仅提取仓库完整内容哈希优化、并发前的 SDK schema 初始化及对应测试 |
| `1041c2d` | 质量压力结果、七项独立任务冻结 | 作为报告证据，实验内容继续保留在调优分支 |
| `b8e9401` | S64/C64 与 C96 的 Nsight 归因 | 同上 |
| `3931429` | 原生 CUDA Graph 范围对照 | 同上 |
| `305e58b` | Prefix Cache、批量预算和同步调度对照 | 同上 |
| `4a1c27a` | 基础设施失败分类、逐任务 F1 分母与重复基线比较 | 属于离线性能分析，继续保留在调优分支 |
| `0a4503c` | 显式区分定位改善与上下文成本增加 | 属于仍在前瞻验证的分析规则，未进入生产奖励或主分支比较逻辑 |

这不是将调优分支整体合并。主分支保留原生服务默认值：S8、B8192、上下文 16384、显存预算 0.70、Mamba `align` Prefix Cache。没有引入 FI 后端、扩大的 Graph 捕获列表、实验并发配置、Nsight 埋点或性能专用依赖。

两个附加修复的生产代码只涉及 `src/service.py`、`src/tools/__init__.py` 和 `src/agent/agent.py`：

- 完整内容哈希改用 `os.scandir`，继续读取全部文件字节，保留文件名、空目录、忽略/未跟踪文件和符号链接目标；不使用 mtime 捷径，也不扩大结果缓存命中范围。
- 在模块导入时完成四个固定工具的原生 SDK schema 注册与最终 Agent schema 初始化，避免并发会话第一次初始化时发生重复注册或未定义类型错误；没有修改 OpenHands 或 vLLM 依赖源码。

这两项修复与对应测试单独提交为 [`f46ca6f`](https://github.com/PeterLeeXX/CodePin/commit/f46ca6f772c63bcf15ca5c266b7938f3b9d5b733)：5 个文件，111 行新增、13 行删除。报告和图表另行提交，便于单独审查生产改动。主分支以快进方式接纳原 `9e56b61` 与该修复，没有重写原提交历史。

## 对提取组合的重新验收

验收在独立源码目录 `/root/autodl-tmp/codepin-stable-main-v108` 进行。47 个代码/环境文件按清单逐件核验，复用既有匹配的 Python 3.12.14 隔离环境且不改动依赖；直接依赖版本和 Git 来源均与项目声明核对。实际服务仍使用同一 SFT 权重和原生 vLLM。

| 验收项 | 本次结果 |
|---|---|
| 静态检查 | ruff 通过 |
| 全部 pytest | **55 passed，0 failed / errors / skipped，24.33 秒** |
| 冷启动并发初始化 | 新解释器中 16 线程创建 64 个真实 SDK Conversation，schema 一致 |
| 真实 MCP | stdio 初始化、工具发现、真实定位、结构化结果、有界上下文、双任务批处理、越界拒绝通过 |
| 缓存正确性 | 真实结果命中、仓库内容变化失效通过；单测覆盖部署身份、请求预算、TTL/LRU 和同大小/恢复时间戳的修改 |
| 原生 vLLM 批处理 | 一个 completions 请求包含两个 prompt，返回两个真实生成结果 |
| 数据与轨迹闭环 | 3 个固定真实任务，生成并评测 3 条，SFT 接受 1 / 拒绝 2，RL 导出 3 |
| SkyRL 读回 | 原生 `TrainingInputBatch`；SFT 170 个监督 token；RL PromptDataset 3 行 |
| 正常真实 rollout | `stop`，reward 2.986668，170 个监督 token，工具观察不参与 Loss |
| 截断真实 rollout | `length`，reward 0，Loss Mask 全零 |
| Judge 接入 | 实际请求和解析 3/3 完成，0 错误；不把同一小模型的打分当独立质量评价 |

三个任务中，python-docx 定位正确，Tweepy 和 glom 提交了不存在的方法并被正确拒绝、过滤；全量文件/类/函数 F1 均为 1/3。功能验收通过不等于模型在所有定位任务上都成功，也不等于已运行完整 SWE-bench。

第一次验收保留为 `stable-main-acceptance-v108`：54 项通过、1 项因独立目录漏带原有 `data/sample/validation.parquet` 而失败。补齐与原 Git 对象 SHA-256 相同的样例数据后，源码和测试不变，完整重跑 `stable-main-acceptance-v109` 通过。没有跳过测试或替换真实推理。

本次功能回归使用当时已经运行的原生 S64/FI/G1024 引擎；这是功能验证，不用于给主分支默认 S8 配置宣称性能收益。启动脚本与已经完成 18 组原部署基线的脚本字节一致。为避免相互干扰，先暂停性能调度器，等正在运行的测量子进程完整结束，再执行功能验收；验收后恢复同一调度器，原完成测量的内容摘要不变。

## 证据与复现

机器可读验收摘要见 [main-acceptance.json](assets/vllm-tuning/20260904-main-acceptance.json)。完整 JUnit、日志、MCP 结果、真实轨迹、parquet 和 SkyRL 结果已拉回：

`tmp/verification/20260904-vllm-performance/stable-main-material-v111/`

47 个归档条目全部核验通过。压缩包 `stable-main-material-v111.tar.gz` 为 379439 bytes，SHA-256 `6e32bad0111ceb79f612410032aa3d11c2029065e9160b784cb916796c885aed`。

原生部署和通用复现步骤见 [SERVING_AND_EVALUATION.md](SERVING_AND_EVALUATION.md)。本次实际命令和环境保存在材料中的 `configs/stable-main-acceptance-protocol-v109.json`、`configs/accept-stable-main-v109.py`；入口为 `bash scripts/run_acceptance.sh`，未开展训练。

完整调优方案、收益、退化与验证边界见 [VLLM_TUNING_REPORT_20260904.md](VLLM_TUNING_REPORT_20260904.md)。其中的候选性能提升不能当作本次主分支已经交付的相同幅度提升。
