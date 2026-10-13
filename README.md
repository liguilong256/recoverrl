# RecoverRL

**Learning to recover when a tool times out but may already have executed.**

用于研究工具执行结果不确定时，Agent 如何核验、等待、重试和停止的开源实验代码。包含确定性环境、效果账本、规则基线、模型评测，以及基于 verl 的多轮 GRPO 训练入口。

**当前状态：v0.1 实验底座。CPU 测试、数据导出、HTTP 接口和训练配置已经验证；没有真实模型得分，没有已完成的 SFT / RL 训练，也没有算法提升结论。GPU 依赖组合与显存配置仍需在服务器上验证。**

## 研究问题

批量创建三个工单时收到超时，可能一个都没创建，也可能创建了一部分，或者请求仍在执行。直接重试会重复写入；直接结束会遗漏工作。

本项目检验：在提供相同工具语义、使用可比预算时，RL 能否让小模型在未见过的工作流中提高“完成任务且没有重复副作用”的比例，或降低恢复成本。

本项目没有提出新的 GRPO 优化器。当前候选改动是基于效果账本的副作用惩罚与成本约束；是否比 SFT、普通 RL 和规则更好，需要后续实证。

## 本地快速开始

核心环境无第三方依赖，支持 Python 3.9+；推荐 Python 3.11 或 3.12。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,data]"
python -m unittest discover -s tests -v
python -m recoverrl evaluate --tasks 4 --output runs/quickstart
```

每个基础任务展开为 **5 种故障状态 × 4 种工具契约 = 20 个 episode**。输出包含逐步动作、观察、效果账本、模型 token 用量（如果端点提供）、评测结果和运行清单。输出目录必须为空，避免覆盖已有实验。

```bash
python -m recoverrl compare runs/quickstart/retry.jsonl runs/quickstart/verify.jsonl
```

比较采用配对的任务簇 bootstrap；同一任务的不同故障与契约不视为独立任务。这不能代替训练随机种子的重复实验。

## 已实现

- 工单环境：单任务、批量、链式依赖、分支依赖；`reset / step / snapshot / restore`。
- 故障：无故障、提交前超时、提交后响应丢失、部分提交、延迟提交。
- 工具契约：幂等键、读可见性延迟、已知或未知的提交时间上界。
- 不可变效果账本：结算在途写入后评分；最终文字声明不会改变任务得分。
- 三种脚本基线、标准 chat-completions HTTP 模型评测、轨迹导出。
- 确定性数据生成、JSONL / verl Parquet 导出、训练集限定的 SFT 示范导出。
- 可选 SFT 入口、verl 自定义 AgentLoop、两张 GPU 的 GRPO 候选配置。
- CPU 测试与跨 Windows / Linux 的 GitHub Actions。

工具使用严格 JSON 动作协议，环境观察作为下一条 user 消息返回。**这一版不是原生 function-calling benchmark，也没有接入真实第三方服务。**

## 环境自检结果

以下是 **100 个合成基础任务、每策略 2,000 个 episode** 的脚本结果。与 LLM 能力或 RL 提升无关；故障分布是人为平衡的压力测试分布。

| 脚本策略 | 任务完成率 | 完成且无重复 | 发生重复的 episode | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: |
| 遇到模糊结果就停止 `once` | 45% | 45% | 0% | 2.150 |
| 直接重试 `retry` | 100% | 40% | 60% | 3.550 |
| 利用工具契约恢复 `verify` | 100% | 100% | 0% | 4.375 |

复现：[基线报告](reports/baseline_smoke.json)，运行 `python scripts/report_baselines.py`。

`verify` 是了解当前工单领域的强规则参考。它在有界合成环境中已经满分，因此这一版环境主要用于正确性检查与训练链路验证。**不能用击败弱重试策略证明方法贡献。** 后续研究需要更广的工具/工作流、同等信息的可迁移规则基线，以及成本与质量的联合比较。

## 服务器训练

按 [服务器运行说明](docs/server_runbook.md) 执行。所有训练都在你自己的服务器上进行。

```bash
git clone https://github.com/liguilong256/recoverrl.git
cd recoverrl
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/setup_server.sh
python scripts/download_model.py
bash scripts/prepare_data.sh
python scripts/check_tokenizer.py

# 先做一次更新，检查显存、token 对齐、奖励和 checkpoint。
python -m recoverrl.training.launch --steps 1 --output runs/gpu-smoke
```

固定了模型 revision 和 verl commit；具体版本见 [configs/versions.json](configs/versions.json)。安装脚本的依赖是候选组合，**尚未在 A800 上执行**。预检会检查 Linux、CUDA、BF16、GPU 数量、verl API 和源码版本；正式实验会记录完整依赖清单。

普通完成奖励与约束奖励使用相同训练入口：

```bash
python -m recoverrl.training.launch --reward completion --seed 42 --output runs/completion-s42
python -m recoverrl.training.launch --reward constrained --seed 42 --output runs/constrained-s42
```

每种条件最终至少做三个训练种子，并保存所有结果。可用 `--dry-run` 在没有 GPU 的机器生成完整命令和 AgentLoop 配置。

## 模型评测

启动本地 vLLM 或其他兼容的 chat-completions 端点，然后执行：

```bash
python -m recoverrl evaluate \
  --dataset data/test_ood.jsonl --policies chat \
  --base-url http://127.0.0.1:8000/v1 \
  --model recoverrl-policy --output runs/model-ood
```

如果端点需要密钥，通过 `RECOVERRL_API_KEY` 环境变量提供。不要将密钥写入命令、配置或仓库。HTTP 错误会中止评测，已完成的轨迹保留；不会将服务不可用计为 Agent 任务失败。

不启用 JSON 自动修复，格式错误会计入环境动作预算。端点缺少 token 用量时，结果保存为 `null`；脚本基线 token 数为零。

当前 HTTP 评测用每回合 token 上限和环境步数约束；RL 则限制整个响应序列（含观察）的 token 数。比较真实模型成本前，需要统一这些预算设置并记录实际用量，不能直接将默认配置视为完全等预算。

## 数据与奖励

```bash
python -m recoverrl generate --tasks 100 --output data
python -m recoverrl export-verl --dataset data/train.jsonl --output data/train.parquet
python -m recoverrl export-sft --dataset data/train.jsonl --output data/sft_train.jsonl
```

- `train / val / test_id` 使用不同种子的 single / batch 任务，属于**模板内**数据划分。
- `test_ood` 留出 chain / fanout 依赖结构，只能支持**合成工作流结构迁移**的结论。
- 这些集合共享工单服务及词汇，不代表跨领域、跨 API 或真实业务泛化。
- 同一基础任务的隐藏故障变化不会进入提示词；`spec_json` 只交给可信环境和评分器。
- SFT 数据来自成功的规则轨迹，明确标注为脚本示范，不是教师大模型生成数据。

默认约束奖励为：

```text
task_success
  - 1.0 * duplicate_effects
  - 1.0 * unrequested_effects
  - 0.01 * tool_calls
  - 0.001 * wait_ticks
```

可配置 token 成本。普通 GRPO 对照只使用 `task_success`。查询和停止不会单独获得正奖励。评分先结算所有在途写入，即使模型提前结束或耗尽步数。

## 验证边界

| 层级 | 状态 |
| --- | --- |
| 环境故障语义、效果账本、恢复与数据隔离 | CPU 实测 |
| JSONL / Parquet、SFT 标签构造、HTTP 客户端 | CPU 实测；HTTP 使用本地假服务 |
| 多轮 rollout 状态隔离、token mask、奖励结算 | 假推理后端的契约测试 |
| verl 配置字段 | 固定源码版本的 Hydra 配置加载通过 |
| 实际 Qwen tokenizer 对齐 | 固定模型 revision 的 80 条对话检查通过 |
| 模型推理、SFT、RL 梯度更新、A800 显存与吞吐 | 待服务器验证 |

详见 [本地验证记录](reports/validation.md)。这些检查不能替代一次真实 GPU 更新。

## 相关工作与贡献边界

已有 [ToolRL](https://arxiv.org/abs/2504.13958)、[ToolRL-DR](https://arxiv.org/abs/2605.11928)、[Bench2Robust](https://arxiv.org/abs/2608.11977) 研究工具学习与恢复训练。[Verified Tool Calls](https://arxiv.org/abs/2608.02645) 和 [LIMBO](https://arxiv.org/abs/2609.29095) 研究模糊执行结果、核验及幂等语义。

RecoverRL 是受这些问题启发的独立实验实现，不是上述项目的官方复现；不声称首次提出故障注入、核验后重试或效果账本评测。算法贡献取决于未来强基线、消融和泛化实验的证据。

## 目录

```text
src/recoverrl/
  environment.py      # 服务状态、故障计划、效果账本
  specs.py            # 私有实验定义与 public_view
  policies.py         # 规则与 HTTP 模型策略
  evaluation.py       # 轨迹、指标与配对统计
  training/           # SFT、verl AgentLoop、启动与预检
configs/              # 源码/模型版本与 GPU 依赖候选
scripts/              # 服务器准备、配置检查、tokenizer 检查
tests/                # CPU 语义与接口测试
reports/              # 可复现的公开报告
docs/                 # 运行手册和实验约定
```

许可证：MIT。依赖库与基础模型遵循各自许可证。
