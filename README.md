# RecoverRL

工具调用超时后，Agent 应该重试、查询，还是等一会儿？RecoverRL 用一个可控的工单服务研究这个问题，并提供 SFT 和多轮 GRPO 的训练、评测代码。

例如，Agent 要创建三个工单，服务却返回了 `TIMEOUT`。此时可能一个都没创建，也可能已经创建了两个，或者请求还在执行。直接重试可能产生重复工单，直接结束又可能漏掉任务。环境会记录实际发生的写入，用它判断 Agent 是否完成了任务、是否重复执行，以及花了多少次调用。

目前已完成环境、规则基线和训练接口的本地检查，模型训练尚未开始。GPU 安装和训练配置需要在服务器上验证。

## 安装与运行

在已安装 Miniconda 的终端中执行：

```bash
git clone https://github.com/liguilong256/recoverrl.git
cd recoverrl
conda env create -f environment.yml
conda activate recoverrl
python -m pip install -e ".[dev,data]"

python -m unittest discover -s tests -v
python -m recoverrl evaluate --tasks 4 --output runs/quickstart
```

`environment.yml` 创建 Python 3.11 环境。项目依赖通过该环境中的 pip 安装；本地运行规则基线不需要 GPU。

每个基础任务会生成 20 个 episode，覆盖 5 种故障状态和 4 种工具契约。`runs/quickstart/` 保存动作、观察、实际写入和统计结果。再次运行时请换一个输出目录。

比较两种策略：

```bash
python -m recoverrl compare runs/quickstart/retry.jsonl runs/quickstart/verify.jsonl
```

## 环境与基线

环境支持单个工单、批量创建、链式依赖和分支依赖。故障包括提交前超时、提交后响应丢失、部分提交和延迟提交，也包含无故障的对照。工具契约决定是否支持幂等键、写入多久可见，以及提交时间是否有上界。

Agent 只能看到工具说明和返回结果。隐藏的故障状态不进入提示词；评分时会结算所有在途写入，因此提前结束也无法抹掉重复操作。

下面是 100 个合成基础任务的规则基线结果，每种策略运行 2,000 个 episode：

| 策略 | 任务完成率 | 完成且无重复 | 发生重复的比例 | 平均调用次数 |
| --- | ---: | ---: | ---: | ---: |
| `once`：遇到不确定结果就停止 | 45% | 45% | 0% | 2.150 |
| `retry`：直接重试 | 100% | 40% | 60% | 3.550 |
| `verify`：按工具契约查询、等待和重试 | 100% | 100% | 0% | 4.375 |

运行 `python scripts/report_baselines.py` 可复现[这份报告](reports/baseline_smoke.json)。这些是脚本得分，没有使用大模型。

`verify` 已经能解出当前有界环境中的全部任务。接下来的实验重点是：学到的策略能否迁移到更复杂的工作流，以及能否减少恢复所需的调用。只超过直接重试的基线，还不足以说明 RL 有用。

## 训练

默认模型为 Qwen3-4B-Instruct-2507，RL 使用 verl 的多轮 AgentLoop 和 GRPO。模型 revision、verl commit 和依赖版本见 [configs/versions.json](configs/versions.json) 与 [configs/gpu-constraints.txt](configs/gpu-constraints.txt)。

在 Linux GPU 服务器上，克隆仓库并创建、激活上述 Conda 环境后执行：

```bash
bash scripts/setup_server.sh
python scripts/download_model.py
bash scripts/prepare_data.sh
python scripts/check_tokenizer.py

# 先跑一次更新，确认训练能完成并保存 checkpoint。
python -m recoverrl.training.launch --steps 1 --output runs/gpu-smoke
```

默认配置面向两张 A800 80GB，尚未实测显存与吞吐。SFT、正式 RL 对照实验、checkpoint 导出和评测步骤见[服务器运行手册](docs/server_runbook.md)。没有 GPU 时，可以给训练命令加 `--dry-run` 检查生成的配置。

训练比较两种奖励：`completion` 只计算任务完成；`constrained` 另外扣除重复写入、额外写入和调用成本。

```bash
python -m recoverrl.training.launch --reward completion --seed 42 --output runs/completion-s42
python -m recoverrl.training.launch --reward constrained --seed 42 --output runs/constrained-s42
```

## 模型评测

生成数据并启动兼容 chat-completions 的服务后，可以评测基础模型或训练后的 checkpoint：

```bash
python -m recoverrl evaluate \
  --dataset data/test_ood.jsonl --policies chat \
  --base-url http://127.0.0.1:8000/v1 \
  --model recoverrl-policy --output runs/model-ood
```

需要密钥时设置 `RECOVERRL_API_KEY`。工具动作采用严格 JSON 格式，观察作为下一条 user 消息返回；格式错误会消耗动作预算。HTTP 服务出错时评测中止，已完成的轨迹会保留。

`test_id` 使用训练模板中的新任务，`test_ood` 留出链式和分支依赖结构。两者仍然使用同一个合成工单服务。数据划分、奖励公式、token 预算差异和统计方法见[实验说明](docs/experiment_protocol.md)。

## 代码与文档

- [src/recoverrl/environment.py](src/recoverrl/environment.py)：服务状态、故障注入和写入账本。
- [src/recoverrl/policies.py](src/recoverrl/policies.py)：规则基线与模型接口。
- [src/recoverrl/training/](src/recoverrl/training/)：SFT、verl AgentLoop、训练启动与预检。
- [tests/](tests/)：环境语义、数据隔离、接口和 rollout 测试。
- [本地验证记录](reports/validation.md)：已执行的检查及待验证项。
- [相关工作](docs/experiment_protocol.md#related-work)：工具学习、故障恢复和执行结果核验。

MIT License。基础模型和依赖库使用各自的许可证。
