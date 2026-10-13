# 服务器训练

目标环境：Linux、2 × A800 80GB、Miniconda。GPU 依赖和显存配置尚未实测，先完成一次更新，再启动正式实验。

## 1. 安装与准备

```bash
git clone https://github.com/liguilong256/recoverrl.git
cd recoverrl
conda env create -f environment.yml
conda activate recoverrl
bash scripts/setup_server.sh
python scripts/download_model.py
python -m recoverrl.study prepare
python -m recoverrl.study plan
```

服务器需要可用的 NVIDIA 驱动；编译 FlashAttention 还需要 CUDA toolkit。安装脚本检查当前 Conda 环境，在其中用 pip 安装固定版本的依赖，不修改驱动。安装失败时先保存错误与驱动/CUDA 版本。

`configs/study.json` 定义数据规模、种子和训练参数。准备结果在 `data/recovery-v1/`，实验计划在 `runs/recovery-v1/plan.json`。在服务器上生成计划，其中使用当前机器的绝对路径。已有同名输出时使用新的目录；已有 Conda 环境只需激活。

```bash
python -m pip check
python -m unittest discover -s tests -v
python scripts/check_verl_config.py --verl .deps/verl
python scripts/check_tokenizer.py --dataset data/recovery-v1/train.jsonl \
  --policy query_first --response-length 6144
python -m recoverrl.study run --stage main --dry-run
```

最后一条只显示六次主实验的命令。全部数据生成和检查可以在 CPU 上执行；模型下载包含权重，需要预留磁盘空间。

## 2. 评测基础模型

在另一个终端启动模型服务：

```bash
conda activate recoverrl
CUDA_VISIBLE_DEVICES=0 vllm serve models/qwen3-4b \
  --served-model-name recoverrl-policy --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --port 8000
```

然后运行验证集：

```bash
python -m recoverrl.study evaluate --checkpoint models/qwen3-4b \
  --splits val --output runs/eval/base
python -m recoverrl.study evaluate --policies once retry verify query_first \
  --splits val --output runs/eval/rules-val
```

`--checkpoint` 应与服务加载的模型一致。评测用该目录的 tokenizer 计算历史长度，统一限制每轮 512、整个响应与观察 6,144 token；用尽预算后结算已有写入。HTTP 服务错误会中止评测并保留已完成轨迹，不能把它当成模型失败样本。已失败的输出不自动续跑。

开始双卡训练前，停止这个 vLLM 服务，释放显存。

## 3. 一次更新与 SFT

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m recoverrl.study run --stage smoke
```

确认一次 optimizer update 完成并保存了 checkpoint，loss/KL/梯度有限，观察 token 的 mask 为 0、模型 token 的 mask 非零。检查奖励组有差异，指标包含 exact success、重复和无效动作；记录峰值显存、单次更新耗时及截断比例。

默认使用 GRPO、每提示四次 rollout、FSDP2、actor microbatch 1、优化器及 reference offload。这些是起始配置。如果 OOM，先检查序列长度和进程占用，再调整 batch/group；最终对照必须使用相同配置，group 至少为 2。

```bash
python -m recoverrl.study run --stage sft
```

SFT 使用去重后的训练集规则示范，全参数训练一轮，仅监督 assistant 内容。模型保存在 `runs/recovery-v1/sft/`。用它替换模型服务的加载路径，再评测 val，并将结果保存到 `runs/eval/sft`；评测后停止服务。

## 4. 主实验与消融

```bash
python -m recoverrl.study run --stage main
```

依次运行 completion / constrained 各三个种子，从同一 SFT checkpoint 初始化。各条件固定更新次数与 rollout 数，实际 token 用量仍需记录。每次训练的完整参数、日志和 checkpoint 分别保存在对应目录。

主对照确认值得继续后，再运行消融：

```bash
python -m recoverrl.study run --stage ablation
```

也可以只选一项：

```bash
python -m recoverrl.study run --jobs constrained-s42
```

调度器按顺序使用双卡，不并发启动多个训练。重复执行会跳过已完成的 job；失败或中断的 job 会报错，要求先检查日志，不会自动覆盖或假装完成。`state.json` 记录的是进程完成状态，GPU checkpoint 的可加载性仍须在下一步确认。修改计划、数据或恢复旧 checkpoint 时，使用单独的实验目录并记录来源。

## 5. 导出与最终评测

用固定版本 verl 的 model merger 导出 actor：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir runs/recovery-v1/constrained-s42/checkpoints/global_step_200/actor \
  --target_dir models/constrained-s42
```

先检查真实 checkpoint 路径。确认导出的目录含 tokenizer 文件；缺失时从实际 SFT 初始化模型复制并记录来源。用 vLLM 加载导出模型，先跑 val 检查可用性。

固定 checkpoint 选择规则后，再执行四个测试集：

```bash
python -m recoverrl.study evaluate --checkpoint models/constrained-s42 \
  --splits test_id test_workflow test_timing test_joint \
  --output runs/final/constrained-s42
```

基础模型、SFT 和每个 RL 种子都按同样的服务配置、tokenizer 与评测命令运行，分别命名输出目录。领域规则的最终评测：

```bash
python -m recoverrl.study evaluate --policies once retry verify query_first \
  --splits test_id test_workflow test_timing test_joint \
  --output runs/final/rules
```

汇总已完成的结果，例如：

```bash
python -m recoverrl.report \
  --inputs sft=runs/final/sft constrained-s42=runs/final/constrained-s42 rules=runs/final/rules \
  --reference sft/chat --output runs/report
```

输出 Markdown、CSV、JSON 和配对 bootstrap 差值。不同训练种子单独保留；报告拒绝不完整结果、不同数据版本或不同评测预算。后续解释按[研究计划](research_plan.md)执行。
