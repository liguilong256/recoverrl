# RecoverRL

用强化学习训练 Agent 处理工具调用失败后的恢复。关注一个容易被忽略的问题：**工具返回超时，不代表操作没有执行。** 直接重试可能重复写入，直接结束又可能漏掉任务。

基于 Qwen3-4B-Instruct-2507 和 verl / GRPO。代码包含模拟环境、规则基线、SFT、RL 训练和评测；模型实验待运行。

## 想研究什么

- **恢复是否正确**：任务完成了没有，有没有重复或额外写入。
- **恢复是否划算**：同样完成任务，能否少查询、少等待。
- **策略能否迁移**：训练后换成多父节点依赖、更长的提交与读可见性延迟，是否仍然有效。

实验对比基础模型、SFT、普通完成奖励 GRPO、加入重复操作与成本惩罚的 GRPO，并保留两种强规则作参考。当前合成任务中，规则已能达到 100% 完成且无重复；训练的价值需要从成本和迁移实验中验证。

## 本地运行

```bash
git clone https://github.com/liguilong256/recoverrl.git
cd recoverrl
conda env create -f environment.yml
conda activate recoverrl
python -m pip install -e ".[dev,data]"
python -m recoverrl evaluate --tasks 4 --output runs/quickstart
```

这一步运行规则基线，不需要 GPU。

## 服务器训练

配置面向 Linux、两张 A800 80GB，GPU 安装与训练仍需实测。

```bash
bash scripts/setup_server.sh
python scripts/download_model.py
python -m recoverrl.study prepare
python -m recoverrl.study plan
python -m recoverrl.study run --stage smoke
```

检查一次更新能正常完成后，运行 SFT 和三个随机种子的 RL 对照：

```bash
python -m recoverrl.study run --stage sft
python -m recoverrl.study run --stage main
```

数据规模和训练参数在 [configs/study.json](configs/study.json)。每次实验会保存数据校验值、启动配置、日志和运行状态。

## 文档

- [研究计划](docs/research_plan.md)：实验问题、对照、消融和两个月的推进顺序。
- [训练指南](docs/server_runbook.md)：从环境安装到 checkpoint 评测。
- [实验约定](docs/experiment_protocol.md)：奖励、指标和数据划分。
- [规则基线结果](reports/research_baselines.json) · [验证记录](reports/validation.md)

[MIT License](LICENSE)
