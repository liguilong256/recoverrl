# RecoverRL

研究 Agent 在工具超时、部分执行后的恢复策略：什么时候该重试，什么时候该先检查结果。基于 verl / GRPO，包含模拟环境、训练和评测代码。

开发中，尚未完成模型训练。

## 快速开始

```bash
git clone https://github.com/liguilong256/recoverrl.git
cd recoverrl
conda env create -f environment.yml
conda activate recoverrl
python -m pip install -e ".[dev,data]"
python -m recoverrl evaluate --tasks 4 --output runs/quickstart
```

上面运行的是规则基线，无需 GPU，结果保存在 `runs/quickstart/`。

## 训练

默认使用 Qwen3-4B-Instruct-2507，配置面向两张 A800 80GB。安装和训练步骤见[训练指南](docs/server_runbook.md)。

[实验设置](docs/experiment_protocol.md) · [验证记录](reports/validation.md)

[MIT License](LICENSE)
