# Server runbook

This recipe targets one Linux host with two A800 80GB GPUs and Python 3.11/3.12.
It has **not yet completed a GPU run**. Start with the one-update smoke test; do not schedule the full experiment matrix first.

## 1. Prepare an isolated environment

```bash
git clone https://github.com/liguilong256/recoverrl.git
cd recoverrl
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/setup_server.sh
```

The script installs a pinned verl checkout under `.deps/verl`, installs a candidate vLLM/PyTorch/Transformers combination, builds FlashAttention and runs a preflight check. A compatible NVIDIA driver and, when compiling FlashAttention, CUDA toolkit are host prerequisites. It does not modify the driver. If installation fails, save the exact error and host CUDA/driver versions before changing constraints.

The stack was selected by inspecting upstream APIs and package dependency metadata. Full CUDA compatibility is not implied by that inspection. Record a successful environment with `python -m pip freeze`; the training launcher saves its own package list.

```bash
python -m pip check
python -m unittest discover -s tests -v
python scripts/check_verl_config.py --verl .deps/verl
python scripts/download_model.py --output models/qwen3-4b
python scripts/check_tokenizer.py --model models/qwen3-4b
bash scripts/prepare_data.sh
```

The model downloader uses the immutable revision in `configs/versions.json`. It downloads weights; ensure sufficient disk space. All artifacts under `models/`, `data/`, `runs/` and `.deps/` are excluded from Git.

## 2. Evaluate the base model first

In a separate terminal in the same virtual environment:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve models/qwen3-4b \
  --served-model-name recoverrl-policy --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --port 8000
```

```bash
python -m recoverrl evaluate --dataset data/val.jsonl --policies chat \
  --model recoverrl-policy --output runs/base-val
```

Inspect failures and formatting before training. Stop the model server before launching two-GPU training, so it does not occupy one of the devices. Reserve `test_id` and `test_ood` for frozen final checkpoints; tune on `val`.

The evaluator processes episodes sequentially in v0.1. Run separate explicitly partitioned datasets for throughput if needed; preserve task IDs for pairing. It does not silently resume a partially failed output directory.

## 3. Optional cold-start SFT

```bash
torchrun --standalone --nproc_per_node=2 -m recoverrl.training.sft \
  --data data/sft_train.jsonl --model models/qwen3-4b \
  --output models/sft --fsdp --epochs 1
```

The SFT entry point uses full-parameter training, Qwen3 FSDP auto-wrapping, BF16, microbatch 1 and gradient accumulation 8. It masks all system/user observations and only trains assistant contents and end markers. It rejects dialogues that exceed the length limit or whose chat template rewrites earlier tokens.

If a cold start is used, compare both RL rewards from the **same SFT checkpoint** using `--model models/sft`. Also evaluate that SFT checkpoint without RL. These demonstrations come from a domain-aware rule policy; they are not evidence of a novel learned recovery algorithm.

## 4. One-update RL smoke test

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m recoverrl.training.launch \
  --model models/qwen3-4b --steps 1 --output runs/gpu-smoke
```

This uses GRPO, four rollouts per prompt, two FSDP2 GPUs, actor microbatch 1, an offloaded optimizer/reference, vLLM TP=1, eager execution and conservative rollout memory use. The values are a starting point, not a measured capacity guarantee.

Check before proceeding:

- one optimizer update and a final checkpoint actually complete;
- loss, KL, rewards and gradients are finite;
- nonzero assistant response masks and zero observation masks;
- `exact_success`, duplicate and invalid-action metrics are present;
- GRPO groups have useful reward variation (all-equal groups provide no relative signal);
- initial validation is plausible and not accidentally reading hidden fault state;
- no engine process remains from the earlier evaluation server.

Failures require inspection, not silently disabling masks or dropping difficult samples. If memory is tight, first lower train batch/group sizes and inspect sequence lengths. Group size must remain at least two. Keep changes identical across final comparison conditions.

## 5. Main comparisons

```bash
for seed in 42 43 44; do
  python -m recoverrl.training.launch --model models/sft \
    --reward completion --seed "$seed" --output "runs/completion-s$seed"
  python -m recoverrl.training.launch --model models/sft \
    --reward constrained --seed "$seed" --output "runs/constrained-s$seed"
done
```

Use the base-model path instead if no SFT was performed. The launcher refuses nonempty output directories and disables automatic checkpoint resumption, preventing accidental mixing of experiments. Native verl resumption can be configured manually after inspecting the generated `launch.json`; do not claim a fresh seed when resuming another run.

Examples of one-factor ablations:

```bash
python -m recoverrl.training.launch --duplicate-penalty 0 --output runs/no-duplicate-penalty
python -m recoverrl.training.launch --call-cost 0 --wait-cost 0 --output runs/no-cost-penalty
```

Each run stores its exact argv, agent config, dependency list and checkpoints. Checkpoint export to Hugging Face format uses the pinned verl model merger; inspect the produced checkpoint directory before invoking it:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir runs/constrained-s42/checkpoints/global_step_200/actor \
  --target_dir models/rl-constrained-s42
```

Then serve the exported model with vLLM and evaluate under the same prompts, sampling parameters and budgets as the base/SFT model. Confirm the checkpoint includes tokenizer files; if absent, copy them from its actual base checkpoint and record the source.

## 6. Interpretation

The scripted verify baseline saturates the bounded synthetic suite. This is an environment correctness result. Strong claims require expanded tasks, independent API/workflow transfer, comparable cost budgets, strong rules and SFT baselines, ablations and multiple training seeds.

Virtual wait ticks are not wall-clock seconds. The timeout is injected inside the simulator, not into model HTTP transport. Fault labels and effect ledgers belong only to the experiment runner. Do not insert them into prompts or teacher examples.
