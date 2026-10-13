"""Construct inspectable verl commands, or execute them on a prepared GPU host."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import sys

from recoverrl.rewards import RewardConfig


def overrides(args, agent_config: Path) -> list[str]:
    values = {
        "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": False,
        "data.train_files": str(Path(args.train).resolve()), "data.val_files": str(Path(args.val).resolve()),
        "data.train_batch_size": args.batch_size, "data.val_batch_size": 16,
        "data.val_max_samples": 80, "data.max_prompt_length": 2048,
        "data.max_response_length": 4096, "data.truncation": "error", "data.seed": args.seed,
        "data.dataloader_num_workers": 0,
        "actor_rollout_ref.model.path": str(Path(args.model).resolve()),
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.ppo_mini_batch_size": args.batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_kl_loss": True, "actor_rollout_ref.actor.kl_loss_coef": 0.01,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.fsdp_config.seed": args.seed,
        "actor_rollout_ref.actor.data_loader_seed": args.seed,
        "actor_rollout_ref.ref.strategy": "fsdp2",
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": args.group_size,
        "actor_rollout_ref.rollout.seed": args.seed,
        "actor_rollout_ref.rollout.temperature": 0.8, "actor_rollout_ref.rollout.top_p": 0.95,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.35,
        "actor_rollout_ref.rollout.max_model_len": 8192,
        "actor_rollout_ref.rollout.max_num_seqs": 16,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.agent.num_workers": 2,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "recoverrl",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(agent_config.resolve()),
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "trainer.n_gpus_per_node": args.gpus, "trainer.nnodes": 1,
        "trainer.project_name": "recoverrl", "trainer.experiment_name": f"{args.reward}-s{args.seed}",
        "trainer.logger": ["console", "tensorboard"], "trainer.total_training_steps": args.steps,
        "trainer.total_epochs": 100,
        "trainer.save_freq": 20, "trainer.test_freq": 20, "trainer.val_before_train": True,
        "trainer.resume_mode": "disable",
        "trainer.default_local_dir": str((Path(args.output) / "checkpoints").resolve()),
    }
    return [f"{key}={json.dumps(value)}" for key, value in values.items()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.parquet")
    parser.add_argument("--val", default="data/val.parquet")
    parser.add_argument("--model", default="models/qwen3-4b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reward", choices=["completion", "constrained"], default="constrained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--duplicate-penalty", type=float, default=1.0)
    parser.add_argument("--call-cost", type=float, default=0.01)
    parser.add_argument("--wait-cost", type=float, default=0.001)
    parser.add_argument("--token-cost-per-1k", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.gpus, args.steps, args.batch_size) < 1 or args.group_size < 2:
        parser.error("Positive GPU/step/batch sizes and group-size >= 2 required")
    if args.batch_size % args.gpus:
        parser.error("batch-size must be divisible by gpus")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty; use a new directory for every run")
    if not args.dry_run:
        from .preflight import check
        check(args.gpus)
        for path in (args.train, args.val, args.model):
            if not Path(path).exists():
                parser.error(f"Missing {path}; prepare data and download the pinned model first")
        from .validate_data import validate_files
        validate_files(args.train, args.val)
    output.mkdir(parents=True, exist_ok=True)
    reward = RewardConfig(mode=args.reward, duplicate_penalty=args.duplicate_penalty,
                          call_cost=args.call_cost, wait_cost=args.wait_cost,
                          token_cost_per_1k=args.token_cost_per_1k)
    agent_config = output / "agent_loop.yaml"
    agent_config.write_text(json.dumps([{
        "name": "recoverrl", "_target_": "recoverrl.training.verl_loop.RecoverAgentLoop",
        "max_tokens_per_turn": 512, "reward_config": asdict(reward),
    }], indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides(args, agent_config)]
    (output / "launch.json").write_text(json.dumps({"arguments": vars(args), "argv": command}, indent=2)
                                       + "\n", encoding="utf-8")
    print(shlex.join(command))
    if not args.dry_run:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (output / "packages.txt").write_text(result.stdout, encoding="utf-8")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
