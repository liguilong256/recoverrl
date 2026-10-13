"""Compose the launcher config against a local pinned verl checkout, without GPUs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    from hydra import compose, initialize_config_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--verl", required=True, help="Local pinned verl source directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "launch"
        subprocess.run([sys.executable, "-m", "recoverrl.training.launch", "--dry-run", "--output", str(output)],
                       check=True, capture_output=True, text=True)
        command = json.loads((output / "launch.json").read_text())["argv"]
        config_dir = str((Path(args.verl) / "verl/trainer/config").resolve())
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="ppo_trainer", overrides=command[3:])
        assert config.actor_rollout_ref.rollout.agent.default_agent_loop == "recoverrl"
        assert config.algorithm.adv_estimator == "grpo"
        assert config.trainer.n_gpus_per_node == 2
    print("verl Hydra config composition passed (no GPU execution).")


if __name__ == "__main__":
    main()
