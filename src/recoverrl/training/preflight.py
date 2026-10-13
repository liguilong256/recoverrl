"""Fail before an expensive launch if the local GPU/API contract is incompatible."""

import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys

VERL_COMMIT = "3e0d79ea8c38264bcaddc64de1b3465589580320"


def check(gpus=2):
    if platform.system() != "Linux" or sys.version_info < (3, 11):
        raise RuntimeError("GPU training requires Linux and Python >= 3.11")
    import torch
    import verl
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

    for name in ("ct_build_initial_tokens", "ct_merge_assistant_token", "ct_merge_context_msg"):
        if not inspect.iscoroutinefunction(getattr(AgentLoopBase, name, None)):
            raise RuntimeError(f"Incompatible verl AgentLoop API: {name}")
    if "reward_score" not in AgentLoopOutput.model_fields:
        raise RuntimeError("verl does not support inline environment reward_score")
    root = Path(verl.__file__).resolve().parents[1]
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != VERL_COMMIT:
        raise RuntimeError(f"verl revision mismatch: {revision}; expected {VERL_COMMIT}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < gpus:
        raise RuntimeError(f"Need {gpus} visible CUDA GPUs")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires BF16 support")
    report = {
        "python": sys.version, "verl_commit": revision, "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "vllm", "transformers", "flash-attn")},
        "gpus": [{"name": torch.cuda.get_device_name(i),
                  "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
                 for i in range(gpus)],
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    check()
