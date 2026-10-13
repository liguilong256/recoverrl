"""Transparent terminal rewards; no reward for querying or declaring success."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "constrained"
    duplicate_penalty: float = 1.0
    unrequested_penalty: float = 1.0
    call_cost: float = 0.01
    wait_cost: float = 0.001
    token_cost_per_1k: float = 0.0

    def __post_init__(self):
        if self.mode not in ("completion", "constrained"):
            raise ValueError("Reward mode must be completion or constrained")
        if any(not math.isfinite(v) or v < 0 for k, v in asdict(self).items() if k != "mode"):
            raise ValueError("Reward coefficients must be finite and nonnegative")


def score(metrics: dict, config: RewardConfig, generated_tokens: int = 0) -> float:
    if generated_tokens < 0:
        raise ValueError("generated_tokens cannot be negative")
    if config.mode == "completion":
        return float(metrics["task_success"])
    return float(
        metrics["task_success"] - config.duplicate_penalty * metrics["duplicate_effects"]
        - config.unrequested_penalty * metrics["unrequested_effects"]
        - config.call_cost * metrics["tool_calls"] - config.wait_cost * metrics["wait_ticks"]
        - config.token_cost_per_1k * generated_tokens / 1000
    )
