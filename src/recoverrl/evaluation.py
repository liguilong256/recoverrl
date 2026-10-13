"""Auditable episode traces and paired task-cluster bootstrap summaries."""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
import random
import statistics
import time

from . import __version__
from .environment import RecoveryEnv
from .protocol import messages_for, parse_action, tool_message
from .rewards import RewardConfig, score
from .specs import TaskSpec


def run_episode(spec: TaskSpec, policy, reward: RewardConfig | None = None) -> dict:
    reward = reward or RewardConfig()
    env = RecoveryEnv(spec)
    observation = env.reset()
    messages = messages_for(spec, observation)
    policy.reset(spec.public_view())
    generated_tokens, prompt_tokens, usage_available = 0, 0, True
    started = time.perf_counter()
    while not env.done:
        decision = policy.decide(copy.deepcopy(messages), copy.deepcopy(observation))
        messages.append({"role": "assistant", "content": decision.text})
        for value in (decision.generated_tokens, decision.prompt_tokens):
            if value is None:
                usage_available = False
        generated_tokens += decision.generated_tokens or 0
        prompt_tokens += decision.prompt_tokens or 0
        observation = env.step(parse_action(decision.text))
        if not env.done:
            messages.append(tool_message(observation))
    elapsed = time.perf_counter() - started
    metrics = env.finalize()
    if reward.token_cost_per_1k and not usage_available:
        raise ValueError("Token-cost reward requires endpoint usage accounting")
    metrics.update({
        "generated_tokens": generated_tokens if usage_available else None,
        "prompt_tokens": prompt_tokens if usage_available else None,
        "wall_seconds": elapsed, "reward": score(metrics, reward, generated_tokens),
    })
    # Cluster all paired modes and contracts from the same underlying task together.
    cluster = hashlib.sha256(json.dumps({"goals": spec.public_view()["goals"], "family": spec.family},
                                       sort_keys=True).encode()).hexdigest()[:20]
    return {
        "version": __version__, "task_id": spec.task_id, "cluster_id": cluster,
        "spec": spec.to_dict(), "reward_config": asdict(reward), "metrics": metrics,
        "messages": messages, "trace": env.history, "ledger": env.ledger,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No episodes to aggregate")
    keys = rows[0]["metrics"].keys()
    means = {}
    for key in keys:
        values = [row["metrics"].get(key) for row in rows]
        means[key] = statistics.mean(values) if all(v is not None for v in values) else None
    return {"episodes": len(rows), "task_clusters": len({r["cluster_id"] for r in rows}), **means}


def paired_bootstrap(left: list[dict], right: list[dict], metric: str = "exact_success",
                     samples: int = 2000, seed: int = 0) -> dict:
    if not left or samples < 1:
        raise ValueError("Need nonempty paired results and positive bootstrap samples")
    left_by_id = {row["task_id"]: row for row in left}
    right_by_id = {row["task_id"]: row for row in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right):
        raise ValueError("Duplicate task IDs: compare one policy / seed at a time")
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("Paired comparison requires exactly matching task IDs")
    clusters = defaultdict(list)
    for task_id, row in left_by_id.items():
        other = right_by_id[task_id]
        if row["cluster_id"] != other["cluster_id"]:
            raise ValueError("Paired rows have inconsistent task clusters")
        a, b = row["metrics"][metric], other["metrics"][metric]
        if a is None or b is None:
            raise ValueError("Metric missing in paired results")
        clusters[row["cluster_id"]].append(b - a)
    groups = list(clusters.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        drawn = [rng.choice(groups) for _ in groups]
        estimates.append(statistics.mean(value for group in drawn for value in group))
    estimates.sort()
    return {
        "metric": metric, "direction": "right minus left", "clusters": len(groups),
        "difference": statistics.mean(value for group in groups for value in group),
        "ci95": [estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]],
        "bootstrap_samples": samples, "seed": seed,
        "scope": "task clusters within this synthetic suite; not training-seed or real-world uncertainty",
    }
