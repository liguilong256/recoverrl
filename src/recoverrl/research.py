"""Recovery study v1: separate workflow and timing shifts, paired hidden faults."""

from __future__ import annotations

import hashlib
import random

from .specs import Contract, FAULTS, FaultPlan, Goal, TaskSpec

STUDY_VERSION = "recovery-v1"
RESEARCH_SPLITS = ("train", "val", "test_id", "test_workflow", "test_timing", "test_joint")


def make_research_suite(split: str, tasks: int = 32, seed: int = 0) -> list[TaskSpec]:
    if split not in RESEARCH_SPLITS or not 1 <= tasks <= 9999:
        raise ValueError("Unknown research split or task count outside [1, 9999]")
    workflow_shift = split in ("test_workflow", "test_joint")
    timing_shift = split in ("test_timing", "test_joint")
    families = ("diamond", "dag") if workflow_shift else ("batch", "chain", "fanout")
    specs = []
    for index in range(tasks):
        namespace = f"{STUDY_VERSION}/{seed}/{split}/{index}"
        task_seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:15], 16)
        rng = random.Random(task_seed)
        family = families[index % len(families)]
        count = rng.randint(4, 6)
        prefix = hashlib.sha256(namespace.encode()).hexdigest()[:12]
        goals = []
        for i in range(count):
            parents = ()
            if family == "chain" and i:
                parents = (goals[-1].request_id,)
            elif family == "fanout" and i:
                parents = (goals[0].request_id,)
            elif family in ("diamond", "dag") and i:
                if i in (1, 2):
                    parents = (goals[0].request_id,)
                elif i == 3:
                    parents = (goals[1].request_id, goals[2].request_id)
                elif family == "diamond":
                    parents = (goals[-1].request_id,)
                else:
                    parents = tuple(g.request_id for g in rng.sample(goals, rng.randint(1, min(3, i))))
            goals.append(Goal(f"req-{prefix}-{i}", f"{rng.choice(('Audit', 'Review', 'Prepare'))} item {i}", parents))
        # Both train and evaluation can fail at any depth, not just on the first write.
        target = rng.choice(goals).request_id
        bound = rng.choice((8, 12)) if timing_shift else rng.choice((2, 4))
        delay = rng.randint(1, bound)
        lags = (3, 6) if timing_shift else (0, 1)
        for keys in (False, True):
            for lag in lags:
                for mode in FAULTS:
                    specs.append(TaskSpec(
                        tuple(goals), Contract(keys, lag, bound), FaultPlan(mode, target, delay),
                        family, task_seed, split, max_steps=24, max_wait=120,
                    ))
    return specs
