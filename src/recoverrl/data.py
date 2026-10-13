"""Seeded development suites with explicit family splits and fault pairing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from .specs import Contract, FaultPlan, Goal, TaskSpec, FAULTS

SPLIT_FAMILIES = {
    "train": ("single", "batch"),
    "val": ("single", "batch"),
    "test_id": ("single", "batch"),
    "test_ood": ("chain", "fanout"),
    "dev": ("single", "batch", "chain", "fanout"),
}
SPLIT_SEEDS = {"train": 10000, "val": 20000, "test_id": 30000, "test_ood": 40000, "dev": 50000}


def make_task(seed: int, family: str, mode: str, contract: Contract, split: str = "dev") -> TaskSpec:
    rng = random.Random(seed)
    count = 1 if family == "single" else rng.randint(2, 4)
    # Task IDs/titles are invariant across fault modes and contracts for pairing.
    prefix = hashlib.sha256(str(seed).encode()).hexdigest()[:8]
    goals = []
    for i in range(count):
        parents = ()
        if family == "chain" and i:
            parents = (goals[-1].request_id,)
        elif family == "fanout" and i:
            parents = (goals[0].request_id,)
        goals.append(Goal(f"req-{prefix}-{i}", f"{rng.choice(['Audit', 'Review', 'Prepare'])} item {i}", parents))
    delay = contract.max_commit_delay if contract.max_commit_delay else 4
    return TaskSpec(tuple(goals), contract, FaultPlan(mode, goals[0].request_id, delay),
                    family, seed, split, max_steps=20)


def make_suite(split: str = "dev", tasks: int = 20, seed: int = 0) -> list[TaskSpec]:
    if split not in SPLIT_FAMILIES or tasks < 1 or tasks >= 10000:
        raise ValueError("Unknown split or task count outside [1, 9999]")
    if seed != 0:
        # Hash the run seed into a disjoint namespace; fixed split offsets remain disjoint.
        seed_base = (int(hashlib.sha256(str(seed).encode()).hexdigest()[:12], 16) + 1) * 100000
    else:
        seed_base = 0
    specs = []
    families = SPLIT_FAMILIES[split]
    for index in range(tasks):
        task_seed = seed_base + SPLIT_SEEDS[split] + index
        family = families[index % len(families)]
        for keys, lag in ((False, 0), (False, 3), (True, 0), (True, 3)):
            for mode in FAULTS:
                specs.append(make_task(task_seed, family, mode, Contract(keys, lag, 4), split))
    return specs


def write_jsonl(path: str | Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def read_specs(path: str | Path) -> list[TaskSpec]:
    with Path(path).open(encoding="utf-8") as stream:
        return [TaskSpec.from_dict(json.loads(line)) for line in stream if line.strip()]


def export_verl(specs: list[TaskSpec], output: str | Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .environment import RecoveryEnv
    from .protocol import messages_for

    rows = []
    for spec in specs:
        rows.append({
            "data_source": "recoverrl", "ability": "tool_recovery",
            "agent_name": "recoverrl",
            "prompt": messages_for(spec, RecoveryEnv(spec).reset()),
            "reward_model": {"style": "rule", "ground_truth": "environment_ledger"},
            # A JSON string prevents Arrow from converting variable-length fields to ndarray objects.
            "extra_info": {"spec_json": json.dumps(spec.to_dict(), sort_keys=True),
                           "index": spec.task_id, "split": spec.split},
        })
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
