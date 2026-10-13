"""Validate training metadata and public prompts before allocating rollouts."""

import json

from recoverrl.environment import RecoveryEnv
from recoverrl.protocol import messages_for
from recoverrl.specs import TaskSpec


def validate_rows(rows, expected_split):
    if not rows:
        raise ValueError("Empty training/validation data")
    ids = set()
    for row in rows:
        spec = TaskSpec.from_dict(json.loads(row["extra_info"]["spec_json"]))
        if spec.split != expected_split:
            raise ValueError(f"Expected {expected_split}, found {spec.split}; refusing split leakage")
        if spec.task_id in ids:
            raise ValueError("Duplicate task spec in dataset")
        ids.add(spec.task_id)
        if row["prompt"] != messages_for(spec, RecoveryEnv(spec).reset()):
            raise ValueError("Prompt disagrees with public task view")
        if row.get("agent_name") != "recoverrl":
            raise ValueError("Wrong agent loop in dataset")
    return ids


def validate_files(train, val):
    import pyarrow.parquet as pq

    train_rows, val_rows = pq.read_table(train).to_pylist(), pq.read_table(val).to_pylist()
    train_ids = validate_rows(train_rows, "train")
    val_ids = validate_rows(val_rows, "val")
    if train_ids & val_ids:
        raise ValueError("Train/validation task overlap")
    def worlds(rows):
        return {
            json.dumps(json.loads(row["extra_info"]["spec_json"])["goals"], sort_keys=True)
            for row in rows
        }
    if worlds(train_rows) & worlds(val_rows):
        raise ValueError("Train/validation share underlying goals despite different metadata")
