import json
from pathlib import Path
import tempfile
import unittest

from recoverrl.data import make_suite, make_task, read_specs, write_jsonl
from recoverrl.environment import RecoveryEnv
from recoverrl.evaluation import paired_bootstrap, run_episode
from recoverrl.policies import ScriptedPolicy
from recoverrl.protocol import messages_for, parse_action
from recoverrl.rewards import RewardConfig, score
from recoverrl.specs import Contract, TaskSpec


class ExperimentTests(unittest.TestCase):
    def test_contract_reference_solves_bounded_development_suite(self):
        for task in make_suite("dev", tasks=8):
            row = run_episode(task, ScriptedPolicy("verify"))
            self.assertEqual(row["metrics"]["exact_success"], 1, task.to_dict())
            self.assertEqual(row["metrics"]["invalid_actions"], 0)

    def test_identical_prompts_across_hidden_faults(self):
        task = make_task(42, "batch", "timeout_before", Contract(False, 3, 4))
        other = TaskSpec.from_dict({**task.to_dict(), "fault": {**task.to_dict()["fault"], "mode": "timeout_after"}})
        self.assertEqual(messages_for(task, RecoveryEnv(task).reset()), messages_for(other, RecoveryEnv(other).reset()))
        self.assertNotIn(task.task_id, json.dumps(messages_for(task, RecoveryEnv(task).reset())))

    def test_splits_and_fault_pairs(self):
        splits = {name: make_suite(name, 4) for name in ("train", "val", "test_id", "test_ood")}
        all_ids = [task.task_id for tasks in splits.values() for task in tasks]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertFalse({t.seed for t in splits["train"]} & {t.seed for t in splits["val"]})
        self.assertFalse({t.family for t in splits["train"]} & {t.family for t in splits["test_ood"]})
        for tasks in splits.values():
            for offset in range(0, len(tasks), 5):
                views = [task.public_view() for task in tasks[offset:offset + 5]]
                self.assertTrue(all(view == views[0] for view in views))

    def test_reproducible_roundtrip(self):
        tasks = make_suite("train", 3, seed=7)
        self.assertEqual(tasks, make_suite("train", 3, seed=7))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            write_jsonl(path, [task.to_dict() for task in tasks])
            self.assertEqual(tasks, read_specs(path))

    def test_reward_cannot_be_increased_by_extra_cost_or_duplicates(self):
        metrics = {"task_success": 1, "duplicate_effects": 0, "unrequested_effects": 0,
                   "tool_calls": 3, "wait_ticks": 0}
        baseline = score(metrics, RewardConfig())
        for key in ("duplicate_effects", "unrequested_effects", "tool_calls", "wait_ticks"):
            changed = {**metrics, key: metrics[key] + 1}
            self.assertLess(score(changed, RewardConfig()), baseline)
        self.assertEqual(score({**metrics, "duplicate_effects": 3}, RewardConfig(mode="completion")), 1)

    def test_strict_json_no_silent_repair(self):
        for text in ('[]', '```json\n{}\n```', '{"a":NaN}', '{"a":1,"a":2}', '{} {}'):
            self.assertIn("invalid_output", parse_action(text))
        self.assertEqual(parse_action('{"tool":"finish","arguments":{"status":"uncertain"}}')["tool"], "finish")

    def test_pairing_requires_identical_unique_ids_and_preserves_clusters(self):
        specs = make_suite("dev", 2)
        left = [run_episode(t, ScriptedPolicy("retry")) for t in specs]
        right = [run_episode(t, ScriptedPolicy("verify")) for t in specs]
        result = paired_bootstrap(left, right, samples=100)
        self.assertEqual(result["clusters"], 2)
        self.assertGreater(result["difference"], 0)
        self.assertEqual(result, paired_bootstrap(left, right, samples=100))
        with self.assertRaises(ValueError):
            paired_bootstrap(left, right[:-1])
        with self.assertRaises(ValueError):
            paired_bootstrap(left + left[:1], right + right[:1])

    def test_unbounded_uncertainty_reference_stops(self):
        task = make_task(42, "chain", "late_commit", Contract(False, 3, None))
        row = run_episode(task, ScriptedPolicy("verify"))
        self.assertEqual(row["metrics"]["duplicate_episode"], 0)
        self.assertEqual(row["metrics"]["task_success"], 0)


if __name__ == "__main__":
    unittest.main()
