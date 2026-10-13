import contextlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from recoverrl.data import make_task, read_specs
from recoverrl.evaluation import run_episode
from recoverrl.policies import ScriptedPolicy
from recoverrl.research import RESEARCH_SPLITS, make_research_suite
from recoverrl.report import collect, write_report
from recoverrl.specs import Contract
from recoverrl.study import check_data, execute, load_config, main as study_main, make_plan, prepare, save_json


class ResearchTests(unittest.TestCase):
    def test_shift_axes_and_disjoint_goal_worlds(self):
        worlds = set()
        for split in RESEARCH_SPLITS:
            specs = make_research_suite(split, 12)
            current = {json.dumps(spec.to_dict()["goals"], sort_keys=True) for spec in specs}
            self.assertFalse(worlds & current)
            worlds |= current
            self.assertEqual(specs, make_research_suite(split, 12))
            held_out = split in ("test_workflow", "test_joint")
            self.assertEqual(any(len(g.parents) > 1 for s in specs for g in s.goals), held_out)
            timing = split in ("test_timing", "test_joint")
            self.assertEqual({s.contract.max_commit_delay for s in specs} <= {8, 12}, timing)
            self.assertTrue(any(s.fault.target != s.goals[0].request_id for s in specs))
            for offset in range(0, len(specs), 5):
                self.assertTrue(all(s.public_view() == specs[offset].public_view() for s in specs[offset:offset + 5]))

    def test_strong_references_solve_all_shift_axes_without_invalid_actions(self):
        for split in RESEARCH_SPLITS:
            for task in make_research_suite(split, 6):
                for policy in ("verify", "query_first"):
                    row = run_episode(task, ScriptedPolicy(policy))
                    self.assertEqual(row["metrics"]["exact_success"], 1, (policy, task.to_dict()))
                    self.assertEqual(row["metrics"]["invalid_actions"], 0)

    def test_early_positive_read_saves_calls_but_empty_read_does_not_authorize_retry(self):
        task = make_task(1, "batch", "timeout_after", Contract(False, 0, 12))
        early = run_episode(task, ScriptedPolicy("query_first"))
        conservative = run_episode(task, ScriptedPolicy("verify"))
        self.assertEqual(early["metrics"]["exact_success"], 1)
        self.assertLess(early["metrics"]["tool_calls"], conservative["metrics"]["tool_calls"])
        self.assertEqual(early["metrics"]["wait_ticks"], 0)
        task = make_task(1, "batch", "late_commit", Contract(False, 6, 12))
        row = run_episode(task, ScriptedPolicy("query_first"))
        self.assertEqual(row["trace"][1]["observation"]["result"]["tickets"], [])
        self.assertEqual(sum(t["action"]["tool"] == "create_tickets" for t in row["trace"]), 1)
        self.assertEqual(row["metrics"]["exact_success"], 1)

    def test_two_tick_gap_still_needs_wait_before_negative_read_is_conclusive(self):
        task = make_task(1, "batch", "late_commit", Contract(False, 1, 2))
        row = run_episode(task, ScriptedPolicy("query_first"))
        self.assertEqual(row["metrics"]["exact_success"], 1)
        self.assertEqual(sum(t["action"]["tool"] == "create_tickets" for t in row["trace"]), 1)
        self.assertGreater(row["metrics"]["wait_ticks"], 0)

    def test_unknown_bound_cannot_turn_empty_read_into_safe_retry(self):
        task = make_task(1, "chain", "timeout_before", Contract(False, 0, None))
        row = run_episode(task, ScriptedPolicy("query_first"))
        self.assertEqual(row["metrics"]["task_success"], 0)
        self.assertEqual(row["metrics"]["duplicate_effects"], 0)
        self.assertEqual(row["trace"][-1]["action"]["arguments"]["status"], "uncertain")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow not installed")
    def test_frozen_data_sft_sources_and_balanced_training_matrix(self):
        config = load_config("configs/study.json")
        config.update(train_tasks=3, val_tasks=3, test_tasks=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            prepare(config, data)
            manifest = check_data(data)
            self.assertEqual(manifest["splits"]["train"]["episodes"], 60)
            demos = [json.loads(line) for line in (data / "sft_train.jsonl").read_text().splitlines()]
            train_ids = {s.task_id for s in read_specs(data / "train.jsonl")}
            self.assertTrue({d["source_task_id"] for d in demos} <= train_ids)
            self.assertEqual(len(demos), len({json.dumps(d["messages"]) for d in demos}))
            self.assertNotIn('"fault"', json.dumps([d["messages"] for d in demos]))
            plan = make_plan(config, data, root / "plan", root / "base")
            main = [job for job in plan["jobs"] if job["stage"] == "main"]
            self.assertEqual(len(main), 6)
            self.assertEqual(len([job for job in plan["jobs"] if job["stage"] == "ablation"]), 6)
            for job in main:
                self.assertEqual(Path(job["argv"][job["argv"].index("--model") + 1]).resolve(),
                                 (root / "plan" / "sft").resolve())
                self.assertEqual(job["argv"][job["argv"].index("--val-max-samples") + 1], "-1")
            with contextlib.redirect_stdout(io.StringIO()):
                execute(root / "plan" / "plan.json", stage="main", dry_run=True)
            self.assertFalse((root / "plan" / "state.json").exists())
            with self.assertRaisesRegex(ValueError, "prerequisite"):
                with contextlib.redirect_stdout(io.StringIO()):
                    execute(root / "plan" / "plan.json", stage="main")
            with (data / "train.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                check_data(data)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow not installed")
    def test_runner_records_success_skips_completed_and_stops_on_failure(self):
        config = load_config("configs/study.json")
        config.update(train_tasks=1, val_tasks=1, test_tasks=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(config, root / "data")
            plan = make_plan(config, root / "data", root / "plan", root / "model")
            target = root / "result"
            marker = target / "done"
            command = f"from pathlib import Path; p=Path({str(target)!r}); p.mkdir(); (p/'done').write_text('ok')"
            plan["jobs"] = [{"id": "first", "stage": "smoke", "requires": [], "output": str(target),
                             "argv": ["-c", command]},
                            {"id": "second", "stage": "smoke", "requires": ["first"],
                             "output": str(root / "failed"), "argv": ["-c", "raise SystemExit(7)"]}]
            plan_path = root / "plan" / "plan.json"
            save_json(plan_path, plan)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    execute(plan_path, stage="smoke")
                self.assertTrue(marker.exists())
                execute(plan_path, job_ids=["first"])
                with self.assertRaisesRegex(ValueError, "interrupted or failed"):
                    execute(plan_path, job_ids=["second"])
            state = json.loads(plan_path.with_name("state.json").read_text())
            self.assertEqual(state["jobs"]["first"]["status"], "completed")
            self.assertEqual(state["jobs"]["second"]["status"], "failed")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow not installed")
    def test_reports_pair_only_complete_evaluations_and_preserve_separate_runs(self):
        config = load_config("configs/study.json")
        config.update(train_tasks=1, val_tasks=1, test_tasks=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(config, root / "data")
            with contextlib.redirect_stdout(io.StringIO()):
                study_main(["evaluate", "--data", str(root / "data"), "--output", str(root / "rules"),
                            "--splits", "val", "test_joint", "--policies", "verify", "query_first"])
            rows = write_report([f"rules={root / 'rules'}"], root / "report", "rules/verify", samples=20)
            self.assertEqual(len(rows), 4)
            self.assertTrue((root / "report" / "summary.csv").is_file())
            self.assertEqual(len(json.loads((root / "report" / "paired.json").read_text())), 8)
            path = root / "rules" / "val" / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["status"] = "running"
            save_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                collect([f"rules={root / 'rules'}"])


if __name__ == "__main__":
    unittest.main()
