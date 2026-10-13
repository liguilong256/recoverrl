import json
import unittest

from recoverrl.data import make_task
from recoverrl.environment import RecoveryEnv
from recoverrl.policies import action
from recoverrl.specs import Contract, FaultPlan, Goal, TaskSpec


def spec(mode="none", keys=False, lag=0, max_steps=20, bound=4):
    return TaskSpec((Goal("a", "A"), Goal("b", "B"), Goal("c", "C")),
                    Contract(keys, lag, bound), FaultPlan(mode, "a", 4), max_steps=max_steps)


def create(ids="abc", key=None):
    arguments = {"items": [{"request_id": i, "title": i.upper(), "parent_ids": []} for i in ids]}
    if key is not None:
        arguments["idempotency_key"] = key
    return action("create_tickets", **arguments)


class EnvironmentTests(unittest.TestCase):
    def test_ambiguous_outcomes_have_identical_public_context(self):
        before, after = RecoveryEnv(spec("timeout_before")), RecoveryEnv(spec("timeout_after"))
        self.assertEqual(before.spec.public_view(), after.spec.public_view())
        self.assertEqual(before.reset(), after.reset())
        self.assertEqual(before.step(create()), after.step(create()))
        self.assertEqual(len(before.ledger), 0)
        self.assertEqual(len(after.ledger), 3)

    def test_partial_and_late_outcomes_also_hide_execution(self):
        observations = [RecoveryEnv(spec(mode)).step(create())
                        for mode in ("timeout_before", "timeout_after", "partial", "late_commit")]
        self.assertTrue(all(value == observations[0] for value in observations))

    def test_invalid_request_does_not_consume_fault(self):
        env = RecoveryEnv(spec("timeout_after"))
        invalid = create()
        invalid["arguments"]["items"][0]["parent_ids"] = ["not-a-ticket"]
        self.assertEqual(env.step(invalid)["result"]["error"], "invalid_action")
        self.assertFalse(env.fault_triggered)
        self.assertEqual(env.step(create())["result"]["error"], "timeout")
        self.assertTrue(env.fault_triggered)

    def test_fault_is_bound_to_logical_operation_not_call_count(self):
        env = RecoveryEnv(spec("timeout_before"))
        env.step(action("list_tickets", request_ids=["a"]))
        env.step(create("b"))
        self.assertFalse(env.fault_triggered)
        self.assertEqual(env.step(create("a"))["result"]["error"], "timeout")

    def test_blind_retry_duplicates_committed_writes(self):
        env = RecoveryEnv(spec("timeout_after"))
        env.step(create())
        env.step(create())
        metrics = env.finalize()
        self.assertEqual(metrics["task_success"], 1)
        self.assertEqual(metrics["exact_success"], 0)
        self.assertEqual(metrics["duplicate_effects"], 3)

    def test_same_key_resumes_partial_batch(self):
        env = RecoveryEnv(spec("partial", keys=True))
        env.step(create(key="stable"))
        self.assertEqual(len(env.ledger), 1)
        first = env.step(create(key="stable"))["result"]
        second = env.step(create(key="stable"))["result"]
        self.assertEqual(first, second)
        self.assertEqual(env.finalize()["exact_success"], 1)

    def test_key_payload_conflict_and_key_rotation(self):
        env = RecoveryEnv(spec("none", keys=True))
        env.step(create("a", key="stable"))
        self.assertEqual(env.step(create("b", key="stable"))["result"]["error"], "invalid_action")
        env.step(create("a", key="new-key"))
        self.assertEqual(env.finalize()["duplicate_effects"], 1)

    def test_keys_rejected_when_contract_lacks_support(self):
        env = RecoveryEnv(spec("timeout_after"))
        self.assertEqual(env.step(create(key="key"))["result"]["error"], "invalid_action")
        self.assertFalse(env.fault_triggered)

    def test_stale_read_until_visibility_bound(self):
        env = RecoveryEnv(spec("timeout_after", lag=3))
        env.step(create())
        self.assertEqual(env.step(action("list_tickets", request_ids=["a"]))["result"]["tickets"], [])
        env.step(action("wait", ticks=2))
        self.assertEqual(len(env.step(action("list_tickets", request_ids=["a"]))["result"]["tickets"]), 1)

    def test_late_commit_defeats_immediate_verify(self):
        env = RecoveryEnv(spec("late_commit"))
        env.step(create())
        self.assertEqual(env.step(action("list_tickets", request_ids=["a"]))["result"]["tickets"], [])
        env.step(create())
        env.step(action("finish", status="completed"))
        metrics = env.finalize()
        self.assertEqual(metrics["duplicate_effects"], 3)
        self.assertEqual(metrics["overclaim"], 1)

    def test_pending_same_key_does_not_add_second_request(self):
        env = RecoveryEnv(spec("late_commit", keys=True))
        env.step(create(key="one"))
        env.step(create(key="one"))
        self.assertEqual(len(env.pending), 1)
        self.assertEqual(env.finalize()["exact_success"], 1)

    def test_early_finish_and_budget_exhaustion_settle_pending(self):
        env = RecoveryEnv(spec("late_commit", max_steps=2))
        env.step(create())
        observation = env.step(create())
        self.assertTrue(observation["done"])
        self.assertEqual(env.finalize()["duplicate_effects"], 3)
        self.assertEqual(env.finalize(), env.finalize())
        with self.assertRaises(RuntimeError):
            env.step(create())

    def test_snapshot_restores_keys_pending_fault_and_clock(self):
        env = RecoveryEnv(spec("late_commit", keys=True, lag=3))
        env.step(create(key="same"))
        snapshot = json.loads(json.dumps(env.snapshot()))
        restored = RecoveryEnv.restore(snapshot)
        for command in [action("wait", ticks=6), create(key="same"), action("finish", status="completed")]:
            self.assertEqual(env.step(command), restored.step(command))
        self.assertEqual(env.finalize(), restored.finalize())
        self.assertEqual(env.ledger, restored.ledger)

    def test_goal_payload_and_dependencies_are_verified(self):
        task = TaskSpec((Goal("a", "A"), Goal("b", "B", ("a",))))
        env = RecoveryEnv(task)
        env.step(create("a"))
        env.step(create("b"))
        self.assertEqual(env.finalize()["task_success"], 0)
        env = RecoveryEnv(task)
        env.step(create("a"))
        command = create("b")
        command["arguments"]["items"][0]["parent_ids"] = ["ticket-1"]
        env.step(command)
        self.assertEqual(env.finalize()["exact_success"], 1)

    def test_unrequested_writes_and_false_success_are_penalized(self):
        env = RecoveryEnv(spec())
        env.step(create())
        env.step(create("z"))
        env.step(action("finish", status="completed"))
        metrics = env.finalize()
        self.assertEqual(metrics["unrequested_effects"], 1)
        self.assertEqual(metrics["task_success"], 0)
        self.assertEqual(metrics["overclaim"], 1)

    def test_agent_cannot_mutate_state_via_returned_observations(self):
        env = RecoveryEnv(spec())
        observation = env.step(create())
        observation["result"]["tickets"][0]["title"] = "tampered"
        self.assertEqual(env.finalize()["exact_success"], 1)

    def test_wait_limits_and_invalid_types(self):
        env = RecoveryEnv(spec())
        for ticks in (True, 0, -1, 61, "4"):
            self.assertEqual(env.step(action("wait", ticks=ticks))["result"]["error"], "invalid_action")
        self.assertEqual(env.waited, 0)

    def test_task_contract_rejects_inconsistent_bound(self):
        with self.assertRaises(ValueError):
            TaskSpec((Goal("a", "A"),), Contract(False, 0, 1), FaultPlan("late_commit", "a", 4))

    def test_reset_erases_private_episode_state(self):
        task = make_task(12, "chain", "late_commit", Contract(True, 3, 4))
        env = RecoveryEnv(task)
        env.step(action("finish", status="uncertain"))
        self.assertEqual(env.reset(), RecoveryEnv(task).reset())
        self.assertEqual(env.history, [])


if __name__ == "__main__":
    unittest.main()
