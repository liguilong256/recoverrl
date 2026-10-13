"""CPU contract tests with a fake backend, NOT a substitute for a GPU smoke run."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from recoverrl.data import make_task
from recoverrl.specs import Contract


class FakeBase:
    def __init__(self, server_manager, response_length=4096, **kwargs):
        self.server_manager = server_manager
        self.rollout_config = SimpleNamespace(response_length=response_length)
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kw: bytes(ids).decode())

    async def ct_build_initial_tokens(self, messages):
        self.initial_messages = messages
        return list(json.dumps(messages).encode())

    async def ct_merge_assistant_token(self, runtime, ids, mask, logprobs, assistant_logprobs=None):
        # Deliberately include a non-generated boundary to test mask/logprob handling.
        result = SimpleNamespace(token_ids=runtime + [32] + ids)
        return result, mask + [0] + [1] * len(ids), (logprobs or []) + [0.0] + assistant_logprobs

    async def ct_merge_context_msg(self, previous, updated, runtime, mask, logprobs):
        ids = list(json.dumps(updated[-1]).encode())
        return SimpleNamespace(token_ids=runtime + ids), mask + [0] * len(ids), logprobs + [0.0] * len(ids)


def load_loop():
    stub = ModuleType("verl.experimental.agent_loop.agent_loop")
    stub.AgentLoopBase = FakeBase
    stub.AgentLoopOutput = SimpleNamespace
    path = Path(__file__).resolve().parents[1] / "src/recoverrl/training/verl_loop.py"
    spec = importlib.util.spec_from_file_location("recoverrl_test_verl_loop", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"verl.experimental.agent_loop.agent_loop": stub}):
        spec.loader.exec_module(module)
    return module.RecoverAgentLoop


class FakeServer:
    def __init__(self, responses):
        self.responses = responses
        self.indices = {}
        self.calls = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        await asyncio.sleep(0)
        index = self.indices.get(request_id, 0)
        self.indices[request_id] = index + 1
        self.calls.append((request_id, sampling_params.copy()))
        text = self.responses[index] if index < len(self.responses) else ""
        ids = list(text.encode())[:sampling_params["max_tokens"]]
        return SimpleNamespace(token_ids=ids, log_probs=[-0.5] * len(ids), extra_fields={})


class LoopTests(unittest.IsolatedAsyncioTestCase):
    def task_and_actions(self, mode="none"):
        task = make_task(42, "single", mode, Contract(False, 0, 4))
        goal = task.goals[0]
        actions = [json.dumps({"tool": "create_tickets", "arguments": {"items": [{
            "request_id": goal.request_id, "title": goal.title, "parent_ids": [],
        }]}}), '{"tool":"finish","arguments":{"status":"completed"}}']
        return task, actions

    async def test_mask_and_terminal_reward_alignment(self):
        task, actions = self.task_and_actions()
        server = FakeServer(actions)
        loop = load_loop()(server_manager=server)
        result = await loop.run({}, extra_info={"spec_json": json.dumps(task.to_dict())})
        self.assertEqual(len(result.response_ids), len(result.response_mask))
        self.assertEqual(len(result.response_ids), len(result.response_logprobs))
        self.assertEqual(sum(result.response_mask), sum(len(a.encode()) for a in actions))
        self.assertEqual(result.response_mask[-1], 1)
        self.assertGreater(result.reward_score, 0)
        trained = bytes(t for t, mask in zip(result.response_ids, result.response_mask) if mask).decode()
        self.assertEqual(trained, "".join(actions))
        self.assertNotIn('"fault"', json.dumps(loop.initial_messages))

    async def test_concurrent_rollouts_do_not_share_world_state(self):
        before, actions = self.task_and_actions("timeout_before")
        after, _ = self.task_and_actions("timeout_after")
        loop = load_loop()(server_manager=FakeServer(actions))
        results = await asyncio.gather(*[
            loop.run({}, extra_info={"spec_json": json.dumps(t.to_dict())}) for t in (before, after)
        ])
        self.assertEqual([r.extra_fields["reward_extra_info"]["exact_success"] for r in results], [0, 1])
        self.assertLess(results[0].reward_score, results[1].reward_score)

    async def test_context_budget_stops_without_losing_executed_action(self):
        task, actions = self.task_and_actions()
        loop = load_loop()(server_manager=FakeServer(actions), response_length=len(actions[0]) + 10)
        result = await loop.run({}, extra_info={"spec_json": json.dumps(task.to_dict())})
        info = result.extra_fields["reward_extra_info"]
        self.assertEqual(info["tool_calls"], 1)
        self.assertEqual(info["exact_success"], 1)
        self.assertEqual(info["truncated"], 1)
        self.assertEqual(result.response_mask[-1], 1)

    async def test_empty_backend_fails_visibly(self):
        task, _ = self.task_and_actions()
        loop = load_loop()(server_manager=FakeServer([""]))
        with self.assertRaisesRegex(RuntimeError, "no trainable"):
            await loop.run({}, extra_info={"spec_json": json.dumps(task.to_dict())})

    async def test_malformed_output_counts_as_action_without_state_corruption(self):
        task, actions = self.task_and_actions()
        loop = load_loop()(server_manager=FakeServer(["not json"] + actions))
        result = await loop.run({}, extra_info={"spec_json": json.dumps(task.to_dict())})
        info = result.extra_fields["reward_extra_info"]
        self.assertEqual(info["invalid_actions"], 1)
        self.assertEqual(info["exact_success"], 1)


if __name__ == "__main__":
    unittest.main()
