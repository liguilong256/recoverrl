import importlib.util
import json
from pathlib import Path
import tempfile
from threading import Thread
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest

from recoverrl.data import export_verl, make_suite, make_task
from recoverrl.environment import RecoveryEnv
from recoverrl.evaluation import run_episode
from recoverrl.policies import ChatPolicy
from recoverrl.protocol import messages_for
from recoverrl.training.sft_data import encode_dialogue
from recoverrl.training.validate_data import validate_rows, validate_files
from recoverrl.specs import Contract


class CharacterTemplate:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
        text = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode())


class InterfaceTests(unittest.TestCase):
    def test_http_context_budget_caps_request_and_settles_pending_writes(self):
        task = make_task(5, "single", "late_commit", Contract(False, 0, 4))
        requests = []

        class BudgetTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [0] * (10 + 20 * len(messages))

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                content = json.dumps({"tool": "create_tickets", "arguments": {"items": [{
                    "request_id": task.goals[0].request_id, "title": task.goals[0].title, "parent_ids": [],
                }]}})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            policy = ChatPolicy(f"http://127.0.0.1:{server.server_port}/v1", "test", max_tokens=512,
                                tokenizer=BudgetTokenizer(), response_length=40)
            row = run_episode(task, policy)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_tokens"], 32)
        self.assertEqual(row["termination"], "token_budget")
        self.assertEqual(row["metrics"]["exact_success"], 1)
        self.assertEqual(row["metrics"]["tool_calls"], 1)

    def test_sft_masks_only_assistant_content_and_end_markers(self):
        messages = [{"role": "system", "content": "PRIVATE_SYSTEM"},
                    {"role": "user", "content": "TOOL_OUTPUT"},
                    {"role": "assistant", "content": "ACTION_1"},
                    {"role": "user", "content": "NEXT_OBSERVATION"},
                    {"role": "assistant", "content": "ACTION_2"}]
        data = encode_dialogue(CharacterTemplate(), messages)
        trained = bytes(token for token in data["labels"] if token != -100).decode()
        self.assertEqual(trained, "ACTION_1</end>ACTION_2</end>")
        self.assertEqual(len(data["labels"]), len(data["input_ids"]))

    def test_sft_refuses_silent_truncation_or_rewriting(self):
        messages = [{"role": "user", "content": "input"}, {"role": "assistant", "content": "answer"}]
        with self.assertRaises(ValueError):
            encode_dialogue(CharacterTemplate(), messages, max_length=1)

        class RewritingTemplate(CharacterTemplate):
            def apply_chat_template(self, *args, **kwargs):
                ids = super().apply_chat_template(*args, **kwargs)
                return [0] + ids if kwargs.get("add_generation_prompt") else ids

        with self.assertRaises(ValueError):
            encode_dialogue(RewritingTemplate(), messages)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow not installed")
    def test_parquet_roundtrip_keeps_fault_only_in_private_metadata(self):
        import pyarrow.parquet as pq

        specs = make_suite("train", 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            export_verl(specs, path)
            rows = pq.read_table(path).to_pylist()
        self.assertEqual(len(rows), len(specs))
        for spec, row in zip(specs, rows):
            self.assertEqual(json.loads(row["extra_info"]["spec_json"]), spec.to_dict())
            self.assertEqual(row["prompt"], messages_for(spec, RecoveryEnv(spec).reset()))
            self.assertNotIn('"fault"', json.dumps(row["prompt"]))
        validate_rows(rows, "train")
        with self.assertRaises(ValueError):
            validate_rows(rows, "val")
        rows[0]["prompt"][0]["content"] += " secret fault hint"
        with self.assertRaisesRegex(ValueError, "Prompt disagrees"):
            validate_rows(rows, "train")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow not installed")
    def test_different_split_labels_cannot_hide_task_overlap(self):
        from recoverrl.specs import TaskSpec

        train = make_suite("train", 1)
        val = [TaskSpec.from_dict({**task.to_dict(), "split": "val"}) for task in train]
        with tempfile.TemporaryDirectory() as directory:
            train_path, val_path = Path(directory) / "train.parquet", Path(directory) / "val.parquet"
            export_verl(train, train_path)
            export_verl(val, val_path)
            with self.assertRaisesRegex(ValueError, "share underlying goals"):
                validate_files(train_path, val_path)

    def test_chat_http_episode_and_token_accounting(self):
        task = make_suite("dev", 1)[0]
        goal = task.goals[0]
        outputs = [json.dumps({"tool": "create_tickets", "arguments": {"items": [{
            "request_id": goal.request_id, "title": goal.title, "parent_ids": [],
        }]}}), '{"tool":"finish","arguments":{"status":"completed"}}']
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                content = outputs[len(requests) - 1]
                data = json.dumps({"choices": [{"message": {"content": content}}],
                                   "usage": {"completion_tokens": 12, "prompt_tokens": 30}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            policy = ChatPolicy(f"http://127.0.0.1:{server.server_port}/v1", "mock-model")
            row = run_episode(task, policy)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(row["metrics"]["exact_success"], 1)
        self.assertEqual(row["metrics"]["generated_tokens"], 24)
        self.assertEqual(row["metrics"]["prompt_tokens"], 60)
        self.assertEqual(len(requests[1]["messages"]), 4)
        self.assertNotIn('"fault"', json.dumps(requests))


if __name__ == "__main__":
    unittest.main()
