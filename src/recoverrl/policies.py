"""Reference policies use public task data and observations only."""

from __future__ import annotations

import copy
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Decision:
    text: str
    generated_tokens: int | None = 0
    prompt_tokens: int | None = 0


class PolicyBudgetExceeded(Exception):
    """No room remains for another model response; settle the environment."""


def action(tool: str, **arguments) -> dict:
    return {"tool": tool, "arguments": arguments}


class ScriptedPolicy:
    """Deterministic, domain-aware references, not learned model results.

    once: stop on ambiguous outcome. retry: repeat failed batch once.
    verify: use stable keys when available; otherwise wait for the documented
    commit + visibility bound, query state and issue only missing operations.
    query_first: query immediately without keys; act on confirmed writes, but
    treat a negative read as inconclusive until the commit + visibility bound.
    """

    def __init__(self, mode: str):
        if mode not in ("once", "retry", "verify", "query_first"):
            raise ValueError(f"Unknown policy: {mode}")
        self.mode = mode

    def reset(self, public_task: dict):
        self.task = copy.deepcopy(public_task)
        self.known = {}
        self.last = None
        self.pending = None
        self.retries = 0
        self.batch_number = 0
        self.phase = "plan"
        self.safe_read_at = None

    def _next(self, observation: dict) -> dict:
        result = observation["result"]
        if self.phase == "early_read":
            if not result.get("ok"):
                return action("finish", status="uncertain")
            records = result["tickets"]
            if len({r["request_id"] for r in records}) != len(records):
                return action("finish", status="uncertain")
            for record in records:
                self.known[record["request_id"]] = record["ticket_id"]
            missing = [item for item in self.pending["arguments"]["items"]
                       if item["request_id"] not in self.known]
            if missing and (self.safe_read_at is None or observation["clock"] < self.safe_read_at):
                if self.safe_read_at is None:
                    return action("finish", status="uncertain")
                self.phase = "query"
                # A wait call and its following query each advance one tick.
                remaining = self.safe_read_at - observation["clock"]
                ticks = max(1, remaining - 2)
                if remaining > 1:
                    if ticks > min(60, observation["remaining_wait"]):
                        return action("finish", status="uncertain")
                    return action("wait", ticks=ticks)
            else:
                self.phase = "plan"
        if self.phase == "retry_after_wait":
            self.phase, self.retries = "plan", 0
            return copy.deepcopy(self.pending)
        if self.last is not None and self.last["tool"] == "create_tickets":
            if result.get("ok"):
                for record in result["tickets"]:
                    self.known[record["request_id"]] = record["ticket_id"]
                self.phase, self.pending, self.retries = "plan", None, 0
            elif self.mode == "once":
                return action("finish", status="uncertain")
            elif self.mode == "retry" or self.task["contract"]["idempotency"]:
                if self.retries >= 2:
                    bound = self.task["contract"]["max_commit_delay"]
                    if (self.mode in ("verify", "query_first") and bound and bound <= 60
                            and observation["remaining_wait"] >= bound):
                        self.phase = "retry_after_wait"
                        return action("wait", ticks=bound)
                    return action("finish", status="uncertain")
                self.retries += 1
                return copy.deepcopy(self.pending)
            else:
                bound = self.task["contract"]["max_commit_delay"]
                if self.mode == "query_first":
                    self.safe_read_at = (None if bound is None else observation["clock"] + bound
                                         + self.task["contract"]["visibility_lag"])
                    self.phase = "early_read"
                    return action("list_tickets", request_ids=[g["request_id"] for g in self.task["goals"]])
                if bound is None:
                    return action("finish", status="uncertain")
                delay = bound + self.task["contract"]["visibility_lag"]
                if delay > observation["remaining_wait"] or delay > 60:
                    return action("finish", status="uncertain")
                self.phase = "query"
                if delay:
                    return action("wait", ticks=delay)
        if self.phase == "query":
            self.phase = "read"
            return action("list_tickets", request_ids=[g["request_id"] for g in self.task["goals"]])
        if self.phase == "read":
            if not result.get("ok"):
                return action("finish", status="uncertain")
            records = result["tickets"]
            if len({r["request_id"] for r in records}) != len(records):
                return action("finish", status="uncertain")
            for record in records:
                self.known[record["request_id"]] = record["ticket_id"]
            self.phase = "plan"
        missing = [g for g in self.task["goals"] if g["request_id"] not in self.known]
        if not missing:
            return action("finish", status="completed")
        ready = [g for g in missing if all(parent in self.known for parent in g["parents"])]
        if not ready:
            return action("finish", status="uncertain")
        args = {"items": [{
            "request_id": g["request_id"], "title": g["title"],
            "parent_ids": [self.known[parent] for parent in g["parents"]],
        } for g in ready]}
        if self.mode in ("verify", "query_first") and self.task["contract"]["idempotency"]:
            args["idempotency_key"] = f"batch-{self.batch_number}"
        self.batch_number += 1
        self.pending = action("create_tickets", **args)
        self.retries = 0
        return copy.deepcopy(self.pending)

    def decide(self, messages: list[dict], observation: dict) -> Decision:
        self.last = self._next(observation)
        return Decision(json.dumps(self.last, sort_keys=True))


class ChatPolicy:
    """Standard chat-completions HTTP interface, e.g. a local vLLM endpoint.

    No automatic HTTP retries: infrastructure failures must not become agent
    environment failures or accidentally repeat model requests in cost reports.
    """

    def __init__(self, base_url: str, model: str, api_key_env: str = "RECOVERRL_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 512, timeout: int = 120,
                 tokenizer=None, response_length: int = 6144):
        self.url = base_url.rstrip("/") + "/chat/completions"
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL ending in /v1")
        self.model, self.temperature, self.max_tokens, self.timeout = model, temperature, max_tokens, timeout
        self.api_key = os.environ.get(api_key_env)
        self.tokenizer, self.response_length = tokenizer, response_length
        self.initial_tokens = None

    def reset(self, public_task: dict):
        self.initial_tokens = None

    def decide(self, messages: list[dict], observation: dict) -> Decision:
        max_tokens = self.max_tokens
        if self.tokenizer is not None:
            count = len(self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                           add_generation_prompt=True))
            if self.initial_tokens is None:
                if count > 2048:
                    raise ValueError("Initial evaluation prompt exceeds 2048 tokens")
                self.initial_tokens = count
            remaining = self.response_length - (count - self.initial_tokens)
            if remaining <= 8:
                raise PolicyBudgetExceeded
            max_tokens = min(max_tokens, remaining - 8)
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": max_tokens}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            # Avoid dumping response bodies, which may contain infrastructure secrets.
            raise RuntimeError(f"Model endpoint returned HTTP {error.code}") from None
        content = data["choices"][0]["message"].get("content")
        if not isinstance(content, str):
            raise RuntimeError("Endpoint must return textual JSON actions in message.content")
        usage = data.get("usage") or {}
        return Decision(content, usage.get("completion_tokens"), usage.get("prompt_tokens"))
