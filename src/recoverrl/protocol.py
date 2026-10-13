"""Identical JSON action protocol for scripted, HTTP and verl policies."""

from __future__ import annotations

import json

from .specs import TaskSpec

SYSTEM_PROMPT = """You operate a ticket service. Satisfy the user's goals exactly once, without unrelated writes.
Return exactly one JSON object per turn: {"tool": "name", "arguments": {...}}.
Do not emit Markdown, explanations, multiple objects, or text outside the JSON object.
Available tools:
create_tickets: {"items": [{"request_id": "logical ID", "title": "exact title", "parent_ids": []}],
                 "idempotency_key": "optional key"}
list_tickets: {"request_ids": ["logical ID", ...]}
wait: {"ticks": positive integer, at most 60}
finish: {"status": "completed" | "incomplete" | "uncertain"}
parent_ids are actual ticket IDs from tool receipts or queries, not logical request IDs.
Every create call creates new records unless a supported idempotency key is reused with the exact same batch.
With key support, retries resume a partially committed batch; changing a key or omitting it loses that protection.
Each tool call advances time by one tick; wait advances an additional ticks amount.
Reads can lag by contract.visibility_lag ticks. Writes can commit after a timeout, up to
contract.max_commit_delay ticks after the call; null means no known bound. A timeout does not reveal execution.
Only use idempotency keys when contract.idempotency is true. Do not assume read freshness before the bound.
The next user message after each action is a tool observation, not a new task. Respect the supplied budgets.
Call finish when done, or report uncertainty if completion cannot be established within the budget."""


def messages_for(spec: TaskSpec, observation: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"task": spec.public_view(), "observation": observation},
                                              ensure_ascii=False, sort_keys=True)},
    ]


def parse_action(text: str) -> dict:
    """No JSON repair or markdown stripping: format failures remain observable."""
    def reject_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
        return value if isinstance(value, dict) else {"invalid_output": text[:200]}
    except (ValueError, TypeError):
        return {"invalid_output": str(text)[:200]}


def tool_message(observation: dict) -> dict:
    return {"role": "user", "content": json.dumps({"tool_observation": observation}, sort_keys=True)}
