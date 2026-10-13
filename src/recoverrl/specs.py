"""Task definitions. Only public_view() may enter the model's context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

FAULTS = ("none", "timeout_before", "timeout_after", "partial", "late_commit")
FAMILIES = ("single", "batch", "chain", "fanout", "diamond", "dag")


@dataclass(frozen=True)
class Goal:
    request_id: str
    title: str
    parents: tuple[str, ...] = ()


@dataclass(frozen=True)
class Contract:
    idempotency: bool = False
    visibility_lag: int = 0
    max_commit_delay: Optional[int] = 4

    def __post_init__(self):
        if type(self.idempotency) is not bool:
            raise ValueError("idempotency must be a bool")
        if type(self.visibility_lag) is not int or not 0 <= self.visibility_lag <= 30:
            raise ValueError("visibility_lag must be an integer in [0, 30]")
        if self.max_commit_delay is not None and (
            type(self.max_commit_delay) is not int or not 0 <= self.max_commit_delay <= 60
        ):
            raise ValueError("max_commit_delay must be None or an integer in [0, 60]")


@dataclass(frozen=True)
class FaultPlan:
    mode: str = "none"
    target: str = ""
    commit_delay: int = 4

    def __post_init__(self):
        if self.mode not in FAULTS:
            raise ValueError(f"Unknown fault: {self.mode}")
        if type(self.commit_delay) is not int or self.commit_delay < 1:
            raise ValueError("commit_delay must be a positive integer")


@dataclass(frozen=True)
class TaskSpec:
    goals: tuple[Goal, ...]
    contract: Contract = field(default_factory=Contract)
    fault: FaultPlan = field(default_factory=FaultPlan)
    family: str = "batch"
    seed: int = 0
    split: str = "dev"
    max_steps: int = 16
    max_wait: int = 120
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.family not in FAMILIES:
            raise ValueError("Unsupported schema version or workflow family")
        if not self.goals or len(self.goals) > 10:
            raise ValueError("A task must have 1 to 10 goals")
        ids = [g.request_id for g in self.goals]
        if len(set(ids)) != len(ids):
            raise ValueError("Goal request IDs must be unique")
        visited = set()
        for goal in self.goals:
            if not goal.request_id or not goal.title or not set(goal.parents) <= visited:
                raise ValueError("Goals must be nonempty and topologically ordered")
            visited.add(goal.request_id)
        if self.fault.mode != "none" and self.fault.target not in visited:
            raise ValueError("Fault target must identify a requested logical operation")
        if self.fault.mode == "late_commit" and self.contract.max_commit_delay is not None:
            if self.fault.commit_delay > self.contract.max_commit_delay:
                raise ValueError("Hidden commit delay exceeds the advertised bound")
        if type(self.max_steps) is not int or self.max_steps < 2:
            raise ValueError("max_steps must be an integer >= 2")
        if type(self.max_wait) is not int or self.max_wait < 0:
            raise ValueError("max_wait must be a nonnegative integer")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, data: dict) -> TaskSpec:
        data = dict(data)
        data["goals"] = tuple(
            Goal(g["request_id"], g["title"], tuple(g.get("parents", []))) for g in data["goals"]
        )
        data["contract"] = Contract(**data.get("contract", {}))
        data["fault"] = FaultPlan(**data.get("fault", {}))
        return cls(**data)

    @property
    def task_id(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:20]

    def public_view(self) -> dict:
        # Exclude even task_id: its hash depends on the hidden fault assignment.
        return {
            "goals": [asdict(goal) for goal in self.goals],
            "contract": asdict(self.contract),
            "max_steps": self.max_steps,
            "max_wait": self.max_wait,
        }
