"""Produce portable, deterministic public results without machine-specific paths."""

import argparse
import json
from pathlib import Path

from recoverrl import __version__
from recoverrl.data import make_suite
from recoverrl.evaluation import aggregate, run_episode
from recoverrl.policies import ScriptedPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--output", default="reports/baseline_smoke.json")
    args = parser.parse_args()
    specs = make_suite("dev", args.tasks)
    report = {
        "version": __version__, "kind": "scripted_environment_sanity_check",
        "base_tasks": args.tasks, "episodes_per_policy": len(specs),
        "scope": "Synthetic deterministic service; no LLM, SFT or RL result.",
        "policies": {},
    }
    for name in ("once", "retry", "verify"):
        rows = [run_episode(spec, ScriptedPolicy(name)) for spec in specs]
        stats = aggregate(rows)
        stats.pop("wall_seconds")
        report["policies"][name] = stats
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
