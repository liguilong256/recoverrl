"""Deterministic reference results for the four held-out research splits."""

import argparse
import json
from pathlib import Path

from recoverrl.evaluation import aggregate, run_episode
from recoverrl.policies import ScriptedPolicy
from recoverrl.research import RESEARCH_SPLITS, STUDY_VERSION, make_research_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--output", default="reports/research_baselines.json")
    args = parser.parse_args()
    report = {"study": STUDY_VERSION, "kind": "scripted_reference", "base_tasks_per_split": args.tasks,
              "note": "Synthetic tasks and scripted policies; no trained-model results.", "splits": {}}
    for split in RESEARCH_SPLITS[2:]:
        specs = make_research_suite(split, args.tasks)
        report["splits"][split] = {}
        for policy in ("once", "retry", "verify", "query_first"):
            rows = [run_episode(spec, ScriptedPolicy(policy)) for spec in specs]
            stats = aggregate(rows)
            stats.pop("wall_seconds")
            report["splits"][split][policy] = stats
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
