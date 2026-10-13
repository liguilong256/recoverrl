"""Collect completed study evaluations without pooling different training seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .evaluation import aggregate, paired_bootstrap
from .study import empty_directory, save_json

METRICS = ("exact_success", "task_success", "duplicate_episode", "tool_calls", "wait_ticks",
           "invalid_actions", "generated_tokens", "prompt_tokens")


def collect(inputs):
    records, traces, dataset_hashes, labels, budgets = [], {}, set(), set(), set()
    for entry in inputs:
        label, separator, location = entry.partition("=")
        if not separator or not label or label in labels:
            raise ValueError("Inputs must be distinct LABEL=EVALUATION_DIRECTORY entries")
        labels.add(label)
        root = Path(location)
        evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
        dataset_hashes.add(evaluation["dataset_manifest_sha256"])
        args = evaluation["arguments"]
        for split in args["splits"]:
            manifest = json.loads((root / split / "manifest.json").read_text(encoding="utf-8"))
            if manifest["status"] != "completed":
                raise ValueError(f"Evaluation is incomplete: {label}/{split}")
            run_args = manifest["arguments"]
            budgets.add((run_args["temperature"], run_args["max_tokens"], run_args["response_length"], run_args["reward"]))
            if "chat" in run_args["policies"] and not run_args.get("tokenizer"):
                raise ValueError("Model evaluations must enable the tokenizer-based context budget")
            for policy in run_args["policies"]:
                path = root / split / f"{policy}.jsonl"
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
                if ([row["task_id"] for row in rows] != manifest["task_ids"]
                        or len(set(manifest["task_ids"])) != len(manifest["task_ids"])):
                    raise ValueError(f"Trace coverage differs from manifest: {path}")
                stats = aggregate(rows)
                series = f"{label}/{policy}"
                records.append({"run": label, "policy": policy, "split": split,
                                "episodes": stats["episodes"], **{key: stats[key] for key in METRICS},
                                "token_budget_rate": sum(r.get("termination") == "token_budget" for r in rows) / len(rows)})
                traces[(series, split)] = [{key: row[key] for key in ("task_id", "cluster_id", "metrics")}
                                           for row in rows]
    if len(dataset_hashes) != 1 or len(budgets) != 1:
        raise ValueError("Reports require the same frozen dataset and evaluation settings")
    return records, traces


def write_report(inputs, output, reference=None, samples=2000):
    records, traces = collect(inputs)
    comparisons = []
    if reference:
        reference_splits = {split for series, split in traces if series == reference}
        if not reference_splits:
            raise ValueError("Reference must identify a LABEL/POLICY series")
        for (series, split), rows in traces.items():
            if series == reference:
                continue
            if split not in reference_splits:
                raise ValueError(f"Reference has no paired results for {split}")
            for metric in ("exact_success", "duplicate_episode", "tool_calls", "wait_ticks"):
                comparison = paired_bootstrap(traces[(reference, split)], rows, metric, samples=samples)
                comparisons.append({"reference": reference, "run": series, "split": split, **comparison})
    output = empty_directory(output)
    save_json(output / "summary.json", records)
    save_json(output / "paired.json", comparisons)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Evaluation results", "", "Each row is one policy/checkpoint on one split; training seeds stay separate.",
             "", "| Run | Split | Episodes | Exact success | Duplicates | Calls | Wait ticks |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in records:
        label = f"{row['run']}/{row['policy']}".replace("|", "/").replace("\n", " ")
        lines.append(f"| {label} | {row['split']} | {row['episodes']} | {row['exact_success']:.1%} | "
                     f"{row['duplicate_episode']:.1%} | {row['tool_calls']:.3f} | {row['wait_ticks']:.3f} |")
    lines += ["", "Costs include failed episodes. Scripted policies have zero model tokens.",
              "Paired deltas and task-cluster intervals are in `paired.json`; they do not measure training-seed uncertainty."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, metavar="LABEL=DIRECTORY")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", help="LABEL/POLICY, for example sft/chat or rules/query_first")
    args = parser.parse_args()
    rows = write_report(args.inputs, args.output, args.reference)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
