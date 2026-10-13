from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

from . import __version__
from .data import SPLIT_FAMILIES, export_verl, make_suite, read_specs, write_jsonl
from .evaluation import aggregate, paired_bootstrap, run_episode
from .policies import ChatPolicy, ScriptedPolicy
from .rewards import RewardConfig


def _save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _revision():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "git-unavailable"
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="RecoverRL reproducible agent-recovery experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Create paired task/fault/contract JSONL suites")
    generate.add_argument("--output", default="data")
    generate.add_argument("--tasks", type=int, default=20, help="Base tasks per split; each expands to 20 episodes")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--splits", nargs="+", choices=SPLIT_FAMILIES, default=list(SPLIT_FAMILIES)[:-1])
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--dataset")
    evaluate.add_argument("--tasks", type=int, default=20)
    evaluate.add_argument("--policies", nargs="+", choices=["once", "retry", "verify", "query_first", "chat"],
                          default=["once", "retry", "verify"])
    evaluate.add_argument("--output", default="runs/dev")
    evaluate.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    evaluate.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--max-tokens", type=int, default=512)
    evaluate.add_argument("--tokenizer", help="Local tokenizer path; enables a shared response/context budget")
    evaluate.add_argument("--response-length", type=int, default=6144)
    evaluate.add_argument("--reward", choices=["completion", "constrained"], default="constrained")
    compare = sub.add_parser("compare")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--metric", default="exact_success")
    export = sub.add_parser("export-verl")
    export.add_argument("--dataset", required=True)
    export.add_argument("--output", required=True)
    sft = sub.add_parser("export-sft")
    sft.add_argument("--dataset", required=True)
    sft.add_argument("--output", required=True)
    sft.add_argument("--policy", choices=["verify", "query_first"], default="verify")
    args = parser.parse_args(argv)
    if args.command == "generate":
        for split in args.splits:
            specs = make_suite(split, args.tasks, args.seed)
            path = Path(args.output) / f"{split}.jsonl"
            write_jsonl(path, [spec.to_dict() for spec in specs])
            print(f"{path}: {args.tasks} base tasks, {len(specs)} episodes")
    elif args.command == "evaluate":
        if args.max_tokens < 1 or args.response_length < 1:
            parser.error("Token limits must be positive")
        tokenizer = None
        if "chat" in args.policies and args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
        specs = read_specs(args.dataset) if args.dataset else make_suite("dev", args.tasks)
        if not specs or len({s.task_id for s in specs}) != len(specs):
            parser.error("Dataset must contain nonempty, unique task specs")
        output = Path(args.output)
        # Do not silently overwrite experiment artifacts.
        if output.exists() and any(output.iterdir()):
            parser.error("Output directory is not empty; choose a new experiment directory")
        output.mkdir(parents=True, exist_ok=True)
        _save_json(output / "manifest.json", {
            "version": __version__, "git_revision": _revision(), "python": platform.python_version(),
            "created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
            "task_ids": [spec.task_id for spec in specs], "status": "running",
        })
        summary = {}
        for name in args.policies:
            policy = (ChatPolicy(args.base_url, args.model, temperature=args.temperature,
                                 max_tokens=args.max_tokens, tokenizer=tokenizer,
                                 response_length=args.response_length) if name == "chat" else ScriptedPolicy(name))
            rows = []
            # Stream each trace so completed work remains inspectable on endpoint failure.
            with (output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for spec in specs:
                    row = run_episode(spec, policy, RewardConfig(mode=args.reward))
                    row["policy"] = name
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
            by_fault = {mode: aggregate([r for r in rows if r["spec"]["fault"]["mode"] == mode])
                        for mode in sorted({r["spec"]["fault"]["mode"] for r in rows})}
            by_family = {family: aggregate([r for r in rows if r["spec"]["family"] == family])
                         for family in sorted({r["spec"]["family"] for r in rows})}
            by_keys = {str(keys).lower(): aggregate([r for r in rows if r["spec"]["contract"]["idempotency"] == keys])
                       for keys in sorted({r["spec"]["contract"]["idempotency"] for r in rows})}
            summary[name] = {"overall": aggregate(rows), "by_fault": by_fault,
                             "by_family": by_family, "by_idempotency": by_keys}
            print(f"{name}: {json.dumps(summary[name]['overall'], sort_keys=True)}")
        _save_json(output / "summary.json", summary)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        _save_json(output / "manifest.json", manifest)
    elif args.command == "compare":
        print(json.dumps(paired_bootstrap(_load_rows(args.left), _load_rows(args.right), args.metric), indent=2))
    elif args.command == "export-verl":
        export_verl(read_specs(args.dataset), args.output)
        print(args.output)
    else:
        specs = read_specs(args.dataset)
        if not specs or any(spec.split != "train" for spec in specs):
            parser.error("SFT export accepts training tasks only")
        rows = []
        for spec in specs:
            episode = run_episode(spec, ScriptedPolicy(args.policy))
            if episode["metrics"]["exact_success"]:
                rows.append({"messages": episode["messages"], "source_task_id": spec.task_id})
        write_jsonl(args.output, rows)
        print(f"{args.output}: {len(rows)} successful scripted training demonstrations; not model trajectories")


if __name__ == "__main__":
    main()
