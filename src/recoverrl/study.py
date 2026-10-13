"""Prepare frozen data and execute an explicit, sequential experiment matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .cli import _revision, main as cli_main
from .data import export_verl, read_specs, write_jsonl
from .evaluation import run_episode
from .policies import ScriptedPolicy
from .research import RESEARCH_SPLITS, STUDY_VERSION, make_research_suite


def save_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def empty_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["study"] != STUDY_VERSION:
        raise ValueError("Unsupported study version")
    if type(config["data_seed"]) is not int:
        raise ValueError("data_seed must be an integer")
    for name in ("train_tasks", "val_tasks", "test_tasks"):
        if type(config[name]) is not int or not 1 <= config[name] <= 9999:
            raise ValueError(f"Invalid {name}")
    if (not config["training_seeds"] or len(set(config["training_seeds"])) != len(config["training_seeds"])
            or any(type(seed) is not int or seed < 0 for seed in config["training_seeds"])):
        raise ValueError("Use distinct nonnegative training seeds")
    for name in ("steps", "batch_size", "group_size", "response_length"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"Invalid {name}")
    if config["batch_size"] % 2 or config["group_size"] < 2 or config["response_length"] < 512:
        raise ValueError("Need an even batch size, group size >= 2 and response length >= 512")
    return config


def prepare(config, output):
    output = empty_directory(output)
    counts = {}
    for split in RESEARCH_SPLITS:
        count = config["train_tasks" if split == "train" else "val_tasks" if split == "val" else "test_tasks"]
        specs = make_research_suite(split, count, config["data_seed"])
        write_jsonl(output / f"{split}.jsonl", (spec.to_dict() for spec in specs))
        if split in ("train", "val"):
            export_verl(specs, output / f"{split}.parquet")
        counts[split] = {"base_tasks": count, "episodes": len(specs)}
    # Use the query-first reference for the shared SFT initialization.
    # Identical successful dialogues from indistinguishable faults appear once.
    demos, seen = [], set()
    train_specs = read_specs(output / "train.jsonl")
    for spec in train_specs:
        row = run_episode(spec, ScriptedPolicy("query_first"))
        if not row["metrics"]["exact_success"]:
            raise RuntimeError(f"Reference failed training task {spec.task_id}; inspect before training")
        key = json.dumps(row["messages"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            demos.append({"messages": row["messages"], "source_task_id": spec.task_id,
                          "source_policy": "query_first"})
    write_jsonl(output / "sft_train.jsonl", demos)
    save_json(output / "manifest.json", {
        "study": STUDY_VERSION, "config": config, "splits": counts, "git_revision": _revision(),
        "sft_demonstrations": len(demos), "status": "completed",
        "files": {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()},
    })
    return counts


def check_data(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["study"] != STUDY_VERSION or manifest["status"] != "completed":
        raise ValueError("Dataset preparation did not complete")
    required = {f"{split}.jsonl" for split in RESEARCH_SPLITS} | {"train.parquet", "val.parquet", "sft_train.jsonl"}
    if set(manifest["files"]) != required:
        raise ValueError("Dataset manifest has missing or unexpected files")
    for filename, expected in manifest["files"].items():
        if digest(directory / filename) != expected:
            raise ValueError(f"Dataset changed after preparation: {filename}")
    return manifest


def make_plan(config, data, output, model):
    data = Path(data).resolve()
    manifest = check_data(data)
    if manifest["config"] != config:
        raise ValueError("Plan config must match the frozen dataset config")
    output = empty_directory(output).resolve()
    model = str(Path(model).resolve())
    sft_model = str(output / "sft")
    common = ["--train", str(data / "train.parquet"), "--val", str(data / "val.parquet"),
              "--batch-size", str(config["batch_size"]), "--group-size", str(config["group_size"]),
              "--response-length", str(config["response_length"]), "--val-max-samples", "-1"]
    jobs = [{"id": "smoke", "stage": "smoke", "requires": [],
             "output": str(output / "smoke"),
             "argv": ["-m", "recoverrl.training.launch", *common, "--model", model, "--steps", "1",
                      "--output", str(output / "smoke")]},
            {"id": "sft", "stage": "sft", "requires": ["smoke"], "output": sft_model,
             "argv": ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                      "-m", "recoverrl.training.sft", "--data", str(data / "sft_train.jsonl"),
                      "--model", model, "--output", sft_model, "--fsdp", "--epochs", "1", "--seed", "42"]}]
    conditions = {
        "completion": ["--reward", "completion"],
        "constrained": ["--reward", "constrained"],
        "no_duplicate": ["--reward", "constrained", "--duplicate-penalty", "0"],
        "no_cost": ["--reward", "constrained", "--call-cost", "0", "--wait-cost", "0"],
    }
    for name, flags in conditions.items():
        for seed in config["training_seeds"]:
            job_id = f"{name}-s{seed}"
            job_output = str(output / job_id)
            jobs.append({"id": job_id, "stage": "main" if name in ("completion", "constrained") else "ablation",
                         "requires": ["sft"], "output": job_output,
                         "argv": ["-m", "recoverrl.training.launch", *common, "--model", sft_model,
                                  "--steps", str(config["steps"]), "--seed", str(seed),
                                  *flags, "--output", job_output]})
    plan = {"study": STUDY_VERSION, "config": config, "data": str(data),
            "dataset_manifest_sha256": digest(data / "manifest.json"), "jobs": jobs}
    save_json(output / "plan.json", plan)
    return plan


def execute(plan_path, stage=None, job_ids=None, dry_run=False):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    check_data(plan["data"])
    if digest(Path(plan["data"]) / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the plan")
    jobs = [job for job in plan["jobs"] if job["stage"] == stage] if stage else [
        job for job in plan["jobs"] if job["id"] in job_ids]
    if not jobs or (job_ids and set(job_ids) != {job["id"] for job in jobs}):
        raise ValueError("No jobs matched or an unknown job was requested")
    state_path = plan_path.with_name("state.json")
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "plan_sha256": digest(plan_path), "jobs": {},
    }
    if state["plan_sha256"] != digest(plan_path):
        raise ValueError("Plan changed after execution started; create a new plan")
    for job in jobs:
        command = [sys.executable, *job["argv"]]
        print(json.dumps({"job": job["id"], "argv": command}))
        if dry_run:
            continue
        previous = state["jobs"].get(job["id"], {})
        if previous.get("status") == "completed":
            if not Path(job["output"]).is_dir() or not any(Path(job["output"]).iterdir()):
                raise ValueError(f"Completed job output is missing: {job['id']}")
            print(f"Already completed: {job['id']}")
            continue
        if previous:
            raise ValueError(f"Job {job['id']} was interrupted or failed; inspect its log and use a new plan")
        for dependency in job["requires"]:
            if state["jobs"].get(dependency, {}).get("status") != "completed":
                raise ValueError(f"Run prerequisite first: {dependency}")
            dependency_output = next(Path(j["output"]) for j in plan["jobs"] if j["id"] == dependency)
            if not dependency_output.is_dir() or not any(dependency_output.iterdir()):
                raise ValueError(f"Prerequisite output is missing: {dependency}")
        job_output = Path(job["output"])
        if job_output.exists() and any(job_output.iterdir()):
            raise ValueError(f"Job output is not empty: {job_output}")
        state["jobs"][job["id"]] = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
        save_json(state_path, state)
        try:
            with plan_path.with_name(f"{job['id']}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                finally:
                    process.stdout.close()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            state["jobs"][job["id"]]["status"] = "failed"
            save_json(state_path, state)
            raise
        state["jobs"][job["id"]].update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(state_path, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", default="configs/study.json")
    prep.add_argument("--output", default="data/recovery-v1")
    plan = sub.add_parser("plan")
    plan.add_argument("--config", default="configs/study.json")
    plan.add_argument("--data", default="data/recovery-v1")
    plan.add_argument("--output", default="runs/recovery-v1")
    plan.add_argument("--model", default="models/qwen3-4b")
    run = sub.add_parser("run")
    run.add_argument("--plan", default="runs/recovery-v1/plan.json")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stage", choices=("smoke", "sft", "main", "ablation"))
    selection.add_argument("--jobs", nargs="+")
    run.add_argument("--dry-run", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data", default="data/recovery-v1")
    evaluate.add_argument("--splits", nargs="+", choices=RESEARCH_SPLITS[1:], default=["val"])
    evaluate.add_argument("--policies", nargs="+", choices=("once", "retry", "verify", "query_first", "chat"),
                          default=["chat"])
    evaluate.add_argument("--model", default="recoverrl-policy")
    evaluate.add_argument("--checkpoint", help="Local checkpoint served by the endpoint; required for chat")
    evaluate.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(load_config(args.config), args.output), indent=2))
    elif args.command == "plan":
        plan = make_plan(load_config(args.config), args.data, args.output, args.model)
        print("\n".join(f"{j['stage']:8s} {j['id']}" for j in plan["jobs"]))
    elif args.command == "run":
        execute(args.plan, args.stage, args.jobs, args.dry_run)
    else:
        dataset = check_data(args.data)
        checkpoint_info = None
        if "chat" in args.policies:
            if not args.checkpoint or not (Path(args.checkpoint) / "config.json").is_file():
                parser.error("Pass --checkpoint pointing to the model directory served by the endpoint")
            checkpoint = Path(args.checkpoint).resolve()
            checkpoint_info = {"path": str(checkpoint), "config_sha256": digest(checkpoint / "config.json"),
                               "weights": {p.name: p.stat().st_size for p in sorted(checkpoint.glob("*.safetensors"))}}
        if len(set(args.splits)) != len(args.splits):
            parser.error("Duplicate evaluation split")
        output = empty_directory(args.output)
        save_json(output / "evaluation.json", {"arguments": vars(args), "checkpoint": checkpoint_info,
                  "git_revision": _revision(),
                  "dataset_manifest_sha256": digest(Path(args.data) / "manifest.json")})
        for split in args.splits:
            tokenizer_args = ["--tokenizer", args.checkpoint] if "chat" in args.policies else []
            cli_main(["evaluate", "--dataset", str(Path(args.data) / f"{split}.jsonl"),
                      "--output", str(output / split), "--model", args.model, "--base-url", args.base_url,
                      *tokenizer_args, "--response-length", str(dataset["config"]["response_length"]),
                      "--temperature", "0", "--max-tokens", "512", "--policies", *args.policies])


if __name__ == "__main__":
    main()
