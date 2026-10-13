"""Download an immutable model revision; do not silently use a moving branch."""

import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/qwen3-4b")
    args = parser.parse_args()
    versions = json.loads((Path(__file__).resolve().parents[1] / "configs/versions.json").read_text())
    snapshot_download(repo_id=versions["model_id"], revision=versions["model_revision"], local_dir=args.output)
    (Path(args.output) / "recoverrl_source.json").write_text(json.dumps(versions, indent=2) + "\n")


if __name__ == "__main__":
    main()
