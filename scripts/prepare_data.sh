#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m recoverrl generate --tasks 100 --output data
for split in train val test_id test_ood; do
  python -m recoverrl export-verl --dataset "data/$split.jsonl" --output "data/$split.parquet"
done
python -m recoverrl export-sft --dataset data/train.jsonl --output data/sft_train.jsonl
