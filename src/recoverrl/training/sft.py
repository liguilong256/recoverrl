"""Optional cold-start full-parameter SFT on exported scripted demonstrations."""

import argparse
import json
import os
from pathlib import Path

from .sft_data import encode_dialogue


def main():
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sft_train.jsonl")
    parser.add_argument("--model", default="models/qwen3-4b")
    parser.add_argument("--output", default="models/sft")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsdp", action="store_true", help="Qwen3 full sharding; launch with torchrun")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("SFT requires CUDA")
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        parser.error("Use an empty output directory")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        parser.error("Empty SFT dataset")
    encoded = [encode_dialogue(tokenizer, row["messages"], args.max_length) for row in rows]

    class Dialogues(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    def collate(batch):
        length = max(len(row["input_ids"]) for row in batch)
        result = {}
        for key, padding in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.tensor([row[key] + [padding] * (length - len(row[key])) for row in batch])
        return result

    fsdp = {}
    if args.fsdp:
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            parser.error("--fsdp requires torchrun with at least two processes")
        fsdp = {"fsdp": "full_shard auto_wrap",
                "fsdp_config": {"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]}}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               attn_implementation="sdpa", trust_remote_code=False)
    model.config.use_cache = False
    training = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-5, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="epoch", save_total_limit=1,
        report_to="none", seed=args.seed, data_seed=args.seed, **fsdp,
    )
    trainer = Trainer(model=model, args=training, train_dataset=Dialogues(), data_collator=collate,
                      processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)
        (Path(args.output) / "recoverrl_sft.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
