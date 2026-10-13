"""CPU-only token/mask check using the actual model's tokenizer; no weights loaded."""

import argparse

from recoverrl.data import make_suite
from recoverrl.evaluation import run_episode
from recoverrl.policies import ScriptedPolicy
from recoverrl.training.sft_data import encode_dialogue


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-4b")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    lengths, prompt_lengths = [], []
    for task in make_suite("dev", 4):
        episode = run_episode(task, ScriptedPolicy("verify"))
        data = encode_dialogue(tokenizer, episode["messages"], 8192)
        assert any(label == -100 for label in data["labels"])
        assert any(label != -100 for label in data["labels"])
        lengths.append(len(data["input_ids"]))
        prompt_lengths.append(len(tokenizer.apply_chat_template(episode["messages"][:2], tokenize=True,
                                                                return_dict=False, add_generation_prompt=True)))
    if max(prompt_lengths) > 2048:
        raise ValueError("Initial prompt exceeds the RL launcher's 2048-token limit")
    print(f"Checked {len(lengths)} dialogues; max prompt={max(prompt_lengths)}, max full={max(lengths)} tokens")


if __name__ == "__main__":
    main()
