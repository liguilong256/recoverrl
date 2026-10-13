"""CPU-only token/mask check using the actual model's tokenizer; no weights loaded."""

import argparse

from recoverrl.data import make_suite, read_specs
from recoverrl.evaluation import run_episode
from recoverrl.policies import ScriptedPolicy
from recoverrl.training.sft_data import encode_dialogue


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-4b")
    parser.add_argument("--dataset", help="Check every episode in this JSONL instead of the small development suite")
    parser.add_argument("--policy", choices=["verify", "query_first"], default="verify")
    parser.add_argument("--response-length", type=int, default=4096)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    lengths, prompt_lengths, response_lengths = [], [], []
    specs = read_specs(args.dataset) if args.dataset else make_suite("dev", 4)
    if not specs or args.response_length < 1:
        parser.error("Need nonempty data and a positive response length")
    for task in specs:
        episode = run_episode(task, ScriptedPolicy(args.policy))
        data = encode_dialogue(tokenizer, episode["messages"], 2048 + args.response_length)
        assert any(label == -100 for label in data["labels"])
        assert any(label != -100 for label in data["labels"])
        lengths.append(len(data["input_ids"]))
        prompt_lengths.append(len(tokenizer.apply_chat_template(episode["messages"][:2], tokenize=True,
                                                                return_dict=False, add_generation_prompt=True)))
        response_lengths.append(len(data["input_ids"]) - prompt_lengths[-1])
    if max(prompt_lengths) > 2048:
        raise ValueError("Initial prompt exceeds the RL launcher's 2048-token limit")
    if max(response_lengths) > args.response_length:
        raise ValueError("A reference dialogue exceeds the chosen response budget")
    print(f"Checked {len(lengths)} dialogues; max prompt={max(prompt_lengths)}, "
          f"max response={max(response_lengths)}, max full={max(lengths)} tokens")


if __name__ == "__main__":
    main()
