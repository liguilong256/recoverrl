"""Assistant-only labels with strict prefix validation instead of guessed spans."""


def encode_dialogue(tokenizer, messages, max_length=8192):
    def encode(dialogue, generation=False):
        return tokenizer.apply_chat_template(dialogue, tokenize=True, return_dict=False,
                                             add_generation_prompt=generation)

    full = encode(messages)
    if len(full) > max_length:
        raise ValueError(f"SFT dialogue length {len(full)} exceeds {max_length}; refusing silent truncation")
    labels = [-100] * len(full)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = encode(messages[:index], generation=True)
        completed = encode(messages[:index + 1])
        if completed[:len(prefix)] != prefix or full[:len(completed)] != completed:
            raise ValueError("Chat template rewrites earlier tokens; cannot safely infer assistant spans")
        labels[len(prefix):len(completed)] = full[len(prefix):len(completed)]
    if not any(label != -100 for label in labels):
        raise ValueError("SFT dialogue contains no trainable assistant tokens")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}
