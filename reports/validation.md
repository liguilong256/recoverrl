# Local validation — 2026-10-13

Environment: Windows, Python 3.12.14. No server connection and no model weights were used.

- 37 CPU tests pass, including deterministic fault semantics, snapshot restoration, pending-write settlement, idempotency, split isolation, Parquet roundtrip and a mock HTTP endpoint.
- Fake-backend AgentLoop tests verify independent concurrent episodes, observation masking, generated-token accounting, token-budget termination and inline terminal reward. These do not execute verl's GPU workers or an optimizer.
- Launcher overrides compose successfully against verl commit `3e0d79ea8c38264bcaddc64de1b3465589580320` using Hydra 1.3.2. This validates configuration names, not CUDA execution.
- Actual Qwen3-4B-Instruct-2507 tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554` checked with Transformers 5.12.0: 80 scripted dialogues; maximum initial prompt 588 tokens, maximum full dialogue 1,509 tokens. SFT assistant-only span checks pass. `return_dict=False` is explicit to avoid the newer tokenizer API's changed default return type.
- 100 base development tasks × 20 fault/contract variants × 3 scripted policies = 6,000 CPU episodes completed. Public deterministic aggregates are in `baseline_smoke.json`.
- Python lint and shell syntax checks pass.

The default bounded synthetic suite is solved by the domain-aware verification reference. Its 100% score is a sanity check; it is not a learned-policy result or evidence for a new algorithm.

Pending on Linux/A800: resolving and installing the entire GPU dependency stack, actual model inference, SFT, one GRPO optimizer update, checkpoint export/reload, memory/throughput measurement and all model comparisons. The server runbook starts with these checks.
