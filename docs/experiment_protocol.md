# Experiment protocol v0.1

## Freeze before evaluation

Record the Git commit, model source/revision, tokenizer, prompt protocol, data checksums, all task IDs, fault/contract composition, reward coefficients, training seed, generated tokens, environment steps, GPU hours and endpoint sampling parameters. Dataset seeds are distinct from training seeds.

Run `completion` and `constrained` from the same initialization with matched training budgets. If SFT is used, retain its validation and final test scores as a separate baseline. Keep tools, observations and tool documentation identical. Charge wrapper-generated queries and waits to the same cost counters.

## Metrics

- `task_success`: every requested effect is present with correct title and parent dependencies, and there are no unrequested writes. Duplicates can still exist.
- `exact_success`: task success with no excess executions in the append-only ledger.
- `duplicate_episode` / `duplicate_effects`: any duplicate logical request / number beyond one.
- `fault_triggered`: the policy reached the target logical operation; clean episodes report zero.
- `overclaim`: explicit `finish(completed)` although exact success is false after settlement.
- `stopped_incomplete`: explicit incomplete/uncertain ending while the task is incomplete. This is descriptive, not an oracle for whether stopping was justified.
- `tool_calls`: includes reads, writes, waits, invalid actions and finish; all consume steps.
- `wait_ticks`: extra simulated waiting beyond the one tick per call.
- generated/prompt tokens: endpoint usage if present; null if unavailable. Scripted policies use zero.

Successful-task averages must not replace all-episode costs. The simulator currently has no deletion tool, so there is no compensation claim or post-hoc erasure of duplicate effects. All pending writes settle at episode end, including after token/step exhaustion.

## Generalization limits

`test_id` changes instances inside training template families. `test_ood` changes dependency structure while retaining the same service and vocabulary. Neither is a held-out real-world domain. Expand the tool interface and independent task families before claiming broad tool-use transfer. Do not add test tasks to RL or SFT data.

The current private metadata includes one focal fault per episode. It is attached to a logical request, not an absolute call index. Pair policies on the exact same task spec. Unknown commit bounds are supported for manual stress tests but excluded from the default balanced suite. There is no reward for guessing the unobservable commit state.

The `partial` fault commits the first `max(1, floor(batch_size / 2))` items. For a singleton request it is therefore equivalent to commit-with-lost-acknowledgement, rather than a genuinely partial write. Keep this degeneracy visible when interpreting per-fault results.

## Required comparisons

Base model; domain-aware rules (labeled as such); recovery SFT; completion-reward GRPO; constrained-reward GRPO. Then isolate duplicate penalty, cost penalty and clean-task mixing. Any future generic recovery wrapper should be compared both with and without RL, under matched information.

Task-cluster bootstrap accounts for repeated fault variants of the same task. It does not account for variability across training runs; report at least three training seeds separately. A degenerate confidence interval on deterministic script scores reflects the artificial suite, not confidence about deployment reliability.

## Research gate

Do not call a better score than blind retry a method contribution. The strong rule already achieves 100% exact success on this small bounded suite. First ask whether a learned policy transfers or trades cost against success better on a broader, held-out distribution. If rules remain sufficient, report that outcome and position the release as experimental infrastructure.
