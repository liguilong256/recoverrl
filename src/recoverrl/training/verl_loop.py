"""Stateful JSON-action AgentLoop for a pinned verl revision.

The upstream Continuous Token helpers own token boundary alignment. We never
retokenize a generated assistant response or reconstruct rollout logprobs.
One environment is allocated per run(), including concurrent grouped rollouts.
"""

from __future__ import annotations

import json
import time
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from recoverrl.environment import RecoveryEnv
from recoverrl.protocol import messages_for, parse_action, tool_message
from recoverrl.rewards import RewardConfig, score
from recoverrl.specs import TaskSpec


class RecoverAgentLoop(AgentLoopBase):
    def __init__(self, *args, max_tokens_per_turn=512, reward_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        if self.max_tokens_per_turn < 1:
            raise ValueError("max_tokens_per_turn must be positive")
        self.reward_config = RewardConfig(**dict(reward_config or {}))

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        spec = TaskSpec.from_dict(json.loads(kwargs["extra_info"]["spec_json"]))
        env = RecoveryEnv(spec)
        messages = messages_for(spec, env.reset())
        # Rebuild exclusively from the public view. extra_info is trusted runner metadata,
        # not text from the model, and is never interpolated into the prompt.
        runtime_ids = await self.ct_build_initial_tokens(messages)
        mask, logprobs = [], None
        turns, sampled_tokens = 0, 0
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0, "compute_score": 0.0}
        extra_fields = {}
        request_id = uuid4().hex
        response_limit = self.rollout_config.response_length
        termination = "environment"
        while not env.done:
            remaining = response_limit - len(mask)
            if remaining <= 0:
                termination = "token_budget"
                break
            params = dict(sampling_params)
            params.pop("max_new_tokens", None)
            # Continuous Token can insert a boundary before the generated span.
            # Leave a small margin rather than dropping a sampled action afterward.
            if remaining <= 8:
                termination = "token_budget"
                break
            params["max_tokens"] = min(remaining - 8, self.max_tokens_per_turn)
            started = time.perf_counter()
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=runtime_ids, sampling_params=params,
            )
            metrics["generate_sequences"] += time.perf_counter() - started
            if not output.token_ids:
                termination = "empty_generation"
                break
            if len(output.token_ids) > params["max_tokens"]:
                raise RuntimeError("Backend exceeded the requested token budget")
            sampled_tokens += len(output.token_ids)
            turns += 1
            extra_fields.update(getattr(output, "extra_fields", {}) or {})
            merged, mask, logprobs = await self.ct_merge_assistant_token(
                runtime_ids, output.token_ids, mask, logprobs,
                assistant_logprobs=output.log_probs if output.log_probs else None,
            )
            runtime_ids = merged.token_ids
            text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": text})
            started = time.perf_counter()
            observation = env.step(parse_action(text))
            metrics["tool_calls"] += time.perf_counter() - started
            if env.done:
                break
            updated = messages + [tool_message(observation)]
            merged, next_mask, next_logprobs = await self.ct_merge_context_msg(
                messages, updated, runtime_ids, mask, logprobs,
            )
            if len(next_mask) >= response_limit:
                # Keep the sampled action and its real effects; omit unconsumed context.
                termination = "token_budget"
                break
            runtime_ids, mask, logprobs = merged.token_ids, next_mask, next_logprobs
            messages = updated
        if not any(mask):
            # A zero-token sample cannot carry a policy gradient; fail visibly.
            raise RuntimeError("Backend produced no trainable assistant tokens")
        # Terminal reward must land on a generated token, never on a tool observation.
        while mask and mask[-1] == 0:
            mask.pop()
            runtime_ids.pop()
            if logprobs is not None:
                logprobs.pop()
        if len(mask) > response_limit or len(mask) > len(runtime_ids):
            raise RuntimeError("Invalid rollout token/mask alignment")
        if logprobs is not None and len(logprobs) != len(mask):
            raise RuntimeError("Rollout logprob alignment mismatch")
        started = time.perf_counter()
        result = env.finalize()
        reward = score(result, self.reward_config, sampled_tokens)
        metrics["compute_score"] = time.perf_counter() - started
        extra_fields["reward_extra_info"] = {
            **result, "generated_tokens": sampled_tokens, "truncated": int(termination != "environment"),
        }
        return AgentLoopOutput(
            prompt_ids=runtime_ids[:-len(mask)], response_ids=runtime_ids[-len(mask):],
            response_mask=mask, response_logprobs=logprobs, reward_score=reward,
            num_turns=turns, metrics=metrics, extra_fields=extra_fields,
        )
