"""Generate N candidate completions per prompt using vLLM."""

import argparse
import json
import logging
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from power_distillation.prompt_formatting import (
    extract_prompt_text,
    format_general_generation_prompt,
    format_math_generation_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_cumulative_logprob(completion):
    cum_lp = completion.cumulative_logprob
    if cum_lp is None and completion.logprobs:
        cum_lp = sum(
            lp_dict[token_id].logprob
            for token_id, lp_dict in zip(completion.token_ids, completion.logprobs)
            if token_id in lp_dict
        )
    if cum_lp is None:
        cum_lp = float("-inf")
    return float(cum_lp)


def format_prompt(tokenizer, prompt_text, prompt_template):
    if prompt_template == "raw":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if prompt_template == "math_wrapped":
        return format_math_generation_prompt(tokenizer, prompt_text)
    if prompt_template == "general_wrapped":
        return format_general_generation_prompt(tokenizer, prompt_text)
    raise ValueError(f"Unknown prompt template: {prompt_template}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="raw",
        choices=["raw", "math_wrapped", "general_wrapped"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    with open(args.prompts) as handle:
        prompts = json.load(handle)

    prompt_texts = [extract_prompt_text(item) for item in prompts]
    formatted = [format_prompt(tokenizer, prompt_text, args.prompt_template) for prompt_text in prompt_texts]
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    params = SamplingParams(
        max_tokens=args.max_tokens,
        n=args.n_samples,
        temperature=args.temperature,
        top_p=1.0,
        seed=args.seed,
        stop_token_ids=stop_ids,
        logprobs=0,
    )
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16")

    logger.info("Generating %s x %s completions", len(formatted), args.n_samples)
    t0 = time.time()
    outputs = llm.generate(formatted, params)
    elapsed = time.time() - t0
    logger.info("Generation done in %.1fs", elapsed)

    results = []
    for idx, req_out in enumerate(outputs):
        log_probs = []
        completions = []
        for comp in req_out.outputs:
            log_probs.append(extract_cumulative_logprob(comp))
            completions.append(comp.text)
        results.append({"prompt": prompt_texts[idx], "log_probs": log_probs, "completions": completions})

    with open(args.output, "w") as handle:
        json.dump({"results": results, "elapsed": elapsed, "proposal_temperature": args.temperature}, handle)
    logger.info("Saved %s prompt results to %s", len(results), args.output)


if __name__ == "__main__":
    main()
