"""Standalone best-of-n / pass@k math evaluation with token statistics."""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from power_distillation.grading.math_grader import grade_answer
from power_distillation.grading.parse_utils import parse_answer
from power_distillation.prompt_formatting import format_math_generation_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-4B-Instruct-2507",
}
SUPPORTED_DATASET_FILES = {"MATH_test_L4_L5.json"}


def estimate_pass_at_k(num_samples, num_correct, k):
    def estimator(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples = [num_samples] * len(num_correct)
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])


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
    return cum_lp


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM math pass@k evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--n_problems", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--best_of_n", type=int, default=1)
    parser.add_argument("--pass_k", type=int, nargs="*", default=None)
    return parser.parse_args()


def validate_release_args(args):
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(
            "--model must be one of the supported release models: "
            + ", ".join(sorted(SUPPORTED_MODELS))
        )
    if Path(args.dataset).name not in SUPPORTED_DATASET_FILES:
        raise ValueError(
            "--dataset must point to the bundled evaluation dataset "
            "MATH_test_L4_L5.json."
        )


def main():
    args = parse_args()
    validate_release_args(args)
    if args.best_of_n < 1:
        raise ValueError("--best_of_n must be >= 1")
    dataset = json.load(open(args.dataset))
    if args.n_problems > 0:
        dataset = dataset[: args.n_problems]
    logger.info("Loaded %s problems from %s", len(dataset), args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [format_math_generation_prompt(tokenizer, item["prompt"]) for item in dataset]
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    sampling_params = SamplingParams(
        temperature=args.temp,
        n=args.best_of_n,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_ids,
        logprobs=0,
    )
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    outputs = llm.generate(prompts, sampling_params)

    results = []
    correct = 0
    by_level = {}
    num_correct_per_problem = []
    for i, (request_output, item) in enumerate(zip(outputs, dataset)):
        candidates = request_output.outputs
        candidate_logprobs = [float(extract_cumulative_logprob(c)) for c in candidates]
        chosen_index = int(max(range(len(candidates)), key=lambda idx: candidate_logprobs[idx]))
        completion = candidates[chosen_index]
        completion_text = completion.text
        chosen_logprob = candidate_logprobs[chosen_index]
        gold_answer = parse_answer(item["answer"]) if "\\boxed" in item["answer"] else item["answer"]
        candidate_rows = []
        candidate_passes = []
        for cand_idx, cand in enumerate(candidates):
            cand_text = cand.text
            cand_answer = parse_answer(cand_text)
            cand_passed = grade_answer(cand_answer, gold_answer)
            candidate_passes.append(bool(cand_passed))
            candidate_rows.append({
                "candidate_index": cand_idx,
                "completion": cand_text,
                "answer": cand_answer,
                "passed": bool(cand_passed),
                "num_tokens": len(cand.token_ids),
                "cumulative_logprob": candidate_logprobs[cand_idx],
            })
        pred_answer = candidate_rows[chosen_index]["answer"]
        passed = candidate_rows[chosen_index]["passed"]
        correct += int(passed)
        num_correct_per_problem.append(sum(candidate_passes))
        level = item.get("level", "unknown")
        by_level.setdefault(level, [0, 0])
        by_level[level][0] += int(passed)
        by_level[level][1] += 1
        results.append({
            "problem_idx": i,
            "prompt": item["prompt"],
            "gold_answer": gold_answer,
            "completion": completion_text,
            "answer": pred_answer,
            "passed": passed,
            "level": level,
            "num_tokens": len(completion.token_ids),
            "cumulative_logprob": chosen_logprob,
            "chosen_index": chosen_index,
            "best_of_n": args.best_of_n,
            "candidate_logprobs": candidate_logprobs,
            "num_correct_candidates": sum(candidate_passes),
            "candidates": candidate_rows,
        })

    n = len(dataset)
    pass_rate = correct / n if n else 0.0
    pass_k_values = list(range(1, args.best_of_n + 1)) if args.pass_k is None else sorted({
        k for k in args.pass_k if 1 <= k <= args.best_of_n
    })
    pass_at_k = {
        f"pass@{k}": float(estimate_pass_at_k(args.best_of_n, num_correct_per_problem, k).mean())
        for k in pass_k_values
    }
    chosen_num_tokens = [row["num_tokens"] for row in results]
    token_stats = {
        "mean": float(statistics.mean(chosen_num_tokens)) if chosen_num_tokens else 0.0,
        "median": float(statistics.median(chosen_num_tokens)) if chosen_num_tokens else 0.0,
        "min": min(chosen_num_tokens) if chosen_num_tokens else 0,
        "max": max(chosen_num_tokens) if chosen_num_tokens else 0,
    }

    print(f"\n{'=' * 60}")
    print(f"{Path(args.dataset).stem} Results")
    print(f"Model: {args.model} | TP: {args.tensor_parallel_size} |")
    print(f"{'=' * 60}")
    print(f"  Pass@1 (best-of-{args.best_of_n}): {correct}/{n} = {pass_rate:.2%}")
    for key, value in pass_at_k.items():
        print(f"  {key}: {value:.2%}")
    print(
        "  Answer tokens: "
        f"mean={token_stats['mean']:.2f} "
        f"median={token_stats['median']:.0f} "
        f"min={token_stats['min']} "
        f"max={token_stats['max']}"
    )
    for level in sorted(by_level):
        level_correct, level_total = by_level[level]
        print(f"  {level}: {level_correct}/{level_total} = {level_correct / level_total:.2%}")
    print(f"{'=' * 60}\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"samples_{ts}.jsonl", "w") as handle:
        for row in results:
            handle.write(json.dumps(row, default=str) + "\n")
    (out_dir / f"summary_{ts}.json").write_text(json.dumps({
        "config": vars(args),
        "dataset": Path(args.dataset).stem,
        "n_problems": n,
        "best_of_n": args.best_of_n,
        "pass_rate": pass_rate,
        "pass_at_k": pass_at_k,
        "correct": correct,
        "by_level": by_level,
        "answer_num_tokens": token_stats,
    }, indent=2))


if __name__ == "__main__":
    main()
