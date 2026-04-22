"""Iterative power distillation."""

import argparse
import gc
import json
import logging
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from power_distillation.prompt_formatting import extract_prompt_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_BASE_MODELS = {
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-4B-Instruct-2507",
}
SUPPORTED_DATASET_FILES = {
    "MATH_hendrycks_train.json",
    "MATH_test_L4_L5.json",
}


def clear_cuda_memory():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def compute_ess(log_probs, alpha):
    log_w = (alpha - 1) * np.array(log_probs)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    return 1.0 / (w ** 2).sum()


def finalize_completion(text):
    return text.rstrip()


def find_alpha(all_log_probs_by_prompt, target_ess):
    lo, hi = 1.001, 8.0
    median_ess = lambda a: np.median([compute_ess(lp, a) for lp in all_log_probs_by_prompt])
    if median_ess(lo) < target_ess:
        return lo
    for _ in range(30):
        mid = (lo + hi) / 2
        if median_ess(mid) > target_ess:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_subprocess(cmd, label):
    logger.info("[%s] Running: %s", label, " ".join(cmd[-8:]))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"[{label}] Failed with exit code {result.returncode}")


def get_state(output_dir):
    state_path = output_dir / "state.json"
    if state_path.exists():
        return json.load(open(state_path))
    return {"round": -1, "cumulative_alpha": 1.0, "model_path": None}


def save_state(output_dir, state):
    json.dump(state, open(output_dir / "state.json", "w"), indent=2)


def validate_release_args(args):
    if args.base_model not in SUPPORTED_BASE_MODELS:
        raise ValueError(
            "--base_model must be one of the supported release models: "
            + ", ".join(sorted(SUPPORTED_BASE_MODELS))
        )
    if Path(args.prompts).name != "MATH_hendrycks_train.json":
        raise ValueError(
            "--prompts must point to the bundled training dataset "
            "MATH_hendrycks_train.json."
        )
    if Path(args.eval_dataset).name != "MATH_test_L4_L5.json":
        raise ValueError(
            "--eval_dataset must point to the bundled evaluation dataset "
            "MATH_test_L4_L5.json."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative power distillation")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--prompts_per_round", type=int, default=5000)
    parser.add_argument("--max_gen_tokens", type=int, default=2048)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument(
        "--sample_prompt_template",
        type=str,
        default="raw",
        choices=["raw", "math_wrapped", "general_wrapped"],
    )
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--target_ess_ratio", type=float, default=0.3)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--eval_max_tokens", type=int, default=3072)
    parser.add_argument("--eval_temperature", type=float, default=0.6)
    parser.add_argument("--eval_top_p", type=float, default=0.95)
    parser.add_argument("--eval_frequency_penalty", type=float, default=0.0)
    parser.add_argument(
        "--eval_prompt_template",
        type=str,
        default="math_wrapped",
        choices=["raw", "math_wrapped", "general_wrapped"],
    )
    parser.add_argument("--eval_use_eos_stop", action="store_true")
    parser.add_argument("--max_rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_next_round(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_prompts = json.load(open(args.prompts))
    state = get_state(out_dir)
    round_idx = state["round"] + 1
    cumulative_alpha = state["cumulative_alpha"]
    if round_idx >= args.max_rounds:
        logger.info("Already completed %s rounds. Done.", args.max_rounds)
        return
    current_model = args.base_model if round_idx == 0 else state["model_path"]
    random.seed(args.seed + round_idx)
    np.random.seed(args.seed + round_idx)

    round_dir = out_dir / f"round_{round_idx:03d}"
    round_dir.mkdir(exist_ok=True)
    logger.info("%s", "=" * 60)
    logger.info("ROUND %s | Model: %s", round_idx, current_model)
    logger.info("Cumulative alpha: %.4f", cumulative_alpha)
    logger.info("Sampling config: temp=%.3f template=%s", args.sample_temperature, args.sample_prompt_template)
    logger.info(
        "Eval config: temp=%.3f top_p=%.2f eos_stop=%s template=%s",
        args.eval_temperature,
        args.eval_top_p,
        args.eval_use_eos_stop,
        args.eval_prompt_template,
    )
    logger.info("%s", "=" * 60)

    samples_path = round_dir / "samples.json"
    if not samples_path.exists():
        n_prompts = min(args.prompts_per_round, len(all_prompts))
        round_prompts = random.sample(all_prompts, n_prompts)
        prompts_path = round_dir / "prompts.json"
        json.dump(round_prompts, open(prompts_path, "w"))
        clear_cuda_memory()
        run_subprocess([
            sys.executable, "-m", "power_distillation.generate_candidates",
            "--model", str(current_model),
            "--tokenizer", args.base_model,
            "--prompts", str(prompts_path),
            "--output", str(samples_path),
            "--n_samples", str(args.n_samples),
            "--max_tokens", str(args.max_gen_tokens),
            "--tp", str(args.tp),
            "--temperature", str(args.sample_temperature),
            "--prompt_template", str(args.sample_prompt_template),
            "--seed", str(args.seed + round_idx),
        ], "sample")
    else:
        logger.info("Reusing existing samples")

    train_data_path = round_dir / "train_data.jsonl"
    if not train_data_path.exists():
        raw = json.load(open(samples_path))
        sample_results = raw["results"]
        all_log_probs = [record["log_probs"] for record in sample_results]
        target_ess = args.target_ess_ratio * args.n_samples
        alpha = find_alpha(all_log_probs, target_ess)
        ess_vals = []
        records = []
        for prompt_idx, record in enumerate(sample_results):
            completions = record["completions"]
            valid_indices = np.arange(len(completions), dtype=int)
            if valid_indices.size == 0:
                continue
            log_probs = np.array(record["log_probs"])[valid_indices]
            log_w = (alpha - 1) * log_probs
            log_w -= log_w.max()
            w = np.exp(log_w)
            w /= w.sum()
            ess_vals.append(1.0 / (w ** 2).sum())
            chosen = np.random.choice(len(valid_indices), p=w)
            completion = finalize_completion(completions[valid_indices[chosen]])
            if completion.strip():
                records.append({
                    "prompt": extract_prompt_text(record["prompt"]),
                    "prompt_idx": prompt_idx,
                    "completion": completion,
                    "weight": 1.0,
                    "sampling_weight": float(w[chosen]),
                })
        with open(train_data_path, "w") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
        round_info = {
            "alpha": alpha,
            "step2_alpha": alpha,
            "ess_mean": float(np.mean(ess_vals)),
            "ess_median": float(np.median(ess_vals)),
            "n_records": len(records),
        }
        json.dump(round_info, open(round_dir / "round_info.json", "w"), indent=2)
        cumulative_alpha *= alpha
        logger.info(
            "Alpha: %.4f | ESS: mean=%.1f median=%.1f | Records: %s | Cum alpha: %.4f",
            alpha,
            np.mean(ess_vals),
            np.median(ess_vals),
            len(records),
            cumulative_alpha,
        )
    else:
        logger.info("Reusing existing training data")
        round_info = json.load(open(round_dir / "round_info.json"))
        cumulative_alpha *= round_info["alpha"]

    model_dir = round_dir / "model"
    if not model_dir.exists():
        n_records = round_info["n_records"]
        n_gpus = args.tp
        eff_grad_accum = max(math.ceil(args.effective_batch / (args.batch_size * n_gpus)), 1)
        steps_per_epoch = max(n_records // (args.batch_size * n_gpus), 1)
        max_steps = steps_per_epoch
        total_opt_steps = max_steps // eff_grad_accum
        warmup_steps = max(total_opt_steps // 20, 1)
        logger.info(
            "Training: %s records, 1 epoch, %s opt steps, target batch=%s, actual batch=%s, warmup=%s",
            n_records,
            total_opt_steps,
            args.effective_batch,
            args.batch_size * n_gpus * eff_grad_accum,
            warmup_steps,
        )
        run_subprocess([
            "torchrun", f"--nproc_per_node={n_gpus}",
            "-m", "power_distillation.train_supervised",
            "--model", str(current_model),
            "--data", str(train_data_path),
            "--output_dir", str(round_dir),
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            "--grad_accum", str(eff_grad_accum),
            "--max_steps", str(max_steps),
            "--warmup_steps", str(warmup_steps),
            "--max_length", str(args.max_seq_length),
            "--save_every", "999999",
            "--log_every", "10",
        ], "train")
        final_path = round_dir / "final"
        if final_path.exists() and not model_dir.exists():
            final_path.rename(model_dir)
    else:
        logger.info("Reusing existing model")

    clear_cuda_memory()
    eval_result_path = round_dir / "eval_result.json"
    eval_result = {}
    if not eval_result_path.exists():
        eval_dir = round_dir / "eval"
        eval_dir.mkdir(exist_ok=True)
        run_subprocess([
            sys.executable, "-m", "power_distillation.evaluate_round",
            "--model", str(model_dir),
            "--dataset", args.eval_dataset,
            "--tensor_parallel_size", str(args.tp),
            "--max_tokens", str(args.eval_max_tokens),
            "--temperature", str(args.eval_temperature),
            "--top_p", str(args.eval_top_p),
            "--frequency_penalty", str(args.eval_frequency_penalty),
            "--prompt_template", str(args.eval_prompt_template),
            "--output_dir", str(eval_dir),
            "--label", f"round_{round_idx:03d}",
        ] + (["--use_eos_stop"] if args.eval_use_eos_stop else []), "eval")
        summaries = sorted(eval_dir.glob("summary_*.json"))
        if summaries:
            eval_result = json.load(open(summaries[-1]))
            json.dump(eval_result, open(eval_result_path, "w"), indent=2)
    else:
        eval_result = json.load(open(eval_result_path))

    state = {
        "round": round_idx,
        "cumulative_alpha": cumulative_alpha,
        "model_path": str(model_dir),
        "alpha": round_info["alpha"],
        "step2_alpha": round_info.get("step2_alpha", round_info["alpha"]),
        "sample_temperature": args.sample_temperature,
        "sample_prompt_template": args.sample_prompt_template,
        "eval_temperature": args.eval_temperature,
        "eval_top_p": args.eval_top_p,
        "eval_use_eos_stop": args.eval_use_eos_stop,
        "eval_prompt_template": args.eval_prompt_template,
        "ess_mean": round_info["ess_mean"],
        "ess_median": round_info["ess_median"],
        "n_records": round_info["n_records"],
        "eval_pass_rate": eval_result.get("pass_rate"),
    }
    save_state(out_dir, state)
    json.dump(state, open(round_dir / "state.json", "w"), indent=2)
    logger.info("ROUND %s COMPLETE", round_idx)
    logger.info("  Alpha: %.4f (cum: %.4f)", round_info["alpha"], cumulative_alpha)
    logger.info("  Eval pass@1: %s", eval_result.get("pass_rate", "N/A"))


def run_iterative_round(args):
    validate_release_args(args)
    while True:
        state = get_state(Path(args.output_dir))
        if state["round"] + 1 >= args.max_rounds:
            logger.info("Completed %s rounds. Done.", args.max_rounds)
            break
        run_next_round(args)


def main():
    run_iterative_round(parse_args())


if __name__ == "__main__":
    main()
