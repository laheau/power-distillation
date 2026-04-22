"""Single-sample math evaluation for iterative training rounds."""

import argparse
import json
import logging
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from power_distillation.grading.math_grader import grade_answer
from power_distillation.grading.parse_utils import parse_answer
from power_distillation.prompt_formatting import (
    extract_prompt_text,
    format_general_generation_prompt,
    format_math_generation_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="raw",
        choices=["raw", "math_wrapped", "general_wrapped"],
    )
    parser.add_argument("--use_eos_stop", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = json.load(open(args.dataset))
    logger.info("Loaded %s problems", len(dataset))
    prompts = [
        format_prompt(tokenizer, extract_prompt_text(item), args.prompt_template)
        for item in dataset
    ]
    sampling_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        frequency_penalty=args.frequency_penalty,
    )
    if args.use_eos_stop and tokenizer.eos_token_id is not None:
        sampling_kwargs["stop_token_ids"] = [tokenizer.eos_token_id]
    params = SamplingParams(**sampling_kwargs)
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0

    correct = 0
    parse_none_count = 0
    by_level = {}
    results = []
    for req_out, item in zip(outputs, dataset):
        out = req_out.outputs[0]
        completion = out.text
        pred = parse_answer(completion)
        if pred is None:
            parse_none_count += 1
        gold = parse_answer(item["answer"]) if "\\boxed" in item["answer"] else item["answer"]
        passed = grade_answer(pred, gold)
        correct += int(passed)
        level = item.get("level", "unknown")
        by_level.setdefault(level, [0, 0])
        by_level[level][0] += int(passed)
        by_level[level][1] += 1
        results.append({
            "prompt": item["prompt"],
            "completion": completion,
            "num_tokens": len(out.token_ids),
            "passed": passed,
            "level": level,
            "pred": str(pred),
            "gold": str(gold),
        })

    n = len(dataset)
    parse_none_rate = parse_none_count / n if n else 0.0
    print(f"\n{'=' * 60}")
    print(f"{Path(args.dataset).stem} | {args.label or args.model}")
    print(f"{'=' * 60}")
    print(f"  Overall: {correct}/{n} = {correct / n:.2%}")
    print(f"  Parse None: {parse_none_count}/{n} = {parse_none_rate:.2%}")
    for level in sorted(by_level):
        c, t = by_level[level]
        print(f"  {level}: {c}/{t} = {c / t:.2%}")
    print(f"{'=' * 60}\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json.dump(
        {
            "label": args.label,
            "model": args.model,
            "correct": correct,
            "total": n,
            "pass_rate": correct / n if n else 0.0,
            "by_level": by_level,
            "time": elapsed,
            "parse_none_count": parse_none_count,
            "parse_none_rate": parse_none_rate,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "frequency_penalty": args.frequency_penalty,
            "use_eos_stop": args.use_eos_stop,
        },
        open(out_dir / f"summary_{ts}.json", "w"),
        indent=2,
    )
    with open(out_dir / f"details_{ts}.jsonl", "w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
