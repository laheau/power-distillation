"""Supervised fine-tuning on resampled prompt/completion pairs."""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from power_distillation.prompt_formatting import (
    extract_prompt_text,
    format_user_assistant_text,
    format_user_prompt_with_generation_marker,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Weighted SFT training")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=2500)
    parser.add_argument("--warmup_steps", type=int, default=250)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class WeightedSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=4096):
        logger.info("Loading data from %s...", data_path)
        self.items = [json.loads(line) for line in open(data_path)]
        logger.info("Loaded %s training samples", len(self.items))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        prompt_text = extract_prompt_text(item["prompt"])
        full_text = format_user_assistant_text(self.tokenizer, prompt_text, item["completion"])
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prompt_for_mask = format_user_prompt_with_generation_marker(self.tokenizer, prompt_text)
        prompt_ids = self.tokenizer(prompt_for_mask, return_tensors="pt")["input_ids"].squeeze(0)
        labels[: len(prompt_ids)] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "weight": torch.tensor(item.get("weight", 1.0), dtype=torch.float32),
        }


def collate_fn(batch, pad_token_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(bsz, max_len, dtype=torch.long)
    weights = torch.zeros(bsz, dtype=torch.float32)
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1
        weights[i] = item["weight"]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "weights": weights}


def get_lr(step, warmup_steps, max_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        logger.info("World size: %s | Device: %s", world_size, device)
        logger.info("Loading model: %s", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    dataset = WeightedSFTDataset(args.data, tokenizer, args.max_length)
    pad_id = tokenizer.pad_token_id
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = args.batch_size * args.grad_accum * world_size
    total_opt_steps = args.max_steps // args.grad_accum
    step = 0
    window_loss = 0.0
    window_opt_steps = 0
    epoch = 0
    optimizer.zero_grad()
    t0 = time.time()
    if is_main:
        logger.info("Training for %s optimizer steps (effective batch=%s)", total_opt_steps, effective_batch)

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dataloader:
            if step >= args.max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            weights = batch["weights"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :].contiguous()
                targets = labels[:, 1:].contiguous()
                loss_per_token = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(logits.size(0), -1)
                mask = (targets != -100).float()
                tokens_per_sample = mask.sum(dim=-1).clamp(min=1)
                loss_per_sample = (loss_per_token * mask).sum(dim=-1) / tokens_per_sample
                loss = (weights * loss_per_sample).sum() / weights.sum()
                loss = loss / args.grad_accum
            loss.backward()
            window_loss += loss.item()
            if (step + 1) % args.grad_accum == 0:
                opt_step = (step + 1) // args.grad_accum
                lr = get_lr(opt_step, args.warmup_steps, total_opt_steps, args.lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                window_opt_steps += 1
                if is_main and opt_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    avg_loss = window_loss / max(window_opt_steps, 1)
                    logger.info(
                        "Step %s/%s | Loss: %.4f | LR: %.2e | Time: %.0fs",
                        opt_step,
                        total_opt_steps,
                        avg_loss,
                        lr,
                        elapsed,
                    )
                    window_loss = 0.0
                    window_opt_steps = 0
                if is_main and opt_step > 0 and opt_step % args.save_every == 0:
                    save_path = out_dir / f"step_{opt_step}"
                    raw_model = model.module if hasattr(model, "module") else model
                    raw_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
            step += 1
        epoch += 1

    if is_main:
        save_path = out_dir / "final"
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info("Saved final checkpoint to %s", save_path)
    cleanup_distributed()


if __name__ == "__main__":
    main()
