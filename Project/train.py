"""Training script for the VLM project.

Usage:
    python train.py --dataset_local_path /shared/datasets/the_cauldron/ai2d
    python train.py --dataset_type flickr \
        --dataset_local_path /shared/datasets/flickr30k
    python train.py --batch_size 1 --max_steps 100  # quick smoke test

The PROVIDED sections handle:
  * argument parsing and config override
  * data loading (CauldronDataset / FlickrDataset + DataLoader)
  * model construction
  * optimizer setup (3 parameter groups with different learning rates)
  * cosine LR schedule with warmup
  * evaluation loop
  * checkpoint saving

The STUDENT SECTION (clearly marked below) is the inner training loop body.
"""

import argparse
import json
import math
import os
import time
from dataclasses import fields

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.config import VLMConfig, TrainConfig
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from data.collator import VQACollator


# ── Cosine LR schedule with linear warmup ────────────────────────────────────
def get_lr(step: int, max_lr: float, max_steps: int) -> float:
    """Return the learning rate for a given step.

    Phase 1: linear ramp from 0 → max_lr over the first 3% of steps.
    Phase 2: cosine decay from max_lr → max_lr/10 over the remaining steps.
    """
    min_lr = max_lr * 0.1
    warmup_steps = max(1, int(max_steps * 0.03))
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (
        1.0 + math.cos(math.pi * decay)
    )


# ── Data loading (PROVIDED) ───────────────────────────────────────────────────
def get_dataloaders(train_cfg: TrainConfig, vlm_cfg: VLMConfig):
    from datasets import load_from_disk, concatenate_datasets

    if not train_cfg.dataset_local_path:
        raise ValueError(
            "dataset_local_path is required. "
            "Run prepare_datasets.py first, then set --dataset_local_path."
        )

    tokenizer = get_tokenizer(vlm_cfg.lm.tokenizer, vlm_cfg.image_token)
    image_processor = get_image_processor(vlm_cfg.vit.img_size)

    if train_cfg.dataset_type == 'flickr':
        print(f"Loading dataset from disk: {train_cfg.dataset_local_path}")
        raw = load_from_disk(train_cfg.dataset_local_path)
        ds = raw["train"] if "train" in raw else raw

        from data.dataset import FlickrDataset
        train_dataset = FlickrDataset(
            ds, tokenizer, image_processor, vlm_cfg
        )
        val_dataset = FlickrDataset(
            ds, tokenizer, image_processor, vlm_cfg
        )
    else:
        # Load and concatenate all cauldron subsets
        splits = []
        base_path = train_cfg.dataset_local_path
        for subset in train_cfg.dataset_subsets:
            subset_path = os.path.join(base_path, subset)
            if not os.path.exists(subset_path):
                print(f"  [skip] {subset} not found at {subset_path}")
                continue
            print(f"  Loading {subset}...")
            raw = load_from_disk(subset_path)
            ds = raw["train"] if "train" in raw else raw
            splits.append(ds)

        if not splits:
            raise ValueError(
                f"No cauldron subsets found under {base_path}/. "
                "Run prepare_datasets.py first."
            )

        ds = concatenate_datasets(splits)

        print(f"Concatenated {len(splits)} subsets → {len(ds)} samples")

        from data.dataset import CauldronDataset
        train_dataset = CauldronDataset(
            ds, tokenizer, image_processor, vlm_cfg
        )
        val_dataset = CauldronDataset(
            ds, tokenizer, image_processor, vlm_cfg
        )

    collator = VQACollator(tokenizer, max_length=train_cfg.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=2,  
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
    )
    return train_loader, val_loader


# ── Main training function ────────────────────────────────────────────────────
def train(train_cfg: TrainConfig, vlm_cfg: VLMConfig):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.backends.mps.enable_fallback_to_cpu = True
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VisionLanguageModel(
        vlm_cfg, load_backbone=vlm_cfg.load_backbone_weights
    )
    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # ── Optimizer — three learning-rate groups ────────────────────────────────
    # Why three groups?
    #   * The modality projector (MP) is randomly initialised → high LR
    #   * The ViT and LM are pretrained → low LR to preserve knowledge
    param_groups = [
        {
            "params": list(model.MP.parameters()),
            "lr": train_cfg.lr_mp,
            "name": "MP",
        },
        {
            "params": list(model.vision_encoder.parameters()),
            "lr": train_cfg.lr_vit,
            "name": "ViT",
        },
        {
            "params": list(model.decoder.parameters()),
            "lr": train_cfg.lr_lm,
            "name": "LM",
        },
    ]
    # max_lrs: the initial (maximum) LR per group, used inside get_lr().
    # Students reference this in TODO 5.
    max_lrs = [  # noqa: F841
        train_cfg.lr_mp, train_cfg.lr_vit, train_cfg.lr_lm
    ]
    optimizer = optim.AdamW(param_groups)
    # all_params: flat list of parameters for gradient clipping.
    # Students reference this in TODO 5.
    all_params = [  # noqa: F841
        p for g in optimizer.param_groups for p in g["params"]
    ]

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg)
    iter_train = iter(train_loader)

    # ── AMP context ───────────────────────────────────────────────────────────
    autocast_dtype = (
        torch.bfloat16 if device.type in ("cuda", "cpu") else torch.float16
    )
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=autocast_dtype
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # ── Training state ────────────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")
    best_mmstar_acc = -1.0
    batch_loss = 0.0   # set by the student section each micro-step
    optimizer.zero_grad()

    print(
        f"Training for {train_cfg.max_steps} optimiser steps "
        f"(gradient_accumulation={train_cfg.gradient_accumulation_steps})"
    )
    t0 = time.time()

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    accum_step = 0   # counts micro-steps within one accumulation cycle
    while global_step < train_cfg.max_steps:
        model.train()

        # ── Get next batch (skip None batches from the collator) ──────────────
        batch = None
        while batch is None:
            try:
                batch = next(iter_train)
            except StopIteration:
                iter_train = iter(train_loader)
                batch = next(iter_train)

        is_update_step = (
            (accum_step + 1) % train_cfg.gradient_accumulation_steps == 0
        )

        # ══════════════════════════════════════════════════════════════════════
        # STUDENT SECTION — implement the training step
        
        # TODO 1 — Move tensors to device:
        input_ids      = batch["input_ids"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        
        # TODO 2 — Forward pass (inside autocast_ctx for mixed precision):
        with autocast_ctx:
            _, loss = model(
            input_ids, pixel_values, attention_mask, labels
            )
        
        # TODO 3 — Scale loss for gradient accumulation:
        loss = loss / train_cfg.gradient_accumulation_steps
        
        # TODO 4 — Backward pass:
        loss.backward()
        
        # TODO 5 — Optimiser step (only on update steps):
        if is_update_step:
            torch.nn.utils.clip_grad_norm_(
                all_params, train_cfg.max_grad_norm
            )
            for g, max_lr in zip(optimizer.param_groups, max_lrs):
                g["lr"] = get_lr(
                global_step, max_lr, train_cfg.max_steps
                            )
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        # TODO 6 — Store the unscaled loss for logging:
        batch_loss = (
            loss.item() * train_cfg.gradient_accumulation_steps
            )
        #
        # TODO 1 — Move all four batch tensors (input_ids, pixel_values,
        #           attention_mask, labels) to the training device.
        #
        # TODO 2 — Run the forward pass inside autocast_ctx for mixed-precision
        #           training. The model returns (logits, loss); discard logits.
        #
        # TODO 3 — Divide the loss by gradient_accumulation_steps before
        #           backpropagating, so gradients accumulate correctly across
        #           micro-steps.
        #
        # TODO 4 — Call backwardmodel on the scaled loss to accumulate gradients.
        #
        # TODO 5 — On update steps only (is_update_step):
        #           Clip gradients with torch.nn.utils.clip_grad_norm_ using
        #           all_params and max_grad_norm. Update each param group's
        #           "lr" by calling get_lr with global_step, the group's
        #           corresponding entry from max_lrs, and max_steps. Then step
        #           the optimizer, zero its gradients, and increment global_step.
        #
        # TODO 6 — Record the unscaled loss in batch_loss for logging by
        #           reversing the accumulation scaling on the detached scalar
        #           (use .item() to detach from the graph).
        # ══════════════════════════════════════════════════════════════════════

        accum_step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if is_update_step and global_step % train_cfg.log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"step {global_step:5d} | loss {batch_loss:.4f}"
                f" | {elapsed:.1f}s"
            )

        # ── Evaluation ────────────────────────────────────────────────────────
        if is_update_step and global_step % train_cfg.eval_interval == 0:
            model.eval()
            val_losses = []
            val_iter = iter(val_loader)
            n_val = min(64, train_cfg.val_size // train_cfg.batch_size)
            for _ in range(n_val):
                vbatch = next(val_iter, None)
                if vbatch is None:
                    break
                with torch.no_grad(), autocast_ctx:
                    _, vloss = model(
                        vbatch["input_ids"].to(device),
                        vbatch["pixel_values"].to(device),
                        vbatch["attention_mask"].to(device),
                        vbatch["labels"].to(device),
                    )
                if vloss is not None:
                    val_losses.append(vloss.item())

            avg_val = (
                sum(val_losses) / len(val_losses)
                if val_losses else float("nan")
            )
            print(f"step {global_step:5d} | val_loss {avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                ckpt = os.path.join(
                    train_cfg.checkpoint_dir, f"best_step{global_step}"
                )
                model.save_pretrained(ckpt)
                print(f"  → new best checkpoint saved to {ckpt}")

            if (
                train_cfg.mmstar_val_path
                and train_cfg.mmstar_eval_interval > 0
                and global_step % train_cfg.mmstar_eval_interval == 0
            ):
                from datasets import load_from_disk

                from eval_mmstar import evaluate_mmstar

                tokenizer = get_tokenizer(vlm_cfg.lm.tokenizer, vlm_cfg.image_token)
                image_processor = get_image_processor(vlm_cfg.vit.img_size)
                raw_mmstar = load_from_disk(train_cfg.mmstar_val_path)
                mmstar_val = raw_mmstar["val"] if "val" in raw_mmstar else raw_mmstar
                mmstar_metrics = evaluate_mmstar(
                    model=model,
                    dataset=mmstar_val,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    device=device,
                    limit=train_cfg.mmstar_eval_limit,
                    show_progress=False,
                )
                mmstar_acc = mmstar_metrics["accuracy"]
                print(
                    f"step {global_step:5d} | mmstar_val_acc "
                    f"{mmstar_acc:.4f}"
                )

                os.makedirs(train_cfg.mmstar_output_dir, exist_ok=True)
                mmstar_path = os.path.join(
                    train_cfg.mmstar_output_dir,
                    f"mmstar_step{global_step}.json",
                )
                with open(mmstar_path, "w") as f:
                    json.dump(
                        {
                            "global_step": global_step,
                            "checkpoint_dir": train_cfg.checkpoint_dir,
                            "mmstar_val_path": train_cfg.mmstar_val_path,
                            "metrics": mmstar_metrics,
                        },
                        f,
                        indent=2,
                    )

                if mmstar_acc > (best_mmstar_acc + 0.5):
                    best_mmstar_acc = mmstar_acc
                    ckpt = os.path.join(
                        train_cfg.checkpoint_dir,
                        f"best_mmstar_step{global_step}",
                    )
                    model.save_pretrained(ckpt)
                    print(f"  → new best MMStar checkpoint saved to {ckpt}")

            model.train()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the VLM project model"
    )
    for f in fields(TrainConfig):
        if f.type in (int, float, bool, str):
            parser.add_argument(
                f"--{f.name}",
                type=f.type,
                default=None,
                help=f"TrainConfig.{f.name} (default: {f.default})",
            )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()
    for f in fields(TrainConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(train_cfg, f.name, val)
    train(train_cfg, vlm_cfg)
