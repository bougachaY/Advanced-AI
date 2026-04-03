"""Training script for the PP7 warmup vision-language model."""

import argparse
import itertools
import os
from contextlib import nullcontext
from dataclasses import asdict

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig
import matplotlib.pyplot as plt

from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.config import TrainConfig, VLMConfig
from models.vision_language import VisionLanguageModel


def parse_args():
    """Parse command-line arguments for a PP7 training run.

    Returns:
        An `argparse.Namespace` containing dataset, optimization, and model
        settings for the training script.
    """
    parser = argparse.ArgumentParser(description="Warmup VLM training script")
    parser.add_argument("--dataset-path", type=str, default="AnyModal/flickr30k")
    parser.add_argument("--dataset-name", nargs="*", default=[])
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default=None,
        help="Hugging Face datasets cache directory containing the pre-downloaded dataset.",
    )
    parser.add_argument("--train-samples", type=int, default=2560)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr-projector", type=float, default=1e-3)
    parser.add_argument("--lr-vision", type=float, default=0.0)
    parser.add_argument("--lr-language", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--vit-model", type=str, default="google/siglip2-base-patch16-512"
    )
    parser.add_argument(
        "--lm-model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--output-name", type=str, default="projector.pt")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def get_device():
    """Select the best available local accelerator.

    Returns:
        A `torch.device` pointing to CUDA, MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_training_dataset(
    dataset_path,
    dataset_names,
    total_samples,
    dataset_cache_dir=None,
):
    """Load and optionally concatenate one or more training dataset configs.

    Args:
        dataset_path: Hugging Face dataset identifier.
        dataset_names: Optional list of config names to load and concatenate.
        total_samples: Maximum number of samples to keep after shuffling.
        dataset_cache_dir: Optional datasets cache directory.

    Returns:
        A shuffled Hugging Face dataset ready to be split.
    """
    datasets = []
    config_names = dataset_names or [None]
    for dataset_name in config_names:
        datasets.append(
            load_dataset(
                dataset_path,
                dataset_name,
                split="train",
                cache_dir=dataset_cache_dir,
            )
        )

    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    dataset = dataset.shuffle(seed=0)
    if total_samples is not None:
        dataset = dataset.select(range(min(total_samples, len(dataset))))

    return dataset


def split_dataset(dataset, train_samples, val_samples, split_seed):
    """Split a dataset into train/validation subsets with deterministic sampling.

    Args:
        dataset: Dataset to split.
        train_samples: Number of training samples to keep, or `None` to keep all.
        val_samples: Number of validation samples to reserve.
        split_seed: Random seed for the split operation.

    Returns:
        A tuple `(train_dataset, val_dataset)` where `val_dataset` may be `None`.

    Raises:
        ValueError: If the requested split sizes are invalid.
    """
    if val_samples < 0:
        raise ValueError("`val_samples` must be non-negative.")

    if val_samples == 0:
        if train_samples is not None and train_samples < len(dataset):
            dataset = dataset.select(range(train_samples))
        return dataset, None

    if len(dataset) <= val_samples:
        raise ValueError("Need more samples than `val_samples` to build a train split.")

    split = dataset.train_test_split(test_size=val_samples, seed=split_seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    if train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(train_samples, len(train_dataset)))
        )

    return train_dataset, val_dataset


def infer_num_image_tokens(vit_model_type):
    """Infer how many patch tokens the vision backbone emits per image.

    Args:
        vit_model_type: Hugging Face identifier for the vision backbone.

    Returns:
        Number of patch tokens produced for one image.
    """
    config = AutoConfig.from_pretrained(vit_model_type)
    vision_config = config.vision_config if hasattr(config, "vision_config") else config
    grid_size = vision_config.image_size // vision_config.patch_size
    return grid_size**2


def build_dataloader(dataset, cfg, train_cfg, shuffle):
    """Wrap a raw HF dataset in the PP7 dataset/collator pipeline.

    Args:
        dataset: Raw dataset split.
        cfg: Model configuration.
        train_cfg: Training configuration.
        shuffle: Whether to shuffle samples each epoch.

    Returns:
        A `DataLoader` yielding PP7 multimodal batches.
    """
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.image_token)
    image_processor = get_image_processor(cfg.vit_model_type)
    vqa_dataset = VQADataset(
        dataset=dataset,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=cfg.mp_image_token_length,
    )
    collator = VQACollator(tokenizer=tokenizer, max_length=cfg.lm_max_length)
    return DataLoader(
        vqa_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        num_workers=train_cfg.num_workers,
        collate_fn=collator,
    )


def run_model_on_batch(model, batch, device, device_type):
    """Move a batch to the device, run the model, and return the scalar loss.

    Args:
        model: Vision-language model to execute.
        batch: Batch dictionary produced by the collator.
        device: Target device for tensors.
        device_type: String passed to autocast, e.g. `"cuda"` or `"cpu"`.

    Returns:
        The scalar training loss tensor for the batch.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    pixel_values = batch["pixel_values"].to(device)

    autocast_context = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast_context:
        _, loss = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )

    return loss


@torch.inference_mode()
def evaluate(model, dataloader, device, device_type):
    """Compute the mean validation loss over the provided dataloader.

    Args:
        model: Vision-language model to evaluate.
        dataloader: Validation dataloader.
        device: Target device for tensors.
        device_type: String passed to autocast.

    Returns:
        Mean validation loss as a Python float, or `nan` if no batch was valid.
    """
    model.eval()
    losses = []

    for batch in dataloader:
        if batch["input_ids"].numel() == 0:
            continue

        loss = run_model_on_batch(model, batch, device, device_type)
        if torch.isfinite(loss):
            losses.append(loss.item())

    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def build_optimizer(model, train_cfg):
    """Build optimizer parameter groups and freeze disabled submodules.

    Args:
        model: Vision-language model whose parameters should be optimized.
        train_cfg: Training configuration containing learning rates and weight decay.

    Returns:
        An `AdamW` optimizer over the enabled parameter groups.
    """
    parameter_groups = []

    if train_cfg.lr_projector > 0:
        parameter_groups.append(
            {
                "params": model.modality_projector.parameters(),
                "lr": train_cfg.lr_projector,
            }
        )
    else:
        for param in model.modality_projector.parameters():
            param.requires_grad = False

    if train_cfg.lr_vision > 0:
        parameter_groups.append(
            {"params": model.vision_backbone.parameters(), "lr": train_cfg.lr_vision}
        )
    else:
        for param in model.vision_backbone.parameters():
            param.requires_grad = False

    if train_cfg.lr_language > 0:
        parameter_groups.append(
            {"params": model.language_model.parameters(), "lr": train_cfg.lr_language}
        )
    else:
        for param in model.language_model.parameters():
            param.requires_grad = False

    return torch.optim.AdamW(parameter_groups, weight_decay=train_cfg.weight_decay)


def save_checkpoint(model, cfg, path):
    """Persist the projector weights and model config for later reuse.

    Args:
        model: Trained vision-language model.
        cfg: Configuration saved alongside the weights.
        path: Destination checkpoint path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "projector": model.modality_projector.state_dict(),
        },
        path,
    )


def main():
    """Execute the full PP7 training loop.

    Returns:
        `None`. The function is executed for its side effects: training,
        evaluation, and checkpoint writing.
    """
    args = parse_args()

    vlm_cfg = VLMConfig(
        vit_model_type=args.vit_model,
        lm_model_type=args.lm_model,
        lm_tokenizer=args.tokenizer or args.lm_model,
        lm_max_length=args.max_length,
    )
    vlm_cfg.mp_image_token_length = infer_num_image_tokens(vlm_cfg.vit_model_type)
    train_cfg = TrainConfig(
        dataset_path=args.dataset_path,
        dataset_names=tuple(args.dataset_name),
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_projector=args.lr_projector,
        lr_vision=args.lr_vision,
        lr_language=args.lr_language,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        output_dir=args.output_dir,
        output_name=args.output_name,
        compile=args.compile,
    )

    device = get_device()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    print(f"Using device: {device}")
    total_samples = train_cfg.train_samples + train_cfg.val_samples
    dataset = load_training_dataset(
        dataset_path=train_cfg.dataset_path,
        dataset_names=train_cfg.dataset_names,
        total_samples=total_samples,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
        train_samples=train_cfg.train_samples,
        val_samples=train_cfg.val_samples,
        split_seed=train_cfg.split_seed,
    )
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples: {len(val_dataset)}")

    train_dataloader = build_dataloader(train_dataset, vlm_cfg, train_cfg, shuffle=True)
    val_dataloader = (
        build_dataloader(val_dataset, vlm_cfg, train_cfg, shuffle=False)
        if val_dataset is not None
        else None
    )

    model = VisionLanguageModel(vlm_cfg).to(device)
    if train_cfg.compile:
        model = torch.compile(model)

    optimizer = build_optimizer(model, train_cfg)
    model.train()

    data_iterator = itertools.cycle(train_dataloader)
    progress = tqdm(range(train_cfg.max_steps), desc="Training", ncols=100)

    optimizer.zero_grad()
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    last_val_loss = None
    for step in progress:
        batch = next(data_iterator)
        if batch["input_ids"].numel() == 0:
            continue

        loss = run_model_on_batch(model, batch, device, device_type)
        train_loss_value = loss.item()

        # Gradient accumulation keeps the effective batch size larger than what fits per step.
        loss = loss / train_cfg.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
            if train_cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg.max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(train_loss_value)
        train_window = (
            train_cfg.eval_interval
            if train_cfg.eval_interval > 0
            else len(train_losses)
        )
        avg_train_loss = sum(train_losses[-train_window:]) / min(
            len(train_losses), train_window
        )

        metrics = {"train": f"{avg_train_loss:.4f}"}
        should_eval = (
            val_dataloader is not None
            and train_cfg.eval_interval > 0
            and (
                (step + 1) % train_cfg.eval_interval == 0
                or step + 1 == train_cfg.max_steps
            )
        )
        if should_eval:
            last_val_loss = evaluate(model, val_dataloader, device, device_type)
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss

            val_losses.append(last_val_loss)

        if last_val_loss is not None:
            metrics["val"] = f"{last_val_loss:.4f}"

        progress.set_postfix(metrics)

    if val_dataloader is not None and best_val_loss < float("inf"):
        print(f"Best val loss: {best_val_loss:.4f}")

    checkpoint_path = os.path.join(train_cfg.output_dir, train_cfg.output_name)
    save_checkpoint(model, vlm_cfg, checkpoint_path)
    print(f"Saved projector checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
