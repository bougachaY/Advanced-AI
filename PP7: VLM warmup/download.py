"""Pre-download Hugging Face assets needed by the PP7 warmup."""

import argparse

from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from data.processors import get_image_processor, get_tokenizer
from models.config import VLMConfig


def parse_args():
    """Parse command-line arguments for cache warmup.

    Returns:
        An `argparse.Namespace` describing which models and datasets to cache.
    """
    parser = argparse.ArgumentParser(description="Warm up PP7 caches for offline execution")
    parser.add_argument("--dataset-path", type=str, default="AnyModal/flickr30k")
    parser.add_argument("--dataset-name", nargs="*", default=[])
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default=None,
        help="Hugging Face datasets cache directory where the dataset should be downloaded.",
    )
    parser.add_argument("--vit-model", type=str, default="google/siglip2-base-patch16-512")
    parser.add_argument("--lm-model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--tokenizer", type=str, default=None)
    return parser.parse_args()


def cache_model_assets(cfg):
    """Download model, tokenizer, and processor assets used during PP7 runs.

    Args:
        cfg: Model configuration describing the required backbone and tokenizer
            identifiers.
    """
    print(f"Caching vision config: {cfg.vit_model_type}")
    AutoConfig.from_pretrained(cfg.vit_model_type)

    print(f"Caching language config: {cfg.lm_model_type}")
    AutoConfig.from_pretrained(cfg.lm_model_type)

    print(f"Caching tokenizer: {cfg.lm_tokenizer}")
    get_tokenizer(cfg.lm_tokenizer, cfg.image_token)

    print(f"Caching image processor: {cfg.vit_model_type}")
    get_image_processor(cfg.vit_model_type)

    print(f"Caching vision model weights: {cfg.vit_model_type}")
    vision_model = AutoModel.from_pretrained(cfg.vit_model_type)
    del vision_model

    print(f"Caching language model weights: {cfg.lm_model_type}")
    language_model = AutoModelForCausalLM.from_pretrained(cfg.lm_model_type)
    del language_model


def cache_dataset(dataset_path, dataset_names, dataset_cache_dir):
    """Download the requested dataset splits into the local Hugging Face cache.

    Args:
        dataset_path: Hugging Face dataset identifier.
        dataset_names: Optional config names to download.
        dataset_cache_dir: Optional destination cache directory.
    """
    config_names = dataset_names or [None]
    for dataset_name in config_names:
        dataset_label = dataset_path if dataset_name is None else f"{dataset_path}/{dataset_name}"
        print(f"Caching dataset split: {dataset_label}")
        load_dataset(
            dataset_path,
            dataset_name,
            split="train",
            cache_dir=dataset_cache_dir,
        )


def main():
    """Warm the caches required to run PP7 on an offline machine later on.

    Returns:
        `None`. The function is executed for its side effects and console output.
    """
    args = parse_args()

    cfg = VLMConfig(
        vit_model_type=args.vit_model,
        lm_model_type=args.lm_model,
        lm_tokenizer=args.tokenizer or args.lm_model,
    )

    cache_model_assets(cfg)
    cache_dataset(
        dataset_path=args.dataset_path,
        dataset_names=args.dataset_name,
        dataset_cache_dir=args.dataset_cache_dir,
    )

    print("Download step complete. Subsequent PP7 runs should use the local cache.")


if __name__ == "__main__":
    main()
