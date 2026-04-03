"""Generation script for testing a PP7 projector checkpoint interactively."""

import argparse

import torch
from PIL import Image

from data.processors import get_image_string
from models.config import VLMConfig
from models.vision_language import VisionLanguageModel


def parse_args():
    """Parse command-line arguments for single-image generation.

    Returns:
        An `argparse.Namespace` describing the checkpoint, prompt, image path,
        and decoding hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Warmup VLM generation script")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="What is in the image?")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true")
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


def load_model(args, device):
    """Instantiate the model and optionally restore projector weights from disk.

    Args:
        args: Parsed command-line arguments.
        device: Device on which the model should be materialized.

    Returns:
        A `VisionLanguageModel` in evaluation mode.
    """
    cfg = VLMConfig()
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config_dict = dict(checkpoint["config"])
        # Older checkpoints may still contain removed config keys.
        config_dict.pop("mp_pixel_shuffle_factor", None)
        cfg = VLMConfig(**config_dict)

    model = VisionLanguageModel(cfg).to(device)
    if args.checkpoint is not None:
        model.modality_projector.load_state_dict(checkpoint["projector"])

    model.eval()
    return model


def main():
    """Run image-conditioned generation from the command line.

    Returns:
        `None`. The decoded text is printed to stdout.
    """
    args = parse_args()
    device = get_device()

    model = load_model(args, device)
    tokenizer = model.tokenizer
    image_processor = model.image_processor

    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    # The prompt starts with placeholder image tokens that will be replaced by visual embeddings.
    image_string = get_image_string(tokenizer, model.cfg.mp_image_token_length)
    messages = [{"role": "user", "content": image_string + args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt_text, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=encoded["input_ids"].to(device),
        pixel_values=pixel_values,
        attention_mask=encoded["attention_mask"].to(device),
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        greedy=args.greedy,
    )
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(output_text)


if __name__ == "__main__":
    main()
