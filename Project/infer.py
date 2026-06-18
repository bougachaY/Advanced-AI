"""Minimal single-sample inference with a trained VLM checkpoint.
Usage:
    python infer.py \
        --checkpoint checkpoints/best_step5000 \
        --image path/to/image.jpg \
        --prompt "What is shown in this image?"

    # Text-only (no image):
    python infer.py \
        --checkpoint checkpoints/best_step5000 \
        --prompt "What is the capital of France?"
"""
import argparse
import torch
from PIL import Image
from data.processors import get_image_processor, get_image_string, get_tokenizer
from models.vision_language_model import VisionLanguageModel


def build_prompt(question: str, tokenizer, cfg, use_image: bool) -> str:
    if use_image:
        image_string = get_image_string(
            cfg.projector.image_token_length, cfg.image_token
        )
        content = image_string + question
    else:
        content = question

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    prompt = tokenizer.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )
    if isinstance(prompt, list):
        return prompt[0]
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single inference with a trained VLM checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint directory.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to the input image (optional; omit for text-only inference).")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt / question.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")
    print(f"Loading checkpoint from {args.checkpoint} ...")

    model = VisionLanguageModel.from_pretrained(args.checkpoint).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm.tokenizer, model.cfg.image_token)

    use_image = args.image is not None

    # --- Image branch ---
    pixel_values = None
    if use_image:
        image_processor = get_image_processor(model.cfg.vit.img_size)
        image = Image.open(args.image).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device)
        print(f"Image loaded: {args.image}")
    else:
        print("No image provided — running text-only inference.")

    prompt = build_prompt(args.prompt, tokenizer, model.cfg, use_image)
    encoded = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids,
            pixel_values,          # None is passed through when text-only
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            greedy=True,
        )

    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\n--- VLM Answer ---")
    print(answer)


if __name__ == "__main__":
    main()