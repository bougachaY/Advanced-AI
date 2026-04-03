"""Tokenizer and image-processor helpers used across PP7."""

from transformers import AutoImageProcessor, AutoTokenizer


TOKENIZER_CACHE = {}
IMAGE_PROCESSOR_CACHE = {}


def get_tokenizer(model_name, image_token):
    """Load and cache a tokenizer, ensuring the image placeholder token exists.

    Args:
        model_name: Hugging Face tokenizer identifier.
        image_token: Special token inserted into prompts to reserve image slots.

    Returns:
        A tokenizer instance augmented with `image_token` and `image_token_id`
        attributes for downstream convenience.
    """
    if model_name not in TOKENIZER_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        vocab = tokenizer.get_vocab()
        if image_token not in vocab:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [image_token]}
            )

        tokenizer.image_token = image_token
        tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        TOKENIZER_CACHE[model_name] = tokenizer

    return TOKENIZER_CACHE[model_name]


def get_image_processor(model_name):
    """Load and cache the image processor paired with the vision backbone.

    Args:
        model_name: Hugging Face model identifier for the vision encoder.

    Returns:
        The corresponding `AutoImageProcessor` instance.
    """
    if model_name not in IMAGE_PROCESSOR_CACHE:
        IMAGE_PROCESSOR_CACHE[model_name] = AutoImageProcessor.from_pretrained(
            model_name
        )
    return IMAGE_PROCESSOR_CACHE[model_name]


def get_image_string(tokenizer, num_image_tokens):
    """Build the repeated image-token prefix inserted into the user prompt.

    Args:
        tokenizer: Tokenizer carrying the configured image placeholder token.
        num_image_tokens: Number of visual token slots to reserve in the prompt.

    Returns:
        A string containing `num_image_tokens` consecutive image placeholder
        tokens.
    """
    return tokenizer.image_token * num_image_tokens
