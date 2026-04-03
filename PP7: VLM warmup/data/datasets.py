"""Dataset wrappers that turn image-text corpora into PP7 training samples."""

import logging

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.processors import get_image_string


class VQADataset(Dataset):
    """Prepare conversational multimodal examples for supervised fine-tuning.

    The wrapper normalizes a small set of supported dataset schemas, injects the
    image placeholder tokens expected by the VLM, and builds token-level loss
    masks so only assistant responses contribute to the objective.
    """

    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        """Initialize the dataset wrapper.

        Args:
            dataset: Underlying Hugging Face dataset or dataset-like object.
            tokenizer: Tokenizer used to render chat prompts and tokenize text.
            image_processor: Vision preprocessor matching the image backbone.
            mp_image_token_length: Number of placeholder image tokens that must
                appear in the prompt for each image.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.caption_prompt = "Describe the image."

    def __len__(self):
        """Return the number of raw samples exposed by the wrapped dataset.

        Returns:
            The number of items in the wrapped dataset.
        """
        return len(self.dataset)

    def _process_image(self, image):
        """Convert a PIL image into the tensor format expected by the backbone.

        Args:
            image: Input image as a `PIL.Image.Image` instance.

        Returns:
            A single image tensor with channel-first layout.

        Raises:
            ValueError: If the provided object is not a PIL image.
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected a PIL image, got {type(image)}")

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        return pixel_values.squeeze(0)

    def _prepare_inputs_and_loss_mask(self, messages):
        """Tokenize a conversation and mark assistant tokens for loss computation.

        Args:
            messages: Chat-style list of role/content dictionaries.

        Returns:
            A tuple of `(input_ids, attention_mask, loss_mask)` tensors where
            `loss_mask` is true only on assistant tokens.
        """
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # Only assistant turns should contribute to the supervised loss.
            prefix_ids = self.tokenizer.apply_chat_template(
                messages[:idx],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )["input_ids"]
            full_ids = self.tokenizer.apply_chat_template(
                messages[: idx + 1],
                tokenize=True,
                add_special_tokens=False,
                return_dict=True,
            )["input_ids"]

            start = len(prefix_ids)
            end = len(full_ids)
            if end > start:
                mask[start:end] = [1] * (end - start)

        input_ids = torch.tensor(conv_ids["input_ids"])
        attention_mask = torch.tensor(conv_ids["attention_mask"])
        loss_mask = torch.tensor(mask, dtype=torch.bool)
        return input_ids, attention_mask, loss_mask

    def _build_cauldron_messages(self, item):
        """Convert a Cauldron-style sample into a chat transcript.

        Args:
            item: Dataset row containing a `texts` field with user/assistant
                pairs.

        Returns:
            A chat transcript as a list of role/content dictionaries.
        """
        messages = []
        for text in item["texts"]:
            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        if not messages:
            return []
        return messages

    def _build_flickr_messages(self, item):
        """Convert a captioning sample into a simple user/assistant exchange.

        Args:
            item: Dataset row containing caption text.

        Returns:
            A two-message conversation asking the model to describe the image,
            or an empty list if no usable caption exists.
        """
        captions = item.get("original_alt_text") or item.get("alt_text") or []
        if isinstance(captions, str):
            captions = [captions]
        if not captions:
            return []

        caption = next((caption.strip() for caption in captions if caption and caption.strip()), None)
        if caption is None:
            return []

        return [
            {"role": "user", "content": self.caption_prompt},
            {"role": "assistant", "content": caption},
        ]

    def _build_messages(self, item, has_image):
        """Build the final chat prompt, including image placeholder tokens.

        Args:
            item: Raw dataset row.
            has_image: Whether an image is available and should be represented by
                placeholder tokens in the prompt.

        Returns:
            A cleaned conversation ready for tokenization.
        """
        if "texts" in item:
            messages = self._build_cauldron_messages(item)
        else:
            messages = self._build_flickr_messages(item)

        if not messages:
            return []

        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                logging.warning("Removed an unexpected image token from the text.")
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")

        if has_image:
            image_string = get_image_string(self.tokenizer, self.mp_image_token_length)
            # The placeholder image tokens are later replaced with visual embeddings.
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def __getitem__(self, idx):
        """Return a single multimodal training example or `None` if unusable.

        Args:
            idx: Sample index in the wrapped dataset.

        Returns:
            A dictionary containing image tensor, token ids, attention mask, and
            loss labels, or `None` if the raw sample cannot be converted.
        """
        item = self.dataset[idx]

        images = item.get("images")
        if images is None and "image" in item:
            images = item["image"]
        if images is None:
            return None
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            return None

        pixel_values = self._process_image(images[0])
        messages = self._build_messages(item, has_image=True)
        if not messages:
            return None

        input_ids, attention_mask, loss_mask = self._prepare_inputs_and_loss_mask(
            messages
        )
        # Ignore non-assistant tokens in the loss with the standard -100 sentinel.
        labels = input_ids.clone().masked_fill(~loss_mask, -100)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
