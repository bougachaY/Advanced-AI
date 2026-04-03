"""Configuration dataclasses for the PP7 warmup training scripts."""

from dataclasses import dataclass


@dataclass
class VLMConfig:
    """Static model settings for the warmup vision-language model.

    Attributes:
        vit_model_type: Hugging Face identifier for the vision backbone.
        lm_model_type: Hugging Face identifier for the language model.
        lm_tokenizer: Tokenizer identifier used for text prompts.
        image_token: Special token reserved for image embeddings in the prompt.
        mp_image_token_length: Number of projected visual tokens injected into
            the language model sequence.
        lm_max_length: Maximum sequence length used during collation.
    """

    vit_model_type: str = "google/siglip2-base-patch16-512"
    lm_model_type: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    image_token: str = "<|image|>"
    mp_image_token_length: int = 64
    lm_max_length: int = 1024


@dataclass
class TrainConfig:
    """Training hyperparameters and dataset settings for PP7 experiments.

    Attributes:
        dataset_path: Hugging Face dataset identifier.
        dataset_names: Optional dataset config names to concatenate.
        train_samples: Number of training samples to keep after splitting.
        val_samples: Number of validation samples to reserve.
        batch_size: Per-step batch size.
        max_steps: Number of optimizer steps to run.
        eval_interval: Validation frequency in steps.
        gradient_accumulation_steps: Number of forward/backward passes to
            accumulate before an optimizer step.
        lr_projector: Learning rate for the modality projector.
        lr_vision: Learning rate for the vision backbone.
        lr_language: Learning rate for the language model.
        weight_decay: Weight decay passed to AdamW.
        max_grad_norm: Gradient clipping threshold.
        num_workers: Number of dataloader workers.
        compile: Whether to compile the model with `torch.compile`.
        split_seed: Random seed used for train/validation splitting.
        output_dir: Directory where checkpoints are written.
        output_name: Filename of the saved projector checkpoint.
    """

    dataset_path: str = "AnyModal/flickr30k"
    dataset_names: tuple[str, ...] = ()
    train_samples: int = 256
    val_samples: int = 64
    batch_size: int = 2
    max_steps: int = 20
    eval_interval: int = 10
    gradient_accumulation_steps: int = 1
    lr_projector: float = 1e-3
    lr_vision: float = 0.0
    lr_language: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_workers: int = 0
    compile: bool = False
    split_seed: int = 0
    output_dir: str = "checkpoints"
    output_name: str = "projector.pt"
