from __future__ import annotations
import logging
import os
import sys
import torch
from PIL import Image

# On remonte d'un niveau (..) et on pointe sur project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project")))

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

logger = logging.getLogger(__name__)

# Le dossier checkpoints est aussi dans project
CHECKPOINT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "project", "checkpoints", "best_step100")
)

_model = None
_tokenizer = None
_image_processor = None
_cfg = None
_device = "cpu"

def load_model():
    global _model, _tokenizer, _image_processor, _cfg, _device
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading VLM from %s on %s", CHECKPOINT_PATH, _device)

    _model = VisionLanguageModel.from_pretrained(CHECKPOINT_PATH).to(_device)
    _model.eval()
    _cfg = _model.cfg

    _tokenizer = get_tokenizer(_cfg.lm.tokenizer, _cfg.image_token)
    _image_processor = get_image_processor(_cfg.vit.img_size)

    logger.info("VLM loaded — %s parameters", f"{sum(p.numel() for p in _model.parameters()):,}")

def generate(question: str, choices: list[str], image: Image.Image,
             max_new_tokens: int = 64, temperature: float = 0.0,
             greedy: bool = True) -> str:
    
    choices_str = "\n".join(choices)
    prompt = f"{question}\n{choices_str}\nAnswer with only the letter of the correct choice."

    pixel_values = _image_processor(image).unsqueeze(0).to(_device)

    image_string = get_image_string(_cfg.projector.image_token_length, _cfg.image_token)
    messages = [{"role": "user", "content": image_string + prompt}]
    
    encoded = _tokenizer.apply_chat_template(
        [messages], tokenize=True, add_generation_prompt=True
    )
    input_ids = torch.tensor(encoded).to(_device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        gen = _model.generate(
            input_ids, pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            temperature=temperature if not greedy else 1.0,
        )

    return _tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()