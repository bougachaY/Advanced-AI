import torch
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from data.processors import get_tokenizer, get_image_processor, get_image_string
from PIL import Image
import numpy as np

cfg = VLMConfig()
model = VisionLanguageModel(cfg, load_backbone=True)  # poids frais, MP corrigé
model.eval()

tokenizer = get_tokenizer(cfg.lm.tokenizer, cfg.image_token)
img_proc = get_image_processor(cfg.vit.img_size)

# Deux images TRÈS différentes (une noire, une blanche)
img_black = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
img_white = Image.fromarray(np.full((512, 512, 3), 255, dtype=np.uint8))

pv_black = img_proc(img_black).unsqueeze(0)
pv_white = img_proc(img_white).unsqueeze(0)

# ── Test 1 : ViT + ModalityProjector distinguent-ils les images ? ──────────
with torch.no_grad():
    feat_black = model.vision_encoder(pv_black)
    feat_white = model.vision_encoder(pv_white)
    embd_black = model.MP(feat_black)
    embd_white = model.MP(feat_white)

print("=== TEST 1 : ViT + Modality Projector ===")
print("Features ViT identiques ?", torch.allclose(feat_black, feat_white))
print("Embeddings projetés identiques ?", torch.allclose(embd_black, embd_white))
print("Diff moyenne features:", (feat_black - feat_white).abs().mean().item())
print("Diff moyenne embeddings:", (embd_black - embd_white).abs().mean().item())

# ── Construction du prompt MMStar-like ──────────────────────────────────────
image_string = get_image_string(cfg.projector.image_token_length, cfg.image_token)
messages = [[{
    "role": "user",
    "content": image_string + "What color is this image?\nA. Black\nB. White\nC. Red\nD. Blue\nAnswer with the letter directly.",
}]]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
if isinstance(prompt, list):
    prompt = prompt[0]

encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

# ── Test 2 : combien de tokens <|image|> sont réellement présents ? ────────
print("\n=== TEST 2 : comptage des tokens image ===")
print("Tokens <|image|> attendus (image_token_length):", cfg.projector.image_token_length)
print("Tokens <|image|> trouvés dans le prompt:", image_string.count(cfg.image_token))
print("Tokens <|image|> trouvés dans input_ids:", (input_ids == tokenizer.image_token_id).sum().item())
print("Shape de input_ids:", input_ids.shape)

# ── Test 3 : le LM utilise-t-il vraiment le signal visuel en sortie ? ──────
with torch.no_grad():
    hidden_black, _ = model(input_ids, pv_black, attention_mask, targets=None)
    hidden_white, _ = model(input_ids, pv_white, attention_mask, targets=None)

print("\n=== TEST 3 : sortie du LM (hidden states) ===")
print("Hidden states identiques ?", torch.allclose(hidden_black, hidden_white))
print("Diff moyenne hidden states:", (hidden_black - hidden_white).abs().mean().item())

# ── Test 4 : génération réelle ──────────────────────────────────────────────
gen_black = model.generate(input_ids, pv_black, attention_mask=attention_mask, max_new_tokens=5, greedy=True)
gen_white = model.generate(input_ids, pv_white, attention_mask=attention_mask, max_new_tokens=5, greedy=True)

print("\n=== TEST 4 : génération ===")
print("Réponse image noire:", tokenizer.decode(gen_black[0], skip_special_tokens=True))
print("Réponse image blanche:", tokenizer.decode(gen_white[0], skip_special_tokens=True))

from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig


# ── Test 5 : comparaison des normes (texte vs visuel) ───────────────────────
token_embd = model.decoder.token_embedding(input_ids)

print("\n=== TEST 5 : échelle des embeddings ===")
print("Norme moyenne token_embd (texte):", token_embd.norm(dim=-1).mean().item())
print("Norme moyenne image_embd (noire):", embd_black.norm(dim=-1).mean().item())
print("Norme moyenne image_embd (blanche):", embd_white.norm(dim=-1).mean().item())
print("Ratio image/texte:", (embd_black.norm(dim=-1).mean() / token_embd.norm(dim=-1).mean()).item())