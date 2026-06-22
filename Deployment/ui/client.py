import base64, io, requests
from PIL import Image

def encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def call_qcm(api_url, question, choices, image, max_new_tokens=64,
             temperature=0.0, greedy=True, timeout=180):
    payload = {
        "question": question,
        "choices": choices,
        "image_base64": encode_image(image),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "greedy": greedy,
    }
    resp = requests.post(f"{api_url.rstrip('/')}/qcm", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()