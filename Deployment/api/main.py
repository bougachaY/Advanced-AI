import base64, io, logging, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from PIL import Image
import model
from schemas import QCMRequest, QCMResponse

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app):
    model.load_model()
    yield

app = FastAPI(title="VLM QCM API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/qcm", response_model=QCMResponse)
def qcm(req: QCMRequest):
    try:
        raw = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    start = time.perf_counter()
    raw_output = model.generate(
        req.question, req.choices, image,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        greedy=req.greedy,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Extrait la lettre (a/b/c/d) depuis la sortie brute
    answer = raw_output.strip().lower()
    letter = next((c for c in answer if c in "abcd"), "?")

    return QCMResponse(answer=letter, raw_output=raw_output, generation_time_ms=elapsed_ms)