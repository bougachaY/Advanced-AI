FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    fastapi>=0.115 \
    "uvicorn[standard]>=0.32" \
    pydantic>=2.9 \
    pillow>=10.0 \
    safetensors>=0.4 \
    "transformers>=4.38,<5" \
    torch==2.3.0 \
    torchvision

COPY api/main.py api/model.py api/schemas.py ./
COPY project/ ./project/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]