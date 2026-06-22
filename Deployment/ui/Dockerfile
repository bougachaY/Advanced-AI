FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    gradio>=4.0 \
    requests>=2.31 \
    pillow>=10.0

COPY ui/app.py ui/client.py ./

EXPOSE 7860
CMD ["python", "app.py"]