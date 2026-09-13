FROM python:3.11-slim

WORKDIR /app

# System deps for torch/transformers (kept minimal for image size)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch build — HF Spaces free tier has no GPU, and the default PyPI
# torch wheel bundles CUDA and is 5-10x larger for no benefit here
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# App code + frontend + fine-tuned model weights
COPY app.py .
COPY index.html .
COPY saved_summarizer_model ./saved_summarizer_model

# Hugging Face Spaces (Docker SDK) expects the container to listen on 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
