# RadReason inference server
# Base weights are mounted at runtime (not baked in): -v /path/Qwen3-VL-4B:/models/base
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV HF_ENDPOINT=https://hf-mirror.com \
    PYTHONUNBUFFERED=1

COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt

COPY src /app/src
COPY deploy /app/deploy

EXPOSE 8000
# MODEL / ADAPTER are provided via -e at `docker run`
CMD ["python", "deploy/app.py"]
