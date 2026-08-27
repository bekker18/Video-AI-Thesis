FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    TORCH_HOME=/app/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgles2 \
    libegl1 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

ENV PATH=/app/.venv/bin:$PATH \
    YOLO_CONFIG_DIR=/app/models/ultralytics \
    YOLO_OFFLINE=true

COPY main.py config.py download_models.py preview.py ./
COPY src/ src/

ENTRYPOINT ["/app/.venv/bin/python", "main.py"]
