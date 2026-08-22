FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    TORCH_HOME=/app/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip3 install --no-deps \
    git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git \
    ultralytics==8.3.40 \
    controlnet_aux==0.0.9

ENV YOLO_CONFIG_DIR=/app/models/ultralytics \
    YOLO_OFFLINE=true

COPY main.py config.py download_models.py pyproject.toml ./
COPY src/ src/

ENTRYPOINT ["python3", "main.py"]
