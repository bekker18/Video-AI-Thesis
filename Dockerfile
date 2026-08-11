FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    TORCH_HOME=/app/models/torch

# The pipeline decodes with OpenCV; ffmpeg is here only as OmniShotCut's fallback decoder.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# OmniShotCut's pyproject needs setuptools>=61, otherwise it builds an empty "UNKNOWN" wheel.
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Cache mount keeps the torch wheels local, so editing requirements.txt rebuilds in
# minutes instead of re-downloading ~2.5GB.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt

# Both are installed without deps so they use opencv-python-headless rather than pulling in
# opencv-python and libGL. Their real dependencies are declared in requirements.txt.
RUN --mount=type=cache,target=/root/.cache/pip pip3 install --no-deps \
        git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git \
        ultralytics==8.3.40

ENV YOLO_CONFIG_DIR=/app/models/ultralytics \
    YOLO_OFFLINE=true

COPY main.py config.py download_models.py pyproject.toml ./
COPY src/ src/

ENTRYPOINT ["python3", "main.py"]
