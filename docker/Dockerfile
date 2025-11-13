# ============================================
# Stage 1: Builder - Contains build tools
# ============================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0 12.0+PTX"

# Install build dependencies and Python 3.10 exactly
RUN apt-get update && apt-get install -y --no-install-recommends -qq \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic-dev \
    libexiv2-dev \
    libgomp1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade setuptools, wheel
RUN pip install --no-cache-dir --upgrade setuptools==69.5.1 wheel

# Install PyTorch (cached layer if versions don't change)
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Clone and build mmcv from specific version tag
RUN pip install --no-cache-dir \
    mmengine==0.10.7 \
    PyYAML \
    ninja

RUN nvcc --version

RUN git clone --depth 1 --branch v2.2.0 https://github.com/open-mmlab/mmcv.git /mmcv
WORKDIR /mmcv
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install --no-cache-dir -e . -v

# Clean up build artifacts to reduce image size
RUN rm -rf /mmcv/build \
    && rm -rf /mmcv/.git \
    && rm -rf /mmcv/docs \
    && rm -rf /mmcv/tests

# Install Python packages
RUN pip install --no-cache-dir \
    tensorboard \
    fairscale==0.4.13 \
    jsonlines \
    scipy \
    shapely==2.1.2 \
    matplotlib \
    fastapi==0.119.1 \
    uvicorn==0.38.0 \
    pydantic==2.12.3 \
    starlette==0.47.3 \
    urllib3==2.2.3 \
    pillow==10.4.0 \
    SimpleITK==2.4.1 \
    pandas==2.3.3 \
    yapf==0.32.0

# ============================================
# Stage 2: Runtime - Minimal runtime image
# ============================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends -qq \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    wget \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libmagic1 \
    libexiv2-27 \
    libgomp1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy mmcv source (needed for editable install)
COPY --from=builder /mmcv /mmcv

# Download pretrained models
RUN mkdir /weights && \
    wget https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth -P /weights

# Install supervisely
RUN pip install --no-cache-dir supervisely==6.73.460