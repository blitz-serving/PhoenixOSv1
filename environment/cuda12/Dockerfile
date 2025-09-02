# ---- CUDA 12.3 + PyTorch (GPU) ----
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 1) Base tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    # ensure repos are present & updated
    sed -Ei 's/^# deb/deb/g' /etc/apt/sources.list; \
    apt-get update; \
    # install clang from Ubuntu repo (clang-14 on 22.04)
    apt-get install -y --no-install-recommends \
        clang \
        lld \
        build-essential \
        ca-certificates \
        curl; \
    rm -rf /var/lib/apt/lists/*

RUN apt update
RUN apt-get install -y --no-install-recommends \
    librdmacm-dev ibverbs-utils; 

# 2) (Optional) keep the container lean: a venv avoids cluttering system site-packages
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --upgrade pip

# 3) Pick a PyTorch wheel channel with bundled CUDA
#    cu121 is universally supported; cu124 is available on newer releases.
ARG TORCH_CUDA_CHANNEL="https://download.pytorch.org/whl/cu124"
# If you prefer extra stability, switch to cu121:
# ARG TORCH_CUDA_CHANNEL="https://download.pytorch.org/whl/cu121"

# 4) Install "default" PyTorch trio (GPU build)
RUN pip install --index-url ${TORCH_CUDA_CHANNEL} \
      torch torchvision torchaudio

## Rust tool chain

ENV RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup
ENV RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup 
RUN mkdir -p /opt/rust

ENV CARGO_HOME=/opt/rust/.cargo
ENV RUSTUP_HOME=/opt/rust/.rustup

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
ENV PATH="/opt/rust/.cargo/bin:${PATH}"
RUN . /opt/rust/.cargo/env

RUN rustup install nightly
RUN rustup default nightly

RUN apt install libnccl2 libnccl-dev -y --allow-change-held-packages

## FIXME: currently lack CUDA support 