# ---- CUDA 12.3 + PyTorch (GPU) ----
FROM phoenixos/pytorch:11.3-ubuntu20.04-jit-disable as base

ARG DEBIAN_FRONTEND=noninteractive

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

## CRIU-related
RUN apt install -y pkg-config 
RUN apt install -y libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler 
RUN apt install protobuf-compiler python3-protobuf libnet1-dev libcap-dev -y
RUN apt install -y libnl-3-dev libnl-genl-3-dev asciidoctor    

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