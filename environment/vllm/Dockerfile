FROM vllm/vllm-openai:v0.19.1

WORKDIR /workspace

RUN sed -i -E 's/(archive|security).ubuntu.com/mirrors.cernet.edu.cn/g' /etc/apt/sources.list && \
    echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf

RUN python3.10 /usr/bin/add-apt-repository ppa:criu/ppa && \
    apt-get install -y --allow-change-held-packages \
        criu \
        git \
        clang \
        cuda-nvml-dev-12-9 \
        librdmacm-dev \
        ibverbs-utils && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    CUDA_INC=/usr/local/cuda/include; \
    CUDA_LIB=/usr/local/cuda/lib64; \
    PY_NVIDIA=/usr/local/lib/python3.12/dist-packages/nvidia; \
    ln -sf ${PY_NVIDIA}/cublas/include/* ${CUDA_INC}/; \
    ln -sf ${PY_NVIDIA}/cudnn/include/* ${CUDA_INC}/; \
    ln -sf ${PY_NVIDIA}/cuda_nvrtc/include/* ${CUDA_INC}/; \
    ln -sf ${PY_NVIDIA}/cublas/lib/libcublas.so.12 ${CUDA_LIB}/libcublas.so; \
    ln -sf ${PY_NVIDIA}/cublas/lib/libcublas.so.12 ${CUDA_LIB}/libcublas.so.12; \
    ln -sf ${PY_NVIDIA}/cublas/lib/libcublasLt.so.12 ${CUDA_LIB}/libcublasLt.so; \
    ln -sf ${PY_NVIDIA}/cublas/lib/libcublasLt.so.12 ${CUDA_LIB}/libcublasLt.so.12; \
    ln -sf ${PY_NVIDIA}/cudnn/lib/libcudnn.so.9 ${CUDA_LIB}/libcudnn.so; \
    ln -sf ${PY_NVIDIA}/cudnn/lib/* ${CUDA_LIB}/; \
    ln -sf ${PY_NVIDIA}/cuda_nvrtc/lib/libnvrtc.so.12 ${CUDA_LIB}/libnvrtc.so; \
    ln -sf ${PY_NVIDIA}/cuda_nvrtc/lib/libnvrtc.so.12 ${CUDA_LIB}/libnvrtc.so.12

ENTRYPOINT ["sleep", "inf"]
