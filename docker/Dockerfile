FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# -------- Install System Dependencies --------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    unzip \
    pkg-config \
    libssl-dev \
    python3-build \
    && rm -rf /var/lib/apt/lists/*

# -------- Install Rust --------
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . "$HOME/.cargo/env" \
    && rustc --version \
    && cargo --version \
    && rustup target add wasm32-wasip2

# Ensure Rust is in PATH for subsequent layers
ENV PATH="/root/.cargo/bin:${PATH}"

# -------- Install Protobuf --------
RUN curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v21.3/protoc-21.3-linux-x86_64.zip \
    && apt-get update && apt-get install -y unzip \
    && unzip protoc-21.3-linux-x86_64.zip -d /usr/local \
    && rm protoc-21.3-linux-x86_64.zip

# Install Python dependencies including protoc plugins
RUN pip install grpcio-tools quart uvicorn

# link protoc-gen-mypy
RUN ln -s /opt/conda/bin/protoc-gen-mypy /usr/local/bin/protoc-gen-pyi

# -------- Build FlashInfer --------
WORKDIR /tmp

RUN pip install --upgrade pip setuptools wheel build uv
# Clone flashinfer with submodules
RUN git clone https://github.com/flashinfer-ai/flashinfer.git --recursive

WORKDIR /tmp/flashinfer
# Set CUDA arch (can override via --build-arg TORCH_CUDA_ARCH_LIST=8.9 etc.)
ARG TORCH_CUDA_ARCH_LIST=8.9
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Build AOT kernels and wheel
RUN python -m flashinfer.aot \
    && python -m build --no-isolation --wheel \
    && pip install dist/flashinfer_*.whl

# Cleanup source to reduce image size
WORKDIR /workspace
RUN rm -rf /tmp/flashinfer

# Clone the pie repository
RUN git clone https://github.com/Leonana69/pie.git /workspace

# -------- Install Backend --------
RUN cd /workspace/backend/backend-python \
    && chmod +x build_proto.sh \
    && ./build_proto.sh \
    && pip install -e .

# -------- Compile Frontend --------
RUN cd /workspace/pie-cli \
    && cargo install --path .

# -------- Compile Examples --------
RUN cd /workspace/example-apps \
    && cargo build --target wasm32-wasip2 --release

# -------- Install lib for client --------
RUN pip install blake3 fastapi

# CMD ["python", "./run.py"]
