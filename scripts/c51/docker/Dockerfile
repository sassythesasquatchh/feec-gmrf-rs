FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_TOOLCHAIN=1.98.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        libfontconfig1-dev \
        libopenblas-dev \
        libpetsc-real-dev \
        libslepc-real-dev \
        pkg-config \
        python3 \
        time \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain ${RUST_TOOLCHAIN}

ENV PATH=/root/.cargo/bin:${PATH}
ENV CARGO_TARGET_DIR=/target
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV RAYON_NUM_THREADS=1
ENV FAER_NUM_THREADS=1
WORKDIR /workspace
