FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# GLiNER2 + training deps
RUN pip install --no-cache-dir \
    gliner2==1.2.6 \
    torch==2.4.0 \
    transformers \
    accelerate \
    wandb \
    scikit-learn \
    httpx

ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID appuser && useradd -m -u $UID -g $GID appuser

WORKDIR /workspace
RUN chown appuser:appuser /workspace

USER appuser

CMD ["bash"]
