FROM unsloth/unsloth

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HOME=/home/unsloth

ARG UID=1000
ARG GID=1000

USER root

RUN set -eux; \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    python -m pip install --no-cache-dir "trl==0.22.2" datasets accelerate "git+https://github.com/huggingface/transformers.git@main"; \
    python -m pip uninstall -y torchao || true; \
    if getent group unsloth >/dev/null; then \
        groupmod -o -g ${GID} unsloth; \
    else \
        groupadd -o -g ${GID} unsloth; \
    fi; \
    if getent passwd unsloth >/dev/null; then \
        usermod -o -u ${UID} -g ${GID} -d /home/unsloth -s /bin/bash unsloth; \
    else \
        useradd -o -m -d /home/unsloth -s /bin/bash -u ${UID} -g ${GID} unsloth; \
    fi; \
    mkdir -p /workspace /cache/huggingface /home/unsloth; \
    touch /home/unsloth/.bashrc; \
    chown -R ${UID}:${GID} /workspace /cache /home/unsloth

WORKDIR /workspace
USER unsloth

CMD ["bash"]
