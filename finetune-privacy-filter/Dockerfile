FROM ml-base:cuda12.4-torch2.4-transformers4.51

USER root

# Privacy Filter support landed after the base image's pinned transformers build.
RUN pip uninstall -y transformers && \
    pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git" \
      accelerate tiktoken safetensors huggingface_hub packaging numpy peft

WORKDIR /workspace

USER appuser

CMD ["bash"]
