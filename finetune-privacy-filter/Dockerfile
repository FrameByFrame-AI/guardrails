FROM ml-base:cuda12.4-torch2.4-transformers4.51

USER root

# Privacy Filter support landed after the base image's pinned transformers build.
RUN pip uninstall -y transformers onnxruntime torch torchvision torchaudio && \
    pip install --no-cache-dir \
      "torch>=2.9" \
      "git+https://github.com/huggingface/transformers.git" \
      "git+https://github.com/huggingface/optimum.git" \
      accelerate tiktoken safetensors huggingface_hub packaging numpy peft \
      onnx onnxruntime-gpu onnxconverter-common onnxscript

# optimum-onnx main has transformers 5.x compat shims (PR #113) but its setup.py
# still caps transformers <4.58. Install with --no-deps to bypass the stale cap.
RUN pip install --no-cache-dir --no-deps "git+https://github.com/huggingface/optimum-onnx.git"

WORKDIR /workspace

USER appuser

CMD ["bash"]
