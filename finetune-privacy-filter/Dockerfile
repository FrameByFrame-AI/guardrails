FROM ml-base:cuda12.6-torch2.11-transformers4.51

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

# The base image ships Python 3.11.0rc1 (pre-release, Aug 2022), which predates
# sys.get_int_max_str_digits/set_int_max_str_digits landing in 3.11.0 final.
# torch._dynamo's polyfill loader imports these unconditionally at import time
# (even for plain model loading, no dynamo/export involved), so every model
# load fails with AttributeError without this shim. Appended (not written fresh)
# to the interpreter's actual sitecustomize.py at /usr/lib/python3.11/ — an
# earlier-on-sys.path copy in dist-packages would be shadowed and never run.
RUN cat >> /usr/lib/python3.11/sitecustomize.py <<'EOF'

# --- privacy-filter shim: pre-3.11.0-final Python lacks these (added in 3.11.0
# final for CVE-2020-10735; this rc1 build predates it), but torch._dynamo's
# polyfill loader imports them unconditionally at import time. ---
import sys as _sys

if not hasattr(_sys, "get_int_max_str_digits"):
    _sys._int_max_str_digits = 4300

    def get_int_max_str_digits():
        return _sys._int_max_str_digits

    def set_int_max_str_digits(maxdigits):
        if maxdigits != 0 and maxdigits < 640:
            raise ValueError("maxdigits must be 0 or >= 640")
        _sys._int_max_str_digits = maxdigits

    _sys.get_int_max_str_digits = get_int_max_str_digits
    _sys.set_int_max_str_digits = set_int_max_str_digits
EOF

WORKDIR /workspace

USER appuser

CMD ["bash"]
