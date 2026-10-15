FROM ml-base:cuda12.6-torch2.11-transformers4.51

USER root

RUN pip install --no-cache-dir \
      "sentence-transformers>=3.0" \
      "pysbd>=0.3" \
      "wtpsplit>=2.0" \
      "scikit-learn>=1.3" \
      "peft>=0.10" \
      "datasets>=2.18" \
      "onnx" "onnxruntime" "optimum[onnxruntime]"

WORKDIR /workspace

USER appuser

CMD ["bash"]
