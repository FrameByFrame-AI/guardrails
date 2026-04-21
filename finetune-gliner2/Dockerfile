FROM ml-base:cuda12.4-torch2.4-transformers4.51

USER root

# GLiNER2 + training deps
RUN pip install --no-cache-dir \
    gliner2==1.2.6 \
    "compressed-tensors==0.13.0"

WORKDIR /workspace

USER appuser

CMD ["bash"]
