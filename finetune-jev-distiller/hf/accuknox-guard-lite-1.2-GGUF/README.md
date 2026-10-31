---
license: apache-2.0
language:
- en
base_model: Accuknoxtechnologies/accuknox-guard-lite-1.2
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-classification
tags:
- guardrail
- safety
- prompt-injection
- jailbreak
- llama.cpp
- gguf
---

# AccuKnox Guard Lite 1.2 (GGUF)

[AccuKnox Guard Lite 1.2](https://huggingface.co/Accuknoxtechnologies/accuknox-guard-lite-1.2) (0.85B) for CPU
inference with [llama.cpp](https://github.com/ggml-org/llama.cpp). It decides whether an input should be blocked
before it reaches an AI agent (harmful request, jailbreak, prompt injection), or whether an assistant's reply is
unsafe. Each answer is a probability; block at 0.5.

| File | Quantization | Size |
|---|---|---|
| `accuknox-guard-lite-1.2-Q8_0.gguf` | Q8_0 (output layer f32) | 1.8 GB |

Q8_0 matches the bf16 model: 18 of 11,586 benchmark decisions differ.

## Quick start

```bash
hf download Accuknoxtechnologies/accuknox-guard-lite-1.2-GGUF --local-dir accuknox-guard-lite-1.2-GGUF
cd accuknox-guard-lite-1.2-GGUF
THREADS=8 docker compose up -d     # THREADS = free physical cores
python3 check.py                   # prints "all checks passed"
```

Then POST to `http://127.0.0.1:8100/v1/systemone` (about 0.35 s per short input on 8 cores). API, thread
settings, throughput and building llama.cpp without Docker: [DEPLOYMENT.md](DEPLOYMENT.md).

## Results

Threshold 0.5; every system run by us on the same items.

| | Lite 1.2 | [1.2](https://huggingface.co/Accuknoxtechnologies/accuknox-guard-1.2) (4.5B, GPU) |
|---|---|---|
| Artificial Analysis guardrail benchmark (local rebuild), avg F1 | 83.2 | 87.6 |
| GuardRate English (local rebuild) | 0.787 | 0.807 |
| Prompt injection (9 public sets) | 0.777 | 0.846 |
| Over-defence (safe inputs passed) | 0.850 | 0.905 |

## Limitations

- English. Russian scores lower (GuardRate RU 0.796 vs 1.2's 0.888); use 1.2 for non-English traffic.
- Blocks more safe-but-alarming prompts than 1.2 (XSTest specificity 92.4% vs 97.2%). Text only.

## License

Apache-2.0, as the base model; see `LICENSE` and `NOTICE`. Part of the training data is licensed for
non-commercial use only (toxic-chat); see the
[Lite 1.2 card](https://huggingface.co/Accuknoxtechnologies/accuknox-guard-lite-1.2) for the training datasets.
