.PHONY: help bench-data train-data train score baselines data-v6 data-v9 train-d3 score-d3 soup guardrate injection compare smoke smoke-gpu shell clean

RUN = docker compose run --rm -T
CPU = $(RUN) cpu
# every GPU command starts the telemetry logger first (gpu_watch.sh; the GPU has crashed with Xid 79).
# The card comes from GPU in .env; override per command, e.g. GPU=<A5000 uuid> make train-d3 ...
GPU_RUN = ./gpu_watch.sh && $(RUN) gpu
GPU_RUNK = ./gpu_watch.sh && $(RUN) -e PYTHONPATH=/workspace/runs/pylibs gpu
RUN_NAME ?= gold-gliner
EPOCHS ?= 3
BATCH_SIZE ?= 8
GRAD_ACCUM ?= 2
# truncation in gliner2 word units; 2048 runs out of memory even on 96 GB
MAX_LEN ?= 1024
BASE = /models/GLiNER2.5-Decide
D = data
# zero-shot open decision models (decision_caller.py): run dir = model; compare includes those already scored
BASELINES = d1 d3 shieldstral
# train_d3.py data: run 5 used split_{train,val}; run 6 uses v6_{train,val} (make data-v6)
D3_TRAIN ?= $(D)/split_train.jsonl
D3_VAL ?= $(D)/split_val.jsonl
# extra train_d3.py options, e.g. D3_ARGS="--grad-accum 2 --eval-every 250" on the 24 GB A5000
D3_ARGS ?=
# d3 GuardRate questions: block (runs 1-7) or split (run 8: content question scored, plus the agent gate)
D3_QUESTIONS ?= block
# GuardRate items seen in training by any run compared (guardrate.py audit), so "unseen" means unseen by all of them
SEEN_TRAIN ?= $(D)/split_train.jsonl $(D)/split_val.jsonl
SEEN ?= $(D)/guardrate_en_seen.json
BASELINE_SYSTEMS = $(foreach b,$(BASELINES),$(if $(wildcard runs/$(b)/benchmark.jsonl),--system $(b)=runs/$(b)/benchmark.jsonl))

help:
	@echo "make bench-data  benchmark: dataset test splits, labels verified by GLM  -> data/benchmark.jsonl"
	@echo "make train-data  training set: dataset train splits (no benchmark prompt), verified -> data/split_{train,val}.jsonl"
	@echo "make train       fine-tune GLiNER2.5-Decide on the datasets' labels    -> runs/\$$RUN_NAME/ (e.g. make train EPOCHS=5)"
	@echo "make score       run the model over the benchmark                       -> runs/\$$RUN_NAME/benchmark.jsonl"
	@echo "make baselines  zero-shot d1-3B, d3-mini and Shieldstral-3B on the benchmark -> runs/{d1,d3,shieldstral}/"
	@echo "make train-d3 score-d3 RUN_NAME=d3-ft   LoRA fine-tune of d3-mini on the same data, then score it"
	@echo "make data-v6     run 6 data: build, decontaminate, hold out whole sources -> data/v6_{train,val}.jsonl"
	@echo "make train-d3 guardrate RUN_NAME=d3-v6 D3_TRAIN=data/v6_train.jsonl D3_VAL=data/v6_val.jsonl \\"
	@echo "     SEEN_TRAIN='data/split_train.jsonl data/split_val.jsonl data/v6_train.jsonl data/v6_val.jsonl' \\"
	@echo "     SEEN=data/guardrate_en_seen_r5r6.json      run 6, scored on items unseen by runs 5 and 6"
	@echo "make guardrate RUN_NAME=d3-ft  GuardRate English rebuild: Shieldstral, the model, XGuard-8B (resumable)"
	@echo "make injection RUN_NAME=d3-ft  public prompt-injection benchmark (BIPIA, InjecAgent, NotInject), resumable"
	@echo "make data-v9     run 9 data: run 5's rows plus planted-instruction documents -> data/v9_{train,val}.jsonl"
	@echo "make soup        AccuKnox Guard 1.1: average runs 5 and 9's weights -> runs/d3-soup59/package (CPU)"
	@echo "                 then GPU=<uuid> scripts/soup.sh scores it (injection, GuardRate)"
	@echo "make compare     Jev, GLM, Bedrock, baselines and the model            -> runs/\$$RUN_NAME/compare.json"
	@echo "make smoke       10 failure cases through every API module (about \$$0.001)"
	@echo "make smoke-gpu   10 failure cases through convert, split, train (2 steps), score, compare"
	@echo "make shell       bash in the GPU container      make clean  delete runs/ (never data/)"

# Label caches (jev_labels, glm_labels, bedrock_labels) are append-only: reruns only pay for new prompts.
bench-data:
	$(CPU) build_benchmark.py --per-dataset 300 --output $(D)/benchmark_candidates.jsonl
	$(CPU) verify_glm.py --input $(D)/benchmark_candidates.jsonl --cache $(D)/glm_labels.jsonl \
		--output $(D)/benchmark.jsonl --disputed $(D)/benchmark_disputed.jsonl
	$(CPU) jev_caller.py --input $(D)/benchmark.jsonl --output $(D)/jev_labels.jsonl
	$(CPU) bedrock_caller.py --input $(D)/benchmark.jsonl --cache $(D)/bedrock_labels.jsonl

train-data:
	$(CPU) build_train.py --per-dataset 2000 --exclude $(D)/benchmark_candidates.jsonl --output $(D)/train_candidates.jsonl
	@# Training keeps every prompt the judges confirm, even from a dataset under 80% (OR-Bench: about half of its
	@# "safe" prompts are judged harmful). The benchmark keeps the 80% rule.
	$(CPU) verify_glm.py --input $(D)/train_candidates.jsonl --cache $(D)/glm_labels.jsonl --workers 16 \
		--min-dataset-agreement 0 --output $(D)/train_verified.jsonl --disputed $(D)/train_disputed.jsonl
	$(CPU) to_gliner.py --input $(D)/train_verified.jsonl --output $(D)/gliner_train.jsonl
	$(CPU) split_data.py --input $(D)/gliner_train.jsonl --manifest $(D)/train_verified.jsonl --out-prefix $(D)/split --test 0

train:
	$(GPU_RUN) train_gliner.py --train $(D)/split_train.jsonl --val $(D)/split_val.jsonl --out runs/$(RUN_NAME) --base $(BASE) \
		--epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --grad-accum $(GRAD_ACCUM) --max-len $(MAX_LEN) --gradient-checkpointing

score:
	$(GPU_RUN) gliner_caller.py --model runs/$(RUN_NAME) --input $(D)/benchmark.jsonl --output runs/$(RUN_NAME)/benchmark.jsonl

# d3-mini's linear-attention kernels (Triton only), kept out of the shared base image
runs/pylibs:
	$(RUN) --entrypoint pip cpu install --quiet --no-deps --target /workspace/runs/pylibs flash-linear-attention==0.5.2 fla-core==0.5.2

baselines: runs/pylibs
	for b in $(BASELINES); do mkdir -p runs/$$b && \
		$(GPU_RUNK) decision_caller.py --model $$b --input $(D)/benchmark.jsonl --output runs/$$b/benchmark.jsonl || exit 1; done

# LoRA fine-tune of d3-mini on the same verified records (train_d3.py); RUN_NAME e.g. d3-ft
data-v6:
	test -f $(D)/train_v6.jsonl || $(CPU) build_v6.py --out $(D)/train_v6.jsonl
	$(CPU) split_v6.py --input $(D)/train_v6.jsonl --train $(D)/v6_train.jsonl --val $(D)/v6_val.jsonl

train-d3: runs/pylibs
	$(GPU_RUNK) train_d3.py --train $(D3_TRAIN) --val $(D3_VAL) --out runs/$(RUN_NAME) $(D3_ARGS)

score-d3: runs/pylibs
	$(GPU_RUNK) decision_caller.py --model d3 --path runs/$(RUN_NAME)/package \
		--input $(D)/benchmark.jsonl --output runs/$(RUN_NAME)/benchmark.jsonl

# GuardRate English rebuild (guardrate.py). Every step resumes: rerun after a crash continues where it stopped.
GR = $(D)/guardrate_en.jsonl
guardrate: runs/pylibs
	test -f $(GR) || $(CPU) guardrate.py build --out $(GR)
	test -f $(SEEN) || $(CPU) guardrate.py audit --train $(SEEN_TRAIN) --out $(SEEN)
	mkdir -p runs/shieldstral runs/xguard runs/$(RUN_NAME)
	$(GPU_RUNK) guardrate.py run --model shieldstral --input $(GR) --output runs/shieldstral/guardrate_en.jsonl
	$(GPU_RUNK) guardrate.py run --model d3 --path runs/$(RUN_NAME)/package --questions $(D3_QUESTIONS) \
		--input $(GR) --output runs/$(RUN_NAME)/guardrate_en.jsonl
	$(GPU_RUNK) guardrate.py run --model xguard --input $(GR) --output runs/xguard/guardrate_en.jsonl
	$(CPU) guardrate.py score --input $(GR) --seen $(SEEN) --output runs/$(RUN_NAME)/guardrate_compare.json \
		--system shieldstral=runs/shieldstral/guardrate_en.jsonl --system $(RUN_NAME)=runs/$(RUN_NAME)/guardrate_en.jsonl \
		--system xguard-8b=runs/xguard/guardrate_en.jsonl \
		$(if $(filter split,$(D3_QUESTIONS)),--system $(RUN_NAME)-gate=runs/$(RUN_NAME)/guardrate_en.jsonl@gate) \
		$(if $(filter-out d3-ft,$(RUN_NAME)),$(if $(wildcard runs/d3-ft/guardrate_en.jsonl),--system d3-ft=runs/d3-ft/guardrate_en.jsonl))

# public prompt-injection benchmark (build_injection_bench.py: BIPIA test, InjecAgent, NotInject), resumable
IB = $(D)/injection_bench.jsonl
injection: runs/pylibs
	test -f $(IB) || $(CPU) build_injection_bench.py --out $(IB)
	$(GPU_RUNK) guardrate.py run --model d3 --path runs/$(RUN_NAME)/package --input $(IB) \
		--output runs/$(RUN_NAME)/injection_bench.jsonl
	$(CPU) injection_results.py --system $(RUN_NAME)=runs/$(RUN_NAME)/injection_bench.jsonl

# run 9 data: run 5's rows plus documents with planted instructions and their clean twins (build_v9.py)
data-v9:
	$(CPU) build_v9.py --out-train $(D)/v9_train.jsonl --out-val $(D)/v9_val.jsonl

# AccuKnox Guard 1.1: equal-weight average of runs 5 and 9 (merge_runs.py); skipped if already merged
SOUP_RUNS ?= d3-ft d3-v9
SOUP_NAME ?= d3-soup59
soup:
	$(CPU) merge_runs.py $(foreach r,$(SOUP_RUNS),--run runs/$(r)/package) --out runs/$(SOUP_NAME)/package

compare:
	$(CPU) compare_systems.py --bench $(D)/benchmark.jsonl --jev $(D)/jev_labels.jsonl --glm $(D)/glm_labels.jsonl \
		--bedrock $(D)/bedrock_labels.jsonl --system $(RUN_NAME)=runs/$(RUN_NAME)/benchmark.jsonl $(BASELINE_SYSTEMS) \
		--output runs/$(RUN_NAME)/compare.json

smoke:
	$(RUN) --entrypoint bash cpu tests/smoke.sh

smoke-gpu: runs/pylibs
	$(RUN) -e PYTHONPATH=/workspace/runs/pylibs --entrypoint bash gpu tests/smoke_gpu.sh

shell:
	docker compose run --rm --entrypoint bash gpu

clean:
	rm -rf runs
