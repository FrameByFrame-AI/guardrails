"""LoRA fine-tune of vllm-sr/d3-mini, exported as a d3 package.

    python train_d3.py --train data/v6_train.jsonl --val data/v6_val.jsonl --out runs/d3-v6

The released d3 package is inference-only, so this loads its own runtime (d3_runtime.D3) for the tokenizer,
prompt renderer, backbone and 255-way answer-code readout, and trains through the same computation as its
inference: the noncausal prompt encoder, the last-position hidden state times the readout, and a softmax over
the question's option codes. Trained: LoRA on every projection of the text backbone, and the readout itself.

Input rows, either format:
- build_v6.py rows {mode, text, response, gold_block, risk, source}. Request rows ask jev_caller.QUESTIONS (block,
  and risk when the row has a category; rows without one add no risk loss). Response rows ask
  jev_caller.RESPONSE_QUESTIONS on the state {"user": text, "assistant": response}, as guardrate.py scores them.
- gliner records (runs 1-5, split_*.jsonl): block and risk from their true labels.
A request row may name its question (build_v8.py, run 8): "attack" (jev_caller.ATTACK_QUESTIONS) or "unsafe"
(jev_caller.CONTENT_QUESTIONS, with the risk question); without one it asks the block question. Validation scores
each row on its own question.

The best checkpoint is merged and written to <out>/package. Best (--select score, the default) = highest
GuardRate-style score over the validation sources at threshold 0.5: per source the harmonic mean of recall and
1 - false-block rate (one of them for single-class sources), then the harmonic mean over sources, so false blocks
count as much as misses and one bad source pulls it down. --select loss = lowest mean per-source block loss (run 6).
The probabilities of that checkpoint on the validation set go to <out>/val_preds.jsonl for choosing a threshold. The package is the original package
files with new weights and no MODEL_MANIFEST.json (its hashes no longer match). decision_caller.py scores it with
--model d3 --path <out>/package. Training truncates the input text to --max-state-tokens (response rows: the prompt to a quarter of it, the
response to the rest); inference never does.
Needs flash-linear-attention on PYTHONPATH (make train-d3 installs it into runs/pylibs).
Resumable: <out>/resume.pt is written every --save-every steps; rerunning the same command continues from it, and
it is deleted once the package is exported.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

from decision_caller import D3_REVISION
from jev_caller import ATTACK_QUESTIONS, CONTENT_QUESTIONS, QUESTIONS, RESPONSE_QUESTIONS

log = logging.getLogger("train-d3")
REPO = "vllm-sr/d3-mini"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj"]
WEIGHT_FILES = ("model.safetensors.index.json", "MODEL_MANIFEST.json", "readout.safetensors")
REQUEST_QUESTIONS = {"block": QUESTIONS, "attack": ATTACK_QUESTIONS, "unsafe": CONTENT_QUESTIONS}


def normalise(rec: dict) -> dict:
    """Either input format -> {mode, question, text, response, block, risk (or None), source}."""
    if "mode" in rec:
        return {"mode": rec["mode"], "question": rec.get("question", "block"), "text": rec["text"],
                "response": rec["response"], "block": rec["gold_block"], "risk": rec["risk"], "source": rec["source"]}
    task = {c["task"]: c["true_label"][0] for c in rec["output"]["classifications"]}
    return {"mode": "request", "question": "block", "text": rec["input"], "response": None,
            "block": task["block"] == "block", "risk": task["risk"], "source": rec.get("source", "all")}


def encode_examples(d3, records, max_state_tokens):
    """Each record -> {seq, target, options, risk_seq, risk_target, risk_options (None without a category),
    block, source}: the block question (or the response question), and the risk question for categorised requests."""
    tok, out, cut = d3.tokenizer, [], 0

    def clip(text, limit):
        nonlocal cut
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= limit:
            return text
        cut += 1
        return tok.decode(ids[:limit])

    for rec in map(normalise, records):
        if rec["mode"] == "response":
            state = {"user": clip(rec["text"], max_state_tokens // 4),
                     "assistant": clip(rec["response"], max_state_tokens - max_state_tokens // 4)}
            questions = RESPONSE_QUESTIONS
        else:
            state = clip(rec["text"], max_state_tokens)
            questions = REQUEST_QUESTIONS[rec["question"]]
            if not rec["risk"]:
                questions = {k: v for k, v in questions.items() if k != "risk"}
        main = next(iter(questions))  # the yes/no question; "risk" is never first
        prep = d3.prepare(state, questions)
        if prep.errors:
            raise SystemExit(f"d3 rejected a prompt: {prep.errors}")
        qb = prep.questions[main]
        e = {"seq": prep.sequences[main], "target": qb.keys.index("true" if rec["block"] else "false"),
             "options": len(qb.keys), "risk_seq": None, "risk_target": None, "risk_options": None,
             "block": rec["block"], "source": rec["source"]}
        if "risk" in questions:
            qr = prep.questions["risk"]
            e.update(risk_seq=prep.sequences["risk"], risk_target=qr.keys.index(rec["risk"]), risk_options=len(qr.keys))
        out.append(e)
    return out, cut


def harmonic(xs):
    return 0.0 if not xs or min(xs) <= 0 else len(xs) / sum(1 / x for x in xs)


def length(e):
    return max(len(e["seq"]), len(e["risk_seq"] or ()))


def batches(examples, size, rng, shuffle=True):
    """Batches of similar length (sorted within shuffled windows of 32 batches) to keep padding low."""
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
    window = size * 32
    for w in range(0, len(idx), window):
        chunk = sorted(idx[w:w + window], key=lambda i: length(examples[i]))
        groups = [chunk[i:i + size] for i in range(0, len(chunk), size)]
        if shuffle:
            rng.shuffle(groups)
        yield from groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8, help="records per optimizer step")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="split each step's batch into this many forward/backward passes (same update, less memory)")
    ap.add_argument("--select", choices=["score", "loss"], default="score", help="checkpoint selection metric")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many evaluations without a better checkpoint (0: never)")
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    ap.add_argument("--readout-lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--risk-weight", type=float, default=0.5)
    ap.add_argument("--max-state-tokens", type=int, default=1536)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--save-every", type=int, default=200, help="steps between resume checkpoints")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ["D3_FAST"] = "0"  # ROCm-only inference path; training uses the plain one

    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from torch.utils.checkpoint import checkpoint

    root = Path(snapshot_download(REPO, revision=D3_REVISION, ignore_patterns=["assets/*"]))
    sys.path.insert(0, str(root))
    from d3_runtime import D3, MAX_OPTIONS, sdpa_backends

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    if (out / "train-summary.json").exists() and (out / "package").exists():
        log.info(f"{out} is already trained and exported; nothing to do (delete it to retrain)")
        return
    out.mkdir(parents=True, exist_ok=True)
    d3 = D3.from_pretrained(str(root), device="cuda:0", verify="fast")
    if d3.encoder is None:
        raise SystemExit("expected a noncausal_full_attention checkpoint")
    dev = d3.device

    train_recs = [json.loads(l) for l in open(args.train)]
    val_recs = [json.loads(l) for l in open(args.val)]
    t0 = time.time()
    train, cut_t = encode_examples(d3, train_recs, args.max_state_tokens)
    val, cut_v = encode_examples(d3, val_recs, args.max_state_tokens)
    log.info(f"{len(train)} train / {len(val)} val records, tokenized in {time.time() - t0:.0f}s; "
             f"truncated to {args.max_state_tokens} tokens: {cut_t} train, {cut_v} val texts; "
             f"with a risk target: {sum(e['risk_seq'] is not None for e in train)} train")

    lm = d3.backbone.language_model
    d3.backbone.requires_grad_(False)
    peft_lm = get_peft_model(lm, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                            target_modules=LORA_TARGETS, bias="none"))
    readout = torch.nn.Parameter(d3.readout.detach().clone().float())
    lora_params = [p for p in peft_lm.parameters() if p.requires_grad]
    log.info(f"trainable: LoRA {sum(p.numel() for p in lora_params):,} + readout {readout.numel():,}")
    embed = d3.backbone.get_input_embeddings()

    def last_hidden(seqs):
        """The noncausal prompt encoder of d3_runtime.PromptEncoder, with each layer checkpointed."""
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), d3.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s):] = torch.as_tensor(s)
            mask[i, width - len(s):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        enc = d3.encoder
        hidden = embed(ids)
        ramp = torch.arange(width, device=dev).view(1, 1, -1).expand(4, len(seqs), -1)
        text_positions, mrope = ramp[0], ramp[1:]
        rope = lm.rotary_emb(hidden, mrope)
        keys = mask.bool()[:, None, None, :]
        scan = enc.recurrent_mask(config=lm.config, inputs_embeds=hidden, attention_mask=mask,
                                  past_key_values=None, position_ids=text_positions)
        for layer, softmax in enc.stack:
            kw = dict(position_embeddings=rope, attention_mask=keys if softmax else scan,
                      position_ids=text_positions, past_key_values=None, use_cache=False)
            if torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, use_reentrant=False, **kw)
            else:
                hidden = layer(hidden, **kw)
        return lm.norm(hidden)[:, -1]

    def logits(seqs, counts):
        z = last_hidden(seqs).float() @ readout.T
        invalid = torch.arange(MAX_OPTIONS, device=dev)[None] >= torch.as_tensor(counts, device=dev)[:, None]
        return z.masked_fill(invalid, float("-inf")) / d3.temperature

    def step_loss(batch):
        """(total loss, mean block loss, per-row block losses, block logits). Rows without a risk target add no
        risk loss."""
        n = len(batch)
        with_risk = [e for e in batch if e["risk_seq"] is not None]
        z = logits([e["seq"] for e in batch] + [e["risk_seq"] for e in with_risk],
                   [e["options"] for e in batch] + [e["risk_options"] for e in with_risk])
        tb = torch.as_tensor([e["target"] for e in batch], device=dev)
        rows = torch.nn.functional.cross_entropy(z[:n], tb, reduction="none")
        lb = rows.mean()
        loss = lb
        if with_risk:
            tr = torch.as_tensor([e["risk_target"] for e in with_risk], device=dev)
            loss = lb + args.risk_weight * torch.nn.functional.cross_entropy(z[n:], tr) * len(with_risk) / n
        return loss, lb, rows, z[:n]

    def evaluate():
        """Per validation source: block loss, recall, false-block rate; the selection metric is the mean of the
        per-source block losses. Also the block probability of every validation row (in val order)."""
        peft_lm.eval()
        loss, probs = [0.0] * len(val), [0.0] * len(val)
        with torch.no_grad(), sdpa_backends(dev):
            for b in batches(val, max(1, args.batch_size * 2 // args.grad_accum), rng, shuffle=False):
                _, _, rows, zb = step_loss([val[i] for i in b])
                p_block = torch.softmax(zb, dim=-1)[:, 1].tolist()  # noul keys: false, true
                for i, l, p in zip(b, rows.tolist(), p_block):
                    loss[i], probs[i] = l, p
        peft_lm.train()
        per = {}
        for s in sorted({e["source"] for e in val}):
            idx = [i for i, e in enumerate(val) if e["source"] == s]
            pos = [i for i in idx if val[i]["block"]]
            neg = [i for i in idx if not val[i]["block"]]
            recall = sum(probs[i] >= 0.5 for i in pos) / len(pos) if pos else None
            fbr = sum(probs[i] >= 0.5 for i in neg) / len(neg) if neg else None
            parts = [x for x in (recall, None if fbr is None else 1 - fbr) if x is not None]
            per[s] = {"block_loss": round(sum(loss[i] for i in idx) / len(idx), 4), "score": round(harmonic(parts), 4),
                      "recall": None if recall is None else round(recall, 4),
                      "false_block_rate": None if fbr is None else round(fbr, 4)}
        return {"block_loss": sum(m["block_loss"] for m in per.values()) / len(per),
                "score": harmonic([m["score"] for m in per.values()]), "sources": per}, probs

    def better(m, best):
        if best is None:
            return True
        if args.select == "loss":
            return m["block_loss"] < best["block_loss"]
        return (m["score"], -m["block_loss"]) > (best["score"], -best["block_loss"])

    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr},
                             {"params": [readout], "lr": args.readout_lr}], weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    warmup = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    # Parity: before any update, the training forward pass must give d3's own inference logits (block, risk and
    # response questions, when the data has them).
    with torch.no_grad(), sdpa_backends(dev):
        both = val + train
        probe = ([e for e in both if e["risk_seq"] is not None][:2] + [e for e in both if e["risk_seq"] is None][:2])
        seqs = [e["seq"] for e in probe] + [e["risk_seq"] for e in probe if e["risk_seq"] is not None]
        counts = [e["options"] for e in probe] + [e["risk_options"] for e in probe if e["risk_seq"] is not None]
        ours = torch.softmax(logits(seqs, counts), dim=-1)
        theirs = torch.softmax(d3.logits(seqs, counts) / d3.temperature, dim=-1)
        gap = (ours - theirs).abs().max().item()
    if gap > 0.02:
        raise SystemExit(f"training forward differs from d3 inference by {gap:.4f} in probability")
    log.info(f"parity with d3 inference OK (max probability gap {gap:.5f})")

    # Preflight: the longest batch first, so an out-of-memory error comes now and not hours in.
    micro = math.ceil(args.batch_size / args.grad_accum)
    longest = sorted(train, key=lambda e: -length(e))[:micro]
    torch.cuda.reset_peak_memory_stats()
    with sdpa_backends(dev):
        step_loss(longest)[0].backward()
    opt.zero_grad(set_to_none=True)
    log.info(f"preflight OK: longest batch ({length(longest[0])} tokens) peaked at "
             f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    # Resumable: the full batch order is fixed by --seed, and a checkpoint (LoRA, readout, optimizer, scheduler,
    # step, best state, history) is written atomically every --save-every steps and at every evaluation.
    order, r = [], random.Random(args.seed)
    while len(order) < total:
        order += list(batches(train, args.batch_size, r))
    order = order[:total]
    ckpt = out / "resume.pt"
    best, best_state, best_probs, history, step, minutes_before = None, None, None, [], 0, 0.0
    if ckpt.exists():
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        if c["args"] != {k: v for k, v in vars(args).items() if k not in ("max_steps", "eval_every", "save_every", "patience")}:
            raise SystemExit(f"{ckpt} was written with different arguments; delete it to start over")
        peft_lm.load_state_dict(c["lora"], strict=False)
        readout.data.copy_(c["readout"].to(dev))
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        best, best_state, history, step, minutes_before = c["best"], c["best_state"], c["history"], c["step"], c["minutes"]
        best_probs = c.get("best_probs")
        log.info(f"resumed from {ckpt} at step {step}/{total} (best so far {best})")

    def save_checkpoint():
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"args": {k: v for k, v in vars(args).items() if k not in ("max_steps", "eval_every", "save_every", "patience")},
                    "lora": {k: v.detach().cpu() for k, v in peft_lm.state_dict().items() if "lora_" in k},
                    "readout": readout.detach().cpu(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "best_state": best_state, "best_probs": best_probs, "history": history, "step": step,
                    "minutes": minutes_before + (time.time() - started) / 60}, tmp)
        os.replace(tmp, ckpt)

    peft_lm.train()
    started, first = time.time(), step
    while step < total:
        b = order[step]
        for j in range(0, len(b), micro):
            part = [train[i] for i in b[j:j + micro]]
            with sdpa_backends(dev):
                loss, lb, _, _ = step_loss(part)
                (loss * len(part) / len(b)).backward()
        torch.nn.utils.clip_grad_norm_(lora_params + [readout], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            rate = (step - first) * args.batch_size / (time.time() - started)
            log.info(f"step {step}/{total} loss {loss.item():.4f} block {lb.item():.4f} "
                     f"{rate:.1f} records/s, eta {(total - step) * args.batch_size / rate / 60:.0f} min")
        if step % args.eval_every == 0 or step == total:
            m, probs = evaluate()
            m = {"step": step, **m}
            history.append(m)
            log.info(f"val {json.dumps(m)}")
            if better(m, best):
                best, best_probs = m, probs
                best_state = ({k: v.detach().cpu().clone() for k, v in peft_lm.state_dict().items()
                               if "lora_" in k}, readout.detach().cpu().clone())
        if step % args.save_every == 0 or step % args.eval_every == 0 or step == total:
            save_checkpoint()
        if (args.patience and step % args.eval_every == 0 and best is not None
                and len(history) - 1 - history.index(best) >= args.patience):
            log.info(f"no better checkpoint in {args.patience} evaluations; stopping at step {step}")
            break

    log.info(f"best {best}; merging and exporting")
    peft_lm.load_state_dict(best_state[0], strict=False)
    readout.data.copy_(best_state[1].to(dev))
    peft_lm.merge_and_unload()

    import shutil

    from huggingface_hub import save_torch_state_dict
    from safetensors.torch import save_file

    pkg = out / "package"
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(root, pkg, ignore=shutil.ignore_patterns("model-*.safetensors", "assets", *WEIGHT_FILES))
    state = {k: v.contiguous() for k, v in d3.backbone.state_dict().items()}
    save_torch_state_dict(state, str(pkg), filename_pattern="model{suffix}.safetensors", max_shard_size="5GB")
    save_file({"weight": readout.detach().to(torch.float32).cpu().contiguous()}, str(pkg / "readout.safetensors"))
    summary = {"base": REPO, "train_examples": len(train), "val_examples": len(val), "steps": step,
               "batch_size": args.batch_size, "grad_accum": args.grad_accum, "select": args.select, "lr": args.lr,
               "readout_lr": args.readout_lr, "lora_r": args.lora_r,
               "risk_weight": args.risk_weight, "max_state_tokens": args.max_state_tokens,
               "minutes": round(minutes_before + (time.time() - started) / 60, 1), "best": best, "history": history,
               "package": str(pkg)}
    with open(out / "val_preds.jsonl", "w") as fh:
        for rec, e, p in zip(val_recs, val, best_probs):
            n = normalise(rec)
            fh.write(json.dumps({"source": e["source"], "mode": n["mode"], "question": n["question"], "block": e["block"],
                                 "p_block": round(p, 6)}) + "\n")
    (out / "train-summary.json").write_text(json.dumps(summary, indent=2))
    ckpt.unlink()  # the package is complete; the checkpoint (~0.4 GB) is no longer needed
    log.info(f"wrote {pkg}")


if __name__ == "__main__":
    main()
