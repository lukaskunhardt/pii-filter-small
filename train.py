"""Train a small multilingual student with exactly the 33 BIOES labels of openai/privacy-filter (same id2label).

Targets per student token, from the teacher labels of label.py:
  hard   BIOES label from the teacher's spans (first token of a span B, last E, single S, else I; O outside)
  soft   mean over the token's characters of the teacher's 9-class distribution (O and the 8 categories)
loss = cross-entropy(hard, 33 labels) + alpha * KL(soft || student's 33 probabilities summed per category)
The checkpoint with the best character F1 against the teacher on the development split is kept; no test data is used.

  python train.py --labels labels.jsonl --seed 43 --out model      # one NVIDIA A10G, about 21 minutes
"""
import argparse
import base64
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

import evaluate
from label import CATS


def bioes_labels():
    """The teacher's 33 labels, read from its config file (works without the teacher's model code)."""
    cfg = json.load(open(hf_hub_download("openai/privacy-filter", "config.json"), encoding="utf-8"))
    id2label = {i: cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))}
    return id2label, {l: i for i, l in id2label.items()}


def targets(r, tok, label2id):
    """Student token ids, hard label ids (-100 for tokens without characters) and soft 9-class targets."""
    text = r["text"]
    probs = np.frombuffer(base64.b64decode(r["probs"]), dtype=np.uint8).reshape(len(text), len(CATS)).astype(np.float32) / 255
    span_of = np.full(len(text), -1)
    for k, (s, e, _) in enumerate(r["spans"]):
        span_of[s:e] = k
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    real = [i for i, (s, e) in enumerate(offs) if e > s]
    prev_real = {i: (real[n - 1] if n else None) for n, i in enumerate(real)}
    next_real = {i: (real[n + 1] if n + 1 < len(real) else None) for n, i in enumerate(real)}
    in_span = lambda j, k: j is not None and any(span_of[x] == k for x in range(*offs[j]))
    hard, soft = [], []
    for i, (s, e) in enumerate(offs):
        if e <= s:
            hard.append(-100)
            soft.append(np.eye(len(CATS))[0])
            continue
        soft.append(probs[s:e].mean(0))
        ks = [k for k in span_of[s:e] if k >= 0]
        if not ks:
            hard.append(label2id["O"])
            continue
        k = max(set(ks), key=ks.count)  # a token that touches two spans belongs to the one with more of its characters
        first, last = not in_span(prev_real[i], k), not in_span(next_real[i], k)
        prefix = "S" if first and last else "B" if first else "E" if last else "I"
        hard.append(label2id[prefix + "-" + r["spans"][k][2]])
    return ids, hard, soft


def windows(ids, hard, soft, tok, max_len):
    """Cut a text into windows of max_len tokens, without overlap, with the special tokens around each."""
    body, out, none = max_len - 2, [], np.eye(len(CATS))[0]
    for s0 in range(0, max(1, len(ids)), body):
        out.append(([tok.cls_token_id] + ids[s0:s0 + body] + [tok.sep_token_id], [-100] + hard[s0:s0 + body] + [-100],
                    np.stack([none] + soft[s0:s0 + body] + [none])))
    return out


def batches(train, batch, pad_id, device):
    order = list(range(len(train)))
    random.shuffle(order)
    for b in range(0, len(order), batch):
        part = [train[i] for i in order[b:b + batch]]
        w = max(len(p[0]) for p in part)
        x = torch.full((len(part), w), pad_id)
        y = torch.full((len(part), w), -100)
        q = torch.zeros(len(part), w, len(CATS))
        for r, (ids, hard, soft) in enumerate(part):
            x[r, :len(ids)], y[r, :len(hard)], q[r, :len(ids)] = torch.tensor(ids), torch.tensor(hard), torch.tensor(soft)
        yield x.to(device), (x != pad_id).long().to(device), y.to(device), q.to(device)


def dev_char_f1(model, tok, labels, dev_rows, device):
    model.eval()
    run = lambda x: model(input_ids=x.to(device), attention_mask=torch.ones_like(x).to(device)).logits[0]
    rows = []
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for r in dev_rows:
            doc = {"text": r["text"], "gold": sorted({tuple(s) for s in r["spans"]}), "ignore": []}
            rows.append((("dev", r["lang"]), evaluate.score(doc, evaluate.read(r["text"], tok, labels, run))))
    model.train()
    return evaluate.summarize(rows)["dev/all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="output of label.py")
    ap.add_argument("--base", default="jhu-clsp/mmBERT-small")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--evals", type=int, default=6, help="development checks spread evenly over training")
    ap.add_argument("--out", default="model")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    id2label, label2id = bioes_labels()
    labels = [id2label[i] for i in range(len(id2label))]
    group = torch.zeros(len(labels), len(CATS))
    for i, l in enumerate(labels):
        group[i, CATS.index(l.split("-", 1)[1] if "-" in l else "O")] = 1
    group = group.to(device)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForTokenClassification.from_pretrained(a.base, num_labels=len(labels), id2label=id2label, label2id=label2id).to(device)

    rows = [json.loads(line) for line in open(a.labels, encoding="utf-8")]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    train = [w for r in rows if r["split"] == "train" for w in windows(*targets(r, tok, label2id), tok, a.max_len)]
    print(f"train windows {len(train)}, dev texts {len(dev_rows)}", flush=True)

    total = int(math.ceil(len(train) / a.batch) * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    eval_at = {int(total * (k + 1) / a.evals) for k in range(a.evals)}
    best, history, step, t0 = -1.0, [], 0, time.time()
    model.train()
    while step < total:
        for x, att, y, q in batches(train, a.batch, tok.pad_token_id, device):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(input_ids=x, attention_mask=att).logits.float()
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            p_cat = (logits.softmax(-1) @ group).clamp_min(1e-8)
            kl = (q * (q.clamp_min(1e-8).log() - p_cat.log())).sum(-1)[att.bool()].mean()
            loss = ce + a.alpha * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} ce {ce.item():.4f} kl {kl.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step in eval_at:
                dev = dev_char_f1(model, tok, labels, dev_rows, device)
                history.append({"step": step, "dev_char_f1": dev["char_f1"], "dev_span_f1": dev["span_f1"]})
                print(f"step {step}: dev char F1 {dev['char_f1']} span F1 {dev['span_f1']}", flush=True)
                if dev["char_f1"] > best:
                    best = dev["char_f1"]
                    model.save_pretrained(a.out); tok.save_pretrained(a.out)
            if step >= total:
                break
    result = {"args": vars(a), "best_dev_char_f1": best, "history": history, "seconds": round(time.time() - t0)}
    json.dump(result, open(f"{a.out}/train_result.json", "w"), indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
