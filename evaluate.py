"""Measure a PII token classifier on four public test samples, with the same reading and scoring for every model.

Test samples (random seed 0, the same documents for every model):
  ai4privacy  ai4privacy/pii-masking-300k validation, 200 per language (6 languages); academic licence, so test only
  nemotron    nvidia/Nemotron-PII test, 500 US and 500 international documents (English)
  gretel      gretelai/synthetic_pii_finance_multilingual test, 150 per language (7 languages)
  kiji        DataikuNLP/kiji-pii-training-data test, 150 per language (6 languages); spans found by searching each value in the text
Gold labels map to the 8 categories of openai/privacy-filter by mapping.json. Unmapped gold spans are ignored.

Metrics per set and language:
  span F1   typed spans, exact start and end after trimming whitespace
  char F1   binary: is this character personal information (any category)

  python evaluate.py --model LukasKunhardt/pii-filter-small --out results/mine.json
  python evaluate.py --model LukasKunhardt/pii-filter-small --onnx onnx/model_int8.onnx --out results/mine_int8.json
  python evaluate.py --model openai/privacy-filter --out results/pf_transformers.json
  python evaluate.py --opf --out results/pf_opf.json          # privacy-filter with its official decoder (pip install opf)
  python evaluate.py --model ... --speed                       # CPU ms per 1,000 characters on 50 Nemotron documents
"""
import argparse
import ast
import json
import os
import random
import re
import time

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MAPPING = json.load(open(os.path.join(HERE, "mapping.json"), encoding="utf-8"))
SETS = ["ai4privacy", "nemotron", "kiji", "gretel"]

# ---------------------------------------------------------------------------------------------------------------------------------
# test samples


def parse_list(x):
    if not isinstance(x, str):
        return list(x)
    try:
        return json.loads(x)
    except ValueError:
        return ast.literal_eval(x)


def make_doc(set_name, lang, text, spans):
    """spans: (start, end, source label). Mapped spans become gold, unmapped spans become regions to ignore."""
    gold, ignore = [], []
    for s, e, label in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e <= s:
            continue
        cat = MAPPING[set_name].get(label)
        if cat:
            gold.append((s, e, cat))
        else:
            ignore.append((s, e))
    return {"set": set_name, "lang": lang, "text": text, "gold": sorted(set(gold)), "ignore": sorted(set(ignore))}


def take(rows, k, rng):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:k]


def load_test(scale=1.0):
    """The four test samples. One random generator in a fixed order, so every run draws the same documents."""
    import pandas as pd
    rng, docs = random.Random(0), []
    get = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")

    repo = "ai4privacy/pii-masking-300k"
    files = sorted(s.rfilename for s in HfApi().dataset_info(repo).siblings if s.rfilename.startswith("data/validation/"))
    for f in files:
        rows = [json.loads(line) for line in open(get(repo, f), encoding="utf-8")]
        for r in take(rows, int(200 * scale), rng):
            spans = [(m["start"], m["end"], m["label"]) for m in r["privacy_mask"]]
            docs.append(make_doc("ai4privacy", r["language"], r["source_text"], spans))

    df = pd.read_parquet(get("nvidia/Nemotron-PII", "data/test-00000-of-00001.parquet"))
    for locale in ("us", "intl"):
        for _, r in take(df[df.locale == locale].iterrows(), int(500 * scale), rng):
            spans = [(s["start"], s["end"], s["label"]) for s in parse_list(r["spans"])]
            docs.append(make_doc("nemotron", "English", r["text"], spans))

    repo = "gretelai/synthetic_pii_finance_multilingual"
    files = sorted(s.rfilename for s in HfApi().dataset_info(repo).siblings
                   if s.rfilename.endswith("_test-00000-of-00001.parquet") and "combined" not in s.rfilename)
    for f in files:
        for _, r in take(pd.read_parquet(get(repo, f)).iterrows(), int(150 * scale), rng):
            spans = [(s["start"], s["end"], s["label"]) for s in parse_list(r["pii_spans"])]
            docs.append(make_doc("gretel", r["language"], r["generated_text"], spans))

    # kiji stores values, not offsets: every occurrence of a value counts as a span
    df = pd.read_parquet(get("DataikuNLP/kiji-pii-training-data", "data/test-00000-of-00001.parquet"))
    for lang in sorted(df.language.unique()):
        for _, r in take(df[df.language == lang].iterrows(), int(150 * scale), rng):
            spans = [(x.start(), x.end(), m["label"]) for m in parse_list(r["privacy_mask"]) if m["value"]
                     for x in re.finditer(re.escape(m["value"]), r["text"])]
            docs.append(make_doc("kiji", lang, r["text"], spans))
    return docs

# ---------------------------------------------------------------------------------------------------------------------------------
# reading a model


def load_model(name, onnx_file=None, device="cpu", threads=None):
    """Returns (tokenizer, labels, run). run(input_ids [1, n] tensor) -> logits [n, labels] tensor."""
    tok = AutoTokenizer.from_pretrained(name)
    cfg = AutoConfig.from_pretrained(name)
    labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
    if onnx_file:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        if not os.path.exists(onnx_file):
            onnx_file = hf_hub_download(name, onnx_file)
        session = ort.InferenceSession(onnx_file, opts, providers=["CPUExecutionProvider"])

        def run(x):
            ids = x.cpu().numpy().astype(np.int64)
            return torch.from_numpy(session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0])
    else:
        model = AutoModelForTokenClassification.from_pretrained(name).to(device).eval()

        def run(x):
            return model(input_ids=x.to(device), attention_mask=torch.ones_like(x).to(device)).logits[0]
    return tok, labels, run


@torch.no_grad()
def token_logits(run, tok, ids, max_len=512, stride=128):
    """Logits per token over windows of max_len tokens that overlap by stride. Each token takes its logits from the window
    where it is farthest from the edge."""
    pre = [x for x in [tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id] if x is not None]
    post = [x for x in [tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id] if x is not None]
    body = max_len - len(pre) - len(post)
    best = {}  # token index -> (distance to the window edge, logits)
    start = 0
    while True:
        window = ids[start:start + body]
        logits = run(torch.tensor([pre + window + post]))[len(pre):len(pre) + len(window)].float().cpu()
        for j in range(len(window)):
            d = min(j, len(window) - 1 - j)
            if start + j not in best or d > best[start + j][0]:
                best[start + j] = (d, logits[j])
        if start + body >= len(ids):
            break
        start += body - stride
    return [best[i][1] for i in range(len(ids))]


def decode(text, offsets, token_labels):
    """BIOES labels per token -> (start, end, category). A span starts at B- or S- or at a change of category, ends at E-, S- or O."""
    spans, cur = [], None
    for (a, b), label in zip(offsets, token_labels):
        if b <= a:
            continue
        if label == "O":
            cur = None
            continue
        prefix, cat = label.split("-", 1) if "-" in label else ("I", label)
        if cur is None or cur[2] != cat or prefix in ("B", "S"):
            cur = [a, b, cat]
            spans.append(cur)
        else:
            cur[1] = b
        if prefix in ("E", "S"):
            cur = None
    out = []
    for a, b, c in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            out.append((a, b, c))
    return out


def read(text, tok, labels, run):
    """Predicted spans of one text: arg-max label per token, then decode."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    logits = token_logits(run, tok, enc["input_ids"]) if enc["input_ids"] else []
    return decode(text, enc["offset_mapping"], [labels[int(x.argmax())] for x in logits])

# ---------------------------------------------------------------------------------------------------------------------------------
# scores


def score(doc, pred):
    gold, ignore = set(doc["gold"]), doc["ignore"]
    # a prediction that only touches ignored (unmapped) gold is neither right nor wrong
    keep = [(a, b, c) for a, b, c in pred
            if any(a < e and s < b for s, e, _ in gold) or not any(a < e and s < b for s, e in ignore)]
    tp = len(gold & set(keep))
    gold_chars = {i for s, e, _ in gold for i in range(s, e)}
    ignore_chars = {i for s, e in ignore for i in range(s, e)} - gold_chars
    pred_chars = {i for a, b, _ in keep for i in range(a, b)} - ignore_chars
    return {"tp": tp, "fp": len(keep) - tp, "fn": len(gold) - tp, "ctp": len(gold_chars & pred_chars),
            "cfp": len(pred_chars - gold_chars), "cfn": len(gold_chars - pred_chars)}


def f1(tp, fp, fn):
    return round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0


def summarize(rows):
    """rows: ((set, language), counts) -> metrics per set/language and per set/all."""
    agg = {}
    for key, counts in rows:
        for k in (key, (key[0], "all")):
            a = agg.setdefault(k, {"tp": 0, "fp": 0, "fn": 0, "ctp": 0, "cfp": 0, "cfn": 0, "n": 0})
            for x in counts:
                a[x] += counts[x]
            a["n"] += 1
    return {f"{s}/{l}": {"n": a["n"], "span_f1": f1(a["tp"], a["fp"], a["fn"]), "char_f1": f1(a["ctp"], a["cfp"], a["cfn"]),
                         "span_p": round(a["tp"] / max(1, a["tp"] + a["fp"]), 4), "span_r": round(a["tp"] / max(1, a["tp"] + a["fn"]), 4)}
            for (s, l), a in sorted(agg.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Hub id or folder of a token classifier (also the tokenizer for --onnx)")
    ap.add_argument("--onnx", help="ONNX file (path, or file name in the --model repo)")
    ap.add_argument("--opf", action="store_true", help="openai/privacy-filter through its official package, Viterbi decoder")
    ap.add_argument("--speed", action="store_true", help="only CPU time per 1,000 characters on 50 Nemotron documents")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()

    docs = load_test()
    device = "cuda" if torch.cuda.is_available() and not (a.speed or a.onnx) else "cpu"
    torch.set_num_threads(a.threads)
    if a.opf:
        from opf import OPF
        model = OPF(device=device, decode_mode="viterbi")
        # opf returns its spans untrimmed; the scorer takes them as they are
        predict = lambda text: [(s.start, s.end, s.label) for s in model.redact(text).detected_spans]
    else:
        tok, labels, run = load_model(a.model, a.onnx, device, a.threads)
        predict = lambda text: read(text, tok, labels, run)

    if a.speed:
        sub = [d for d in docs if d["set"] == "nemotron"][:50]
        predict(sub[0]["text"])  # warm-up
        t0 = time.time()
        for d in sub:
            predict(d["text"])
        res = {"cpu_ms_per_1k_chars": round(1000 * (time.time() - t0) / (sum(len(d["text"]) for d in sub) / 1000), 1),
               "threads": a.threads, "docs": len(sub)}
    else:
        res = summarize([((d["set"], d["lang"]), score(d, predict(d["text"]))) for d in docs])
        res["mean"] = {m: round(float(np.mean([res[f"{s}/all"][m] for s in SETS])), 4) for m in ("char_f1", "span_f1")}
        for s in SETS:
            print(f"{s:12s} char F1 {res[s + '/all']['char_f1']:.4f}  span F1 {res[s + '/all']['span_f1']:.4f}")
    print(json.dumps(res["mean"] if "mean" in res else res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
