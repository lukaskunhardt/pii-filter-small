"""Label the generated texts with the teacher, openai/privacy-filter.

For every text it stores
  spans   the teacher's spans, read exactly like evaluate.py reads any model (arg-max per token, windows of 512 tokens)
  probs   per character the teacher's distribution over 9 classes (O and the 8 categories, B/I/E/S summed), uint8 x255, base64
The student has a different tokenizer, so it learns from character-level targets: hard labels from the spans and the soft
category distribution. About 5 % of the texts become the development split (hash of the id), used only to pick checkpoints.

  python label.py --texts texts.jsonl --out labels.jsonl     # one GPU; about 24,000 texts
"""
import argparse
import base64
import hashlib
import json

import numpy as np
import torch

import evaluate

TEACHER = "openai/privacy-filter"
CATS = ["O", "account_number", "private_address", "private_date", "private_email", "private_person", "private_phone", "private_url", "secret"]


def is_dev(text_id):
    return int(hashlib.md5(("dev:" + text_id).encode()).hexdigest(), 16) % 100 < 5


def to_categories(labels):
    """Matrix that sums the 33 BIOES labels into the 9 classes."""
    m = torch.zeros(len(labels), len(CATS))
    for j, label in enumerate(labels):
        m[j, CATS.index(label.split("-", 1)[1] if "-" in label else "O")] = 1
    return m


def label_text(text, tok, labels, run, cat_matrix):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    logits = evaluate.token_logits(run, tok, enc["input_ids"]) if enc["input_ids"] else []
    probs = np.zeros((len(text), len(CATS)), dtype=np.float32)
    probs[:, 0] = 1.0
    for (a, b), x in zip(enc["offset_mapping"], logits):
        if b > a:
            probs[a:b] = (x.softmax(-1) @ cat_matrix).numpy()
    spans = evaluate.decode(text, enc["offset_mapping"], [labels[int(x.argmax())] for x in logits])
    q = np.clip(np.round(probs * 255), 0, 255).astype(np.uint8)
    return [list(s) for s in spans], base64.b64encode(q.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="output of make_texts.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, labels, _ = evaluate.load_model(TEACHER, device="cpu")
    from transformers import AutoModelForTokenClassification
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForTokenClassification.from_pretrained(TEACHER, torch_dtype=dtype).to(device).eval()
    # one window per forward pass, no padding: a batched run with padding changed the spans of 11 of 30 texts
    run = lambda x: model(input_ids=x.to(device), attention_mask=torch.ones_like(x).to(device)).logits[0]
    cat_matrix = to_categories(labels)

    texts = [json.loads(line) for line in open(a.texts, encoding="utf-8")]
    with open(a.out, "w", encoding="utf-8") as fh:
        for i, t in enumerate(texts):
            spans, probs = label_text(t["text"], tok, labels, run, cat_matrix)
            fh.write(json.dumps({"id": t["id"], "lang": t["lang"], "neg": t["neg"], "split": "dev" if is_dev(t["id"]) else "train",
                                 "text": t["text"], "spans": spans, "probs": probs}, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                print(f"{i}/{len(texts)}", flush=True)


if __name__ == "__main__":
    main()
