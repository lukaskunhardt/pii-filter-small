"""Check that this repo gives the same results as the research code that produced the released numbers.

  python -I tests/test_equivalence.py --research <research repo> --tokenizer <folder of the released model> --texts <generated texts .jsonl>

1. Test samples: evaluate.load_test draws the same documents with the same gold and ignored spans as the research sampler.
2. Scores: evaluate.score and summarize give the same numbers as the research scorer, for gold as prediction and for
   several kinds of damaged predictions (shifted boundaries, wrong categories, dropped spans, random spans).
3. Reading: with a fake model (logits from token id and window position), evaluate.read gives the same spans as the research
   reader, also for texts longer than one window.
4. Teacher labels: label.py gives the same spans as the research reader with the same fake model.
5. Training targets: train.py builds the same token ids, hard labels and soft targets as the research training script.
No model is trained or run; the scores in step 2 use no model output.
"""
import argparse
import ast
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import evaluate  # noqa: E402
import label  # noqa: E402
import train  # noqa: E402


def fingerprint(d):
    return hashlib.md5(json.dumps([d["set"], d["lang"], d["text"], list(map(list, d["gold"])), list(map(list, d["ignore"]))]).encode()).hexdigest()


def damaged(d, rng):
    """A few kinds of wrong predictions, built from the gold spans."""
    cats = ["account_number", "private_address", "private_date", "private_email", "private_person", "private_phone", "private_url", "secret"]
    n = len(d["text"])
    out = []
    for s, e, c in d["gold"] + [(s, e, "private_person") for s, e in d["ignore"]]:
        r = rng.random()
        if r < 0.2:
            continue
        if r < 0.4:
            s, e = max(0, s + rng.randint(-3, 3)), min(n, e + rng.randint(-3, 3))
        elif r < 0.5:
            c = rng.choice(cats)
        if e > s:
            out.append((s, e, c))
    for _ in range(rng.randint(0, 3)):
        s = rng.randrange(max(1, n))
        out.append((s, min(n, s + rng.randint(1, 15)), rng.choice(cats)))
    return out


class FakeModel:
    """Logits = table[token id] + table[position in window]; enough to make window choice and decoding matter."""
    def __init__(self, n_labels, seed):
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.randn(300_000, n_labels, generator=g)
        self.pos = torch.randn(512, n_labels, generator=g) * 2
        self.ids[:, 0] += 1.5  # mostly O, as in real text

    def __call__(self, input_ids, attention_mask=None):
        x = input_ids[0]
        return type("Out", (), {"logits": (self.ids[x] + self.pos[: len(x)])[None]})()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--n-texts", type=int, default=60)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.research, "pii"))
    import stufe1
    from transformers import AutoTokenizer

    ok = True

    # 1. samples
    mine = evaluate.load_test()
    theirs = stufe1.load(1.0)
    same = len(mine) == len(theirs) and all(fingerprint(x) == fingerprint(y) for x, y in zip(mine, theirs))
    print(f"1 samples: {len(mine)} documents, identical: {same}")
    ok &= same
    del theirs

    # 2. scores
    rng = random.Random(1)
    preds = {"gold": [[tuple(g) for g in d["gold"]] for d in mine]}
    for k in range(3):
        preds[f"damaged{k}"] = [damaged(d, rng) for d in mine]
    for name, p in preds.items():
        a1 = evaluate.summarize([((d["set"], d["lang"]), evaluate.score(d, x)) for d, x in zip(mine, p)])
        a2 = stufe1.summarize([((d["set"], d["lang"]), stufe1.score(d, x)) for d, x in zip(mine, p)])
        print(f"2 scores ({name}): {len(a1)} rows, identical: {a1 == a2}")
        ok &= a1 == a2
    del mine, preds

    # 3. reading, 4. teacher labels
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    labels = list(train.bioes_labels()[0].values())
    texts = [json.loads(line)["text"] for _, line in zip(range(a.n_texts), open(a.texts, encoding="utf-8"))]
    texts.append("\n\n".join(texts[:12]))  # longer than one window of 512 tokens
    fake = FakeModel(len(labels), seed=0)
    research = stufe1.Reader.__new__(stufe1.Reader)
    research.tok, research.model, research.device, research.labels = tok, fake, "cpu", labels
    run = lambda x: fake(x).logits[0]
    cat_matrix = label.to_categories(labels)
    n_tokens = len(tok(texts[-1], add_special_tokens=False)["input_ids"])
    same_read = same_label = n_spans = 0
    rows = []
    for i, t in enumerate(texts):
        ref = research.spans(t)
        same_read += evaluate.read(t, tok, labels, run) == ref
        spans, probs = label.label_text(t, tok, labels, run, cat_matrix)
        same_label += [tuple(s) for s in spans] == ref
        n_spans += len(ref)
        rows.append({"id": f"x{i}", "lang": "en", "text": t, "spans": spans, "probs": probs})
    print(f"3 reading: {same_read}/{len(texts)} texts give identical spans ({n_spans} spans; longest text {n_tokens} tokens)")
    print(f"4 teacher labels: {same_label}/{len(texts)} texts give identical spans")
    ok &= same_read == len(texts) and same_label == len(texts)

    # 5. training targets: run the research function `examples` from its source with the same names
    src = open(os.path.join(a.research, "pii", "train.py"), encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "examples")
    _, label2id = train.bioes_labels()
    scope = {"np": np, "tok": tok, "label2id": label2id, "CATS": label.CATS, "base64": __import__("base64"),
             "a": argparse.Namespace(max_len=512)}
    exec(compile(ast.Module([node], []), "research_train", "exec"), scope)
    same_t = n_hard = 0
    for r in rows:
        ref = scope["examples"](r)
        got = train.windows(*train.targets(r, tok, label2id), tok, 512)
        same_t += len(ref) == len(got) and all(x[0] == y[0] and x[1] == y[1] and np.array_equal(x[2], y[2]) for x, y in zip(ref, got))
        n_hard += sum(h not in (-100, label2id["O"]) for w in ref for h in w[1])
    print(f"5 training targets: {same_t}/{len(rows)} texts identical (ids, hard labels, soft targets; {n_hard} non-O hard labels)")
    ok &= same_t == len(rows)

    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
