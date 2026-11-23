"""Export a trained student to the two ONNX files of the Hub repo and compare them with PyTorch on the development split.

  python export_onnx.py --model model --out onnx --check labels.jsonl
Writes onnx/model.onnx (fp32) and onnx/model_quantized.onnx (int8 embedding table, all other weights fp32). These are the file
names transformers.js looks for, so `dtype: "q8"` there loads the int8 file.
"""
import argparse
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoModelForTokenClassification, AutoTokenizer



class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model_dir, out):
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, attn_implementation="eager").eval()
    x = tok(["Herr Max Muster, max@example.com", "x"], padding=True, return_tensors="pt")
    dyn = {n: {0: "batch", 1: "tokens"} for n in ("input_ids", "attention_mask", "logits")}
    with torch.no_grad():
        torch.onnx.export(Logits(model), (x["input_ids"], x["attention_mask"]), f"{out}/model.onnx", input_names=["input_ids", "attention_mask"],
                          output_names=["logits"], dynamic_axes=dyn, opset_version=17, dynamo=False)


def quantize(out):
    # The embedding table holds 98 of the 141 million parameters, so int8 there halves the file. I also tried int8 MatMul layers:
    # the file got smaller and faster, but only 97.6 % of the tokens kept their PyTorch label (embeddings only: 99.93 %).
    pre = f"{out}/model_pre.onnx"
    quant_pre_process(f"{out}/model.onnx", pre, skip_symbolic_shape=True)
    head = [n.name for n in onnx.load(pre, load_external_data=False).graph.node
            if n.op_type in ("MatMul", "Gemm") and ("classifier" in n.name or "/head/" in n.name)]
    quantize_dynamic(pre, f"{out}/model_quantized.onnx", weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                     op_types_to_quantize=["Gather"], nodes_to_exclude=head)
    os.remove(pre)


def check(model_dir, out, labels_file, threads):
    """Per file: share of tokens with the same arg-max label as PyTorch (first 512 tokens of each development text), CPU time and size."""
    texts = [r["text"] for r in map(json.loads, open(labels_file, encoding="utf-8")) if r["split"] == "dev"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    torch.set_num_threads(threads)
    enc = [tok(t, truncation=True, max_length=512, return_tensors="np") for t in texts]
    with torch.no_grad():
        ref = [model(input_ids=torch.tensor(e["input_ids"]), attention_mask=torch.tensor(e["attention_mask"])).logits[0].argmax(-1).numpy()
               for e in enc]
    res = {}
    for f in ("model.onnx", "model_quantized.onnx"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        session = ort.InferenceSession(f"{out}/{f}", opts, providers=["CPUExecutionProvider"])
        same = n = 0
        t0 = time.time()
        for e, r in zip(enc, ref):
            z = session.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]
            same += int((z.argmax(-1) == r).sum())
            n += len(r)
        res[f"onnx/{f}"] = {"token_agreement": round(same / n, 5), "ms_per_text": round(1000 * (time.time() - t0) / len(enc), 1),
                            "mb": round(os.path.getsize(f"{out}/{f}") / 1e6, 1)}
        print(f, json.dumps(res[f"onnx/{f}"]), flush=True)
    return {"texts": len(texts), "files": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder of the trained student")
    ap.add_argument("--out", default="onnx")
    ap.add_argument("--check", help="labels.jsonl from label.py; compares both files with PyTorch on its development split")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    export(a.model, a.out)
    quantize(a.out)
    if a.check:
        json.dump(check(a.model, a.out, a.check, a.threads), open(f"{a.out}/quantization_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
