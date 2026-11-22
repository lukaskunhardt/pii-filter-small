"""Export a trained student to ONNX and quantise it to int8 for CPU, then compare both files with PyTorch on the development split.

  python export_onnx.py --model model --out onnx --check labels.jsonl
Writes model.onnx (fp32, an intermediate file) and model_int8.onnx.
"""
import argparse
import json
import os
import time

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoModelForTokenClassification, AutoTokenizer

import evaluate


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
    tok.save_pretrained(out)
    model.config.save_pretrained(out)


def quantize(out):
    # A plain dynamic int8 file was unusable (development char F1 0.389 against 0.904 for fp32). What works: shape preprocessing,
    # per-channel weights, MatMul only, reduce_range, and the classification head kept in fp32.
    quant_pre_process(f"{out}/model.onnx", f"{out}/model_pre.onnx", skip_symbolic_shape=True)
    head = [n.name for n in onnx.load(f"{out}/model_pre.onnx", load_external_data=False).graph.node
            if n.op_type in ("MatMul", "Gemm") and ("classifier" in n.name or "/head/" in n.name)]
    quantize_dynamic(f"{out}/model_pre.onnx", f"{out}/model_int8.onnx", weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                     op_types_to_quantize=["MatMul"], nodes_to_exclude=head)
    os.remove(f"{out}/model_pre.onnx")
    return head


def check(model_dir, out, labels_file, threads):
    """Character F1 against the teacher on the development split, and CPU time, for PyTorch, ONNX fp32 and ONNX int8."""
    dev = [r for r in map(json.loads, open(labels_file, encoding="utf-8")) if r["split"] == "dev"]
    torch.set_num_threads(threads)
    res = {}
    for name, onnx_file in [("torch", None), ("onnx_fp32", f"{out}/model.onnx"), ("onnx_int8", f"{out}/model_int8.onnx")]:
        tok, labels, run = evaluate.load_model(model_dir if onnx_file is None else out, onnx_file, "cpu", threads)
        rows = []
        t0 = time.time()
        for r in dev:
            doc = {"text": r["text"], "gold": sorted({tuple(s) for s in r["spans"]}), "ignore": []}
            rows.append((("dev", r["lang"]), evaluate.score(doc, evaluate.read(r["text"], tok, labels, run))))
        res[name] = evaluate.summarize(rows)["dev/all"]
        res[name]["seconds"] = round(time.time() - t0)
        print(name, json.dumps(res[name]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder of the trained student")
    ap.add_argument("--out", default="onnx")
    ap.add_argument("--check", help="labels.jsonl from label.py; compares the files on its development split")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    export(a.model, a.out)
    print("kept in fp32:", quantize(a.out))
    if a.check:
        json.dump(check(a.model, a.out, a.check, a.threads), open(f"{a.out}/onnx_dev.json", "w"), indent=1)


if __name__ == "__main__":
    main()
