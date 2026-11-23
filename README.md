# pii-filter-small

Code for [LukasKunhardt/pii-filter-small](https://huggingface.co/LukasKunhardt/pii-filter-small), a small multilingual model that finds personal information in text with the 33 labels of openai/privacy-filter.
It writes the training texts, labels them with openai/privacy-filter, trains the student, exports it to ONNX and measures it on four public test sets.

## Install

```bash
pip install torch transformers huggingface_hub numpy pandas requests onnx onnxruntime matplotlib
```

## Example

```python
import evaluate

tok, labels, run = evaluate.load_model("LukasKunhardt/pii-filter-small")
print(evaluate.read("Hi, I'm Anna Müller, born 12 May 1984. Call me at +49 151 2345 6789.", tok, labels, run))
# [(8, 19, 'private_person'), (26, 37, 'private_date'), (50, 67, 'private_phone')]
# with ONNX Runtime: evaluate.load_model("LukasKunhardt/pii-filter-small", onnx_file="onnx/model_quantized.onnx")
```

`read` returns `(start, end, category)` spans, joined over the BIOES labels and trimmed of spaces. It reads long texts in windows of 512 tokens that overlap by 128, as in training.

## Files

| File | What it does |
|---|---|
| `make_texts.py` | Writes texts with invented personal information through OpenRouter (`OPENROUTER_API_KEY`), and drops texts that overlap the test samples |
| `label.py` | Labels the texts with openai/privacy-filter: spans and per-character category probabilities |
| `train.py` | Trains the student on the teacher's spans (BIOES cross-entropy) and probabilities (KL), picks the checkpoint on development texts |
| `export_onnx.py` | Writes `onnx/model.onnx` (fp32) and `onnx/model_quantized.onnx` (int8 embedding table) and compares both with PyTorch on the development texts |
| `evaluate.py` | Draws the test samples, maps their labels (`mapping.json`), reads any model the same way, scores it; also runs privacy-filter through `opf` |
| `plot_pareto.py` | Draws the charts of the model card from `results/` |
| `results/` | The measurement files behind the model card and `RESULTS.md` |
| `tests/test_equivalence.py` | Checks this code against the research code that produced the released model |

## Reproduce

```bash
python make_texts.py write --n 24000 --out texts_raw.jsonl
python make_texts.py dedup texts_raw.jsonl texts.jsonl
python label.py --texts texts.jsonl --out labels.jsonl              # one GPU
python train.py --labels labels.jsonl --seed 43 --out model         # one NVIDIA A10G, about 21 minutes
python export_onnx.py --model model --out onnx --check labels.jsonl
python evaluate.py --model model --out results/mine.json
python evaluate.py --opf --out results/opf.json                     # pip install opf
```

The generated texts depend on the model provider, so a new run gives similar, not identical, data.

## License

MIT, Lukas Kunhardt. The test sets keep their own licences; ai4privacy/pii-masking-300k is for academic use.
