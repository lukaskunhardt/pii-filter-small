# Results

All numbers come from the JSON files in `results/`. Scores are on fixed samples (random seed 0) of four test sets, read and scored by `evaluate.py`.
Means are plain means over the four sets.

## Test, all rows

Character F1 (binary: personal or not):

| Model | ai4privacy | Nemotron | Kiji | Gretel | Mean of 4 |
|---|---|---|---|---|---|
| pii-filter-small (seed 43, shipped) | 0.929 | 0.817 | 0.830 | 0.661 | 0.809 |
| mean of 3 seeds | 0.931 | 0.813 | 0.833 | 0.660 | 0.809 |
| int8 file `onnx/model_quantized.onnx` (v1.0.3) | 0.924 | 0.823 | 0.844 | 0.649 | 0.810 |
| int8 file `model_int8.onnx` (v1.0 to v1.0.2, no longer shipped) | 0.915 | 0.791 | 0.815 | 0.645 | 0.791 |
| privacy-filter, official opf decoder | 0.952 | 0.823 | 0.832 | 0.637 | 0.811 |
| privacy-filter through transformers (argmax) | 0.868 | 0.696 | 0.737 | 0.606 | 0.726 |
| privacy-filter, opf argmax | 0.952 | 0.823 | 0.834 | 0.634 | 0.811 |
| seed 42 | 0.929 | 0.821 | 0.832 | 0.658 | 0.810 |
| seed 44 | 0.934 | 0.800 | 0.837 | 0.659 | 0.808 |

Span F1 (typed, exact boundaries):

| Model | ai4privacy | Nemotron | Kiji | Gretel | Mean of 4 |
|---|---|---|---|---|---|
| pii-filter-small (seed 43, shipped) | 0.735 | 0.613 | 0.374 | 0.429 | 0.538 |
| mean of 3 seeds | 0.731 | 0.612 | 0.375 | 0.433 | 0.538 |
| int8 file `onnx/model_quantized.onnx` (v1.0.3) | 0.709 | 0.621 | 0.386 | 0.404 | 0.530 |
| int8 file `model_int8.onnx` (v1.0 to v1.0.2, no longer shipped) | 0.686 | 0.583 | 0.352 | 0.388 | 0.502 |
| privacy-filter, official opf decoder | 0.883 | 0.661 | 0.397 | 0.465 | 0.601 |
| privacy-filter through transformers (argmax) | 0.735 | 0.471 | 0.327 | 0.353 | 0.471 |
| privacy-filter, opf argmax | 0.873 | 0.638 | 0.393 | 0.399 | 0.576 |
| seed 42 | 0.725 | 0.615 | 0.376 | 0.432 | 0.537 |
| seed 44 | 0.734 | 0.608 | 0.374 | 0.438 | 0.538 |

"privacy-filter, opf argmax" is the `opf` package with its argmax decoder instead of Viterbi. Seed 43 is the released model.
The v1.0.3 int8 row is `results/test_onnx_quantized.json`; I did not run `onnx/model.onnx` (fp32) on the test sets.

## Test per language

Character F1 / span F1. Columns: my model (seed 43), the int8 file of v1.0, privacy-filter with `opf` Viterbi, privacy-filter through `transformers`, the int8 file of v1.0.3.
Gretel names French "France" in its data.

| Set / language | Documents | pii-filter-small | int8 v1.0 | privacy-filter (opf) | privacy-filter (transformers) | int8 v1.0.3 |
|---|---|---|---|---|---|---|
| ai4privacy/Dutch | 200 | 0.906 / 0.686 | 0.891 / 0.647 | 0.933 / 0.841 | 0.871 / 0.724 | 0.904 / 0.668 |
| ai4privacy/English | 200 | 0.934 / 0.736 | 0.917 / 0.706 | 0.961 / 0.897 | 0.783 / 0.614 | 0.930 / 0.717 |
| ai4privacy/French | 200 | 0.940 / 0.745 | 0.925 / 0.677 | 0.963 / 0.897 | 0.872 / 0.748 | 0.931 / 0.708 |
| ai4privacy/German | 200 | 0.940 / 0.745 | 0.924 / 0.682 | 0.958 / 0.886 | 0.900 / 0.788 | 0.929 / 0.707 |
| ai4privacy/Italian | 200 | 0.920 / 0.720 | 0.911 / 0.681 | 0.935 / 0.879 | 0.883 / 0.773 | 0.914 / 0.691 |
| ai4privacy/Spanish | 200 | 0.933 / 0.779 | 0.922 / 0.726 | 0.960 / 0.903 | 0.906 / 0.790 | 0.933 / 0.763 |
| gretel/Dutch | 150 | 0.652 / 0.415 | 0.634 / 0.383 | 0.620 / 0.441 | 0.592 / 0.338 | 0.631 / 0.387 |
| gretel/English | 150 | 0.686 / 0.469 | 0.657 / 0.400 | 0.653 / 0.518 | 0.598 / 0.369 | 0.687 / 0.446 |
| gretel/France | 150 | 0.662 / 0.429 | 0.653 / 0.397 | 0.690 / 0.493 | 0.682 / 0.422 | 0.659 / 0.394 |
| gretel/German | 150 | 0.655 / 0.427 | 0.641 / 0.392 | 0.607 / 0.459 | 0.567 / 0.336 | 0.640 / 0.405 |
| gretel/Italian | 150 | 0.660 / 0.418 | 0.649 / 0.376 | 0.625 / 0.421 | 0.597 / 0.312 | 0.646 / 0.388 |
| gretel/Spanish | 150 | 0.686 / 0.467 | 0.671 / 0.421 | 0.669 / 0.494 | 0.651 / 0.384 | 0.668 / 0.454 |
| gretel/Swedish | 150 | 0.627 / 0.384 | 0.609 / 0.351 | 0.593 / 0.433 | 0.555 / 0.322 | 0.610 / 0.364 |
| kiji/Danish | 150 | 0.790 / 0.311 | 0.749 / 0.293 | 0.784 / 0.324 | 0.715 / 0.280 | 0.801 / 0.323 |
| kiji/Dutch | 150 | 0.850 / 0.395 | 0.830 / 0.370 | 0.838 / 0.433 | 0.809 / 0.414 | 0.859 / 0.414 |
| kiji/English | 150 | 0.840 / 0.361 | 0.836 / 0.338 | 0.834 / 0.371 | 0.669 / 0.253 | 0.855 / 0.370 |
| kiji/French | 150 | 0.821 / 0.374 | 0.818 / 0.368 | 0.829 / 0.392 | 0.730 / 0.311 | 0.839 / 0.382 |
| kiji/German | 150 | 0.870 / 0.390 | 0.855 / 0.361 | 0.856 / 0.391 | 0.752 / 0.331 | 0.881 / 0.401 |
| kiji/Spanish | 150 | 0.805 / 0.414 | 0.793 / 0.377 | 0.846 / 0.462 | 0.767 / 0.389 | 0.826 / 0.424 |
| nemotron/English | 1000 | 0.817 / 0.613 | 0.791 / 0.583 | 0.823 / 0.661 | 0.696 / 0.471 | 0.823 / 0.621 |

## Speed

CPU milliseconds per 1,000 characters, 50 Nemotron test documents, 3 threads (`results/speed.json`, `results/test_mine.json`, `results/speed_onnx.json`).
The int8 column of runs 1 and 2 is the v1.0 file. Run 3 came later, possibly on a different machine, so compare its numbers with the other runs with care.

| Run | pii-filter-small int8 | pii-filter-small PyTorch | privacy-filter (opf, Viterbi) | privacy-filter (transformers) |
|---|---|---|---|---|
| Run 1 (with the test scoring) | 120.6 | | | 1,280.6 |
| Run 2 (speed only, one warm-up document) | 121.2 | 224.7 | 16,894.4 | |
| Run 3 (v1.0.3 files, speed only, one warm-up document) | 201.7 (`model_quantized.onnx`); 230.9 (`model.onnx`) | | | |

In my gate the int8 file had to be at most 240 ms and at least 5 times faster than privacy-filter in the same run: 120.6 against 1,280.6 ms (10.6 times).
Before training, privacy-filter through `transformers` needed 1,198.6 ms in a separate run (`results/test_privacy_filter_transformers.json`).

## Development (own texts, teacher spans as reference)

1,150 held-out generated texts. Character F1 / span F1 against the teacher's spans, at the checkpoint each run picked.

| Run | Parameters | Dev character F1 | Dev span F1 | Training time |
|---|---|---|---|---|
| mmBERT-small seed 42 | 140,654,241 | 0.9034 | 0.6407 | 21 min |
| mmBERT-small seed 43 | 140,654,241 | 0.9041 | 0.6382 | 21 min |
| mmBERT-small seed 44 | 140,654,241 | 0.9026 | 0.6373 | 21 min |
| multilingual-e5-small seed 42 (not released, not tested) | 117,518,625 | 0.8956 | 0.6053 | 10 min |

ONNX on development texts, seed 43 (`results/onnx_dev.json`): PyTorch 0.9042, ONNX fp32 0.9042, ONNX int8 0.8971 character F1.
A first int8 file without that recipe reached only 0.389.

For v1.0.3 I compared several int8 recipes with PyTorch on 242 other development texts (`onnx/quantization_check.json` in the Hub repo): share of tokens with the same arg-max label, CPU time and size. The rule, written before the comparison, was to ship the smallest int8 file with at least 0.998 agreement. Only the file with an int8 embedding table and fp32 everywhere else passed (0.99926, 268 MB). The v1.0 recipe (int8 MatMul) kept 0.97614. `onnx/model.onnx` agreed on every token (1.0).

## Gate and test use

The rules are in the Hub repo under `rules/` (German), committed to my private research repository before each step:

1. Label mapping and test sampling, then privacy-filter through `transformers` on the test samples, to set the gate (22 Nov 2026, commit d397926).
2. Gate, before any training (commit dea4fab): mean character F1 at least 0.697, no set more than 0.06 below privacy-filter, mean span F1 at least 0.422, int8 at most 240 ms per 1,000 characters, 5 times faster than privacy-filter, and within 0.01 mean character F1 of PyTorch.
3. Rule for the single test opening of my models (commit 5cdb42c). Result: all points passed except the last; int8 lost 0.018 (0.7915 against 0.8092, both as stored in `results/test_mine.json`).
   The v1.0.3 int8 file gains 0.0007 on the same measure (0.8099 against 0.8092). The gate was not decided again.
4. Decision rule after the failed point, before measuring privacy-filter with `opf` (commit 55d98a8): publish only if the mean of my 3 seeds stays within 0.03 of the `opf` row on average and within 0.06 on every set. It did (0.809 against 0.811; largest gap ai4privacy, 0.931 against 0.952).
5. Speed run with `opf` on CPU, on 50 test documents, no scores.
6. v1.0.3: choice of the ONNX files on development texts only (commits 8e9ef58, 59ee73c), then one test run of the chosen int8 file and one speed run, under a rule written before (commit a6e2afc). Nothing about the files changed after that run.
7. v1.0.3: one run of the official PIIMB evaluation (below), and one run of the `pipeline()` path on two clinical test sets (below).
8. v1.0.4: one more PIIMB run with the new label names in `config.json` (E- → I-, S- → B-), under a rule written before the run; v1.0.3 labels gave a mean F2 of 0.674, the new ones 0.676. Only the names changed, not the weights. The clinical `pipeline()` numbers in this file were measured with the v1.0.3 names.

The overlap check in `make_texts.py` read the test texts (not their labels) and dropped 124 of 24,000 generated texts (`results/overlap_check.json`).

## PII Masking Benchmark (PIIMB)

I ran the official `piimb evaluate` with default settings on the released weights (fp32). PIIMB scores masking per character, regardless of category, with F2 (recall counts more than precision), on six tasks. The other rows come from the public results dataset piimb/pii-masking-benchmark-results; I ranked all 37 entries by mean F2 (`results/piimb.json`). My submission to the leaderboard is pending. I removed overlap only against my own four test samples, not against the PIIMB data.

| Rank | Model | Parameters | ai4privacy-en | ai4privacy-multi | gretel | mapa-eur-lex | nemotron-pii | privy | Mean F2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | OpenMed/privacy-filter-multilingual-v2 | 1,400 M | 0.975 | 0.970 | 0.973 | 0.373 | 0.907 | 0.818 | 0.836 |
| 2 | OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1 | 434 M | 0.944 | 0.899 | 0.945 | 0.524 | 0.963 | 0.734 | 0.835 |
| 3 | knowledgator/gliner-stream-pii-v1.0 | 677 M | 0.958 | 0.921 | 0.939 | 0.411 | 0.833 | 0.904 | 0.828 |
| 4 | OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 | 141 M | 0.949 | 0.888 | 0.953 | 0.478 | 0.972 | 0.659 | 0.816 |
| 5 | OpenMed/privacy-filter-nemotron-v2 | 1,400 M | 0.978 | 0.976 | 0.983 | 0.384 | 0.941 | 0.593 | 0.809 |
| 6 | fastino/gliner2-privacy-filter-PII-multi | 307 M | 0.914 | 0.904 | 0.936 | 0.545 | 0.854 | 0.646 | 0.800 |
| 7 | nvidia/gliner-PII | 445 M | 0.945 | 0.913 | 0.920 | 0.531 | 0.772 | 0.714 | 0.799 |
| 8 | antoniogr7/pii-identifier-v0.1 | 150 M | 0.945 | 0.814 | 0.939 | 0.380 | 0.970 | 0.726 | 0.796 |
| 9 | hivetrace/gliner-guard-omni | 307 M | 0.955 | 0.926 | 0.891 | 0.559 | 0.759 | 0.636 | 0.788 |
| 10 | fastino/gliner2-multi-v1 | 307 M | 0.903 | 0.878 | 0.870 | 0.569 | 0.829 | 0.661 | 0.785 |
| 11 | knowledgator/gliner-pii-large-v1.0 | 440 M | 0.909 | 0.879 | 0.951 | 0.517 | 0.839 | 0.581 | 0.779 |
| 12 | kalyan-ks/ettin-68m-nemotron-pii | 68 M | 0.919 | 0.845 | 0.918 | 0.382 | 0.939 | 0.663 | 0.777 |
| 13 | VidaiUK/stirling-minimo-pii-10m | 92 M | 0.917 | 0.776 | 0.979 | 0.202 | 0.930 | 0.792 | 0.766 |
| 14 | kalyan-ks/ettin-32m-nemotron-pii | 32 M | 0.911 | 0.794 | 0.917 | 0.294 | 0.932 | 0.680 | 0.755 |
| 15 | fastino/gliner2-large-v1 | 486 M | 0.868 | 0.798 | 0.873 | 0.474 | 0.857 | 0.602 | 0.746 |
| 16 | gravitee-io/bert-small-pii-detection | 29 M | 0.900 | 0.729 | 0.818 | 0.273 | 0.693 | 0.987 | 0.733 |
| 17 | OpenMed/privacy-filter-nemotron | 1,400 M | 0.889 | 0.719 | 0.888 | 0.228 | 0.957 | 0.549 | 0.705 |
| 18 | OpenMed/privacy-filter-multilingual | 1,400 M | 0.942 | 0.844 | 0.864 | 0.321 | 0.766 | 0.468 | 0.701 |
| 19 | kalyan-ks/ettin-17m-nemotron-pii | 17 M | 0.865 | 0.723 | 0.868 | 0.234 | 0.895 | 0.553 | 0.690 |
| 20 | bardsai/eu-pii-anonimization-multilang | 278 M | 0.831 | 0.843 | 0.890 | 0.254 | 0.715 | 0.575 | 0.685 |
| 21 | scanpatch/pii-ner-nemotron | 559 M | 0.820 | 0.806 | 0.741 | 0.573 | 0.578 | 0.561 | 0.680 |
| 22 | knowledgator/gliner-pii-base-v1.0 | 166 M | 0.795 | 0.727 | 0.894 | 0.332 | 0.777 | 0.546 | 0.678 |
| 23 | **LukasKunhardt/pii-filter-small** (v1.0.4 labels) | 141 M | 0.902 | 0.845 | 0.903 | 0.290 | 0.616 | 0.499 | 0.676 |
| 24 | urchade/gliner_multi_pii-v1 | 289 M | 0.735 | 0.762 | 0.795 | 0.424 | 0.731 | 0.555 | 0.667 |
| 25 | openai/privacy-filter | 1,399 M | 0.843 | 0.805 | 0.869 | 0.313 | 0.607 | 0.536 | 0.662 |
| 26 | fastino/gliner2-base-v1 | 208 M | 0.750 | 0.723 | 0.891 | 0.292 | 0.739 | 0.499 | 0.649 |
| 27 | tabularisai/eu-pii-safeguard | 559 M | 0.805 | 0.833 | 0.848 | 0.143 | 0.610 | 0.545 | 0.631 |
| 28 | gretelai/gretel-gliner-bi-small-v1.0 | 189 M | 0.716 | 0.644 | 0.975 | 0.312 | 0.719 | 0.404 | 0.628 |
| 29 | lakshyakh93/deberta_finetuned_pii | 139 M | 0.960 | 0.682 | 0.930 | 0.133 | 0.742 | 0.305 | 0.625 |
| 30 | gretelai/gretel-gliner-bi-base-v1.0 | 242 M | 0.624 | 0.596 | 0.978 | 0.359 | 0.703 | 0.356 | 0.603 |
| 31 | LiquidAI/LFM2.5-Encoder-350M-PII-Detector | 355 M | 0.819 | 0.808 | 0.811 | 0.099 | 0.500 | 0.545 | 0.597 |
| 32 | presidio/en_core_web_lg | 106 M | 0.734 | 0.622 | 0.758 | 0.154 | 0.729 | 0.518 | 0.586 |
| 33 | nationaldesignstudio/rampart | 18 M | 0.798 | 0.810 | 0.794 | 0.151 | 0.456 | 0.439 | 0.575 |
| 34 | HikmaAI/hikmaai-distilbert-pii | 135 M | 0.805 | 0.788 | 0.591 | 0.140 | 0.393 | 0.401 | 0.520 |
| 35 | iiiorg/piiranha-v1-detect-personal-information | 278 M | 0.803 | 0.813 | 0.607 | 0.146 | 0.360 | 0.369 | 0.516 |
| 36 | ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii | 150 M | 0.700 | 0.520 | 0.689 | 0.063 | 0.441 | 0.338 | 0.458 |
| 37 | ai4privacy/llama-ai4privacy-multilingual-anonymiser-openpii | 150 M | 0.562 | 0.488 | 0.576 | 0.067 | 0.339 | 0.272 | 0.384 |

## Clinical notes, through `pipeline()`

An extra check outside the training domain, not part of the gate: the plain `pipeline("token-classification", aggregation_strategy="simple", stride=128)`, arg-max, on the GraSCCo test letters (German, real letters with replaced identifiers, CC BY 4.0) and the MDeID test split (synthetic clinical notes, CC BY-NC 4.0, test only). Scored with my clinical benchmark code, not `evaluate.py` (`results/test_pipeline_clinical.json`). A document has a leak if a name, ID, phone or fax number, e-mail address or street stays less than half redacted. The model was not trained for clinical text, and most documents leak something.

| Set | Language | Documents | Character precision | Character recall | Character F1 | Documents with a leak |
|---|---|---|---|---|---|---|
| GraSCCo test | all | 41 | 0.930 | 0.713 | 0.807 | 0.732 |
| MDeID test | all | 2,856 | 0.943 | 0.725 | 0.820 | 0.773 |
| MDeID test | de | 402 | 0.944 | 0.732 | 0.824 | 0.789 |
| MDeID test | en | 416 | 0.942 | 0.678 | 0.789 | 0.757 |
| MDeID test | es | 408 | 0.949 | 0.754 | 0.840 | 0.740 |
| MDeID test | fr | 410 | 0.945 | 0.766 | 0.846 | 0.720 |
| MDeID test | it | 404 | 0.941 | 0.750 | 0.835 | 0.752 |
| MDeID test | nl | 405 | 0.947 | 0.671 | 0.786 | 0.864 |
| MDeID test | pt | 411 | 0.937 | 0.722 | 0.816 | 0.791 |

## Licences

jhu-clsp/mmBERT-small MIT, openai/privacy-filter Apache 2.0, DeepSeek V4.1 Flash MIT. ai4privacy/pii-masking-300k: own licence for academic use, test only. nvidia/Nemotron-PII CC BY 4.0, gretelai/synthetic_pii_finance_multilingual Apache 2.0, DataikuNLP/kiji-pii-training-data Apache 2.0. piimb/pii-masking-benchmark CC BY-NC 4.0, test only.
