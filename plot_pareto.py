"""Draw the quality/size and quality/speed charts of the model card from the files in results/.

  python plot_pareto.py --out pareto
Writes pareto_size.png/.svg and pareto_speed.png/.svg.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = ["ai4privacy", "nemotron", "kiji", "gretel"]
MINE, LIGHT, OTHER = "#0072B2", "#56B4E9", "#D55E00"  # blue, sky blue and vermilion, distinct for colour-blind readers


def load(name):
    return json.load(open(os.path.join(HERE, "results", name), encoding="utf-8"))


def mean_char_f1(res):
    return sum(res[f"{s}/all"]["char_f1"] for s in SETS) / len(SETS)


def points():
    mine, opf, tf = load("test_mine.json"), load("test_privacy_filter_opf.json"), load("test_privacy_filter_transformers.json")
    int8, speed, speed_onnx = load("test_onnx_quantized.json"), load("speed.json"), load("speed_onnx.json")
    params_mine = load("train_mmbert_small_seed43.json")["params"]
    params_pf = tf["openai/privacy-filter"]["params"]
    # name, params, mean char F1, CPU ms per 1,000 characters, marker (circle: other model, filled circle: mine, diamond: my int8 file)
    return [
        ("pii-filter-small", params_mine, mean_char_f1(mine["ours_s43"]), speed["ours_s43_torch_cpu"], "mine"),
        ("int8 file", params_mine, mean_char_f1(int8), speed_onnx["onnx_model_quantized_cpu"], "int8"),
        ("privacy-filter (opf)", params_pf, mean_char_f1(opf["opf_viterbi"]), speed["privacy_filter_opf_viterbi_cpu"], "other"),
        ("privacy-filter (transformers)", params_pf, mean_char_f1(tf["openai/privacy-filter"]),
         mine["privacy_filter_cpu_ms_per_1k_chars"], "other"),
    ]


def front(pts):
    """Pareto front for 'smaller x and higher y is better', as a step line."""
    best, out = -1, []
    for x, y in sorted((p[0], p[1]) for p in pts):
        if y > best:
            out.append((x, y))
            best = y
    return out


def draw(pts, xi, xlabel, title, caption, offsets, xlim, path):
    plt.rcParams.update({"font.size": 17})
    fig, ax = plt.subplots(figsize=(10, 6.25), dpi=160)
    xy = [(p[xi], p[2]) for p in pts]
    f = front(xy)
    ax.step([x for x, _ in f], [y for _, y in f], where="post", color="#999999", lw=1.5, ls="--", zorder=1, label="Pareto front")
    style = {"mine": dict(s=300, color=MINE, edgecolor=MINE, marker="o"), "int8": dict(s=100, color=LIGHT, edgecolor=MINE, marker="D"),
             "other": dict(s=120, color="white", edgecolor=OTHER, marker="o")}
    ys = {p[0]: y for p, (_, y) in zip(pts, xy)}
    for (name, *_, kind), (x, y) in zip(pts, xy):
        ax.scatter(x, y, lw=2.5, zorder=3, **style[kind])
        if kind == "int8":
            continue
        # my two files sit almost on top of each other, so they share one label
        text = f"{name}\nPyTorch {y:.3f}, int8 file {ys['int8 file']:.3f}" if kind == "mine" else f"{name}\n{y:.3f}"
        dx, dy, ha = offsets[name]
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=15, fontweight="bold" if kind == "mine" else "normal", color="#222222")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.70, 0.83)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean character F1, 4 test sets")
    ax.set_title(title, fontsize=19, pad=12)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=12, color="#555555", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path + ".png")
    fig.savefig(path + ".svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pareto")
    a = ap.parse_args()
    pts = points()
    caption = ("ai4privacy, Nemotron-PII, Kiji, Gretel test samples (4,150 documents).\n"
               "Blue: my model (large circle PyTorch weights, diamond int8 ONNX file).")
    draw(pts, 1, "Parameters (log scale)", "PII detection: quality against model size", caption,
         {"pii-filter-small": (16, -48, "left"), "privacy-filter (opf)": (8, 30, "right"), "privacy-filter (transformers)": (-16, 0, "right")},
         (6e7, 3e9), a.out + "_size")
    draw(pts, 3, "CPU milliseconds per 1,000 characters (log scale, 3 threads)", "PII detection: quality against CPU time", caption,
         {"pii-filter-small": (-10, -48, "left"), "privacy-filter (opf)": (8, 30, "right"), "privacy-filter (transformers)": (16, 0, "left")},
         (60, 6e4), a.out + "_speed")

if __name__ == "__main__":
    main()
