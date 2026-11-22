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
MINE, OTHER = "#0072B2", "#D55E00"  # blue and vermilion, distinct for colour-blind readers


def load(name):
    return json.load(open(os.path.join(HERE, "results", name), encoding="utf-8"))


def mean_char_f1(res):
    return sum(res[f"{s}/all"]["char_f1"] for s in SETS) / len(SETS)


def points():
    mine, opf, tf = load("test_mine.json"), load("test_privacy_filter_opf.json"), load("test_privacy_filter_transformers.json")
    speed = load("speed.json")
    params_mine = load("train_mmbert_small_seed43.json")["params"]
    params_pf = tf["openai/privacy-filter"]["params"]
    # name, params, mean char F1, CPU ms per 1,000 characters, mine?
    return [
        ("pii-filter-small", params_mine, mean_char_f1(mine["ours_s43"]), speed["ours_s43_torch_cpu"], True),
        ("pii-filter-small int8", params_mine, mean_char_f1(mine["ours_s43_onnx_int8"]), speed["ours_s43_onnx_int8_cpu"], True),
        ("privacy-filter (opf)", params_pf, mean_char_f1(opf["opf_viterbi"]), speed["privacy_filter_opf_viterbi_cpu"], False),
        ("privacy-filter (transformers)", params_pf, mean_char_f1(tf["openai/privacy-filter"]),
         mine["privacy_filter_cpu_ms_per_1k_chars"], False),
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
    for (name, *_, mine), (x, y) in zip(pts, xy):
        ax.scatter(x, y, s=170 if mine else 120, color=MINE if mine else "white", edgecolor=MINE if mine else OTHER, lw=2.5, zorder=3)
        dx, dy, ha = offsets[name]
        ax.annotate(f"{name}\n{y:.3f}", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=15, fontweight="bold" if mine else "normal", color="#222222")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.70, 0.83)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean character F1, 4 test sets")
    ax.set_title(title, fontsize=19, pad=12)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.015, caption, ha="center", fontsize=12, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path + ".png")
    fig.savefig(path + ".svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pareto")
    a = ap.parse_args()
    pts = points()
    caption = "ai4privacy, Nemotron-PII, Kiji, Gretel test samples (4,150 documents). Filled: my model."
    draw(pts, 1, "Parameters (log scale)", "PII detection: quality against model size", caption,
         {"pii-filter-small": (14, 26, "left"), "pii-filter-small int8": (16, -16, "left"),
          "privacy-filter (opf)": (8, 30, "right"), "privacy-filter (transformers)": (-16, 0, "right")},
         (6e7, 3e9), a.out + "_size")
    draw(pts, 3, "CPU milliseconds per 1,000 characters (log scale, 3 threads)", "PII detection: quality against CPU time", caption,
         {"pii-filter-small": (14, 26, "left"), "pii-filter-small int8": (12, -30, "left"),
          "privacy-filter (opf)": (8, 30, "right"), "privacy-filter (transformers)": (16, 0, "left")},
         (60, 6e4), a.out + "_speed")


if __name__ == "__main__":
    main()
