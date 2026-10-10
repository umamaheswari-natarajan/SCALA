import os
import json
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures",
    "ablation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

KMEANS_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_rag",
    "kmeans_overall_summary.json"
)

GMM_TOP1_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top1",
    "rag_top1_overall_summary.json"
)

GMM_TOP3_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top3",
    "rag_top3_overall_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Font settings for TWO side-by-side figures
LABEL_SIZE = 16
TICK_SIZE = 14
VALUE_SIZE = 13


def add_labels(ax, bars, fmt="{:.2f}%"):

    for bar in bars:

        h = bar.get_height()

        ax.annotate(
            fmt.format(h),
            xy=(
                bar.get_x() + bar.get_width() / 2,
                h
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            fontweight="bold"
        )


def save(fig, name):

    png = os.path.join(
        OUTPUT_DIR,
        name + ".png"
    )

    pdf = os.path.join(
        OUTPUT_DIR,
        name + ".pdf"
    )

    # High-resolution PNG
    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight"
    )

    # Vector PDF
    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)


# ============================================================
# LOAD RESULTS
# ============================================================

kmeans = load_json(KMEANS_SUMMARY)
gmm_top1 = load_json(GMM_TOP1_SUMMARY)
gmm_top3 = load_json(GMM_TOP3_SUMMARY)


# ============================================================
# METHODS
# ============================================================

methods = [
    "CLIP +\nK-Means",
    "SCALA\nTop-1",
    "SCALA\nTop-3"
]


# ============================================================
# DATA
# ============================================================

candidate_reduction = [
    kmeans["avg_candidate_reduction"] * 100,
    gmm_top1["avg_candidate_reduction"] * 100,
    gmm_top3["avg_candidate_reduction"] * 100
]

top5_overlap = [
    kmeans["avg_clip_kmeans_topk_overlap"] * 100,
    gmm_top1["avg_flat_gmm_topk_overlap"] * 100,
    gmm_top3["avg_flat_gmm_topk_overlap"] * 100
]


# ============================================================
# FIGURE 1: CANDIDATE REDUCTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.2, 4.8)
)

bars = ax.bar(
    methods,
    candidate_reduction,
    width=0.65
)

add_labels(
    ax,
    bars
)

ax.set_ylabel(
    "Candidate Reduction (%)",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

ax.set_ylim(
    0,
    105
)

ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

ax.set_axisbelow(True)

ax.set_xlabel(
    "Method",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)


fig.tight_layout()

save(
    fig,
    "ablation_organization_candidate_reduction"
)


# ============================================================
# FIGURE 2: TOP-5 RETRIEVAL PRESERVATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.2, 4.8)
)

bars = ax.bar(
    methods,
    top5_overlap,
    width=0.65
)

add_labels(
    ax,
    bars
)

ax.set_ylabel(
    "Top-5 Retrieval Preservation (%)",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

ax.set_ylim(
    0,
    105
)

ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

ax.set_axisbelow(True)

ax.set_xlabel(
    "Method",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

fig.tight_layout()

save(
    fig,
    "ablation_organization_top5_overlap"
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nKNOWLEDGE-SPACE ORGANIZATION ABLATION")
print("=" * 75)

for i, method in enumerate(methods):

    print(
        f"{method.replace(chr(10), ' '):20s} "
        f"Candidate Reduction = {candidate_reduction[i]:.2f}% | "
        f"Top-5 Preservation = {top5_overlap[i]:.2f}%"
    )

print("\nFigures saved to:")
print(OUTPUT_DIR)