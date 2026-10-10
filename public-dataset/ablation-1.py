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

STANDARD_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "standard_rag",
    "standard_rag_overall_summary.json"
)

STANDARD_JUDGE = os.path.join(
    BASE_DIR,
    "results",
    "standard_rag_llm_judge",
    "standard_rag_llm_judge_summary.json"
)

PLAIN_GMM_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_rag",
    "plain_gmm_overall_summary.json"
)

PLAIN_GMM_JUDGE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_llm_judge",
    "plain_gmm_llm_judge_summary.json"
)

KMEANS_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_rag",
    "kmeans_overall_summary.json"
)

KMEANS_JUDGE = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_llm_judge",
    "kmeans_llm_judge_summary.json"
)

SCALA_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top1",
    "rag_top1_overall_summary.json"
)

SCALA_JUDGE = os.path.join(
    BASE_DIR,
    "results",
    "rag_top1_llm_judge",
    "rag_top1_llm_judge_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# def add_labels(ax, bars, fmt):
#     for bar in bars:
#         value = bar.get_height()

#         ax.annotate(
#             fmt.format(value),
#             xy=(
#                 bar.get_x() + bar.get_width() / 2,
#                 value
#             ),
#             xytext=(0, 4),
#             textcoords="offset points",
#             ha="center",
#             va="bottom",
#             fontsize=9
#         )


# ============================================================
# LOAD
# ============================================================

standard = load_json(STANDARD_SUMMARY)
plain_gmm = load_json(PLAIN_GMM_SUMMARY)
kmeans = load_json(KMEANS_SUMMARY)
scala = load_json(SCALA_SUMMARY)

standard_j = load_json(STANDARD_JUDGE)
plain_gmm_j = load_json(PLAIN_GMM_JUDGE)
kmeans_j = load_json(KMEANS_JUDGE)
scala_j = load_json(SCALA_JUDGE)


# ============================================================
# METHOD ORDER
# ============================================================

methods = [
    "Standard\nRAG",
    "Vanilla GMM\nTop-1",
    "CLIP +\nK-Means",
    "SCALA"
]

# ============================================================
# PUBLICATION FONT SETTINGS
# Designed for THREE side-by-side subfigures
# ============================================================

LABEL_SIZE = 17
TICK_SIZE = 14
VALUE_SIZE = 13

def add_labels(ax, bars, fmt, offset=4):
    for bar in bars:
        value = bar.get_height()

        ax.annotate(
            fmt.format(value),
            xy=(
                bar.get_x() + bar.get_width() / 2,
                value
            ),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            fontweight="bold"
        )


# ============================================================
# DATA
# ============================================================

rouge_values = [
    standard["avg_rougeL"],
    plain_gmm["avg_rougeL"],
    kmeans["avg_rougeL"],
    scala["gmm_avg_rougeL"]
]

relevant_values = [
    standard_j["relevant_percentage"],
    plain_gmm_j["relevant_percentage"],
    kmeans_j["relevant_percentage"],
    scala_j["relevant_percentage"]
]

candidate_reduction = [
    standard["avg_candidate_reduction"] * 100,
    plain_gmm["avg_candidate_reduction"] * 100,
    kmeans["avg_candidate_reduction"] * 100,
    scala["avg_candidate_reduction"] * 100
]


# ============================================================
# FIGURE 1: ROUGE-L
# ============================================================

fig, ax = plt.subplots(figsize=(6.0, 4.8))

bars = ax.bar(
    methods,
    rouge_values,
    width=0.65
)

add_labels(
    ax,
    bars,
    "{:.4f}"
)

ax.set_ylabel(
    "ROUGE-L",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE
)

# Give enough space for values above bars
ax.set_ylim(
    0,
    max(rouge_values) * 1.12
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

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ablation_component_rouge.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ablation_component_rouge.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 2: RELEVANT RESOLUTIONS
# ============================================================

fig, ax = plt.subplots(figsize=(6.0, 4.8))

bars = ax.bar(
    methods,
    relevant_values,
    width=0.65
)

add_labels(
    ax,
    bars,
    "{:.1f}%"
)

ax.set_ylabel(
    "Relevant Resolutions (%)",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

ax.set_ylim(0, 105)

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

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ablation_component_relevant.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ablation_component_relevant.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 3: CANDIDATE REDUCTION
# ============================================================

fig, ax = plt.subplots(figsize=(6.0, 4.8))

bars = ax.bar(
    methods,
    candidate_reduction,
    width=0.65
)

add_labels(
    ax,
    bars,
    "{:.2f}%"
)

ax.set_ylabel(
    "Candidate Reduction (%)",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

ax.set_ylim(0, 105)

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

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ablation_component_candidate_reduction.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ablation_component_candidate_reduction.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# PRINT VALUES
# ============================================================

print("\nCOMPONENT ABLATION")
print("=" * 75)

for i, method in enumerate(methods):
    print(
        f"{method.replace(chr(10), ' '):25s} "
        f"ROUGE-L = {rouge_values[i]:.4f} | "
        f"Relevant = {relevant_values[i]:.2f}% | "
        f"Candidate Reduction = {candidate_reduction[i]:.2f}%"
    )

print("\nFigures saved to:")
print(OUTPUT_DIR)