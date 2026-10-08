import os
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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


PLAIN_TOP1_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_rag",
    "plain_gmm_overall_summary.json"
)

PLAIN_TOP1_JUDGE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_llm_judge",
    "plain_gmm_llm_judge_summary.json"
)


PLAIN_TOP3_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_top3_rag",
    "plain_gmm_top3_overall_summary.json"
)

PLAIN_TOP3_JUDGE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_top3_llm_judge",
    "plain_gmm_top3_llm_judge_summary.json"
)


PROP_TOP1_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top1",
    "rag_top1_overall_summary.json"
)

PROP_TOP1_JUDGE = os.path.join(
    BASE_DIR,
    "results",
    "rag_top1_llm_judge",
    "rag_top1_llm_judge_summary.json"
)


PROP_TOP3_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top3",
    "rag_top3_overall_summary.json"
)

PROP_TOP3_JUDGE = os.path.join(
    BASE_DIR,
    "results",
    "rag_top3_llm_judge",
    "rag_top3_llm_judge_summary.json"
)


IFKG_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "ifkg_inference",
    "ifkg_overall_summary.json"
)

IFKG_JUDGE = os.path.join(
    BASE_DIR,
    "results",
    "ifkg_llm_judge",
    "ifkg_llm_judge_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save(fig, name):

    png = os.path.join(
        OUTPUT_DIR,
        name + ".png"
    )

    pdf = os.path.join(
        OUTPUT_DIR,
        name + ".pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:")
    print(png)
    print(pdf)


def add_labels(
    ax,
    bars,
    fmt="{:.3f}",
    fontsize=8
):

    for bar in bars:

        h = bar.get_height()

        ax.annotate(
            fmt.format(h),
            (
                bar.get_x()
                +
                bar.get_width() / 2,
                h
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=fontsize
        )


# ============================================================
# LOAD
# ============================================================

standard = load_json(
    STANDARD_SUMMARY
)

kmeans = load_json(
    KMEANS_SUMMARY
)

plain1 = load_json(
    PLAIN_TOP1_SUMMARY
)

plain3 = load_json(
    PLAIN_TOP3_SUMMARY
)

prop1 = load_json(
    PROP_TOP1_SUMMARY
)

prop3 = load_json(
    PROP_TOP3_SUMMARY
)

ifkg = load_json(
    IFKG_SUMMARY
)


standard_j = load_json(
    STANDARD_JUDGE
)

kmeans_j = load_json(
    KMEANS_JUDGE
)

plain1_j = load_json(
    PLAIN_TOP1_JUDGE
)

plain3_j = load_json(
    PLAIN_TOP3_JUDGE
)

prop1_j = load_json(
    PROP_TOP1_JUDGE
)

prop3_j = load_json(
    PROP_TOP3_JUDGE
)

ifkg_j = load_json(
    IFKG_JUDGE
)


# ============================================================
# METHOD LABELS
# ============================================================

METHODS_ALL = [
    "Standard\nRAG",
    "CLIP +\nK-Means",
    "Plain GMM\nTop-1",
    "Plain GMM\nTop-3",
    "Proposed\nTop-1",
    "Proposed\nTop-3",
    "IFKG"
]


# ============================================================
# FIGURE 1
# CANDIDATE REDUCTION
#
# IFKG excluded because not comparable.
# ============================================================

methods_candidate = [
    "Standard\nRAG",
    "CLIP +\nK-Means",
    "Plain GMM\nTop-1",
    "Plain GMM\nTop-3",
    "Proposed\nTop-1",
    "Proposed\nTop-3"
]


candidate_reduction = [
    standard["avg_candidate_reduction"] * 100,
    kmeans["avg_candidate_reduction"] * 100,
    plain1["avg_candidate_reduction"] * 100,
    plain3["avg_candidate_reduction"] * 100,
    prop1["avg_candidate_reduction"] * 100,
    prop3["avg_candidate_reduction"] * 100
]


fig, ax = plt.subplots(
    figsize=(10.5, 5)
)

bars = ax.bar(
    methods_candidate,
    candidate_reduction
)

add_labels(
    ax,
    bars,
    "{:.2f}%"
)

ax.set_ylabel(
    "Candidate Reduction (%)"
)

ax.set_ylim(
    0,
    105
)

ax.set_title(
    "Retrieval Candidate Reduction"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save(
    fig,
    "fig_candidate_reduction"
)


# ============================================================
# FIGURE 2
# TOP-5 RETRIEVAL OVERLAP
#
# Standard RAG + IFKG excluded.
# ============================================================

overlap_methods = [
    "CLIP +\nK-Means",
    "Plain GMM\nTop-1",
    "Plain GMM\nTop-3",
    "Proposed\nTop-1",
    "Proposed\nTop-3"
]


overlap_values = [
    kmeans["avg_clip_kmeans_topk_overlap"] * 100,
    plain1["avg_minilm_gmm_topk_overlap"] * 100,
    plain3["avg_minilm_top3_gmm_topk_overlap"] * 100,
    prop1["avg_flat_gmm_topk_overlap"] * 100,
    prop3["avg_flat_gmm_topk_overlap"] * 100
]


fig, ax = plt.subplots(
    figsize=(9.5, 5)
)

bars = ax.bar(
    overlap_methods,
    overlap_values
)

add_labels(
    ax,
    bars,
    "{:.2f}%"
)

ax.set_ylabel(
    "Top-5 Retrieval Overlap (%)"
)

ax.set_ylim(
    0,
    105
)

ax.set_title(
    "Top-5 Retrieval Preservation"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save(
    fig,
    "fig_top5_retrieval_overlap"
)


# ============================================================
# FIGURE 3
# ROUGE-L
# ============================================================

rouge_values = [
    standard["avg_rougeL"],
    kmeans["avg_rougeL"],
    plain1["avg_rougeL"],
    plain3["avg_rougeL"],
    prop1["gmm_avg_rougeL"],
    prop3["gmm_avg_rougeL"],
    ifkg["avg_rougeL"]
]


fig, ax = plt.subplots(
    figsize=(11.5, 5)
)

bars = ax.bar(
    METHODS_ALL,
    rouge_values
)

add_labels(
    ax,
    bars,
    "{:.4f}"
)

ax.set_ylabel(
    "ROUGE-L"
)

ax.set_title(
    "Generated Resolution Quality: ROUGE-L"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save(
    fig,
    "fig_rouge_all_methods"
)


# ============================================================
# FIGURE 4
# BERTSCORE
# ============================================================

bert_values = [
    standard["avg_bertscore_f1"],
    kmeans["avg_bertscore_f1"],
    plain1["avg_bertscore_f1"],
    plain3["avg_bertscore_f1"],
    prop1["gmm_avg_bertscore_f1"],
    prop3["gmm_avg_bertscore_f1"],
    ifkg["avg_bertscore_f1"]
]


fig, ax = plt.subplots(
    figsize=(11.5, 5)
)

bars = ax.bar(
    METHODS_ALL,
    bert_values
)

add_labels(
    ax,
    bars,
    "{:.4f}"
)

ax.set_ylabel(
    "BERTScore F1"
)

ax.set_ylim(
    0,
    1
)

ax.set_title(
    "Generated Resolution Quality: BERTScore F1"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save(
    fig,
    "fig_bertscore_all_methods"
)


# ============================================================
# FIGURE 5
# LLM JUDGE
# ============================================================

categories = [
    "Relevant",
    "Closer",
    "Inadequate"
]


judge_methods = [
    standard_j,
    kmeans_j,
    plain1_j,
    plain3_j,
    prop1_j,
    prop3_j,
    ifkg_j
]


judge_labels = [
    "Standard RAG",
    "CLIP + K-Means",
    "Plain GMM Top-1",
    "Plain GMM Top-3",
    "Proposed Top-1",
    "Proposed Top-3",
    "IFKG"
]


judge_values = []


for item in judge_methods:

    judge_values.append([
        item["relevant_percentage"],
        item["closer_percentage"],
        item["inadequate_percentage"]
    ])


x = np.arange(
    len(categories)
)

width = 0.11


fig, ax = plt.subplots(
    figsize=(14, 6)
)


bars_all = []


offsets = [
    -3 * width,
    -2 * width,
    -1 * width,
    0,
    1 * width,
    2 * width,
    3 * width
]


for i in range(
    len(judge_values)
):

    bars = ax.bar(
        x + offsets[i],
        judge_values[i],
        width,
        label=judge_labels[i]
    )

    bars_all.append(
        bars
    )


for bars in bars_all:

    add_labels(
        ax,
        bars,
        "{:.1f}%",
        fontsize=6
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    categories
)

ax.set_ylabel(
    "Queries (%)"
)

ax.set_ylim(
    0,
    105
)

ax.set_title(
    "LLM-Judge Evaluation"
)

ax.legend(
    ncol=4,
    fontsize=8
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save(
    fig,
    "fig_llm_judge_all_methods"
)


# ============================================================
# FIGURE 6
# IFKG-SPECIFIC RETRIEVAL HEALTH
# ============================================================

ifkg_metrics = [
    "KG Hit Rate",
    "Cypher Success"
]

ifkg_values = [
    ifkg["kg_hit_rate"] * 100,
    ifkg["cypher_success_rate"] * 100
]


fig, ax = plt.subplots(
    figsize=(6.5, 5)
)

bars = ax.bar(
    ifkg_metrics,
    ifkg_values
)

add_labels(
    ax,
    bars,
    "{:.2f}%"
)

ax.set_ylim(
    0,
    105
)

ax.set_ylabel(
    "Success Rate (%)"
)

ax.set_title(
    "IFKG Retrieval Reliability"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

save(
    fig,
    "fig_ifkg_retrieval_reliability"
)


print("\nQUALITY / RETRIEVAL FIGURES COMPLETE")