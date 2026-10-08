# ============================================================
# 08_plot_results.py
#
# Generates:
#   Figure 1: GMM model selection
#             (BIC, Silhouette, Intra-cluster cosine)
#
#   Figure 2: Validation routing trade-off
#             (candidate reduction vs routing recall)
#
#   Figure 2B: Top-5 routing recall across reference K
#
#   Figure 3: Retrieval efficiency comparison
#             (CLIP + K-Means vs Retrieval-Aware GMM)
#
#   Figure 4A: ROUGE-L comparison
#
#   Figure 4B: BERTScore comparison
#
#   Figure 4C: LLM-Judge comparison
#
#   Figure 5: End-to-end latency comparison
#
# No model training or API calls are made.
# ============================================================

import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"


# ------------------------------------------------------------
# GMM model selection
# ------------------------------------------------------------

GMM_FILE = os.path.join(
    BASE_DIR,
    "results",
    "gmm",
    "gmm_model_selection.csv"
)


# ------------------------------------------------------------
# GMM routing validation
# ------------------------------------------------------------

ROUTING_FILE = os.path.join(
    BASE_DIR,
    "results",
    "routing_validation",
    "routing_validation_summary.xlsx"
)


# ------------------------------------------------------------
# CLIP RAG + Retrieval-Aware GMM final results
# ------------------------------------------------------------

RAG_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference",
    "rag_overall_summary.json"
)


# ------------------------------------------------------------
# CLIP RAG + GMM LLM judge
# ------------------------------------------------------------

JUDGE_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "llm_judge",
    "llm_judge_summary.json"
)


# ------------------------------------------------------------
# K-Means RAG final results
# ------------------------------------------------------------

KMEANS_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_rag",
    "kmeans_overall_summary.json"
)


# ------------------------------------------------------------
# K-Means LLM judge
# ------------------------------------------------------------

KMEANS_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_llm_judge",
    "kmeans_llm_judge_summary.json"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

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
# VERIFY FILES
# ============================================================

required_files = [
    GMM_FILE,
    ROUTING_FILE,
    RAG_SUMMARY_FILE,
    JUDGE_SUMMARY_FILE,
    KMEANS_SUMMARY_FILE,
    KMEANS_JUDGE_FILE
]


missing_files = [
    f
    for f in required_files
    if not os.path.exists(f)
]


if missing_files:

    print("\nMissing files:")

    for f in missing_files:
        print(f)

    raise FileNotFoundError(
        "One or more required plotting inputs are missing."
    )


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150
})


METHOD_CLIP = "CLIP RAG"

METHOD_KMEANS = (
    "CLIP +\nK-Means"
)

METHOD_GMM = (
    "Retrieval-Aware\nGMM Clustering"
)


# ============================================================
# HELPERS
# ============================================================

def save_figure(
    fig,
    filename
):

    png_file = os.path.join(
        OUTPUT_DIR,
        filename + ".png"
    )

    pdf_file = os.path.join(
        OUTPUT_DIR,
        filename + ".pdf"
    )


    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight"
    )


    print("Saved:")
    print(" ", png_file)
    print(" ", pdf_file)


def add_bar_labels(
    ax,
    bars,
    fmt="{:.2f}",
    offset=4
):

    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            fmt.format(height),

            xy=(
                bar.get_x()
                +
                bar.get_width() / 2,
                height
            ),

            xytext=(
                0,
                offset
            ),

            textcoords="offset points",

            ha="center",
            va="bottom",

            fontsize=9
        )


def find_column(
    df,
    possible_names
):

    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_map:
            return lower_map[
                name.lower()
            ]

    raise KeyError(
        f"Could not find any of these columns: "
        f"{possible_names}\n"
        f"Available columns: "
        f"{df.columns.tolist()}"
    )


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 90)
print("LOADING RESULT FILES")
print("=" * 90)


gmm_df = pd.read_csv(
    GMM_FILE
)


routing_df = pd.read_excel(
    ROUTING_FILE
)


with open(
    RAG_SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as f:

    rag = json.load(f)


with open(
    JUDGE_SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as f:

    judge = json.load(f)


with open(
    KMEANS_SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as f:

    kmeans = json.load(f)


with open(
    KMEANS_JUDGE_FILE,
    "r",
    encoding="utf-8"
) as f:

    kmeans_judge = json.load(f)


print("\nGMM columns:")
print(
    gmm_df.columns.tolist()
)

print("\nRouting columns:")
print(
    routing_df.columns.tolist()
)


# ============================================================
# FIGURE 1
# GMM MODEL SELECTION
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FIGURE 1: GMM MODEL SELECTION"
)

print(
    "=" * 90
)


k_col = find_column(
    gmm_df,
    [
        "n_components",
        "components",
        "k",
        "K"
    ]
)


bic_col = find_column(
    gmm_df,
    [
        "bic",
        "BIC"
    ]
)


sil_col = find_column(
    gmm_df,
    [
        "silhouette_cosine",
        "silhouette_score",
        "silhouette"
    ]
)


intra_col = find_column(
    gmm_df,
    [
        "avg_intra_cluster_cosine",
        "intra_cluster_cosine",
        "mean_intra_cluster_cosine"
    ]
)


selected_k = 55


selected_row = gmm_df[
    gmm_df[k_col]
    ==
    selected_k
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        15,
        4.3
    )
)


# ------------------------------------------------------------
# BIC
# ------------------------------------------------------------

ax = axes[0]


ax.plot(
    gmm_df[k_col],
    gmm_df[bic_col],
    marker="o",
    linewidth=1.8
)


ax.axvline(
    selected_k,
    linestyle="--",
    alpha=0.6
)


if len(selected_row) == 1:

    selected_bic = (
        selected_row[
            bic_col
        ]
        .iloc[0]
    )

    ax.scatter(
        [selected_k],
        [selected_bic],
        s=90,
        zorder=5
    )

    ax.annotate(
        "Selected K=55",

        (
            selected_k,
            selected_bic
        ),

        xytext=(
            -60,
            20
        ),

        textcoords=
            "offset points",

        arrowprops=dict(
            arrowstyle="->"
        )
    )


ax.set_xlabel(
    "Number of GMM Components (K)"
)

ax.set_ylabel(
    "BIC"
)

ax.set_title(
    "(a) Bayesian Information Criterion"
)

ax.grid(
    alpha=0.25
)


# ------------------------------------------------------------
# Silhouette
# ------------------------------------------------------------

ax = axes[1]


ax.plot(
    gmm_df[k_col],
    gmm_df[sil_col],
    marker="o",
    linewidth=1.8
)


ax.axvline(
    selected_k,
    linestyle="--",
    alpha=0.6
)


if len(selected_row) == 1:

    selected_sil = (
        selected_row[
            sil_col
        ]
        .iloc[0]
    )

    ax.scatter(
        [selected_k],
        [selected_sil],
        s=90,
        zorder=5
    )


ax.set_xlabel(
    "Number of GMM Components (K)"
)

ax.set_ylabel(
    "Cosine Silhouette Score"
)

ax.set_title(
    "(b) Cluster Separation"
)

ax.grid(
    alpha=0.25
)


# ------------------------------------------------------------
# Intra-cluster cosine
# ------------------------------------------------------------

ax = axes[2]


ax.plot(
    gmm_df[k_col],
    gmm_df[intra_col],
    marker="o",
    linewidth=1.8
)


ax.axvline(
    selected_k,
    linestyle="--",
    alpha=0.6
)


if len(selected_row) == 1:

    selected_intra = (
        selected_row[
            intra_col
        ]
        .iloc[0]
    )

    ax.scatter(
        [selected_k],
        [selected_intra],
        s=90,
        zorder=5
    )


ax.set_xlabel(
    "Number of GMM Components (K)"
)

ax.set_ylabel(
    "Average Intra-Cluster Cosine Similarity"
)

ax.set_title(
    "(c) Semantic Cluster Tightness"
)

ax.grid(
    alpha=0.25
)


fig.tight_layout()


save_figure(
    fig,
    "fig1_gmm_model_selection"
)


plt.close(fig)


# ============================================================
# FIGURE 2
# VALIDATION ROUTING TRADE-OFF
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FIGURE 2: VALIDATION ROUTING TRADE-OFF"
)

print(
    "=" * 90
)


reference_col = find_column(
    routing_df,
    [
        "reference_k",
        "reference_K"
    ]
)


strategy_col = find_column(
    routing_df,
    [
        "strategy"
    ]
)


recall_col = find_column(
    routing_df,
    [
        "routing_recall"
    ]
)


reduction_col = find_column(
    routing_df,
    [
        "avg_candidate_reduction",
        "candidate_reduction"
    ]
)


# ------------------------------------------------------------
# Use conservative reference K = 10
# ------------------------------------------------------------

routing_10 = (
    routing_df[
        routing_df[
            reference_col
        ]
        ==
        10
    ]
    .copy()
)


desired_strategies = [
    "top_1_clusters",
    "top_2_clusters",
    "top_3_clusters",
    "top_5_clusters"
]


fixed_df = (
    routing_10[
        routing_10[
            strategy_col
        ]
        .isin(
            desired_strategies
        )
    ]
    .copy()
)


fixed_df[
    strategy_col
] = pd.Categorical(
    fixed_df[
        strategy_col
    ],

    categories=
        desired_strategies,

    ordered=True
)


fixed_df = (
    fixed_df
    .sort_values(
        strategy_col
    )
)


fixed_df[
    "routing_recall_pct"
] = (
    fixed_df[
        recall_col
    ]
    *
    100
)


fixed_df[
    "candidate_reduction_pct"
] = (
    fixed_df[
        reduction_col
    ]
    *
    100
)


label_map = {

    "top_1_clusters":
        "Top-1",

    "top_2_clusters":
        "Top-2",

    "top_3_clusters":
        "Top-3",

    "top_5_clusters":
        "Top-5"
}


fig, ax = plt.subplots(
    figsize=(
        7.3,
        5.2
    )
)


for _, row in fixed_df.iterrows():

    strategy = row[
        strategy_col
    ]

    x = row[
        "candidate_reduction_pct"
    ]

    y = row[
        "routing_recall_pct"
    ]


    if (
        strategy
        ==
        "top_5_clusters"
    ):

        ax.scatter(
            x,
            y,
            s=150,
            marker="*",
            zorder=5
        )

    else:

        ax.scatter(
            x,
            y,
            s=75,
            zorder=4
        )


    ax.annotate(
        label_map[
            strategy
        ],

        (
            x,
            y
        ),

        xytext=(
            6,
            6
        ),

        textcoords=
            "offset points"
    )


ax.plot(
    fixed_df[
        "candidate_reduction_pct"
    ],

    fixed_df[
        "routing_recall_pct"
    ],

    linestyle="--",

    alpha=0.6
)


ax.set_xlabel(
    "Candidate Reduction (%)"
)

ax.set_ylabel(
    "Routing Recall (%)"
)

ax.set_title(
    "Validation Routing Trade-off "
    "(Reference Top-10)"
)

ax.grid(
    alpha=0.25
)


fig.tight_layout()


save_figure(
    fig,
    "fig2_validation_routing_tradeoff"
)


plt.close(fig)


# ============================================================
# FIGURE 2B
# TOP-5 ROUTING ACROSS REFERENCE K
# ============================================================

top_strategy_df = (
    routing_df[
        routing_df[
            strategy_col
        ]
        ==
        "top_5_clusters"
    ]
    .copy()
)


top_strategy_df = (
    top_strategy_df
    .sort_values(
        reference_col
    )
)


fig, ax = plt.subplots(
    figsize=(
        6.5,
        4.7
    )
)


bars = ax.bar(

    top_strategy_df[
        reference_col
    ].astype(str),

    top_strategy_df[
        recall_col
    ]
    *
    100
)


add_bar_labels(
    ax,
    bars,
    "{:.2f}%"
)


ax.set_xlabel(
    "Reference Retrieval K"
)

ax.set_ylabel(
    "Routing Recall (%)"
)

ax.set_title(
    "Top-5 Cluster Routing Across Reference K"
)

ax.set_ylim(
    0,
    100
)

ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()


save_figure(
    fig,
    "fig2b_top5_routing_recall"
)


plt.close(fig)


# ============================================================
# FIGURE 3
# RETRIEVAL EFFICIENCY
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FIGURE 3: RETRIEVAL EFFICIENCY"
)

print(
    "=" * 90
)


methods = [
    METHOD_KMEANS,
    METHOD_GMM
]


candidate_reduction = [

    kmeans[
        "avg_candidate_reduction"
    ]
    *
    100,

    rag[
        "avg_candidate_reduction"
    ]
    *
    100
]


retrieval_overlap = [

    kmeans[
        "avg_clip_kmeans_topk_overlap"
    ]
    *
    100,

    rag[
        "avg_flat_gmm_topk_overlap"
    ]
    *
    100
]


x = np.arange(
    len(methods)
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(
        7.5,
        5
    )
)


bars1 = ax.bar(
    x - width / 2,
    candidate_reduction,
    width,
    label="Candidate Reduction"
)


bars2 = ax.bar(
    x + width / 2,
    retrieval_overlap,
    width,
    label="Top-5 Retrieval Overlap"
)


add_bar_labels(
    ax,
    bars1,
    "{:.2f}%"
)


add_bar_labels(
    ax,
    bars2,
    "{:.2f}%"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    methods
)

ax.set_ylabel(
    "Percentage (%)"
)

ax.set_ylim(
    0,
    105
)

ax.set_title(
    "Retrieval Efficiency"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()


save_figure(
    fig,
    "fig3_retrieval_efficiency_comparison"
)


plt.close(fig)


# ============================================================
# FIGURE 4A
# ROUGE-L
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FIGURE 4A: ROUGE-L"
)

print(
    "=" * 90
)


methods_quality = [
    METHOD_CLIP,
    METHOD_KMEANS,
    METHOD_GMM
]


rouge_values = [

    rag[
        "flat_avg_rougeL"
    ],

    kmeans[
        "avg_rougeL"
    ],

    rag[
        "gmm_avg_rougeL"
    ]
]


fig, ax = plt.subplots(
    figsize=(
        7.5,
        5
    )
)


bars = ax.bar(
    methods_quality,
    rouge_values
)


add_bar_labels(
    ax,
    bars,
    "{:.4f}"
)


ax.set_ylabel(
    "ROUGE-L"
)


ax.set_ylim(
    0,
    max(
        rouge_values
    )
    *
    1.20
)


ax.set_title(
    "Generated Resolution Quality: ROUGE-L"
)


ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()


save_figure(
    fig,
    "fig4a_rouge_comparison"
)


plt.close(fig)


# ============================================================
# FIGURE 4B
# BERTSCORE
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FIGURE 4B: BERTSCORE"
)

print(
    "=" * 90
)


bert_values = [

    rag[
        "flat_avg_bertscore_f1"
    ],

    kmeans[
        "avg_bertscore_f1"
    ],

    rag[
        "gmm_avg_bertscore_f1"
    ]
]


fig, ax = plt.subplots(
    figsize=(
        7.5,
        5
    )
)


bars = ax.bar(
    methods_quality,
    bert_values
)


add_bar_labels(
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
    "Generated Resolution Quality: BERTScore"
)


ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()


save_figure(
    fig,
    "fig4b_bertscore_comparison"
)


plt.close(fig)


# ============================================================
# FIGURE 4C
# LLM JUDGE
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FIGURE 4C: LLM JUDGE"
)

print(
    "=" * 90
)


categories = [
    "Relevant",
    "Closer",
    "Inadequate"
]


clip_judge = [

    judge[
        "flat_rag"
    ][
        "relevant_percentage"
    ],

    judge[
        "flat_rag"
    ][
        "closer_percentage"
    ],

    judge[
        "flat_rag"
    ][
        "inadequate_percentage"
    ]
]


kmeans_judge_values = [

    kmeans_judge[
        "relevant_percentage"
    ],

    kmeans_judge[
        "closer_percentage"
    ],

    kmeans_judge[
        "inadequate_percentage"
    ]
]


gmm_judge = [

    judge[
        "gmm_rag"
    ][
        "relevant_percentage"
    ],

    judge[
        "gmm_rag"
    ][
        "closer_percentage"
    ],

    judge[
        "gmm_rag"
    ][
        "inadequate_percentage"
    ]
]


x = np.arange(
    len(categories)
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(
        8.5,
        5
    )
)


bars1 = ax.bar(
    x - width,
    clip_judge,
    width,
    label="CLIP RAG"
)


bars2 = ax.bar(
    x,
    kmeans_judge_values,
    width,
    label="CLIP + K-Means"
)


bars3 = ax.bar(
    x + width,
    gmm_judge,
    width,
    label="Retrieval-Aware GMM Clustering"
)


add_bar_labels(
    ax,
    bars1,
    "{:.2f}%"
)


add_bar_labels(
    ax,
    bars2,
    "{:.2f}%"
)


add_bar_labels(
    ax,
    bars3,
    "{:.2f}%"
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
    100
)

ax.set_title(
    "LLM-Judge Evaluation"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()


save_figure(
    fig,
    "fig4c_llm_judge_comparison"
)


plt.close(fig)


# ============================================================
# FIGURE 5
# END-TO-END LATENCY
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FIGURE 5: END-TO-END LATENCY"
)

print(
    "=" * 90
)


latency_methods = [
    METHOD_CLIP,
    METHOD_KMEANS,
    METHOD_GMM
]


mean_e2e = [

    rag[
        "flat_e2e_latency"
    ][
        "mean_ms"
    ]
    /
    1000.0,

    kmeans[
        "e2e_latency"
    ][
        "mean_ms"
    ]
    /
    1000.0,

    rag[
        "gmm_e2e_latency"
    ][
        "mean_ms"
    ]
    /
    1000.0
]


p95_e2e = [

    rag[
        "flat_e2e_latency"
    ][
        "p95_ms"
    ]
    /
    1000.0,

    kmeans[
        "e2e_latency"
    ][
        "p95_ms"
    ]
    /
    1000.0,

    rag[
        "gmm_e2e_latency"
    ][
        "p95_ms"
    ]
    /
    1000.0
]


x = np.arange(
    len(
        latency_methods
    )
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(
        8.5,
        5
    )
)


bars1 = ax.bar(
    x - width / 2,
    mean_e2e,
    width,
    label="Mean E2E"
)


bars2 = ax.bar(
    x + width / 2,
    p95_e2e,
    width,
    label="P95 E2E"
)


add_bar_labels(
    ax,
    bars1,
    "{:.3f}s"
)


add_bar_labels(
    ax,
    bars2,
    "{:.3f}s"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    latency_methods
)

ax.set_ylabel(
    "Latency (seconds)"
)

ax.set_title(
    "End-to-End RAG Latency"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()


save_figure(
    fig,
    "fig5_e2e_latency_comparison"
)


plt.close(fig)


# ============================================================
# DONE
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "ALL FIGURES GENERATED"
)

print(
    "=" * 90
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nGenerated figures:"
)


print(
    "1. fig1_gmm_model_selection"
)

print(
    "2. fig2_validation_routing_tradeoff"
)

print(
    "3. fig2b_top5_routing_recall"
)

print(
    "4. fig3_retrieval_efficiency_comparison"
)

print(
    "5. fig4a_rouge_comparison"
)

print(
    "6. fig4b_bertscore_comparison"
)

print(
    "7. fig4c_llm_judge_comparison"
)

print(
    "8. fig5_e2e_latency_comparison"
)