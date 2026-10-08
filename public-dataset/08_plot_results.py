# ============================================================
# 08_plot_results.py
#
# Final comparison:
#
#   Standard RAG
#   Plain GMM-RAG
#   CLIP + K-Means
#   Retrieval-Aware GMM Top-3
#   Retrieval-Aware GMM Top-5
#
# Also retains:
#   GMM model-selection plots
#   GMM validation-routing plots
#
# No API calls.
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


GMM_FILE = os.path.join(
    BASE_DIR,
    "results",
    "gmm",
    "gmm_model_selection.csv"
)


ROUTING_FILE = os.path.join(
    BASE_DIR,
    "results",
    "routing_validation",
    "routing_validation_summary.xlsx"
)


# ============================================================
# TOP-5 PROPOSED
# ============================================================

TOP5_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference",
    "rag_overall_summary.json"
)


TOP5_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "llm_judge",
    "llm_judge_summary.json"
)


# ============================================================
# TOP-3 PROPOSED
# ============================================================

TOP3_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top3",
    "rag_top3_overall_summary.json"
)


TOP3_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_top3_llm_judge",
    "rag_top3_llm_judge_summary.json"
)


# ============================================================
# K-MEANS
# ============================================================

KMEANS_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_rag",
    "kmeans_overall_summary.json"
)


KMEANS_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_llm_judge",
    "kmeans_llm_judge_summary.json"
)


# ============================================================
# PLAIN GMM
# ============================================================

PLAIN_GMM_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_rag",
    "plain_gmm_overall_summary.json"
)


PLAIN_GMM_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_llm_judge",
    "plain_gmm_llm_judge_summary.json"
)


# ============================================================
# STANDARD RAG
# ============================================================

STANDARD_RAG_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "standard_rag",
    "standard_rag_overall_summary.json"
)


STANDARD_RAG_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "standard_rag_llm_judge",
    "standard_rag_llm_judge_summary.json"
)


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
# METHOD NAMES
# ============================================================

METHOD_STANDARD = "Standard\nRAG"

METHOD_PLAIN_GMM = "Plain\nGMM-RAG"

METHOD_KMEANS = "CLIP +\nK-Means"

METHOD_TOP3 = "Retrieval-Aware\nGMM Top-3"

METHOD_TOP5 = "Retrieval-Aware\nGMM Top-5"


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.dpi": 150
})


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
    print(png_file)
    print(pdf_file)


def add_bar_labels(
    ax,
    bars,
    fmt="{:.2f}",
    offset=4,
    fontsize=8
):

    for bar in bars:

        height = bar.get_height()

        if np.isnan(height):
            continue

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
            fontsize=fontsize
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
        f"Could not find columns: "
        f"{possible_names}"
    )


# ============================================================
# VERIFY FILES
# ============================================================

required_files = [
    GMM_FILE,
    ROUTING_FILE,

    TOP5_SUMMARY_FILE,
    TOP5_JUDGE_FILE,

    TOP3_SUMMARY_FILE,
    TOP3_JUDGE_FILE,

    KMEANS_SUMMARY_FILE,
    KMEANS_JUDGE_FILE,

    PLAIN_GMM_SUMMARY_FILE,
    PLAIN_GMM_JUDGE_FILE,

    STANDARD_RAG_SUMMARY_FILE,
    STANDARD_RAG_JUDGE_FILE
]


missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]


if missing_files:

    print("\nMissing files:")

    for path in missing_files:
        print(path)

    raise FileNotFoundError(
        "One or more result files are missing."
    )


# ============================================================
# LOAD FILES
# ============================================================

gmm_df = pd.read_csv(
    GMM_FILE
)

routing_df = pd.read_excel(
    ROUTING_FILE
)


with open(
    TOP5_SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as f:
    top5 = json.load(f)


with open(
    TOP5_JUDGE_FILE,
    "r",
    encoding="utf-8"
) as f:
    top5_judge = json.load(f)


with open(
    TOP3_SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as f:
    top3 = json.load(f)


with open(
    TOP3_JUDGE_FILE,
    "r",
    encoding="utf-8"
) as f:
    top3_judge = json.load(f)


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


with open(
    PLAIN_GMM_SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as f:
    plain_gmm = json.load(f)


with open(
    PLAIN_GMM_JUDGE_FILE,
    "r",
    encoding="utf-8"
) as f:
    plain_gmm_judge = json.load(f)


with open(
    STANDARD_RAG_SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as f:
    standard_rag = json.load(f)


with open(
    STANDARD_RAG_JUDGE_FILE,
    "r",
    encoding="utf-8"
) as f:
    standard_rag_judge = json.load(f)


# ============================================================
# FIGURE 1
# GMM MODEL SELECTION
# ============================================================

k_col = find_column(
    gmm_df,
    ["n_components", "components", "k"]
)

bic_col = find_column(
    gmm_df,
    ["bic"]
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
    gmm_df[k_col] == selected_k
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.3)
)


# BIC
ax = axes[0]

ax.plot(
    gmm_df[k_col],
    gmm_df[bic_col],
    marker="o"
)

ax.axvline(
    selected_k,
    linestyle="--",
    alpha=0.6
)

if len(selected_row) == 1:

    value = selected_row[
        bic_col
    ].iloc[0]

    ax.scatter(
        selected_k,
        value,
        s=90
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


# Silhouette
ax = axes[1]

ax.plot(
    gmm_df[k_col],
    gmm_df[sil_col],
    marker="o"
)

ax.axvline(
    selected_k,
    linestyle="--",
    alpha=0.6
)

if len(selected_row) == 1:

    value = selected_row[
        sil_col
    ].iloc[0]

    ax.scatter(
        selected_k,
        value,
        s=90
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


# Intra-cluster
ax = axes[2]

ax.plot(
    gmm_df[k_col],
    gmm_df[intra_col],
    marker="o"
)

ax.axvline(
    selected_k,
    linestyle="--",
    alpha=0.6
)

if len(selected_row) == 1:

    value = selected_row[
        intra_col
    ].iloc[0]

    ax.scatter(
        selected_k,
        value,
        s=90
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

reference_col = find_column(
    routing_df,
    ["reference_k"]
)

strategy_col = find_column(
    routing_df,
    ["strategy"]
)

recall_col = find_column(
    routing_df,
    ["routing_recall"]
)

reduction_col = find_column(
    routing_df,
    [
        "avg_candidate_reduction",
        "candidate_reduction"
    ]
)


routing_10 = routing_df[
    routing_df[
        reference_col
    ] == 10
].copy()


desired_strategies = [
    "top_1_clusters",
    "top_2_clusters",
    "top_3_clusters",
    "top_5_clusters"
]


fixed_df = routing_10[
    routing_10[
        strategy_col
    ].isin(
        desired_strategies
    )
].copy()


fixed_df[
    strategy_col
] = pd.Categorical(
    fixed_df[
        strategy_col
    ],
    categories=desired_strategies,
    ordered=True
)


fixed_df = fixed_df.sort_values(
    strategy_col
)


fixed_df[
    "routing_recall_pct"
] = (
    fixed_df[
        recall_col
    ] * 100
)


fixed_df[
    "candidate_reduction_pct"
] = (
    fixed_df[
        reduction_col
    ] * 100
)


label_map = {
    "top_1_clusters": "Top-1",
    "top_2_clusters": "Top-2",
    "top_3_clusters": "Top-3",
    "top_5_clusters": "Top-5"
}


fig, ax = plt.subplots(
    figsize=(7.3, 5.2)
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

    marker = (
        "*"
        if strategy
        in [
            "top_3_clusters",
            "top_5_clusters"
        ]
        else
        "o"
    )

    size = (
        130
        if strategy
        in [
            "top_3_clusters",
            "top_5_clusters"
        ]
        else
        75
    )

    ax.scatter(
        x,
        y,
        marker=marker,
        s=size
    )

    ax.annotate(
        label_map[
            strategy
        ],
        (x, y),
        xytext=(6, 6),
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
# FIGURE 3A
# CANDIDATE REDUCTION
# ============================================================

methods_all = [
    METHOD_STANDARD,
    METHOD_PLAIN_GMM,
    METHOD_KMEANS,
    METHOD_TOP3,
    METHOD_TOP5
]


candidate_reduction = [

    standard_rag[
        "avg_candidate_reduction"
    ] * 100,

    plain_gmm[
        "avg_candidate_reduction"
    ] * 100,

    kmeans[
        "avg_candidate_reduction"
    ] * 100,

    top3[
        "avg_candidate_reduction"
    ] * 100,

    top5[
        "avg_candidate_reduction"
    ] * 100
]


fig, ax = plt.subplots(
    figsize=(9.5, 5)
)


bars = ax.bar(
    methods_all,
    candidate_reduction
)


add_bar_labels(
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

save_figure(
    fig,
    "fig3a_candidate_reduction_top3_top5"
)

plt.close(fig)


# ============================================================
# FIGURE 3B
# TOP-5 RETRIEVAL PRESERVATION
# ============================================================

overlap_methods = [
    METHOD_PLAIN_GMM,
    METHOD_KMEANS,
    METHOD_TOP3,
    METHOD_TOP5
]


retrieval_overlap = [

    plain_gmm[
        "avg_minilm_gmm_topk_overlap"
    ] * 100,

    kmeans[
        "avg_clip_kmeans_topk_overlap"
    ] * 100,

    top3[
        "avg_flat_gmm_topk_overlap"
    ] * 100,

    top5[
        "avg_flat_gmm_topk_overlap"
    ] * 100
]


fig, ax = plt.subplots(
    figsize=(9, 5)
)


bars = ax.bar(
    overlap_methods,
    retrieval_overlap
)


add_bar_labels(
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

save_figure(
    fig,
    "fig3b_top5_retrieval_overlap_top3_top5"
)

plt.close(fig)


# ============================================================
# FIGURE 4A
# ROUGE-L
# ============================================================

rouge_values = [

    standard_rag[
        "avg_rougeL"
    ],

    plain_gmm[
        "avg_rougeL"
    ],

    kmeans[
        "avg_rougeL"
    ],

    top3[
        "gmm_avg_rougeL"
    ],

    top5[
        "gmm_avg_rougeL"
    ]
]


fig, ax = plt.subplots(
    figsize=(9.5, 5)
)


bars = ax.bar(
    methods_all,
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
    ) * 1.20
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
    "fig4a_rouge_top3_top5"
)

plt.close(fig)


# ============================================================
# FIGURE 4B
# BERTSCORE
# ============================================================

bert_values = [

    standard_rag[
        "avg_bertscore_f1"
    ],

    plain_gmm[
        "avg_bertscore_f1"
    ],

    kmeans[
        "avg_bertscore_f1"
    ],

    top3[
        "gmm_avg_bertscore_f1"
    ],

    top5[
        "gmm_avg_bertscore_f1"
    ]
]


fig, ax = plt.subplots(
    figsize=(9.5, 5)
)


bars = ax.bar(
    methods_all,
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
    "fig4b_bertscore_top3_top5"
)

plt.close(fig)


# ============================================================
# FIGURE 4C
# LLM JUDGE
# ============================================================

categories = [
    "Relevant",
    "Closer",
    "Inadequate"
]


standard_values = [
    standard_rag_judge[
        "relevant_percentage"
    ],
    standard_rag_judge[
        "closer_percentage"
    ],
    standard_rag_judge[
        "inadequate_percentage"
    ]
]


plain_values = [
    plain_gmm_judge[
        "relevant_percentage"
    ],
    plain_gmm_judge[
        "closer_percentage"
    ],
    plain_gmm_judge[
        "inadequate_percentage"
    ]
]


kmeans_values = [
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


top3_values = [
    top3_judge[
        "relevant_percentage"
    ],
    top3_judge[
        "closer_percentage"
    ],
    top3_judge[
        "inadequate_percentage"
    ]
]


top5_values = [
    top5_judge[
        "gmm_rag"
    ][
        "relevant_percentage"
    ],
    top5_judge[
        "gmm_rag"
    ][
        "closer_percentage"
    ],
    top5_judge[
        "gmm_rag"
    ][
        "inadequate_percentage"
    ]
]


x = np.arange(
    len(categories)
)

width = 0.15


fig, ax = plt.subplots(
    figsize=(11, 5.5)
)


bars1 = ax.bar(
    x - 2 * width,
    standard_values,
    width,
    label="Standard RAG"
)


bars2 = ax.bar(
    x - width,
    plain_values,
    width,
    label="Plain GMM-RAG"
)


bars3 = ax.bar(
    x,
    kmeans_values,
    width,
    label="CLIP + K-Means"
)


bars4 = ax.bar(
    x + width,
    top3_values,
    width,
    label="Retrieval-Aware GMM Top-3"
)


bars5 = ax.bar(
    x + 2 * width,
    top5_values,
    width,
    label="Retrieval-Aware GMM Top-5"
)


for bars in [
    bars1,
    bars2,
    bars3,
    bars4,
    bars5
]:

    add_bar_labels(
        ax,
        bars,
        "{:.2f}%",
        offset=3,
        fontsize=7
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
    ncol=2
)

ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()

save_figure(
    fig,
    "fig4c_llm_judge_top3_top5"
)

plt.close(fig)


# ============================================================
# FIGURE 5
# E2E LATENCY
# ============================================================

mean_e2e = [

    standard_rag[
        "e2e_latency"
    ][
        "mean_ms"
    ] / 1000,

    plain_gmm[
        "e2e_latency"
    ][
        "mean_ms"
    ] / 1000,

    kmeans[
        "e2e_latency"
    ][
        "mean_ms"
    ] / 1000,

    top3[
        "gmm_e2e_latency"
    ][
        "mean_ms"
    ] / 1000,

    top5[
        "gmm_e2e_latency"
    ][
        "mean_ms"
    ] / 1000
]


p95_e2e = [

    standard_rag[
        "e2e_latency"
    ][
        "p95_ms"
    ] / 1000,

    plain_gmm[
        "e2e_latency"
    ][
        "p95_ms"
    ] / 1000,

    kmeans[
        "e2e_latency"
    ][
        "p95_ms"
    ] / 1000,

    top3[
        "gmm_e2e_latency"
    ][
        "p95_ms"
    ] / 1000,

    top5[
        "gmm_e2e_latency"
    ][
        "p95_ms"
    ] / 1000
]


x = np.arange(
    len(methods_all)
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(10, 5)
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
    methods_all
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
    "fig5_e2e_latency_top3_top5"
)

plt.close(fig)


print(
    "\nALL FIGURES GENERATED"
)