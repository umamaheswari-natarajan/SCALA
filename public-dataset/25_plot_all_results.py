import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# GMM MODEL SELECTION / ROUTING VALIDATION
# ============================================================

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
# STANDARD RAG
# ============================================================

STANDARD_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "standard_rag",
    "standard_rag_overall_summary.json"
)

STANDARD_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "standard_rag_llm_judge",
    "standard_rag_llm_judge_summary.json"
)


# ============================================================
# CLIP + K-MEANS
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
# PLAIN GMM TOP-1
# ============================================================

PLAIN_TOP1_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_rag",
    "plain_gmm_overall_summary.json"
)

PLAIN_TOP1_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_llm_judge",
    "plain_gmm_llm_judge_summary.json"
)


# ============================================================
# PLAIN GMM TOP-3
# ============================================================

PLAIN_TOP3_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_top3_rag",
    "plain_gmm_top3_overall_summary.json"
)

PLAIN_TOP3_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_top3_llm_judge",
    "plain_gmm_top3_llm_judge_summary.json"
)


# ============================================================
# PROPOSED TOP-1
# ============================================================

PROPOSED_TOP1_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top1",
    "rag_top1_overall_summary.json"
)

PROPOSED_TOP1_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_top1_llm_judge",
    "rag_top1_llm_judge_summary.json"
)


# ============================================================
# PROPOSED TOP-3
# ============================================================

PROPOSED_TOP3_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top3",
    "rag_top3_overall_summary.json"
)

PROPOSED_TOP3_JUDGE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_top3_llm_judge",
    "rag_top3_llm_judge_summary.json"
)


# ============================================================
# OUTPUT
# ============================================================

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
# METHOD LABELS
# ============================================================

METHOD_STANDARD = "Standard\nRAG"

METHOD_KMEANS = "CLIP +\nK-Means"

METHOD_PLAIN_TOP1 = "Plain GMM\nTop-1"

METHOD_PLAIN_TOP3 = "Plain GMM\nTop-3"

METHOD_PROP_TOP1 = "Proposed\nTop-1"

METHOD_PROP_TOP3 = "Proposed\nTop-3"


METHODS_ALL = [
    METHOD_STANDARD,
    METHOD_KMEANS,
    METHOD_PLAIN_TOP1,
    METHOD_PLAIN_TOP3,
    METHOD_PROP_TOP1,
    METHOD_PROP_TOP3
]


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

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


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

    print("\nSaved:")
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
            (
                bar.get_x()
                +
                bar.get_width() / 2,
                height
            ),
            xytext=(0, offset),
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
        column.lower(): column
        for column in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_map:

            return lower_map[
                name.lower()
            ]

    raise KeyError(
        f"Could not find any of: "
        f"{possible_names}\n"
        f"Available columns: "
        f"{df.columns.tolist()}"
    )


# ============================================================
# VERIFY FILES
# ============================================================

required_files = [
    GMM_FILE,
    ROUTING_FILE,

    STANDARD_SUMMARY_FILE,
    STANDARD_JUDGE_FILE,

    KMEANS_SUMMARY_FILE,
    KMEANS_JUDGE_FILE,

    PLAIN_TOP1_SUMMARY_FILE,
    PLAIN_TOP1_JUDGE_FILE,

    PLAIN_TOP3_SUMMARY_FILE,
    PLAIN_TOP3_JUDGE_FILE,

    PROPOSED_TOP1_SUMMARY_FILE,
    PROPOSED_TOP1_JUDGE_FILE,

    PROPOSED_TOP3_SUMMARY_FILE,
    PROPOSED_TOP3_JUDGE_FILE
]


missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]


if missing_files:

    print("\nMissing result files:")

    for path in missing_files:
        print(path)

    raise FileNotFoundError(
        "One or more required result files are missing."
    )


# ============================================================
# LOAD DATA
# ============================================================

gmm_df = pd.read_csv(
    GMM_FILE
)

routing_df = pd.read_excel(
    ROUTING_FILE
)


standard = load_json(
    STANDARD_SUMMARY_FILE
)

standard_judge = load_json(
    STANDARD_JUDGE_FILE
)


kmeans = load_json(
    KMEANS_SUMMARY_FILE
)

kmeans_judge = load_json(
    KMEANS_JUDGE_FILE
)


plain_top1 = load_json(
    PLAIN_TOP1_SUMMARY_FILE
)

plain_top1_judge = load_json(
    PLAIN_TOP1_JUDGE_FILE
)


plain_top3 = load_json(
    PLAIN_TOP3_SUMMARY_FILE
)

plain_top3_judge = load_json(
    PLAIN_TOP3_JUDGE_FILE
)


prop_top1 = load_json(
    PROPOSED_TOP1_SUMMARY_FILE
)

prop_top1_judge = load_json(
    PROPOSED_TOP1_JUDGE_FILE
)


prop_top3 = load_json(
    PROPOSED_TOP3_SUMMARY_FILE
)

prop_top3_judge = load_json(
    PROPOSED_TOP3_JUDGE_FILE
)


# ============================================================
# FIGURE 1
# GMM MODEL SELECTION
# ============================================================

k_col = find_column(
    gmm_df,
    [
        "n_components",
        "components",
        "k"
    ]
)

bic_col = find_column(
    gmm_df,
    [
        "bic"
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
    gmm_df[
        k_col
    ] == selected_k
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.3)
)


# ------------------------------------------------------------
# BIC
# ------------------------------------------------------------

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

    ax.annotate(
        "Selected K=55",
        (
            selected_k,
            value
        ),
        xytext=(-65, 18),
        textcoords="offset points",
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
# SILHOUETTE
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# INTRA-CLUSTER COSINE
# ------------------------------------------------------------

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
    [
        "reference_k"
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

    ax.scatter(
        x,
        y,
        s=90
    )

    ax.annotate(
        label_map[
            strategy
        ],
        (
            x,
            y
        ),
        xytext=(6, 6),
        textcoords="offset points"
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

candidate_reduction = [

    standard[
        "avg_candidate_reduction"
    ] * 100,

    kmeans[
        "avg_candidate_reduction"
    ] * 100,

    plain_top1[
        "avg_candidate_reduction"
    ] * 100,

    plain_top3[
        "avg_candidate_reduction"
    ] * 100,

    prop_top1[
        "avg_candidate_reduction"
    ] * 100,

    prop_top3[
        "avg_candidate_reduction"
    ] * 100
]


fig, ax = plt.subplots(
    figsize=(10.5, 5)
)


bars = ax.bar(
    METHODS_ALL,
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
    "fig3a_candidate_reduction"
)

plt.close(fig)


# ============================================================
# FIGURE 3B
# TOP-5 RETRIEVAL PRESERVATION
#
# Standard RAG excluded because it is itself unrestricted.
# ============================================================

OVERLAP_METHODS = [
    METHOD_KMEANS,
    METHOD_PLAIN_TOP1,
    METHOD_PLAIN_TOP3,
    METHOD_PROP_TOP1,
    METHOD_PROP_TOP3
]


retrieval_overlap = [

    kmeans[
        "avg_clip_kmeans_topk_overlap"
    ] * 100,

    plain_top1[
        "avg_minilm_gmm_topk_overlap"
    ] * 100,

    plain_top3[
        "avg_minilm_top3_gmm_topk_overlap"
    ] * 100,

    prop_top1[
        "avg_flat_gmm_topk_overlap"
    ] * 100,

    prop_top3[
        "avg_flat_gmm_topk_overlap"
    ] * 100
]


fig, ax = plt.subplots(
    figsize=(9.5, 5)
)


bars = ax.bar(
    OVERLAP_METHODS,
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
    "fig3b_top5_retrieval_overlap"
)

plt.close(fig)


# ============================================================
# FIGURE 4A
# ROUGE-L
# ============================================================

rouge_values = [

    standard[
        "avg_rougeL"
    ],

    kmeans[
        "avg_rougeL"
    ],

    plain_top1[
        "avg_rougeL"
    ],

    plain_top3[
        "avg_rougeL"
    ],

    prop_top1[
        "gmm_avg_rougeL"
    ],

    prop_top3[
        "gmm_avg_rougeL"
    ]
]


fig, ax = plt.subplots(
    figsize=(10.5, 5)
)


bars = ax.bar(
    METHODS_ALL,
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
    "fig4a_rouge_all_methods"
)

plt.close(fig)


# ============================================================
# FIGURE 4B
# BERTSCORE
# ============================================================

bert_values = [

    standard[
        "avg_bertscore_f1"
    ],

    kmeans[
        "avg_bertscore_f1"
    ],

    plain_top1[
        "avg_bertscore_f1"
    ],

    plain_top3[
        "avg_bertscore_f1"
    ],

    prop_top1[
        "gmm_avg_bertscore_f1"
    ],

    prop_top3[
        "gmm_avg_bertscore_f1"
    ]
]


fig, ax = plt.subplots(
    figsize=(10.5, 5)
)


bars = ax.bar(
    METHODS_ALL,
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
    "fig4b_bertscore_all_methods"
)

plt.close(fig)


# ============================================================
# FIGURE 4C
# LLM-JUDGE
# ============================================================

categories = [
    "Relevant",
    "Closer",
    "Inadequate"
]


standard_values = [
    standard_judge[
        "relevant_percentage"
    ],
    standard_judge[
        "closer_percentage"
    ],
    standard_judge[
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


plain_top1_values = [
    plain_top1_judge[
        "relevant_percentage"
    ],
    plain_top1_judge[
        "closer_percentage"
    ],
    plain_top1_judge[
        "inadequate_percentage"
    ]
]


plain_top3_values = [
    plain_top3_judge[
        "relevant_percentage"
    ],
    plain_top3_judge[
        "closer_percentage"
    ],
    plain_top3_judge[
        "inadequate_percentage"
    ]
]


prop_top1_values = [
    prop_top1_judge[
        "relevant_percentage"
    ],
    prop_top1_judge[
        "closer_percentage"
    ],
    prop_top1_judge[
        "inadequate_percentage"
    ]
]


prop_top3_values = [
    prop_top3_judge[
        "relevant_percentage"
    ],
    prop_top3_judge[
        "closer_percentage"
    ],
    prop_top3_judge[
        "inadequate_percentage"
    ]
]


x = np.arange(
    len(categories)
)

width = 0.13


fig, ax = plt.subplots(
    figsize=(12.5, 5.8)
)


bars1 = ax.bar(
    x - 2.5 * width,
    standard_values,
    width,
    label="Standard RAG"
)


bars2 = ax.bar(
    x - 1.5 * width,
    kmeans_values,
    width,
    label="CLIP + K-Means"
)


bars3 = ax.bar(
    x - 0.5 * width,
    plain_top1_values,
    width,
    label="Plain GMM Top-1"
)


bars4 = ax.bar(
    x + 0.5 * width,
    plain_top3_values,
    width,
    label="Plain GMM Top-3"
)


bars5 = ax.bar(
    x + 1.5 * width,
    prop_top1_values,
    width,
    label="Proposed Top-1"
)


bars6 = ax.bar(
    x + 2.5 * width,
    prop_top3_values,
    width,
    label="Proposed Top-3"
)


for bars in [
    bars1,
    bars2,
    bars3,
    bars4,
    bars5,
    bars6
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
    ncol=3,
    fontsize=8
)

ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()

save_figure(
    fig,
    "fig4c_llm_judge_all_methods"
)

plt.close(fig)


# ============================================================
# FIGURE 5
# END-TO-END LATENCY
#
# NOTE:
# This includes LLM/API generation time.
# ============================================================

mean_e2e = [

    standard[
        "e2e_latency"
    ][
        "mean_ms"
    ] / 1000,

    kmeans[
        "e2e_latency"
    ][
        "mean_ms"
    ] / 1000,

    plain_top1[
        "e2e_latency"
    ][
        "mean_ms"
    ] / 1000,

    plain_top3[
        "e2e_latency"
    ][
        "mean_ms"
    ] / 1000,

    prop_top1[
        "gmm_e2e_latency"
    ][
        "mean_ms"
    ] / 1000,

    prop_top3[
        "gmm_e2e_latency"
    ][
        "mean_ms"
    ] / 1000
]


p95_e2e = [

    standard[
        "e2e_latency"
    ][
        "p95_ms"
    ] / 1000,

    kmeans[
        "e2e_latency"
    ][
        "p95_ms"
    ] / 1000,

    plain_top1[
        "e2e_latency"
    ][
        "p95_ms"
    ] / 1000,

    plain_top3[
        "e2e_latency"
    ][
        "p95_ms"
    ] / 1000,

    prop_top1[
        "gmm_e2e_latency"
    ][
        "p95_ms"
    ] / 1000,

    prop_top3[
        "gmm_e2e_latency"
    ][
        "p95_ms"
    ] / 1000
]


x = np.arange(
    len(METHODS_ALL)
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(11, 5)
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
    "{:.3f}s",
    fontsize=7
)


add_bar_labels(
    ax,
    bars2,
    "{:.3f}s",
    fontsize=7
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    METHODS_ALL
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
    "fig5_e2e_latency_all_methods"
)

plt.close(fig)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 90)
print("ALL FIGURES GENERATED")
print("=" * 90)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)