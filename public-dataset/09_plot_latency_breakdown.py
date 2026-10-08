import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"


# ============================================================
# TOP-5 RETRIEVAL-AWARE GMM
# ============================================================

TOP5_INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference",
    "rag_query_results.xlsx"
)


# ============================================================
# TOP-3 RETRIEVAL-AWARE GMM
# ============================================================

TOP3_INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top3",
    "rag_top3_query_results.xlsx"
)


# ============================================================
# K-MEANS
# ============================================================

KMEANS_INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_rag",
    "kmeans_query_results.xlsx"
)


# ============================================================
# PLAIN GMM
# ============================================================

PLAIN_GMM_INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_rag",
    "plain_gmm_query_results.xlsx"
)


# ============================================================
# STANDARD RAG
# ============================================================

STANDARD_RAG_INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "standard_rag",
    "standard_rag_query_results.xlsx"
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
# VERIFY
# ============================================================

required_files = [
    TOP5_INPUT_FILE,
    TOP3_INPUT_FILE,
    KMEANS_INPUT_FILE,
    PLAIN_GMM_INPUT_FILE,
    STANDARD_RAG_INPUT_FILE
]


missing = [
    path
    for path in required_files
    if not os.path.exists(path)
]


if missing:

    print(
        "\nMissing files:"
    )

    for path in missing:
        print(path)

    raise FileNotFoundError(
        "One or more input files are missing."
    )


# ============================================================
# LOAD
# ============================================================

top5_df = pd.read_excel(
    TOP5_INPUT_FILE
)


top3_df = pd.read_excel(
    TOP3_INPUT_FILE
)


kmeans_df = pd.read_excel(
    KMEANS_INPUT_FILE
)


plain_gmm_df = pd.read_excel(
    PLAIN_GMM_INPUT_FILE
)


standard_df = pd.read_excel(
    STANDARD_RAG_INPUT_FILE
)


print("=" * 90)
print("LATENCY DATA")
print("=" * 90)


print(
    "Standard RAG:",
    len(standard_df)
)

print(
    "Plain GMM:",
    len(plain_gmm_df)
)

print(
    "K-Means:",
    len(kmeans_df)
)

print(
    "Top-3 GMM:",
    len(top3_df)
)

print(
    "Top-5 GMM:",
    len(top5_df)
)


# ============================================================
# COUNT CHECK
# ============================================================

query_counts = {
    len(standard_df),
    len(plain_gmm_df),
    len(kmeans_df),
    len(top3_df),
    len(top5_df)
}


if len(
    query_counts
) != 1:

    raise ValueError(
        "Query counts differ across methods."
    )


# ============================================================
# HELPERS
# ============================================================

def mean_ms(
    df,
    column
):

    return float(
        df[
            column
        ].mean()
    )


def p95_ms(
    df,
    column
):

    return float(
        np.percentile(
            df[
                column
            ],
            95
        )
    )


# ============================================================
# COMPONENTS
# ============================================================

stages = [
    "Encoding",
    "Routing",
    "Retrieval",
    "Prompt"
]


# Standard RAG
standard_components = [

    mean_ms(
        standard_df,
        "encoding_ms"
    ),

    0.0,

    mean_ms(
        standard_df,
        "standard_rag_retrieval_ms"
    ),

    mean_ms(
        standard_df,
        "standard_rag_prompt_ms"
    )
]


# Plain GMM
plain_components = [

    mean_ms(
        plain_gmm_df,
        "encoding_ms"
    ),

    mean_ms(
        plain_gmm_df,
        "plain_gmm_routing_ms"
    ),

    mean_ms(
        plain_gmm_df,
        "plain_gmm_retrieval_ms"
    ),

    mean_ms(
        plain_gmm_df,
        "plain_gmm_prompt_ms"
    )
]


# K-Means
kmeans_components = [

    mean_ms(
        kmeans_df,
        "encoding_ms"
    ),

    mean_ms(
        kmeans_df,
        "kmeans_routing_ms"
    ),

    mean_ms(
        kmeans_df,
        "kmeans_retrieval_ms"
    ),

    mean_ms(
        kmeans_df,
        "kmeans_prompt_ms"
    )
]


# Top-3
top3_components = [

    mean_ms(
        top3_df,
        "encoding_ms"
    ),

    mean_ms(
        top3_df,
        "gmm_top3_routing_ms"
    ),

    mean_ms(
        top3_df,
        "gmm_top3_retrieval_ms"
    ),

    mean_ms(
        top3_df,
        "gmm_top3_prompt_ms"
    )
]


# Top-5
top5_components = [

    mean_ms(
        top5_df,
        "encoding_ms"
    ),

    mean_ms(
        top5_df,
        "gmm_routing_ms"
    ),

    mean_ms(
        top5_df,
        "gmm_retrieval_ms"
    ),

    mean_ms(
        top5_df,
        "gmm_prompt_ms"
    )
]


# ============================================================
# E2E MEAN
# ============================================================

standard_mean = (
    mean_ms(
        standard_df,
        "standard_rag_e2e_ms"
    ) / 1000
)


plain_mean = (
    mean_ms(
        plain_gmm_df,
        "plain_gmm_e2e_ms"
    ) / 1000
)


kmeans_mean = (
    mean_ms(
        kmeans_df,
        "kmeans_e2e_ms"
    ) / 1000
)


top3_mean = (
    mean_ms(
        top3_df,
        "gmm_top3_e2e_ms"
    ) / 1000
)


top5_mean = (
    mean_ms(
        top5_df,
        "gmm_e2e_ms"
    ) / 1000
)


# ============================================================
# E2E P95
# ============================================================

standard_p95 = (
    p95_ms(
        standard_df,
        "standard_rag_e2e_ms"
    ) / 1000
)


plain_p95 = (
    p95_ms(
        plain_gmm_df,
        "plain_gmm_e2e_ms"
    ) / 1000
)


kmeans_p95 = (
    p95_ms(
        kmeans_df,
        "kmeans_e2e_ms"
    ) / 1000
)


top3_p95 = (
    p95_ms(
        top3_df,
        "gmm_top3_e2e_ms"
    ) / 1000
)


top5_p95 = (
    p95_ms(
        top5_df,
        "gmm_e2e_ms"
    ) / 1000
)


# ============================================================
# PRINT
# ============================================================

print(
    "\nMEAN COMPONENT LATENCY (ms)"
)


for i, stage in enumerate(
    stages
):

    print(
        f"{stage:12s} | "
        f"Standard={standard_components[i]:.4f} | "
        f"PlainGMM={plain_components[i]:.4f} | "
        f"KMeans={kmeans_components[i]:.4f} | "
        f"Top3={top3_components[i]:.4f} | "
        f"Top5={top5_components[i]:.4f}"
    )


print(
    "\nEND-TO-END"
)

print(
    f"Standard Mean : {standard_mean:.3f}s"
)

print(
    f"Plain GMM Mean: {plain_mean:.3f}s"
)

print(
    f"K-Means Mean  : {kmeans_mean:.3f}s"
)

print(
    f"Top-3 Mean    : {top3_mean:.3f}s"
)

print(
    f"Top-5 Mean    : {top5_mean:.3f}s"
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 5.7)
)


# ============================================================
# PANEL A
# COMPONENT LATENCY
# ============================================================

ax = axes[0]


x = np.arange(
    len(stages)
)


width = 0.15


bars1 = ax.bar(
    x - 2 * width,
    standard_components,
    width,
    label="Standard RAG"
)


bars2 = ax.bar(
    x - width,
    plain_components,
    width,
    label="Plain GMM-RAG"
)


bars3 = ax.bar(
    x,
    kmeans_components,
    width,
    label="CLIP + K-Means"
)


bars4 = ax.bar(
    x + width,
    top3_components,
    width,
    label="Retrieval-Aware GMM Top-3"
)


bars5 = ax.bar(
    x + 2 * width,
    top5_components,
    width,
    label="Retrieval-Aware GMM Top-5"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    stages
)


ax.set_ylabel(
    "Mean Latency (ms)"
)


ax.set_title(
    "(a) Pipeline-Stage Latency"
)


ax.legend(
    ncol=2,
    fontsize=8
)


ax.grid(
    axis="y",
    alpha=0.25
)


for bars in [
    bars1,
    bars2,
    bars3,
    bars4,
    bars5
]:

    for bar in bars:

        height = (
            bar.get_height()
        )

        if height <= 0:
            continue

        ax.annotate(
            f"{height:.3f}",
            (
                bar.get_x()
                +
                bar.get_width() / 2,
                height
            ),
            xytext=(
                0,
                4
            ),
            textcoords=
                "offset points",
            ha="center",
            fontsize=6.5
        )


# ============================================================
# PANEL B
# E2E
# ============================================================

ax = axes[1]


methods = [
    "Standard\nRAG",
    "Plain\nGMM-RAG",
    "CLIP +\nK-Means",
    "Retrieval-Aware\nGMM Top-3",
    "Retrieval-Aware\nGMM Top-5"
]


mean_values = [
    standard_mean,
    plain_mean,
    kmeans_mean,
    top3_mean,
    top5_mean
]


p95_values = [
    standard_p95,
    plain_p95,
    kmeans_p95,
    top3_p95,
    top5_p95
]


x = np.arange(
    len(methods)
)


width = 0.34


bars1 = ax.bar(
    x - width / 2,
    mean_values,
    width,
    label="Mean E2E"
)


bars2 = ax.bar(
    x + width / 2,
    p95_values,
    width,
    label="P95 E2E"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    methods
)


ax.set_ylabel(
    "Latency (seconds)"
)


ax.set_title(
    "(b) End-to-End RAG Latency"
)


ax.legend()


ax.grid(
    axis="y",
    alpha=0.25
)


for bars in [
    bars1,
    bars2
]:

    for bar in bars:

        height = (
            bar.get_height()
        )

        ax.annotate(
            f"{height:.3f}s",
            (
                bar.get_x()
                +
                bar.get_width() / 2,
                height
            ),
            xytext=(
                0,
                4
            ),
            textcoords=
                "offset points",
            ha="center",
            fontsize=7
        )


# ============================================================
# SAVE
# ============================================================

fig.tight_layout()


PNG_FILE = os.path.join(
    OUTPUT_DIR,
    "fig_latency_all_five_methods.png"
)


PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "fig_latency_all_five_methods.pdf"
)


fig.savefig(
    PNG_FILE,
    dpi=300,
    bbox_inches="tight"
)


fig.savefig(
    PDF_FILE,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "\nSaved:"
)

print(
    PNG_FILE
)

print(
    PDF_FILE
)