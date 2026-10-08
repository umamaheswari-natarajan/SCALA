import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

RAG_INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference",
    "rag_query_results.xlsx"
)

KMEANS_INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_rag",
    "kmeans_query_results.xlsx"
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
# LOAD DATA
# ============================================================

rag_df = pd.read_excel(
    RAG_INPUT_FILE
)

kmeans_df = pd.read_excel(
    KMEANS_INPUT_FILE
)


print("=" * 80)
print("LATENCY DATA")
print("=" * 80)

print(
    "CLIP/GMM queries:",
    len(rag_df)
)

print(
    "K-Means queries:",
    len(kmeans_df)
)


if len(rag_df) != len(kmeans_df):

    raise ValueError(
        "RAG and K-Means result files "
        "do not contain the same number of queries."
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
# PIPELINE COMPONENT LATENCY
#
# LLM generation excluded here because it is much larger
# than routing/retrieval and hides the interesting differences.
# ============================================================

stages = [
    "Encoding",
    "Routing",
    "Retrieval",
    "Prompt"
]


# ------------------------------------------------------------
# CLIP RAG
# ------------------------------------------------------------

clip_components = [

    mean_ms(
        rag_df,
        "encoding_ms"
    ),

    0.0,

    mean_ms(
        rag_df,
        "flat_retrieval_ms"
    ),

    mean_ms(
        rag_df,
        "flat_prompt_ms"
    )
]


# ------------------------------------------------------------
# CLIP + K-Means
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Retrieval-Aware GMM
# ------------------------------------------------------------

gmm_components = [

    mean_ms(
        rag_df,
        "encoding_ms"
    ),

    mean_ms(
        rag_df,
        "gmm_routing_ms"
    ),

    mean_ms(
        rag_df,
        "gmm_retrieval_ms"
    ),

    mean_ms(
        rag_df,
        "gmm_prompt_ms"
    )
]


# ============================================================
# END-TO-END LATENCY
# ============================================================

# ------------------------------------------------------------
# Mean E2E
# ------------------------------------------------------------

clip_mean_e2e = (
    mean_ms(
        rag_df,
        "flat_e2e_ms"
    )
    /
    1000.0
)


kmeans_mean_e2e = (
    mean_ms(
        kmeans_df,
        "kmeans_e2e_ms"
    )
    /
    1000.0
)


gmm_mean_e2e = (
    mean_ms(
        rag_df,
        "gmm_e2e_ms"
    )
    /
    1000.0
)


# ------------------------------------------------------------
# P95 E2E
# ------------------------------------------------------------

clip_p95_e2e = (
    p95_ms(
        rag_df,
        "flat_e2e_ms"
    )
    /
    1000.0
)


kmeans_p95_e2e = (
    p95_ms(
        kmeans_df,
        "kmeans_e2e_ms"
    )
    /
    1000.0
)


gmm_p95_e2e = (
    p95_ms(
        rag_df,
        "gmm_e2e_ms"
    )
    /
    1000.0
)


# ============================================================
# PRINT VALUES
# ============================================================

print(
    "\nMEAN COMPONENT LATENCY (ms)"
)


for i, stage in enumerate(stages):

    print(
        f"{stage:12s} | "
        f"CLIP RAG = "
        f"{clip_components[i]:.4f} | "
        f"CLIP + K-Means = "
        f"{kmeans_components[i]:.4f} | "
        f"Retrieval-Aware GMM = "
        f"{gmm_components[i]:.4f}"
    )


print(
    "\nEND-TO-END LATENCY"
)


print(
    f"CLIP RAG Mean           : "
    f"{clip_mean_e2e:.3f} s"
)

print(
    f"CLIP + K-Means Mean     : "
    f"{kmeans_mean_e2e:.3f} s"
)

print(
    f"Retrieval-Aware GMM Mean: "
    f"{gmm_mean_e2e:.3f} s"
)


print(
    f"\nCLIP RAG P95            : "
    f"{clip_p95_e2e:.3f} s"
)

print(
    f"CLIP + K-Means P95      : "
    f"{kmeans_p95_e2e:.3f} s"
)

print(
    f"Retrieval-Aware GMM P95 : "
    f"{gmm_p95_e2e:.3f} s"
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        14,
        5.4
    )
)


# ============================================================
# PANEL A
# PIPELINE-STAGE LATENCY
# ============================================================

ax = axes[0]


x = np.arange(
    len(stages)
)


width = 0.24


bars1 = ax.bar(
    x - width,
    clip_components,
    width,
    label="CLIP RAG"
)


bars2 = ax.bar(
    x,
    kmeans_components,
    width,
    label="CLIP + K-Means"
)


bars3 = ax.bar(
    x + width,
    gmm_components,
    width,
    label="Retrieval-Aware GMM Clustering"
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


ax.legend()


ax.grid(
    axis="y",
    alpha=0.25
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

for bars in [
    bars1,
    bars2,
    bars3
]:

    for bar in bars:

        height = (
            bar.get_height()
        )

        if height > 0:

            ax.annotate(
                f"{height:.3f}",

                (
                    bar.get_x()
                    +
                    bar.get_width()
                    /
                    2,

                    height
                ),

                xytext=(
                    0,
                    4
                ),

                textcoords=
                    "offset points",

                ha="center",

                fontsize=8
            )


# ============================================================
# PANEL B
# END-TO-END LATENCY
# ============================================================

ax = axes[1]


methods = [
    "CLIP RAG",
    "CLIP +\nK-Means",
    "Retrieval-Aware\nGMM Clustering"
]


mean_e2e = [
    clip_mean_e2e,
    kmeans_mean_e2e,
    gmm_mean_e2e
]


p95_e2e = [
    clip_p95_e2e,
    kmeans_p95_e2e,
    gmm_p95_e2e
]


x = np.arange(
    len(methods)
)


width = 0.34


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


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

for bar in bars1:

    height = (
        bar.get_height()
    )

    ax.annotate(
        f"{height:.3f}s",

        (
            bar.get_x()
            +
            bar.get_width()
            /
            2,

            height
        ),

        xytext=(
            0,
            4
        ),

        textcoords=
            "offset points",

        ha="center",

        fontsize=9
    )


for bar in bars2:

    height = (
        bar.get_height()
    )

    ax.annotate(
        f"{height:.3f}s",

        (
            bar.get_x()
            +
            bar.get_width()
            /
            2,

            height
        ),

        xytext=(
            0,
            4
        ),

        textcoords=
            "offset points",

        ha="center",

        fontsize=9
    )


# ============================================================
# SAVE
# ============================================================

fig.tight_layout()


PNG_FILE = os.path.join(
    OUTPUT_DIR,
    "fig_latency_components_and_e2e_3methods.png"
)


PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "fig_latency_components_and_e2e_3methods.pdf"
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