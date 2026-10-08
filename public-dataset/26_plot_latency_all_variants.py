import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"


# ============================================================
# INPUT FILES
# ============================================================

STANDARD_FILE = os.path.join(
    BASE_DIR,
    "results",
    "standard_rag",
    "standard_rag_query_results.xlsx"
)


KMEANS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_rag",
    "kmeans_query_results.xlsx"
)


PLAIN_TOP1_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_rag",
    "plain_gmm_query_results.xlsx"
)


PLAIN_TOP3_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_top3_rag",
    "plain_gmm_top3_query_results.xlsx"
)


PROPOSED_TOP1_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top1",
    "rag_top1_query_results.xlsx"
)


PROPOSED_TOP3_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top3",
    "rag_top3_query_results.xlsx"
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
# VERIFY FILES
# ============================================================

required_files = [
    STANDARD_FILE,
    KMEANS_FILE,
    PLAIN_TOP1_FILE,
    PLAIN_TOP3_FILE,
    PROPOSED_TOP1_FILE,
    PROPOSED_TOP3_FILE
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
        "One or more latency files are missing."
    )


# ============================================================
# LOAD DATA
# ============================================================

standard_df = pd.read_excel(
    STANDARD_FILE
)

kmeans_df = pd.read_excel(
    KMEANS_FILE
)

plain_top1_df = pd.read_excel(
    PLAIN_TOP1_FILE
)

plain_top3_df = pd.read_excel(
    PLAIN_TOP3_FILE
)

prop_top1_df = pd.read_excel(
    PROPOSED_TOP1_FILE
)

prop_top3_df = pd.read_excel(
    PROPOSED_TOP3_FILE
)


# ============================================================
# QUERY COUNT CHECK
# ============================================================

query_counts = {
    len(standard_df),
    len(kmeans_df),
    len(plain_top1_df),
    len(plain_top3_df),
    len(prop_top1_df),
    len(prop_top3_df)
}


print("=" * 90)
print("QUERY COUNTS")
print("=" * 90)

print(
    "Standard RAG:",
    len(standard_df)
)

print(
    "CLIP + K-Means:",
    len(kmeans_df)
)

print(
    "Plain GMM Top-1:",
    len(plain_top1_df)
)

print(
    "Plain GMM Top-3:",
    len(plain_top3_df)
)

print(
    "Proposed Top-1:",
    len(prop_top1_df)
)

print(
    "Proposed Top-3:",
    len(prop_top3_df)
)


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
# PIPELINE STAGES
# ============================================================

stages = [
    "Encoding",
    "Routing",
    "Retrieval",
    "Prompt"
]


# ============================================================
# STANDARD RAG
# ============================================================

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


# ============================================================
# CLIP + K-MEANS
# ============================================================

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


# ============================================================
# PLAIN GMM TOP-1
# ============================================================

plain_top1_components = [

    mean_ms(
        plain_top1_df,
        "encoding_ms"
    ),

    mean_ms(
        plain_top1_df,
        "plain_gmm_routing_ms"
    ),

    mean_ms(
        plain_top1_df,
        "plain_gmm_retrieval_ms"
    ),

    mean_ms(
        plain_top1_df,
        "plain_gmm_prompt_ms"
    )
]


# ============================================================
# PLAIN GMM TOP-3
# ============================================================

plain_top3_components = [

    mean_ms(
        plain_top3_df,
        "encoding_ms"
    ),

    mean_ms(
        plain_top3_df,
        "plain_gmm_top3_routing_ms"
    ),

    mean_ms(
        plain_top3_df,
        "plain_gmm_top3_retrieval_ms"
    ),

    mean_ms(
        plain_top3_df,
        "plain_gmm_top3_prompt_ms"
    )
]


# ============================================================
# PROPOSED TOP-1
# ============================================================

prop_top1_components = [

    mean_ms(
        prop_top1_df,
        "encoding_ms"
    ),

    mean_ms(
        prop_top1_df,
        "gmm_top1_routing_ms"
    ),

    mean_ms(
        prop_top1_df,
        "gmm_top1_retrieval_ms"
    ),

    mean_ms(
        prop_top1_df,
        "gmm_top1_prompt_ms"
    )
]


# ============================================================
# PROPOSED TOP-3
# ============================================================

prop_top3_components = [

    mean_ms(
        prop_top3_df,
        "encoding_ms"
    ),

    mean_ms(
        prop_top3_df,
        "gmm_top3_routing_ms"
    ),

    mean_ms(
        prop_top3_df,
        "gmm_top3_retrieval_ms"
    ),

    mean_ms(
        prop_top3_df,
        "gmm_top3_prompt_ms"
    )
]


# ============================================================
# MEAN E2E
# ============================================================

standard_mean = (
    mean_ms(
        standard_df,
        "standard_rag_e2e_ms"
    )
    /
    1000
)


kmeans_mean = (
    mean_ms(
        kmeans_df,
        "kmeans_e2e_ms"
    )
    /
    1000
)


plain_top1_mean = (
    mean_ms(
        plain_top1_df,
        "plain_gmm_e2e_ms"
    )
    /
    1000
)


plain_top3_mean = (
    mean_ms(
        plain_top3_df,
        "plain_gmm_top3_e2e_ms"
    )
    /
    1000
)


prop_top1_mean = (
    mean_ms(
        prop_top1_df,
        "gmm_top1_e2e_ms"
    )
    /
    1000
)


prop_top3_mean = (
    mean_ms(
        prop_top3_df,
        "gmm_top3_e2e_ms"
    )
    /
    1000
)


# ============================================================
# P95 E2E
# ============================================================

standard_p95 = (
    p95_ms(
        standard_df,
        "standard_rag_e2e_ms"
    )
    /
    1000
)


kmeans_p95 = (
    p95_ms(
        kmeans_df,
        "kmeans_e2e_ms"
    )
    /
    1000
)


plain_top1_p95 = (
    p95_ms(
        plain_top1_df,
        "plain_gmm_e2e_ms"
    )
    /
    1000
)


plain_top3_p95 = (
    p95_ms(
        plain_top3_df,
        "plain_gmm_top3_e2e_ms"
    )
    /
    1000
)


prop_top1_p95 = (
    p95_ms(
        prop_top1_df,
        "gmm_top1_e2e_ms"
    )
    /
    1000
)


prop_top3_p95 = (
    p95_ms(
        prop_top3_df,
        "gmm_top3_e2e_ms"
    )
    /
    1000
)


# ============================================================
# PRINT COMPONENT LATENCY
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "MEAN COMPONENT LATENCY (ms)"
)

print(
    "=" * 90
)


for i, stage in enumerate(
    stages
):

    print(
        f"{stage:12s} | "
        f"Standard={standard_components[i]:.4f} | "
        f"KMeans={kmeans_components[i]:.4f} | "
        f"Plain1={plain_top1_components[i]:.4f} | "
        f"Plain3={plain_top3_components[i]:.4f} | "
        f"Prop1={prop_top1_components[i]:.4f} | "
        f"Prop3={prop_top3_components[i]:.4f}"
    )


# ============================================================
# PRINT E2E
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "END-TO-END LATENCY"
)

print(
    "=" * 90
)


print(
    f"Standard Mean : "
    f"{standard_mean:.3f}s"
)

print(
    f"K-Means Mean  : "
    f"{kmeans_mean:.3f}s"
)

print(
    f"Plain Top-1   : "
    f"{plain_top1_mean:.3f}s"
)

print(
    f"Plain Top-3   : "
    f"{plain_top3_mean:.3f}s"
)

print(
    f"Proposed Top-1: "
    f"{prop_top1_mean:.3f}s"
)

print(
    f"Proposed Top-3: "
    f"{prop_top3_mean:.3f}s"
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(17, 6)
)


# ============================================================
# PANEL A
# COMPONENT LATENCY
# ============================================================

ax = axes[0]


x = np.arange(
    len(stages)
)


width = 0.12


bars1 = ax.bar(
    x - 2.5 * width,
    standard_components,
    width,
    label="Standard RAG"
)


bars2 = ax.bar(
    x - 1.5 * width,
    kmeans_components,
    width,
    label="CLIP + K-Means"
)


bars3 = ax.bar(
    x - 0.5 * width,
    plain_top1_components,
    width,
    label="Plain GMM Top-1"
)


bars4 = ax.bar(
    x + 0.5 * width,
    plain_top3_components,
    width,
    label="Plain GMM Top-3"
)


bars5 = ax.bar(
    x + 1.5 * width,
    prop_top1_components,
    width,
    label="Proposed Top-1"
)


bars6 = ax.bar(
    x + 2.5 * width,
    prop_top3_components,
    width,
    label="Proposed Top-3"
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
    bars5,
    bars6
]:

    for bar in bars:

        height = bar.get_height()

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
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=6,
            rotation=90
        )


# ============================================================
# PANEL B
# E2E LATENCY
# ============================================================

ax = axes[1]


methods = [
    "Standard\nRAG",
    "CLIP +\nK-Means",
    "Plain GMM\nTop-1",
    "Plain GMM\nTop-3",
    "Proposed\nTop-1",
    "Proposed\nTop-3"
]


mean_values = [
    standard_mean,
    kmeans_mean,
    plain_top1_mean,
    plain_top3_mean,
    prop_top1_mean,
    prop_top3_mean
]


p95_values = [
    standard_p95,
    kmeans_p95,
    plain_top1_p95,
    plain_top3_p95,
    prop_top1_p95,
    prop_top3_p95
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

        height = bar.get_height()

        ax.annotate(
            f"{height:.3f}s",
            (
                bar.get_x()
                +
                bar.get_width() / 2,
                height
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7
        )


# ============================================================
# SAVE
# ============================================================

fig.tight_layout()


PNG_FILE = os.path.join(
    OUTPUT_DIR,
    "fig_latency_six_methods.png"
)


PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "fig_latency_six_methods.pdf"
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