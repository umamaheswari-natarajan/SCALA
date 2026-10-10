import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

PLAIN1_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_rag",
    "plain_gmm_query_results.xlsx"
)

PLAIN3_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_top3_rag",
    "plain_gmm_top3_query_results.xlsx"
)

PROP1_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top1",
    "rag_top1_query_results.xlsx"
)

PROP3_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top3",
    "rag_top3_query_results.xlsx"
)

IFKG_FILE = os.path.join(
    BASE_DIR,
    "results",
    "ifkg_inference",
    "ifkg_query_results.xlsx"
)


# ============================================================
# LOAD
# ============================================================

standard_df = pd.read_excel(
    STANDARD_FILE
)

kmeans_df = pd.read_excel(
    KMEANS_FILE
)

plain1_df = pd.read_excel(
    PLAIN1_FILE
)

plain3_df = pd.read_excel(
    PLAIN3_FILE
)

prop1_df = pd.read_excel(
    PROP1_FILE
)

prop3_df = pd.read_excel(
    PROP3_FILE
)

ifkg_df = pd.read_excel(
    IFKG_FILE
)


# ============================================================
# HELPERS
# ============================================================

def mean_ms(
    df,
    col
):

    return float(
        df[col].mean()
    )


def p95_ms(
    df,
    col
):

    return float(
        np.percentile(
            df[col],
            95
        )
    )


def add_labels(
    ax,
    bars,
    fmt="{:.3f}"
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
            fontsize=7
        )


# ============================================================
# E2E
# ============================================================

methods = [
    "Standard\nRAG",
    "CLIP +\nK-Means",
    "Plain GMM\nTop-1",
    "Plain GMM\nTop-3",
    "Proposed\nTop-1",
    "Proposed\nTop-3",
    "IFKG"
]


mean_e2e = [
    mean_ms(
        standard_df,
        "standard_rag_e2e_ms"
    ) / 1000,

    mean_ms(
        kmeans_df,
        "kmeans_e2e_ms"
    ) / 1000,

    mean_ms(
        plain1_df,
        "plain_gmm_e2e_ms"
    ) / 1000,

    mean_ms(
        plain3_df,
        "plain_gmm_top3_e2e_ms"
    ) / 1000,

    mean_ms(
        prop1_df,
        "gmm_top1_e2e_ms"
    ) / 1000,

    mean_ms(
        prop3_df,
        "gmm_top3_e2e_ms"
    ) / 1000,

    mean_ms(
        ifkg_df,
        "ifkg_e2e_ms"
    ) / 1000
]


p95_e2e = [
    p95_ms(
        standard_df,
        "standard_rag_e2e_ms"
    ) / 1000,

    p95_ms(
        kmeans_df,
        "kmeans_e2e_ms"
    ) / 1000,

    p95_ms(
        plain1_df,
        "plain_gmm_e2e_ms"
    ) / 1000,

    p95_ms(
        plain3_df,
        "plain_gmm_top3_e2e_ms"
    ) / 1000,

    p95_ms(
        prop1_df,
        "gmm_top1_e2e_ms"
    ) / 1000,

    p95_ms(
        prop3_df,
        "gmm_top3_e2e_ms"
    ) / 1000,

    p95_ms(
        ifkg_df,
        "ifkg_e2e_ms"
    ) / 1000
]


x = np.arange(
    len(methods)
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(12.5, 5.5)
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

add_labels(
    ax,
    bars1,
    "{:.3f}s"
)

add_labels(
    ax,
    bars2,
    "{:.3f}s"
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
    "End-to-End Inference Latency"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()


png = os.path.join(
    OUTPUT_DIR,
    "fig_inference_e2e_all_methods.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "fig_inference_e2e_all_methods.pdf"
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


# ============================================================
# VECTOR-BASED COMPONENT LATENCY
# ============================================================

stages = [
    "Encoding",
    "Routing",
    "Retrieval",
    "Prompt"
]


standard_comp = [
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


kmeans_comp = [
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


plain1_comp = [
    mean_ms(
        plain1_df,
        "encoding_ms"
    ),
    mean_ms(
        plain1_df,
        "plain_gmm_routing_ms"
    ),
    mean_ms(
        plain1_df,
        "plain_gmm_retrieval_ms"
    ),
    mean_ms(
        plain1_df,
        "plain_gmm_prompt_ms"
    )
]


plain3_comp = [
    mean_ms(
        plain3_df,
        "encoding_ms"
    ),
    mean_ms(
        plain3_df,
        "plain_gmm_top3_routing_ms"
    ),
    mean_ms(
        plain3_df,
        "plain_gmm_top3_retrieval_ms"
    ),
    mean_ms(
        plain3_df,
        "plain_gmm_top3_prompt_ms"
    )
]


prop1_comp = [
    mean_ms(
        prop1_df,
        "encoding_ms"
    ),
    mean_ms(
        prop1_df,
        "gmm_top1_routing_ms"
    ),
    mean_ms(
        prop1_df,
        "gmm_top1_retrieval_ms"
    ),
    mean_ms(
        prop1_df,
        "gmm_top1_prompt_ms"
    )
]


prop3_comp = [
    mean_ms(
        prop3_df,
        "encoding_ms"
    ),
    mean_ms(
        prop3_df,
        "gmm_top3_routing_ms"
    ),
    mean_ms(
        prop3_df,
        "gmm_top3_retrieval_ms"
    ),
    mean_ms(
        prop3_df,
        "gmm_top3_prompt_ms"
    )
]


vector_methods = [
    "Standard RAG",
    "CLIP + K-Means",
    "Plain GMM Top-1",
    "Plain GMM Top-3",
    "Proposed Top-1",
    "Proposed Top-3"
]


vector_values = [
    standard_comp,
    kmeans_comp,
    plain1_comp,
    plain3_comp,
    prop1_comp,
    prop3_comp
]


x = np.arange(
    len(stages)
)

width = 0.12


fig, ax = plt.subplots(
    figsize=(13, 5.7)
)


offsets = [
    -2.5 * width,
    -1.5 * width,
    -0.5 * width,
    0.5 * width,
    1.5 * width,
    2.5 * width
]


for i, vals in enumerate(
    vector_values
):

    ax.bar(
        x + offsets[i],
        vals,
        width,
        label=vector_methods[i]
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
    "Vector-Based Retrieval Pipeline Latency"
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


png = os.path.join(
    OUTPUT_DIR,
    "fig_vector_component_latency.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "fig_vector_component_latency.pdf"
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


# ============================================================
# IFKG COMPONENT LATENCY
# ============================================================

ifkg_stages = [
    "Text-to-Cypher",
    "KG Query",
    "Prompt",
    "Final LLM"
]


ifkg_components = [
    mean_ms(
        ifkg_df,
        "ifkg_text_to_cypher_ms"
    ),
    mean_ms(
        ifkg_df,
        "ifkg_kg_query_ms"
    ),
    mean_ms(
        ifkg_df,
        "ifkg_prompt_ms"
    ),
    mean_ms(
        ifkg_df,
        "ifkg_llm_ms"
    )
]


fig, ax = plt.subplots(
    figsize=(8, 5)
)

bars = ax.bar(
    ifkg_stages,
    ifkg_components
)

add_labels(
    ax,
    bars,
    "{:.1f} ms"
)

ax.set_ylabel(
    "Mean Latency (ms)"
)

ax.set_title(
    "IFKG Inference Latency Decomposition"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()


png = os.path.join(
    OUTPUT_DIR,
    "fig_ifkg_component_latency.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "fig_ifkg_component_latency.pdf"
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


print("\nINFERENCE LATENCY FIGURES COMPLETE")