import os
import json

import pandas as pd
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

PLAIN_GMM_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm",
    "plain_gmm_summary.json"
)

KMEANS_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "kmeans",
    "kmeans_summary.json"
)

GMM_SELECTION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "gmm",
    "gmm_model_selection.csv"
)

IFKG_SUMMARY = os.path.join(
    BASE_DIR,
    "results",
    "ifkg",
    "ifkg_build_summary.json"
)

# Actual contrastive / CLIP training-time file
CLIP_TRAIN_TIME_FILE = os.path.join(
    BASE_DIR,
    "results",
    "contrastive_training",
    "training_time.txt"
)

# Newly measured time to encode all 3049 training faults
# using the trained MiniLM backbone + fault projection head
CLIP_ENCODING_TIME_FILE = os.path.join(
    BASE_DIR,
    "results",
    "offline_timing",
    "clip_training_kb_encoding_time.json"
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


def read_clip_training_time(path):
    """
    Reads ONLY the actual total training time.

    Expected line inside training_time.txt:

    Total training time seconds: 1038.0015

    This deliberately ignores values such as validation MRR.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing CLIP training-time file:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip().startswith(
                "Total training time seconds:"
            ):

                value = (
                    line
                    .split(":", 1)[1]
                    .strip()
                )

                return float(value)

    raise ValueError(
        "Could not find the line:\n"
        "'Total training time seconds:'\n"
        f"in:\n{path}"
    )


def add_bar_labels(
    ax,
    bars,
    fmt="{:.2f}"
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
            fontsize=8
        )


# ============================================================
# LOAD SAVED RESULTS
# ============================================================

plain = load_json(
    PLAIN_GMM_SUMMARY
)

kmeans = load_json(
    KMEANS_SUMMARY
)

ifkg = load_json(
    IFKG_SUMMARY
)

clip_encoding_summary = load_json(
    CLIP_ENCODING_TIME_FILE
)

gmm_df = pd.read_csv(
    GMM_SELECTION_FILE
)


# ============================================================
# CLIP / CONTRASTIVE TRAINING TIME
# ============================================================

clip_train_seconds = (
    read_clip_training_time(
        CLIP_TRAIN_TIME_FILE
    )
)


# ============================================================
# CLIP TRAINING-KB EMBEDDING GENERATION TIME
#
# Mean of the 3 timing runs from:
#
# 33_measure_clip_kb_encoding_time.py
#
# This includes:
# training fault text
# -> fine-tuned MiniLM
# -> fault projection head
# -> normalized learned representation
# ============================================================

clip_embedding_seconds = float(
    clip_encoding_summary[
        "mean_time_seconds"
    ]
)


# ============================================================
# SELECT PROPOSED GMM K=55 ONLY
# ============================================================

k_col = None

for candidate in [
    "k",
    "n_components",
    "components"
]:

    if candidate in gmm_df.columns:

        k_col = candidate
        break


if k_col is None:

    raise KeyError(
        "Could not find the GMM component-number column "
        "in gmm_model_selection.csv."
    )


fit_col = None

for candidate in [
    "fit_time_seconds",
    "training_time_seconds",
    "gmm_fit_time_seconds"
]:

    if candidate in gmm_df.columns:

        fit_col = candidate
        break


if fit_col is None:

    raise KeyError(
        "Could not find the GMM fit-time column "
        "in gmm_model_selection.csv."
    )


selected_row = gmm_df[
    gmm_df[k_col] == 55
]


if len(selected_row) != 1:

    raise ValueError(
        "Could not uniquely find the K=55 row "
        "in gmm_model_selection.csv."
    )


proposed_gmm_fit_seconds = float(
    selected_row[
        fit_col
    ].iloc[0]
)


# ============================================================
# STANDARD RAG OFFLINE TIME
#
# Offline preparation:
#
# 3049 training faults
# -> pretrained MiniLM
# -> KB embeddings
#
# No clustering model.
# ============================================================

standard_embedding_seconds = float(
    plain[
        "embedding_generation_time_sec"
    ]
)

standard_seconds = (
    standard_embedding_seconds
)


# ============================================================
# PLAIN GMM OFFLINE TIME
#
# pretrained MiniLM KB embedding
# +
# GMM fitting
# ============================================================

plain_gmm_fit_seconds = float(
    plain[
        "gmm_training_time_sec"
    ]
)

plain_seconds = (
    standard_embedding_seconds
    +
    plain_gmm_fit_seconds
)


# ============================================================
# CLIP + K-MEANS OFFLINE TIME
#
# contrastive / CLIP training
# +
# encode 3049 training faults using trained representation
# +
# K-Means fitting
# ============================================================

kmeans_fit_seconds = float(
    kmeans[
        "training_time_seconds"
    ]
)

kmeans_seconds = (
    clip_train_seconds
    +
    clip_embedding_seconds
    +
    kmeans_fit_seconds
)


# ============================================================
# PROPOSED CLIP + GMM OFFLINE TIME
#
# contrastive / CLIP training
# +
# encode 3049 training faults using trained representation
# +
# selected GMM-55 fitting
# ============================================================

proposed_seconds = (
    clip_train_seconds
    +
    clip_embedding_seconds
    +
    proposed_gmm_fit_seconds
)


# ============================================================
# IFKG OFFLINE TIME
#
# chunking
# +
# GPT entity/relation extraction
# +
# Neo4j insertion / construction
# ============================================================

ifkg_seconds = float(
    ifkg[
        "total_offline_construction_time_seconds"
    ]
)


# ============================================================
# PRINT COMPONENTS
# ============================================================

print("=" * 90)
print("OFFLINE TIME COMPONENTS")
print("=" * 90)

print("\nSTANDARD RAG")
print(
    "MiniLM KB embedding:",
    f"{standard_embedding_seconds:.4f} sec"
)
print(
    "TOTAL:",
    f"{standard_seconds:.4f} sec"
)


print("\nPLAIN GMM")
print(
    "MiniLM KB embedding:",
    f"{standard_embedding_seconds:.4f} sec"
)
print(
    "GMM fit:",
    f"{plain_gmm_fit_seconds:.4f} sec"
)
print(
    "TOTAL:",
    f"{plain_seconds:.4f} sec"
)


print("\nCLIP + K-MEANS")
print(
    "CLIP training:",
    f"{clip_train_seconds:.4f} sec"
)
print(
    "CLIP KB embedding:",
    f"{clip_embedding_seconds:.4f} sec"
)
print(
    "K-Means fit:",
    f"{kmeans_fit_seconds:.4f} sec"
)
print(
    "TOTAL:",
    f"{kmeans_seconds:.4f} sec"
)


print("\nPROPOSED CLIP + GMM")
print(
    "CLIP training:",
    f"{clip_train_seconds:.4f} sec"
)
print(
    "CLIP KB embedding:",
    f"{clip_embedding_seconds:.4f} sec"
)
print(
    "GMM-55 fit:",
    f"{proposed_gmm_fit_seconds:.4f} sec"
)
print(
    "TOTAL:",
    f"{proposed_seconds:.4f} sec"
)


print("\nIFKG")
print(
    "TOTAL:",
    f"{ifkg_seconds:.4f} sec"
)


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("FINAL OFFLINE TIME COMPARISON")
print("=" * 90)

print(
    f"{'Method':<25}"
    f"{'Seconds':>15}"
    f"{'Minutes':>15}"
)

print("-" * 55)

final_results = [
    (
        "Standard RAG",
        standard_seconds
    ),
    (
        "Plain GMM",
        plain_seconds
    ),
    (
        "CLIP + K-Means",
        kmeans_seconds
    ),
    (
        "Proposed CLIP + GMM",
        proposed_seconds
    ),
    (
        "IFKG",
        ifkg_seconds
    )
]


for name, seconds in final_results:

    print(
        f"{name:<25}"
        f"{seconds:>15.4f}"
        f"{seconds / 60:>15.4f}"
    )


# ============================================================
# METHOD LABELS
# ============================================================

methods = [
    "Standard\nRAG",
    "Plain\nGMM",
    "CLIP +\nK-Means",
    "Proposed\nCLIP + GMM",
    "IFKG"
]


offline_minutes = [
    standard_seconds / 60,
    plain_seconds / 60,
    kmeans_seconds / 60,
    proposed_seconds / 60,
    ifkg_seconds / 60
]


# ============================================================
# FIGURE 1
# TOTAL OFFLINE TIME - LINEAR SCALE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.5, 5.5)
)

bars = ax.bar(
    methods,
    offline_minutes
)

add_bar_labels(
    ax,
    bars,
    "{:.2f}"
)

ax.set_ylabel(
    "Offline Preparation Time (minutes)"
)

ax.set_title(
    "Offline Training / Construction Cost"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()


png = os.path.join(
    OUTPUT_DIR,
    "fig_offline_total_time.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "fig_offline_total_time.pdf"
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
# FIGURE 2
# TOTAL OFFLINE TIME - LOG SCALE
#
# IFKG is much more expensive, so log scale allows the
# vector-based methods to remain visible.
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.5, 5.5)
)

bars = ax.bar(
    methods,
    offline_minutes
)

ax.set_yscale(
    "log"
)

add_bar_labels(
    ax,
    bars,
    "{:.2f}"
)

ax.set_ylabel(
    "Offline Preparation Time (minutes, log scale)"
)

ax.set_title(
    "Offline Training / Construction Cost (Log Scale)"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()


png = os.path.join(
    OUTPUT_DIR,
    "fig_offline_total_time_log.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "fig_offline_total_time_log.pdf"
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
# FIGURE 3
# IFKG OFFLINE DECOMPOSITION
# ============================================================

ifkg_parts = [
    "Chunking",
    "GPT Extraction",
    "Neo4j Insertion"
]


ifkg_part_seconds = [
    float(
        ifkg[
            "chunking_time_seconds"
        ]
    ),

    float(
        ifkg[
            "llm_extraction_time_seconds"
        ]
    ),

    float(
        ifkg[
            "neo4j_insertion_time_seconds"
        ]
    )
]


ifkg_part_minutes = [
    value / 60
    for value in ifkg_part_seconds
]


fig, ax = plt.subplots(
    figsize=(8, 5)
)

bars = ax.bar(
    ifkg_parts,
    ifkg_part_minutes
)

add_bar_labels(
    ax,
    bars,
    "{:.2f}"
)

ax.set_ylabel(
    "Time (minutes)"
)

ax.set_title(
    "IFKG Offline Construction-Time Breakdown"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()


png = os.path.join(
    OUTPUT_DIR,
    "fig_ifkg_offline_breakdown.png"
)

pdf = os.path.join(
    OUTPUT_DIR,
    "fig_ifkg_offline_breakdown.pdf"
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
# FINAL
# ============================================================

print("\n" + "=" * 90)
print("OFFLINE-TIME FIGURES COMPLETE")
print("=" * 90)

print(
    "Figures saved in:"
)

print(
    OUTPUT_DIR
)