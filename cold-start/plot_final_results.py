import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STANDARD_FILE = os.path.join(
    BASE_DIR,
    "synthetic_standard_rag_gpt5mini_argsort",
    "standard_rag_results.csv"
)

PCA_GMM_FILE = os.path.join(
    BASE_DIR,
    "synthetic_clip_pca50_gmm_K50_gpt5mini_argsort",
    "clip_pca50_gmm_results.csv"
)

IFKG_FILE = os.path.join(
    BASE_DIR,
    "synthetic_ifkg_gpt5mini",
    "ifkg_results_scored.csv"
)

RAG_JUDGE_SUMMARY = os.path.join(
    BASE_DIR,
    "synthetic_llm_judge_final",
    "llm_judge_summary.json"
)

IFKG_JUDGE_SUMMARY = os.path.join(
    BASE_DIR,
    "synthetic_ifkg_llm_judge",
    "ifkg_llm_judge_summary.json"
)

PCA_OFFLINE_SUMMARY = os.path.join(
    BASE_DIR,
    "synthetic_proposed_pca50_gmm_K50",
    "offline_summary.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150
})


METHODS = [
    "Standard RAG",
    "SCALA",
    "IFKG"
]


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    STANDARD_FILE,
    PCA_GMM_FILE,
    IFKG_FILE,
    RAG_JUDGE_SUMMARY,
    IFKG_JUDGE_SUMMARY
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

standard = pd.read_csv(STANDARD_FILE)
pca = pd.read_csv(PCA_GMM_FILE)
ifkg = pd.read_csv(IFKG_FILE)


# Standard successful rows
standard = standard[
    standard["status"] == "success"
].copy()


print("=" * 80)
print("FINAL THREE-METHOD PLOT DATA")
print("=" * 80)

print("Standard RAG queries       :", len(standard))
print("CLIP+PCA50+GMM queries     :", len(pca))
print("IFKG queries               :", len(ifkg))

print(
    "IFKG successful generation :",
    (ifkg["generation_status"] == "success").sum()
)

print(
    "IFKG no-KG matches         :",
    (ifkg["generation_status"] == "no_kg_match").sum()
)


# ============================================================
# HELPERS
# ============================================================

def save_plot(filename):

    png = os.path.join(
        OUTPUT_DIR,
        filename + ".png"
    )

    pdf = os.path.join(
        OUTPUT_DIR,
        filename + ".pdf"
    )

    plt.tight_layout()

    plt.savefig(
        png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", png)


def add_bar_labels(ax, bars, decimals=3):

    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            f"{height:.{decimals}f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )


# ============================================================
# CALCULATE METRICS
# ============================================================

# ------------------------------------------------------------
# RETRIEVAL
#
# Standard/PCA retrieval is measured after query encoding.
#
# IFKG retrieval =
# Text-to-Cypher LLM + Neo4j execution.
#
# Convert all to milliseconds for common plot.
# ------------------------------------------------------------

retrieval_mean_ms = [
    standard["retrieval_sec"].mean() * 1000,
    pca["retrieval_sec"].mean() * 1000,
    ifkg["retrieval_sec"].mean() * 1000
]

retrieval_p95_ms = [
    standard["retrieval_sec"].quantile(0.95) * 1000,
    pca["retrieval_sec"].quantile(0.95) * 1000,
    ifkg["retrieval_sec"].quantile(0.95) * 1000
]


# ------------------------------------------------------------
# E2E
# ------------------------------------------------------------

e2e_mean = [
    standard["e2e_latency_sec"].mean(),
    pca["e2e_sec"].mean(),
    ifkg["e2e_sec"].mean()
]

e2e_p95 = [
    standard["e2e_latency_sec"].quantile(0.95),
    pca["e2e_sec"].quantile(0.95),
    ifkg["e2e_sec"].quantile(0.95)
]


# ------------------------------------------------------------
# ENCODING
#
# IFKG has no equivalent embedding-encoding stage.
# Use NaN rather than pretending that its encoding cost is zero.
# ------------------------------------------------------------

encoding_mean = [
    standard["query_encoding_sec"].mean(),
    pca["encoding_sec"].mean(),
    np.nan
]

encoding_p95 = [
    standard["query_encoding_sec"].quantile(0.95),
    pca["encoding_sec"].quantile(0.95),
    np.nan
]


# ------------------------------------------------------------
# PROMPT
# ------------------------------------------------------------

prompt_mean = [
    standard["prompt_build_sec"].mean(),
    pca["prompt_sec"].mean(),
    ifkg["prompt_sec"].mean()
]


# ------------------------------------------------------------
# FINAL GENERATION
# ------------------------------------------------------------

generation_mean = [
    standard["llm_latency_sec"].mean(),
    pca["gpt_sec"].mean(),
    ifkg["generation_sec"].mean()
]

generation_p95 = [
    standard["llm_latency_sec"].quantile(0.95),
    pca["gpt_sec"].quantile(0.95),
    ifkg["generation_sec"].quantile(0.95)
]


# ------------------------------------------------------------
# IFKG RETRIEVAL DECOMPOSITION
# ------------------------------------------------------------

ifkg_text2cypher_mean = (
    ifkg["text2cypher_sec"].mean()
)

ifkg_neo4j_mean = (
    ifkg["kg_execution_sec"].mean()
)

ifkg_text2cypher_p95 = (
    ifkg["text2cypher_sec"].quantile(0.95)
)

ifkg_neo4j_p95 = (
    ifkg["kg_execution_sec"].quantile(0.95)
)


# ------------------------------------------------------------
# QUALITY
#
# IMPORTANT:
# IFKG scored CSV already assigns zero to the 31 no-KG cases.
# Therefore the mean below is over all 1050 queries.
# ------------------------------------------------------------

rouge = [
    standard["rouge_l"].mean(),
    pca["rouge_l"].mean(),
    ifkg["rouge_l"].mean()
]

bert = [
    standard["bertscore_f1"].mean(),
    pca["bertscore_f1"].mean(),
    ifkg["bertscore_f1"].mean()
]


# ------------------------------------------------------------
# RAG CANDIDATE SEARCH SPACE
#
# IFKG intentionally excluded because graph retrieval results
# are not directly equivalent to cosine candidate records.
# ------------------------------------------------------------

RAG_METHODS = [
    "Standard RAG",
    "SCALA"
]

candidate_count = [
    10500,
    pca["candidate_count"].mean()
]


# ============================================================
# LOAD LLM-JUDGE RESULTS
# ============================================================

with open(
    RAG_JUDGE_SUMMARY,
    "r",
    encoding="utf-8"
) as f:

    rag_judge = json.load(f)


with open(
    IFKG_JUDGE_SUMMARY,
    "r",
    encoding="utf-8"
) as f:

    ifkg_judge = json.load(f)


relevant = [
    rag_judge[
        "standard_rag"
    ][
        "relevant_percentage"
    ],

    rag_judge[
        "clip_pca50_gmm"
    ][
        "relevant_percentage"
    ],

    ifkg_judge[
        "relevant_percentage"
    ]
]


closer = [
    rag_judge[
        "standard_rag"
    ][
        "closer_percentage"
    ],

    rag_judge[
        "clip_pca50_gmm"
    ][
        "closer_percentage"
    ],

    ifkg_judge[
        "closer_percentage"
    ]
]


inadequate = [
    rag_judge[
        "standard_rag"
    ][
        "inadequate_percentage"
    ],

    rag_judge[
        "clip_pca50_gmm"
    ][
        "inadequate_percentage"
    ],

    ifkg_judge[
        "inadequate_percentage"
    ]
]


relevant_plus_closer = [
    rag_judge[
        "standard_rag"
    ][
        "relevant_plus_closer_percentage"
    ],

    rag_judge[
        "clip_pca50_gmm"
    ][
        "relevant_plus_closer_percentage"
    ],

    ifkg_judge[
        "relevant_plus_closer_percentage"
    ]
]


# ============================================================
# PRINT VALUES USED IN PLOTS
# ============================================================

print("\nMEAN RETRIEVAL")
for m, v in zip(METHODS, retrieval_mean_ms):
    print(f"{m:25s}: {v:.4f} ms")


print("\nP95 RETRIEVAL")
for m, v in zip(METHODS, retrieval_p95_ms):
    print(f"{m:25s}: {v:.4f} ms")


print("\nMEAN E2E")
for m, v in zip(METHODS, e2e_mean):
    print(f"{m:25s}: {v:.6f} sec")


print("\nP95 E2E")
for m, v in zip(METHODS, e2e_p95):
    print(f"{m:25s}: {v:.6f} sec")


print("\nQUALITY")
for i, m in enumerate(METHODS):
    print(
        f"{m:25s}: "
        f"ROUGE-L={rouge[i]:.6f}, "
        f"BERTScore F1={bert[i]:.6f}"
    )


print("\nLLM JUDGE")
for i, m in enumerate(METHODS):
    print(
        f"{m:25s}: "
        f"Relevant={relevant[i]:.2f}%, "
        f"Closer={closer[i]:.2f}%, "
        f"Inadequate={inadequate[i]:.2f}%"
    )


# ============================================================
# FIGURE 1
# MEAN + P95 RETRIEVAL LATENCY
#
# Use seconds because IFKG retrieval is seconds while RAG is ms.
# Labels show exact values.
# ============================================================

x = np.arange(
    len(METHODS)
)

width = 0.34


retrieval_mean_sec = (
    np.array(retrieval_mean_ms)
    /
    1000
)

retrieval_p95_sec = (
    np.array(retrieval_p95_ms)
    /
    1000
)


fig, ax = plt.subplots(
    figsize=(9, 5.5)
)


bars1 = ax.bar(
    x - width / 2,
    retrieval_mean_sec,
    width,
    label="Mean"
)


bars2 = ax.bar(
    x + width / 2,
    retrieval_p95_sec,
    width,
    label="P95"
)


add_bar_labels(
    ax,
    bars1,
    decimals=3
)

add_bar_labels(
    ax,
    bars2,
    decimals=3
)


ax.set_ylabel(
    "Retrieval Latency (s)"
)

ax.set_title(
    "Retrieval Latency Comparison"
)

ax.set_xticks(x)
ax.set_xticklabels(METHODS)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


save_plot(
    "01_retrieval_latency"
)


# ============================================================
# FIGURE 2
# MEAN + P95 E2E LATENCY
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)


bars1 = ax.bar(
    x - width / 2,
    e2e_mean,
    width,
    label="Mean"
)


bars2 = ax.bar(
    x + width / 2,
    e2e_p95,
    width,
    label="P95"
)


add_bar_labels(
    ax,
    bars1,
    decimals=3
)

add_bar_labels(
    ax,
    bars2,
    decimals=3
)


ax.set_ylabel(
    "End-to-End Latency (s)"
)

ax.set_title(
    "End-to-End Latency Comparison"
)

ax.set_xticks(x)
ax.set_xticklabels(METHODS)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


save_plot(
    "02_e2e_latency"
)


## ============================================================
# FIGURE 3A
# COMPLETE MEAN ONLINE LATENCY BREAKDOWN
# ============================================================

LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 12

encoding_plot = np.nan_to_num(
    np.array(encoding_mean),
    nan=0.0
)

retrieval_sec = (
    np.array(retrieval_mean_ms)
    / 1000
)

prompt_arr = np.array(
    prompt_mean
)

generation_arr = np.array(
    generation_mean
)


fig, ax = plt.subplots(
    figsize=(6.2, 4.8)
)


# Encoding
ax.bar(
    x,
    encoding_plot,
    label="Embedding Encoding"
)

# Retrieval
ax.bar(
    x,
    retrieval_sec,
    bottom=encoding_plot,
    label="Retrieval"
)

bottom2 = (
    encoding_plot
    +
    retrieval_sec
)

# Prompt
ax.bar(
    x,
    prompt_arr,
    bottom=bottom2,
    label="Prompt"
)

bottom3 = (
    bottom2
    +
    prompt_arr
)

# Final generation
ax.bar(
    x,
    generation_arr,
    bottom=bottom3,
    label="Final Generation"
)


# ============================================================
# AXES
# ============================================================

ax.set_ylabel(
    "Mean Latency (s)",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

ax.set_xlabel(
    "Method",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

ax.set_xticks(x)

ax.set_xticklabels(
    METHODS,
    fontsize=TICK_SIZE
)

ax.tick_params(
    axis="y",
    labelsize=TICK_SIZE
)

ax.legend(
    fontsize=LEGEND_SIZE
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

ax.set_axisbelow(True)


# ============================================================
# SAVE
# ============================================================

fig.tight_layout()

PNG_FILE = os.path.join(
    OUTPUT_DIR,
    "03a_mean_latency_breakdown.png"
)

PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "03a_mean_latency_breakdown.pdf"
)

fig.savefig(
    PNG_FILE,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    PDF_FILE,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Saved:", PNG_FILE)
print("Saved:", PDF_FILE)

# ============================================================
# FIGURE 3B
# RAG NON-GENERATION LATENCY BREAKDOWN
#
# IFKG is excluded because its retrieval includes a
# Text-to-Cypher LLM call and is architecturally different.
# ============================================================

LABEL_SIZE = 16
TICK_SIZE = 14
VALUE_SIZE = 12
LEGEND_SIZE = 12

rag_x = np.arange(2)

rag_encoding_ms = np.array([
    standard["query_encoding_sec"].mean() * 1000,
    pca["encoding_sec"].mean() * 1000
])

rag_retrieval_ms = np.array([
    standard["retrieval_sec"].mean() * 1000,
    pca["retrieval_sec"].mean() * 1000
])

rag_prompt_ms = np.array([
    standard["prompt_build_sec"].mean() * 1000,
    pca["prompt_sec"].mean() * 1000
])


fig, ax = plt.subplots(
    figsize=(6.2, 4.8)
)


bars1 = ax.bar(
    rag_x - width,
    rag_encoding_ms,
    width,
    label="Encoding"
)

bars2 = ax.bar(
    rag_x,
    rag_retrieval_ms,
    width,
    label="Retrieval"
)

bars3 = ax.bar(
    rag_x + width,
    rag_prompt_ms,
    width,
    label="Prompt"
)


# ============================================================
# VALUE LABELS
# ============================================================

for bars in [bars1, bars2, bars3]:

    for bar in bars:

        value = bar.get_height()

        ax.annotate(
            f"{value:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                value
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            fontweight="bold"
        )


# ============================================================
# AXES
# ============================================================

ax.set_ylabel(
    "Mean Latency (ms)",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

ax.set_xlabel(
    "Method",
    fontsize=LABEL_SIZE,
    fontweight="bold"
)

ax.set_xticks(
    rag_x
)

ax.set_xticklabels(
    RAG_METHODS,
    fontsize=TICK_SIZE
)

ax.tick_params(
    axis="y",
    labelsize=TICK_SIZE
)

# Extra room for the value labels
ax.set_ylim(
    0,
    max(rag_encoding_ms) * 1.15
)

ax.legend(
    fontsize=LEGEND_SIZE
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

ax.set_axisbelow(True)


# ============================================================
# SAVE
# ============================================================

fig.tight_layout()

PNG_FILE = os.path.join(
    OUTPUT_DIR,
    "03b_rag_non_generation_latency.png"
)

PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "03b_rag_non_generation_latency.pdf"
)

fig.savefig(
    PNG_FILE,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    PDF_FILE,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Saved:", PNG_FILE)
print("Saved:", PDF_FILE)

# ============================================================
# FIGURE 3C
# IFKG RETRIEVAL DECOMPOSITION
#
# Retrieval = Text-to-Cypher + Neo4j execution
# ============================================================

ifkg_components = [
    "Text → Cypher",
    "Neo4j Execution"
]

ifkg_mean_components = [
    ifkg_text2cypher_mean,
    ifkg_neo4j_mean
]

ifkg_p95_components = [
    ifkg_text2cypher_p95,
    ifkg_neo4j_p95
]


component_x = np.arange(2)


fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)


bars1 = ax.bar(
    component_x - width / 2,
    ifkg_mean_components,
    width,
    label="Mean"
)


bars2 = ax.bar(
    component_x + width / 2,
    ifkg_p95_components,
    width,
    label="P95"
)


add_bar_labels(
    ax,
    bars1,
    decimals=3
)

add_bar_labels(
    ax,
    bars2,
    decimals=3
)


ax.set_ylabel(
    "Latency (s)"
)

ax.set_title(
    "IFKG Retrieval Latency Breakdown"
)

ax.set_xticks(
    component_x
)

ax.set_xticklabels(
    ifkg_components
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


save_plot(
    "03c_ifkg_retrieval_breakdown"
)


# ============================================================
# FIGURE 4
# RAG CANDIDATE SEARCH SPACE
#
# IFKG is not included because graph retrieval result counts
# are not equivalent to RAG candidate search space.
# ============================================================

candidate_x = np.arange(
    len(RAG_METHODS)
)


fig, ax = plt.subplots(
    figsize=(8, 5.5)
)


bars = ax.bar(
    candidate_x,
    candidate_count
)


for i, bar in enumerate(bars):

    height = bar.get_height()

    if i == 0:

        label = (
            f"{height:,.0f}\n"
            "(Full KB)"
        )

    else:

        reduction = (
            1
            -
            height / 10500
        ) * 100

        label = (
            f"{height:,.2f}\n"
            f"({reduction:.2f}% reduction)"
        )


    ax.annotate(
        label,
        xy=(
            bar.get_x()
            +
            bar.get_width() / 2,
            height
        ),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9
    )


ax.set_ylabel(
    "Average Candidate Records"
)

ax.set_title(
    "RAG Retrieval Search-Space Comparison"
)

ax.set_xticks(
    candidate_x
)

ax.set_xticklabels(
    RAG_METHODS
)

ax.grid(
    axis="y",
    alpha=0.25
)


save_plot(
    "04_candidate_search_space"
)


# ============================================================
# FIGURE 5
# AUTOMATIC QUALITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)


bars1 = ax.bar(
    x - width / 2,
    rouge,
    width,
    label="ROUGE-L"
)


bars2 = ax.bar(
    x + width / 2,
    bert,
    width,
    label="BERTScore F1"
)


add_bar_labels(
    ax,
    bars1,
    decimals=4
)

add_bar_labels(
    ax,
    bars2,
    decimals=4
)


ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Automatic Resolution Quality"
)

ax.set_xticks(x)
ax.set_xticklabels(METHODS)

ax.set_ylim(
    0,
    1.05
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


save_plot(
    "05_automatic_quality"
)


# ============================================================
# FIGURE 6
# LLM-AS-JUDGE
# ============================================================

judge_width = 0.25


fig, ax = plt.subplots(
    figsize=(9, 5.5)
)


bars1 = ax.bar(
    x - judge_width,
    relevant,
    judge_width,
    label="Relevant"
)


bars2 = ax.bar(
    x,
    closer,
    judge_width,
    label="Closer"
)


bars3 = ax.bar(
    x + judge_width,
    inadequate,
    judge_width,
    label="Inadequate"
)


add_bar_labels(
    ax,
    bars1,
    decimals=2
)

add_bar_labels(
    ax,
    bars2,
    decimals=2
)

add_bar_labels(
    ax,
    bars3,
    decimals=2
)


ax.set_ylabel(
    "Judge Decisions (%)"
)

ax.set_title(
    "LLM-as-Judge Evaluation"
)

ax.set_xticks(x)
ax.set_xticklabels(METHODS)

ax.set_ylim(
    0,
    105
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


save_plot(
    "06_llm_judge"
)


# ============================================================
# OPTIONAL FIGURE 7
# PCA DIMENSION / EXPLAINED VARIANCE
# ============================================================

if os.path.exists(
    PCA_OFFLINE_SUMMARY
):

    with open(
        PCA_OFFLINE_SUMMARY,
        "r",
        encoding="utf-8"
    ) as f:

        pca_summary = json.load(f)


    explained = None

    for key in [
        "explained_variance",
        "explained_variance_ratio",
        "pca_explained_variance"
    ]:

        if key in pca_summary:

            explained = float(
                pca_summary[key]
            )

            break


    if explained is not None:

        if explained <= 1:
            explained *= 100


        fig, ax = plt.subplots(
            figsize=(6.5, 5)
        )


        bars = ax.bar(
            [
                "Retained Variance",
                "Discarded Variance"
            ],
            [
                explained,
                100 - explained
            ]
        )


        add_bar_labels(
            ax,
            bars,
            decimals=2
        )


        ax.set_ylabel(
            "Variance (%)"
        )

        ax.set_ylim(
            0,
            105
        )

        ax.set_title(
            "PCA Compression: 256D → 50D"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )


        save_plot(
            "07_pca_explained_variance"
        )


# ============================================================
# FINAL SUMMARY TABLE
#
# NaN is intentionally used for IFKG encoding and candidate
# count because those quantities are not directly applicable.
# ============================================================

summary = pd.DataFrame({

    "Method":
        METHODS,

    "Mean Encoding (ms)":
        [
            standard["query_encoding_sec"].mean() * 1000,
            pca["encoding_sec"].mean() * 1000,
            np.nan
        ],

    "P95 Encoding (ms)":
        [
            standard["query_encoding_sec"].quantile(0.95) * 1000,
            pca["encoding_sec"].quantile(0.95) * 1000,
            np.nan
        ],

    "Mean Retrieval (s)":
        [
            standard["retrieval_sec"].mean(),
            pca["retrieval_sec"].mean(),
            ifkg["retrieval_sec"].mean()
        ],

    "P95 Retrieval (s)":
        [
            standard["retrieval_sec"].quantile(0.95),
            pca["retrieval_sec"].quantile(0.95),
            ifkg["retrieval_sec"].quantile(0.95)
        ],

    "Mean Final Generation (s)":
        generation_mean,

    "P95 Final Generation (s)":
        generation_p95,

    "Mean E2E (s)":
        e2e_mean,

    "P95 E2E (s)":
        e2e_p95,

    "Avg Candidates":
        [
            10500,
            pca["candidate_count"].mean(),
            np.nan
        ],

    "ROUGE-L":
        rouge,

    "BERTScore F1":
        bert,

    "LLM Judge Relevant (%)":
        relevant,

    "LLM Judge Closer (%)":
        closer,

    "LLM Judge Inadequate (%)":
        inadequate,

    "LLM Judge Relevant+Closer (%)":
        relevant_plus_closer
})


SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "final_results_summary.csv"
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    "\n" + "=" * 80
)

print(
    "ALL FINAL PLOTS COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nPlots saved in:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSummary:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nSummary CSV:"
)

print(
    SUMMARY_FILE
)