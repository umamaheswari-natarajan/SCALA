from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "latency_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

EXPECTED_QUERIES = 872


# ============================================================
# INPUT FILES
# ============================================================

STANDARD_FILE = os.path.join(
    RESULTS_DIR,
    "standard_rag",
    "standard_rag_query_results.csv"
)

SCALA_FILE = os.path.join(
    RESULTS_DIR,
    "rag_inference_nearest_mean_cosine_top15",
    "scala_query_results.csv"
)

IFKG_FILE = os.path.join(
    RESULTS_DIR,
    "ifkg_inference",
    "ifkg_query_results.csv"
)


# ============================================================
# METHODS
# ============================================================

methods = [
    "Standard RAG",
    "SCALA",
    "IFKG"
]


# ============================================================
# HELPER: CHECK REQUIRED COLUMNS
# ============================================================

def check_columns(
    df,
    required_columns,
    method_name
):

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{method_name}: missing columns: {missing}"
        )


# ============================================================
# LOAD STANDARD RAG
# ============================================================

print("\n" + "=" * 90)
print("LOADING STANDARD RAG")
print("=" * 90)

standard_df = pd.read_csv(
    STANDARD_FILE
)

print("Rows:", len(standard_df))

if len(standard_df) != EXPECTED_QUERIES:
    raise ValueError(
        f"Standard RAG: expected {EXPECTED_QUERIES} rows, "
        f"found {len(standard_df)}"
    )

check_columns(
    standard_df,
    [
        "encoding_ms",
        "standard_rag_retrieval_ms",
        "standard_rag_prompt_ms",
        "standard_rag_llm_ms"
    ],
    "Standard RAG"
)


# ============================================================
# LOAD SCALA
# ============================================================

print("\n" + "=" * 90)
print("LOADING SCALA")
print("=" * 90)

scala_df = pd.read_csv(
    SCALA_FILE
)

print("Rows:", len(scala_df))

if len(scala_df) != EXPECTED_QUERIES:
    raise ValueError(
        f"SCALA: expected {EXPECTED_QUERIES} rows, "
        f"found {len(scala_df)}"
    )

check_columns(
    scala_df,
    [
        "encoding_ms",
        "retrieval_ms",
        "prompt_ms",
        "llm_ms"
    ],
    "SCALA"
)


# ============================================================
# LOAD IFKG
# ============================================================

print("\n" + "=" * 90)
print("LOADING IFKG")
print("=" * 90)

ifkg_df = pd.read_csv(
    IFKG_FILE
)

print("Rows:", len(ifkg_df))

if len(ifkg_df) != EXPECTED_QUERIES:
    raise ValueError(
        f"IFKG: expected {EXPECTED_QUERIES} rows, "
        f"found {len(ifkg_df)}"
    )

check_columns(
    ifkg_df,
    [
        "ifkg_text_to_cypher_ms",
        "ifkg_kg_query_ms",
        "ifkg_prompt_ms",
        "ifkg_llm_ms"
    ],
    "IFKG"
)


# ============================================================
# CONVERT STANDARD RAG LATENCIES TO NUMERIC
# ============================================================

standard_columns = [
    "encoding_ms",
    "standard_rag_retrieval_ms",
    "standard_rag_prompt_ms",
    "standard_rag_llm_ms"
]

for col in standard_columns:

    standard_df[col] = pd.to_numeric(
        standard_df[col],
        errors="coerce"
    )


# ============================================================
# CONVERT SCALA LATENCIES TO NUMERIC
# ============================================================

scala_columns = [
    "encoding_ms",
    "retrieval_ms",
    "prompt_ms",
    "llm_ms"
]

for col in scala_columns:

    scala_df[col] = pd.to_numeric(
        scala_df[col],
        errors="coerce"
    )


# ============================================================
# CONVERT IFKG LATENCIES TO NUMERIC
# ============================================================

ifkg_columns = [
    "ifkg_text_to_cypher_ms",
    "ifkg_kg_query_ms",
    "ifkg_prompt_ms",
    "ifkg_llm_ms"
]

for col in ifkg_columns:

    ifkg_df[col] = pd.to_numeric(
        ifkg_df[col],
        errors="coerce"
    )


# ============================================================
# FIGURE 1
# RETRIEVAL LATENCY
#
# Standard RAG:
#     vector retrieval
#
# SCALA:
#     cluster-based retrieval
#
# IFKG:
#     Neo4j KG query
# ============================================================

standard_retrieval_ms = (
    standard_df[
        "standard_rag_retrieval_ms"
    ]
)

scala_retrieval_ms = (
    scala_df[
        "retrieval_ms"
    ]
)

ifkg_retrieval_ms = (
   ifkg_df[
        "ifkg_kg_query_ms"
    ]
)


# ============================================================
# FIGURE 2
# E2E LATENCY EXCLUDING FINAL LLM
#
# Standard RAG:
#     encoding + retrieval + prompt
#
# SCALA:
#     encoding + retrieval + prompt
#
# IFKG:
#     text-to-Cypher + KG query + prompt
#
# IMPORTANT:
# Only the FINAL answer-generation LLM is excluded.
# IFKG text-to-Cypher remains part of the processing pipeline.
# ============================================================

standard_e2e_excl_llm_ms = (
    standard_df[
        "encoding_ms"
    ]
    +
    standard_df[
        "standard_rag_retrieval_ms"
    ]
    +
    standard_df[
        "standard_rag_prompt_ms"
    ]
)

scala_e2e_excl_llm_ms = (
    scala_df[
        "encoding_ms"
    ]
    +
    scala_df[
        "retrieval_ms"
    ]
    +
    scala_df[
        "prompt_ms"
    ]
)

ifkg_e2e_excl_llm_ms = (
    ifkg_df[
        "ifkg_text_to_cypher_ms"
    ]
    +
    ifkg_df[
        "ifkg_kg_query_ms"
    ]
    +
    ifkg_df[
        "ifkg_prompt_ms"
    ]
)


# ============================================================
# FIGURE 3
# FINAL LLM GENERATION LATENCY
# ============================================================

standard_llm_ms = (
    standard_df[
        "standard_rag_llm_ms"
    ]
)

scala_llm_ms = (
    scala_df[
        "llm_ms"
    ]
)

ifkg_llm_ms = (
    ifkg_df[
        "ifkg_llm_ms"
    ]
)


# ============================================================
# CHECK LATENCY DATA
# ============================================================

latency_series = {

    "Standard RAG Retrieval":
        standard_retrieval_ms,

    "SCALA Retrieval":
        scala_retrieval_ms,

    "IFKG Retrieval":
        ifkg_retrieval_ms,

    "Standard RAG E2E excl. final LLM":
        standard_e2e_excl_llm_ms,

    "SCALA E2E excl. final LLM":
        scala_e2e_excl_llm_ms,

    "IFKG E2E excl. final LLM":
        ifkg_e2e_excl_llm_ms,

    "Standard RAG LLM":
        standard_llm_ms,

    "SCALA LLM":
        scala_llm_ms,

    "IFKG LLM":
        ifkg_llm_ms
}


print("\n" + "=" * 90)
print("CHECKING LATENCY DATA")
print("=" * 90)

for name, series in latency_series.items():

    valid = int(
        series.notna().sum()
    )

    missing = int(
        series.isna().sum()
    )

    print(
        f"{name:42s} "
        f"Valid = {valid:4d} | "
        f"Missing = {missing}"
    )

    if missing > 0:
        raise ValueError(
            f"{name} contains "
            f"{missing} missing values."
        )


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize(series):

    values = (
        series
        .astype(float)
        .to_numpy()
    )

    mean_value = float(
        np.mean(values)
    )

    p95_value = float(
        np.percentile(
            values,
            95
        )
    )

    return (
        mean_value,
        p95_value
    )


# ============================================================
# CALCULATE MEAN + P95
# ============================================================

retrieval_stats = [
    summarize(
        standard_retrieval_ms
    ),
    summarize(
        scala_retrieval_ms
    ),
    summarize(
        ifkg_retrieval_ms
    )
]


e2e_stats = [
    summarize(
        standard_e2e_excl_llm_ms
    ),
    summarize(
        scala_e2e_excl_llm_ms
    ),
    summarize(
        ifkg_e2e_excl_llm_ms
    )
]


llm_stats = [
    summarize(
        standard_llm_ms
    ),
    summarize(
        scala_llm_ms
    ),
    summarize(
        ifkg_llm_ms
    )
]


# ============================================================
# ARRAYS FOR PLOTTING
# ============================================================

retrieval_mean = np.array([
    x[0]
    for x in retrieval_stats
])

retrieval_p95 = np.array([
    x[1]
    for x in retrieval_stats
])


e2e_mean = np.array([
    x[0]
    for x in e2e_stats
])

e2e_p95 = np.array([
    x[1]
    for x in e2e_stats
])


# Convert LLM latency from milliseconds to seconds

llm_mean = np.array([
    x[0] / 1000.0
    for x in llm_stats
])

llm_p95 = np.array([
    x[1] / 1000.0
    for x in llm_stats
])


# ============================================================
# PRINT VALUES
# ============================================================

print("\n" + "=" * 90)
print("CALCULATED LATENCY VALUES")
print("=" * 90)


print(
    "\nRETRIEVAL LATENCY (ms)"
)

for i, method in enumerate(methods):

    print(
        f"{method:15s} "
        f"Mean = {retrieval_mean[i]:.3f} ms | "
        f"P95 = {retrieval_p95[i]:.3f} ms"
    )


print(
    "\nE2E LATENCY EXCLUDING FINAL LLM (ms)"
)

for i, method in enumerate(methods):

    print(
        f"{method:15s} "
        f"Mean = {e2e_mean[i]:.3f} ms | "
        f"P95 = {e2e_p95[i]:.3f} ms"
    )


print(
    "\nLLM GENERATION LATENCY (s)"
)

for i, method in enumerate(methods):

    print(
        f"{method:15s} "
        f"Mean = {llm_mean[i]:.3f} s | "
        f"P95 = {llm_p95[i]:.3f} s"
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_df = pd.DataFrame({

    "method":
        methods,

    "retrieval_mean_ms":
        retrieval_mean,

    "retrieval_p95_ms":
        retrieval_p95,

    "e2e_excluding_final_llm_mean_ms":
        e2e_mean,

    "e2e_excluding_final_llm_p95_ms":
        e2e_p95,

    "llm_generation_mean_s":
        llm_mean,

    "llm_generation_p95_s":
        llm_p95
})


SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "latency_comparison_summary.csv"
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams.update({

    "font.size": 11,

    "axes.labelsize": 12,

    "axes.labelweight": "bold",

    "xtick.labelsize": 11,

    "ytick.labelsize": 11,

    "legend.fontsize": 11
})


x = np.arange(
    len(methods)
)

width = 0.34


# ============================================================
# BAR LABEL HELPER
# ============================================================

def add_bar_labels(
    ax,
    bars,
    decimals=3
):

    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            f"{height:.{decimals}f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )


# ============================================================
# FIGURE 1
# RETRIEVAL LATENCY
# LOG SCALE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.4, 4.7)
)


bars_mean = ax.bar(
    x - width / 2,
    retrieval_mean,
    width,
    label="Mean"
)


bars_p95 = ax.bar(
    x + width / 2,
    retrieval_p95,
    width,
    label="P95"
)


ax.set_yscale(
    "log"
)


ax.set_ylabel(
    "Retrieval Latency (ms)"
)


ax.set_xlabel(
    "Method"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    methods,
    fontweight="bold"
)


ax.grid(
    axis="y",
    which="major",
    linestyle="--",
    alpha=0.35
)

ax.set_axisbelow(
    True
)


ax.legend(
    loc="upper left"
)


all_values = np.concatenate([
    retrieval_mean,
    retrieval_p95
])

positive_values = (
    all_values[
        all_values > 0
    ]
)


ax.set_ylim(
    positive_values.min() / 2.0,
    positive_values.max() * 2.0
)


add_bar_labels(
    ax,
    bars_mean,
    decimals=2
)

add_bar_labels(
    ax,
    bars_p95,
    decimals=2
)


plt.tight_layout()


RETRIEVAL_PNG = os.path.join(
    OUTPUT_DIR,
    "retrieval_latency_log.png"
)

RETRIEVAL_PDF = os.path.join(
    OUTPUT_DIR,
    "retrieval_latency_log.pdf"
)


plt.savefig(
    RETRIEVAL_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    RETRIEVAL_PDF,
    bbox_inches="tight"
)


plt.show()
plt.close()


# ============================================================
# FIGURE 2
# E2E LATENCY EXCLUDING FINAL LLM
# LOG SCALE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.4, 4.7)
)


bars_mean = ax.bar(
    x - width / 2,
    e2e_mean,
    width,
    label="Mean"
)


bars_p95 = ax.bar(
    x + width / 2,
    e2e_p95,
    width,
    label="P95"
)


ax.set_yscale(
    "log"
)


ax.set_ylabel(
    "End-to-End Latency (ms)"
)


ax.set_xlabel(
    "Method"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    methods,
    fontweight="bold"
)


ax.grid(
    axis="y",
    which="major",
    linestyle="--",
    alpha=0.35
)

ax.set_axisbelow(
    True
)


ax.legend(
    loc="upper left"
)


all_values = np.concatenate([
    e2e_mean,
    e2e_p95
])

positive_values = (
    all_values[
        all_values > 0
    ]
)


ax.set_ylim(
    positive_values.min() / 2.0,
    positive_values.max() * 2.0
)


add_bar_labels(
    ax,
    bars_mean,
    decimals=2
)

add_bar_labels(
    ax,
    bars_p95,
    decimals=2
)


plt.tight_layout()


E2E_PNG = os.path.join(
    OUTPUT_DIR,
    "e2e_latency_excluding_final_llm_log.png"
)

E2E_PDF = os.path.join(
    OUTPUT_DIR,
    "e2e_latency_excluding_final_llm_log.pdf"
)


plt.savefig(
    E2E_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    E2E_PDF,
    bbox_inches="tight"
)


plt.show()
plt.close()


# ============================================================
# FIGURE 3
# LLM GENERATION LATENCY
# LINEAR SCALE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.4, 4.7)
)


bars_mean = ax.bar(
    x - width / 2,
    llm_mean,
    width,
    label="Mean"
)


bars_p95 = ax.bar(
    x + width / 2,
    llm_p95,
    width,
    label="P95"
)


ax.set_ylabel(
    "LLM Generation Latency (s)"
)


ax.set_xlabel(
    "Method"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    methods,
    fontweight="bold"
)


ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.30
)

ax.set_axisbelow(
    True
)


ax.legend(
    loc="upper left"
)


max_llm = float(
    max(llm_p95)
)


ax.set_ylim(
    0,
    max_llm * 1.20
)


add_bar_labels(
    ax,
    bars_mean,
    decimals=3
)

add_bar_labels(
    ax,
    bars_p95,
    decimals=3
)


plt.tight_layout()


LLM_PNG = os.path.join(
    OUTPUT_DIR,
    "llm_generation_latency.png"
)

LLM_PDF = os.path.join(
    OUTPUT_DIR,
    "llm_generation_latency.pdf"
)


plt.savefig(
    LLM_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    LLM_PDF,
    bbox_inches="tight"
)


plt.show()
plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    "\nSummary CSV:"
)

print(
    SUMMARY_FILE
)


print(
    "\nFigure 1 - Retrieval Latency:"
)

print(
    RETRIEVAL_PNG
)

print(
    RETRIEVAL_PDF
)


print(
    "\nFigure 2 - E2E Excluding Final LLM:"
)

print(
    E2E_PNG
)

print(
    E2E_PDF
)


print(
    "\nFigure 3 - LLM Generation Latency:"
)

print(
    LLM_PNG
)

print(
    LLM_PDF
)


print(
    "\nCOMPLETE"
)