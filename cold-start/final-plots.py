from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SCALA_FILE = os.path.join(
    BASE_DIR,
    "results",
    "scala64_vs_10500_cosine_top1_k2_final1050",
    "query_results.csv"
)

IFKG_FILE = os.path.join(
    BASE_DIR,
    "synthetic_ifkg_gpt5mini",
    "ifkg_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "latency_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. PLOT CONFIGURATION
# ============================================================

METHODS = [
    "SCALA-64",
    "SCALA-10,500",
    "IFKG"
]

BAR_WIDTH = 0.34

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150
})


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("LOADING LATENCY FILES")
print("=" * 90)

scala = pd.read_csv(SCALA_FILE)
ifkg = pd.read_csv(IFKG_FILE)

print(f"SCALA rows : {len(scala)}")
print(f"IFKG rows  : {len(ifkg)}")


# ============================================================
# 4. VALIDATE COLUMNS
# ============================================================

required_scala = [
    "method",
    "encoding_ms",
    "retrieval_ms",
    "prompt_ms",
    "llm_ms",
    "e2e_ms"
]

required_ifkg = [
    "text2cypher_sec",
    "kg_execution_sec",
    "retrieval_sec",
    "prompt_sec",
    "generation_sec",
    "e2e_sec"
]

for col in required_scala:
    if col not in scala.columns:
        raise ValueError(f"Missing SCALA column: {col}")

for col in required_ifkg:
    if col not in ifkg.columns:
        raise ValueError(f"Missing IFKG column: {col}")


# ============================================================
# 5. SPLIT SCALA METHODS
# ============================================================

scala64 = (
    scala[
        scala["method"] == "scala_64"
    ]
    .copy()
    .reset_index(drop=True)
)

scala10500 = (
    scala[
        scala["method"] == "scala_10500"
    ]
    .copy()
    .reset_index(drop=True)
)

print("\nSCALA-64 queries     :", len(scala64))
print("SCALA-10,500 queries :", len(scala10500))
print("IFKG queries         :", len(ifkg))

if len(scala64) != 1050:
    raise ValueError(
        f"Expected 1050 SCALA-64 rows, found {len(scala64)}"
    )

if len(scala10500) != 1050:
    raise ValueError(
        f"Expected 1050 SCALA-10,500 rows, found {len(scala10500)}"
    )

if len(ifkg) != 1050:
    raise ValueError(
        f"Expected 1050 IFKG rows, found {len(ifkg)}"
    )


# ============================================================
# 6. SCALA PER-QUERY LATENCIES
# ============================================================

# retrieval_ms already represents complete SCALA retrieval:
#
# cluster-mean cosine routing
# + Top-1 cluster selection
# + candidate pooling
# + candidate cosine retrieval
# + Top-2 selection

scala64["retrieval_total_ms"] = scala64["retrieval_ms"]

scala10500["retrieval_total_ms"] = (
    scala10500["retrieval_ms"]
)


# SCALA E2E excluding final LLM:
#
# encoding
# + complete retrieval
# + prompt construction

scala64["nonllm_e2e_ms"] = (
    scala64["encoding_ms"]
    + scala64["retrieval_ms"]
    + scala64["prompt_ms"]
)

scala10500["nonllm_e2e_ms"] = (
    scala10500["encoding_ms"]
    + scala10500["retrieval_ms"]
    + scala10500["prompt_ms"]
)


# LLM generation latency in seconds

scala64["generation_sec_plot"] = (
    scala64["llm_ms"] / 1000.0
)

scala10500["generation_sec_plot"] = (
    scala10500["llm_ms"] / 1000.0
)


# Full E2E including LLM

scala64["full_e2e_sec"] = (
    scala64["e2e_ms"] / 1000.0
)

scala10500["full_e2e_sec"] = (
    scala10500["e2e_ms"] / 1000.0
)


# ============================================================
# 7. IFKG PER-QUERY LATENCIES
# ============================================================

# ------------------------------------------------------------
# Diagnostic:
#
# In the stored IFKG results:
#
# retrieval_sec =
# text2cypher_sec + kg_execution_sec
#
# We verify this relationship, but DO NOT use retrieval_sec
# as the retrieval-only latency in the comparison figure.
# ------------------------------------------------------------

ifkg["retrieval_reconstructed_sec"] = (
    ifkg["text2cypher_sec"]
    + ifkg["kg_execution_sec"]
)

retrieval_difference = (
    ifkg["retrieval_sec"]
    - ifkg["retrieval_reconstructed_sec"]
).abs()

print("\n" + "=" * 90)
print("IFKG STORED RETRIEVAL VERIFICATION")
print("=" * 90)

print(
    "Maximum |retrieval_sec - "
    "(text2cypher_sec + kg_execution_sec)| =",
    retrieval_difference.max()
)

print(
    "Mean difference =",
    retrieval_difference.mean()
)


# ------------------------------------------------------------
# IMPORTANT:
# IFKG retrieval latency for the retrieval-only comparison
# is KG query execution ONLY.
#
# seconds -> milliseconds
# ------------------------------------------------------------

ifkg["retrieval_total_ms"] = (
    ifkg["kg_execution_sec"] * 1000.0
)


# ------------------------------------------------------------
# IFKG E2E excluding final LLM:
#
# text-to-Cypher
# + KG execution
# + prompt construction
#
# seconds -> milliseconds
# ------------------------------------------------------------

ifkg["nonllm_e2e_ms"] = (
    (
        ifkg["text2cypher_sec"]
        + ifkg["kg_execution_sec"]
        + ifkg["prompt_sec"]
    )
    * 1000.0
)


# IFKG LLM generation is already seconds

ifkg["generation_sec_plot"] = (
    ifkg["generation_sec"]
)


# IFKG complete E2E already seconds

ifkg["full_e2e_sec"] = (
    ifkg["e2e_sec"]
)


# ============================================================
# 8. HELPER: MEAN AND P95
# ============================================================

def mean_p95(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise ValueError(
            "No valid latency values found."
        )

    return (
        float(np.mean(values)),
        float(np.percentile(values, 95))
    )


# ============================================================
# 9. HELPER: BAR LABELS
# ============================================================

def add_bar_labels(
    ax,
    bars,
    values,
    decimals=2,
    log_scale=False
):

    max_value = max(values)

    for bar, value in zip(
        bars,
        values
    ):

        x_position = (
            bar.get_x()
            + bar.get_width() / 2
        )

        if log_scale:
            y_position = value * 1.10
        else:
            y_position = (
                value
                + max_value * 0.025
            )

        ax.text(
            x_position,
            y_position,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )


# ============================================================
# 10. RETRIEVAL LATENCY
# ============================================================

retrieval_series = [
    scala64["retrieval_total_ms"],
    scala10500["retrieval_total_ms"],
    ifkg["retrieval_total_ms"]
]

retrieval_mean = []
retrieval_p95 = []

for values in retrieval_series:

    mean_value, p95_value = mean_p95(values)

    retrieval_mean.append(mean_value)
    retrieval_p95.append(p95_value)


# ============================================================
# 11. E2E EXCLUDING FINAL LLM
# ============================================================

nonllm_series = [
    scala64["nonllm_e2e_ms"],
    scala10500["nonllm_e2e_ms"],
    ifkg["nonllm_e2e_ms"]
]

nonllm_mean = []
nonllm_p95 = []

for values in nonllm_series:

    mean_value, p95_value = mean_p95(values)

    nonllm_mean.append(mean_value)
    nonllm_p95.append(p95_value)


# ============================================================
# 12. LLM GENERATION
# ============================================================

llm_series = [
    scala64["generation_sec_plot"],
    scala10500["generation_sec_plot"],
    ifkg["generation_sec_plot"]
]

llm_mean = []
llm_p95 = []

for values in llm_series:

    mean_value, p95_value = mean_p95(values)

    llm_mean.append(mean_value)
    llm_p95.append(p95_value)


# ============================================================
# 13. FULL E2E INCLUDING LLM
# ============================================================

full_e2e_series = [
    scala64["full_e2e_sec"],
    scala10500["full_e2e_sec"],
    ifkg["full_e2e_sec"]
]

full_e2e_mean = []
full_e2e_p95 = []

for values in full_e2e_series:

    mean_value, p95_value = mean_p95(values)

    full_e2e_mean.append(mean_value)
    full_e2e_p95.append(p95_value)


# ============================================================
# 14. SCALA COMPONENT BREAKDOWN
# ============================================================

print("\n" + "=" * 90)
print("SCALA NON-LLM COMPONENT BREAKDOWN — MEAN LATENCY (ms)")
print("=" * 90)

component_table = pd.DataFrame({

    "Method": [
        "SCALA-64",
        "SCALA-10,500"
    ],

    "Encoding": [
        scala64["encoding_ms"].mean(),
        scala10500["encoding_ms"].mean()
    ],

    "Retrieval": [
        scala64["retrieval_ms"].mean(),
        scala10500["retrieval_ms"].mean()
    ],

    "Prompt": [
        scala64["prompt_ms"].mean(),
        scala10500["prompt_ms"].mean()
    ],

    "E2E excl. LLM": [
        scala64["nonllm_e2e_ms"].mean(),
        scala10500["nonllm_e2e_ms"].mean()
    ]
})

print(
    component_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 15. IFKG COMPONENT BREAKDOWN
# ============================================================

print("\n" + "=" * 90)
print("IFKG COMPONENT BREAKDOWN")
print("=" * 90)

print(
    f"Text-to-Cypher          : "
    f"{ifkg['text2cypher_sec'].mean() * 1000:.3f} ms"
)

print(
    f"KG execution            : "
    f"{ifkg['kg_execution_sec'].mean() * 1000:.3f} ms"
)

print(
    f"Stored retrieval        : "
    f"{ifkg['retrieval_sec'].mean() * 1000:.3f} ms"
)

print(
    f"Retrieval used in plot  : "
    f"{ifkg['retrieval_total_ms'].mean():.3f} ms"
)

print(
    f"Prompt construction     : "
    f"{ifkg['prompt_sec'].mean() * 1000:.3f} ms"
)

print(
    f"Generation              : "
    f"{ifkg['generation_sec'].mean():.3f} s"
)

print(
    f"E2E including LLM       : "
    f"{ifkg['e2e_sec'].mean():.3f} s"
)

print(
    f"E2E excluding LLM       : "
    f"{ifkg['nonllm_e2e_ms'].mean():.3f} ms"
)


# ============================================================
# 16. CROSS-CHECK E2E DEFINITIONS
# ============================================================

print("\n" + "=" * 90)
print("E2E CROSS-CHECK")
print("=" * 90)

# Compare reconstructed IFKG non-LLM E2E with
# e2e_sec - generation_sec.

ifkg["nonllm_from_total_ms"] = (
    (
        ifkg["e2e_sec"]
        - ifkg["generation_sec"]
    )
    * 1000.0
)

ifkg_e2e_difference = (
    ifkg["nonllm_e2e_ms"]
    - ifkg["nonllm_from_total_ms"]
).abs()

print(
    "IFKG reconstructed non-LLM E2E mean : "
    f"{ifkg['nonllm_e2e_ms'].mean():.6f} ms"
)

print(
    "IFKG (E2E - generation) mean        : "
    f"{ifkg['nonllm_from_total_ms'].mean():.6f} ms"
)

print(
    "Mean absolute difference            : "
    f"{ifkg_e2e_difference.mean():.6f} ms"
)

print(
    "Maximum absolute difference         : "
    f"{ifkg_e2e_difference.max():.6f} ms"
)


# ============================================================
# 17. FINAL SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({

    "Method": METHODS,

    "Retrieval Mean (ms)":
        retrieval_mean,

    "Retrieval P95 (ms)":
        retrieval_p95,

    "E2E excl. LLM Mean (ms)":
        nonllm_mean,

    "E2E excl. LLM P95 (ms)":
        nonllm_p95,

    "LLM Mean (s)":
        llm_mean,

    "LLM P95 (s)":
        llm_p95,

    "Full E2E Mean (s)":
        full_e2e_mean,

    "Full E2E P95 (s)":
        full_e2e_p95
})

print("\n" + "=" * 90)
print("FINAL LATENCY SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 18. SAVE SUMMARY
# ============================================================

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "cold_start_latency_summary.csv"
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

COMPONENT_FILE = os.path.join(
    OUTPUT_DIR,
    "scala_latency_component_summary.csv"
)

component_table.to_csv(
    COMPONENT_FILE,
    index=False
)


# ============================================================
# 19. COMMON X POSITIONS
# ============================================================

x = np.arange(
    len(METHODS)
)


# ============================================================
# 20. FIGURE 1 — RETRIEVAL LATENCY
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 4.8)
)

bars_mean = ax.bar(
    x - BAR_WIDTH / 2,
    retrieval_mean,
    BAR_WIDTH,
    label="Mean"
)

bars_p95 = ax.bar(
    x + BAR_WIDTH / 2,
    retrieval_p95,
    BAR_WIDTH,
    label="P95"
)

ax.set_yscale("log")

ax.set_ylabel(
    "Retrieval Latency (ms)",
    fontsize=12,
    fontweight="bold"
)

ax.set_xticks(x)
ax.set_xticklabels(METHODS, fontsize=11, fontweight="bold")

ax.set_xlabel("Method", fontsize=12, fontweight="bold")

ax.grid(
    axis="y",
    which="major",
    linestyle="--",
    alpha=0.35
)

ax.legend()

add_bar_labels(
    ax,
    bars_mean,
    retrieval_mean,
    decimals=2,
    log_scale=True
)

add_bar_labels(
    ax,
    bars_p95,
    retrieval_p95,
    decimals=2,
    log_scale=True
)

current_bottom, current_top = ax.get_ylim()

ax.set_ylim(
    current_bottom,
    current_top * 1.35
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cold_start_retrieval_latency.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cold_start_retrieval_latency.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 21. FIGURE 2 — E2E LATENCY EXCLUDING FINAL LLM
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 4.8)
)

bars_mean = ax.bar(
    x - BAR_WIDTH / 2,
    nonllm_mean,
    BAR_WIDTH,
    label="Mean"
)

bars_p95 = ax.bar(
    x + BAR_WIDTH / 2,
    nonllm_p95,
    BAR_WIDTH,
    label="P95"
)

ax.set_yscale("log")

ax.set_ylabel(
    "End-to-End Latency (ms)",
    fontsize=12,
    fontweight="bold"
)

ax.set_xticks(x)
ax.set_xticklabels(METHODS, fontsize=11, fontweight="bold")

ax.set_xlabel("Method", fontsize=12, fontweight="bold")

ax.grid(
    axis="y",
    which="major",
    linestyle="--",
    alpha=0.35
)

ax.legend()

add_bar_labels(
    ax,
    bars_mean,
    nonllm_mean,
    decimals=2,
    log_scale=True
)

add_bar_labels(
    ax,
    bars_p95,
    nonllm_p95,
    decimals=2,
    log_scale=True
)

current_bottom, current_top = ax.get_ylim()

ax.set_ylim(
    current_bottom,
    current_top * 1.35
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cold_start_e2e_excluding_llm.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cold_start_e2e_excluding_llm.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 22. FIGURE 3 — LLM GENERATION LATENCY
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 4.8)
)

bars_mean = ax.bar(
    x - BAR_WIDTH / 2,
    llm_mean,
    BAR_WIDTH,
    label="Mean"
)

bars_p95 = ax.bar(
    x + BAR_WIDTH / 2,
    llm_p95,
    BAR_WIDTH,
    label="P95"
)

ax.set_ylabel(
    "Latency (s)"
)

ax.set_xticks(x)
ax.set_xticklabels(METHODS)

ax.set_xlabel("Method")

ax.set_title(
    "LLM Generation Latency"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

ax.legend()

add_bar_labels(
    ax,
    bars_mean,
    llm_mean,
    decimals=2,
    log_scale=False
)

add_bar_labels(
    ax,
    bars_p95,
    llm_p95,
    decimals=2,
    log_scale=False
)

ax.set_ylim(
    0,
    max(
        max(llm_mean),
        max(llm_p95)
    ) * 1.18
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cold_start_llm_generation_latency.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cold_start_llm_generation_latency.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 23. FIGURE 4 — FULL E2E INCLUDING LLM
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 4.8)
)

bars_mean = ax.bar(
    x - BAR_WIDTH / 2,
    full_e2e_mean,
    BAR_WIDTH,
    label="Mean"
)

bars_p95 = ax.bar(
    x + BAR_WIDTH / 2,
    full_e2e_p95,
    BAR_WIDTH,
    label="P95"
)

ax.set_ylabel(
    "Latency (s)"
)

ax.set_xticks(x)
ax.set_xticklabels(METHODS)

ax.set_xlabel("Method")

ax.set_title(
    "End-to-End Latency Including LLM"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

ax.legend()

add_bar_labels(
    ax,
    bars_mean,
    full_e2e_mean,
    decimals=2,
    log_scale=False
)

add_bar_labels(
    ax,
    bars_p95,
    full_e2e_p95,
    decimals=2,
    log_scale=False
)

ax.set_ylim(
    0,
    max(
        max(full_e2e_mean),
        max(full_e2e_p95)
    ) * 1.18
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cold_start_full_e2e_latency.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cold_start_full_e2e_latency.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 24. COMPLETE
# ============================================================

print("\n" + "=" * 90)
print("PLOTS SAVED")
print("=" * 90)

print("Output directory:", OUTPUT_DIR)

print(
    "\n1. cold_start_retrieval_latency.png/.pdf"
)

print(
    "2. cold_start_e2e_excluding_llm.png/.pdf"
)

print(
    "3. cold_start_llm_generation_latency.png/.pdf"
)

print(
    "4. cold_start_full_e2e_latency.png/.pdf"
)

print(
    "\nSummary CSV:",
    SUMMARY_FILE
)

print(
    "SCALA component CSV:",
    COMPONENT_FILE
)

print("\nCOMPLETE")