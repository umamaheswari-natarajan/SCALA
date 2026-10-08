import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC"
    r"\SCALA\synthetic-dataset"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_synthetic_plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DPI = 300


# ============================================================
# METHODS
# ============================================================

methods = [
    "Standard RAG",
    "Proposed Top-1",
    "Proposed Top-3"
]

proposed_methods = [
    "Proposed Top-1",
    "Proposed Top-3"
]


# ============================================================
# FINAL VALUES AVAILABLE SO FAR
# ============================================================

# ------------------------------------------------------------
# Quality
# ------------------------------------------------------------

rouge_l = [
    0.343160,
    0.35112789855524895,
    0.34865596370978114
]

bertscore_f1 = [
    0.898992,
    0.9000068306922913,
    0.8998936414718628
]


# ------------------------------------------------------------
# Encoding latency (ms)
# ------------------------------------------------------------

encoding_ms = [
    14.468945,
    12.988583429583482,
    12.988583429583482
]


# ------------------------------------------------------------
# Retrieval latency (ms)
# ------------------------------------------------------------

retrieval_mean_ms = [
    0.583334,
    0.8704347627991367,
    1.8360314300904672
]

retrieval_p95_ms = [
    0.718065,
    1.3214549981057644,
    2.8625650273170318
]


# ------------------------------------------------------------
# Proposed retrieval components (ms)
# ------------------------------------------------------------

routing_ms = [
    0.7021059985050843,
    1.3505704765252413
]

candidate_build_ms = [
    0.013173143712005444,
    0.05972790420942363
]

candidate_cosine_ms = [
    0.15515562058204696,
    0.4257330493558021
]


# ------------------------------------------------------------
# Prompt latency (ms)
#
# Standard did not explicitly save prompt latency.
# Proposed values are directly measured.
# ------------------------------------------------------------

prompt_ms = [
    np.nan,
    0.35165942883828566,
    0.4202475222492857
]


# ------------------------------------------------------------
# LLM latency
# ------------------------------------------------------------

llm_mean_sec = [
    1.632040,
    1.9561626418110072,
    1.8928414461896978
]


# ------------------------------------------------------------
# E2E latency
# ------------------------------------------------------------

e2e_mean_sec = [
    1.647700,
    1.970373319432228,
    1.908086308571621
]

e2e_p95_sec = [
    2.185477,
    2.998899435024941,
    2.7217228798952406
]


# ------------------------------------------------------------
# Candidate-space statistics
# ------------------------------------------------------------

avg_candidate_count = [
    117.80952380952381,
    354.9952380952381
]

candidate_reduction_pct = [
    98.87800453514739,
    96.61909297052155
]

global_top1_preserved_pct = [
    45.33333333333333,
    69.61904761904762
]

avg_top5_overlap_pct = [
    42.19047619047619,
    65.94285714285715
]


# ============================================================
# STANDARD PROMPT / OTHER OVERHEAD
#
# Standard summary did not include prompt_ms.
#
# So:
#
# residual =
# E2E
# - encoding
# - retrieval
# - LLM
#
# This is NOT claimed to be pure prompt construction time.
# ============================================================

standard_residual_ms = (
    e2e_mean_sec[0] * 1000
    - encoding_ms[0]
    - retrieval_mean_ms[0]
    - llm_mean_sec[0] * 1000
)

print(
    f"Standard Prompt/Other residual: "
    f"{standard_residual_ms:.6f} ms"
)


# ============================================================
# E2E COMPONENTS IN MILLISECONDS
# ============================================================

e2e_encoding_ms = np.array([
    encoding_ms[0],
    encoding_ms[1],
    encoding_ms[2]
])

e2e_retrieval_ms = np.array([
    retrieval_mean_ms[0],
    retrieval_mean_ms[1],
    retrieval_mean_ms[2]
])

e2e_prompt_other_ms = np.array([
    standard_residual_ms,
    prompt_ms[1],
    prompt_ms[2]
])

e2e_llm_ms = np.array([
    llm_mean_sec[0] * 1000,
    llm_mean_sec[1] * 1000,
    llm_mean_sec[2] * 1000
])


# ============================================================
# SAVE PLOT
# ============================================================

def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 1. ROUGE-L
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5)
)

bars = ax.bar(
    methods,
    rouge_l
)

ax.set_ylabel(
    "ROUGE-L"
)

ax.set_title(
    "ROUGE-L Comparison"
)

ax.set_ylim(
    0.33,
    0.36
)

ax.bar_label(
    bars,
    labels=[
        f"{v:.4f}"
        for v in rouge_l
    ],
    padding=3
)

save_plot(
    "01_rouge_l_comparison.png"
)


# ============================================================
# 2. BERTScore F1
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5)
)

bars = ax.bar(
    methods,
    bertscore_f1
)

ax.set_ylabel(
    "BERTScore F1"
)

ax.set_title(
    "BERTScore F1 Comparison"
)

ax.set_ylim(
    0.895,
    0.902
)

ax.bar_label(
    bars,
    labels=[
        f"{v:.4f}"
        for v in bertscore_f1
    ],
    padding=3
)

save_plot(
    "02_bertscore_f1_comparison.png"
)


# ============================================================
# 3. MEAN RETRIEVAL LATENCY
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5)
)

bars = ax.bar(
    methods,
    retrieval_mean_ms
)

ax.set_ylabel(
    "Mean retrieval latency (ms)"
)

ax.set_title(
    "Retrieval Latency Comparison"
)

ax.bar_label(
    bars,
    labels=[
        f"{v:.3f}"
        for v in retrieval_mean_ms
    ],
    padding=3
)

save_plot(
    "03_mean_retrieval_latency.png"
)


# ============================================================
# 4. MEAN + P95 RETRIEVAL LATENCY
# ============================================================

x = np.arange(
    len(methods)
)

width = 0.34

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

b1 = ax.bar(
    x - width / 2,
    retrieval_mean_ms,
    width,
    label="Mean"
)

b2 = ax.bar(
    x + width / 2,
    retrieval_p95_ms,
    width,
    label="P95"
)

ax.set_xticks(x)

ax.set_xticklabels(
    methods
)

ax.set_ylabel(
    "Retrieval latency (ms)"
)

ax.set_title(
    "Mean and P95 Retrieval Latency"
)

ax.legend()

for bars in [b1, b2]:

    ax.bar_label(
        bars,
        labels=[
            f"{bar.get_height():.3f}"
            for bar in bars
        ],
        padding=3,
        fontsize=9
    )

save_plot(
    "04_retrieval_mean_p95.png"
)


# ============================================================
# 5. PROPOSED RETRIEVAL BREAKDOWN
# ============================================================

x = np.arange(
    len(proposed_methods)
)

routing_arr = np.array(
    routing_ms
)

build_arr = np.array(
    candidate_build_ms
)

cosine_arr = np.array(
    candidate_cosine_ms
)

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)

ax.bar(
    x,
    routing_arr,
    label="GMM routing"
)

ax.bar(
    x,
    build_arr,
    bottom=routing_arr,
    label="Candidate construction"
)

ax.bar(
    x,
    cosine_arr,
    bottom=(
        routing_arr
        + build_arr
    ),
    label="Candidate cosine + Top-5"
)

ax.set_xticks(x)

ax.set_xticklabels(
    proposed_methods
)

ax.set_ylabel(
    "Mean retrieval latency (ms)"
)

ax.set_title(
    "Proposed Retrieval-Latency Breakdown"
)

ax.legend()

for i, total in enumerate([
    retrieval_mean_ms[1],
    retrieval_mean_ms[2]
]):

    stack_height = (
        routing_arr[i]
        + build_arr[i]
        + cosine_arr[i]
    )

    ax.text(
        i,
        stack_height + 0.04,
        f"{total:.3f} ms",
        ha="center",
        va="bottom",
        fontsize=9
    )

save_plot(
    "05_proposed_retrieval_breakdown.png"
)


# ============================================================
# 6. E2E LATENCY
# ============================================================

x = np.arange(
    len(methods)
)

width = 0.34

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

b1 = ax.bar(
    x - width / 2,
    e2e_mean_sec,
    width,
    label="Mean"
)

b2 = ax.bar(
    x + width / 2,
    e2e_p95_sec,
    width,
    label="P95"
)

ax.set_xticks(x)

ax.set_xticklabels(
    methods
)

ax.set_ylabel(
    "End-to-end latency (s)"
)

ax.set_title(
    "Mean and P95 End-to-End Latency"
)

ax.legend()

for bars in [b1, b2]:

    ax.bar_label(
        bars,
        labels=[
            f"{bar.get_height():.3f}"
            for bar in bars
        ],
        padding=3,
        fontsize=9
    )

save_plot(
    "06_e2e_latency.png"
)


# ============================================================
# 7. E2E COMPONENT BREAKDOWN
#
# Standard:
# Prompt/other = residual
#
# Proposed:
# Prompt = directly measured
# ============================================================

x = np.arange(
    len(methods)
)

fig, ax = plt.subplots(
    figsize=(9, 5.8)
)

ax.bar(
    x,
    e2e_llm_ms,
    label="LLM generation"
)

ax.bar(
    x,
    e2e_encoding_ms,
    bottom=e2e_llm_ms,
    label="Query encoding"
)

bottom_2 = (
    e2e_llm_ms
    + e2e_encoding_ms
)

ax.bar(
    x,
    e2e_retrieval_ms,
    bottom=bottom_2,
    label="Retrieval"
)

bottom_3 = (
    bottom_2
    + e2e_retrieval_ms
)

ax.bar(
    x,
    e2e_prompt_other_ms,
    bottom=bottom_3,
    label="Prompt / other overhead"
)

ax.set_xticks(x)

ax.set_xticklabels(
    methods
)

ax.set_ylabel(
    "Mean latency (ms)"
)

ax.set_title(
    "End-to-End Latency Component Breakdown"
)

ax.legend()

totals_ms = np.array(
    e2e_mean_sec
) * 1000

for i, total in enumerate(
    totals_ms
):

    ax.text(
        i,
        total + 20,
        f"{total / 1000:.3f} s",
        ha="center",
        va="bottom",
        fontsize=9
    )

save_plot(
    "07_e2e_component_breakdown.png"
)


# ============================================================
# 8. CANDIDATE REDUCTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

bars = ax.bar(
    proposed_methods,
    candidate_reduction_pct
)

ax.set_ylabel(
    "Candidate reduction (%)"
)

ax.set_title(
    "Candidate-Space Reduction"
)

ax.set_ylim(
    94,
    100
)

ax.bar_label(
    bars,
    labels=[
        f"{v:.2f}%"
        for v in candidate_reduction_pct
    ],
    padding=3
)

save_plot(
    "08_candidate_reduction.png"
)


# ============================================================
# 9. AVERAGE CANDIDATE COUNT
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

bars = ax.bar(
    proposed_methods,
    avg_candidate_count
)

ax.set_ylabel(
    "Average candidate count"
)

ax.set_title(
    "Average Routed Candidate Pool"
)

ax.bar_label(
    bars,
    labels=[
        f"{v:.2f}"
        for v in avg_candidate_count
    ],
    padding=3
)

save_plot(
    "09_average_candidate_count.png"
)


# ============================================================
# 10. RETRIEVAL PRESERVATION
# ============================================================

x = np.arange(
    len(proposed_methods)
)

width = 0.34

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)

b1 = ax.bar(
    x - width / 2,
    global_top1_preserved_pct,
    width,
    label="Global Top-1 preserved"
)

b2 = ax.bar(
    x + width / 2,
    avg_top5_overlap_pct,
    width,
    label="Average Top-5 overlap"
)

ax.set_xticks(x)

ax.set_xticklabels(
    proposed_methods
)

ax.set_ylabel(
    "Preservation / overlap (%)"
)

ax.set_title(
    "Retrieval Preservation Under GMM Routing"
)

ax.set_ylim(
    0,
    80
)

ax.legend()

for bars in [b1, b2]:

    ax.bar_label(
        bars,
        labels=[
            f"{bar.get_height():.2f}%"
            for bar in bars
        ],
        padding=3,
        fontsize=9
    )

save_plot(
    "10_retrieval_preservation.png"
)


# ============================================================
# 11. ENCODING LATENCY
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5)
)

bars = ax.bar(
    methods,
    encoding_ms
)

ax.set_ylabel(
    "Mean query-encoding latency (ms)"
)

ax.set_title(
    "Query-Encoding Latency"
)

ax.bar_label(
    bars,
    labels=[
        f"{v:.3f}"
        for v in encoding_ms
    ],
    padding=3
)

save_plot(
    "11_encoding_latency.png"
)


# ============================================================
# 12. LLM LATENCY
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5)
)

bars = ax.bar(
    methods,
    llm_mean_sec
)

ax.set_ylabel(
    "Mean LLM latency (s)"
)

ax.set_title(
    "LLM Generation Latency"
)

ax.bar_label(
    bars,
    labels=[
        f"{v:.3f}"
        for v in llm_mean_sec
    ],
    padding=3
)

save_plot(
    "12_llm_latency.png"
)


# ============================================================
# 13. SEARCH-SPACE SIZE
# ============================================================

search_space_count = [
    10500,
    avg_candidate_count[0],
    avg_candidate_count[1]
]

fig, ax = plt.subplots(
    figsize=(7.5, 5)
)

bars = ax.bar(
    methods,
    search_space_count
)

ax.set_ylabel(
    "Average records searched"
)

ax.set_title(
    "Retrieval Search-Space Comparison"
)

ax.bar_label(
    bars,
    labels=[
        f"{v:,.0f}"
        for v in search_space_count
    ],
    padding=3
)

save_plot(
    "13_search_space_comparison.png"
)


# ============================================================
# FINAL CONSOLIDATED TABLE
#
# Judge columns intentionally excluded for now.
# ============================================================

final_table = pd.DataFrame({

    "Method":
        methods,

    "ROUGE-L":
        rouge_l,

    "BERTScore F1":
        bertscore_f1,

    "Encoding Mean (ms)":
        encoding_ms,

    "Retrieval Mean (ms)":
        retrieval_mean_ms,

    "Retrieval P95 (ms)":
        retrieval_p95_ms,

    "Prompt / Other Mean (ms)": [
        standard_residual_ms,
        prompt_ms[1],
        prompt_ms[2]
    ],

    "LLM Mean (s)":
        llm_mean_sec,

    "E2E Mean (s)":
        e2e_mean_sec,

    "E2E P95 (s)":
        e2e_p95_sec,

    "Avg Candidate Count": [
        np.nan,
        avg_candidate_count[0],
        avg_candidate_count[1]
    ],

    "Candidate Reduction (%)": [
        np.nan,
        candidate_reduction_pct[0],
        candidate_reduction_pct[1]
    ],

    "Global Top-1 Preserved (%)": [
        np.nan,
        global_top1_preserved_pct[0],
        global_top1_preserved_pct[1]
    ],

    "Average Top-5 Overlap (%)": [
        np.nan,
        avg_top5_overlap_pct[0],
        avg_top5_overlap_pct[1]
    ]
})


# ============================================================
# SAVE TABLES
# ============================================================

final_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_synthetic_results_table_current.csv"
    ),
    index=False
)

final_table.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "final_synthetic_results_table_current.xlsx"
    ),
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "CURRENT SYNTHETIC PLOTS AND TABLES COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nStandard prompt latency was not separately saved."
)

print(
    f"Standard Prompt/Other residual = "
    f"{standard_residual_ms:.6f} ms"
)

print(
    "\nLLM-judge plots are intentionally excluded "
    "until the new argpartition outputs are judged."
)

print(
    "\nCurrent consolidated table:"
)

print(
    final_table.to_string(
        index=False
    )
)