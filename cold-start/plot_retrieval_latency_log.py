import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = (
    r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC"
    r"\SCALA\synthetic-dataset"
)

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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

standard = pd.read_csv(STANDARD_FILE)
pca = pd.read_csv(PCA_GMM_FILE)
ifkg = pd.read_csv(IFKG_FILE)


# Standard RAG: successful rows
standard = standard[
    standard["status"] == "success"
].copy()


# ============================================================
# METHODS
# ============================================================

METHODS = [
    "Standard RAG",
    "SCALA",
    "IFKG"
]


# ============================================================
# RETRIEVAL LATENCY
# Convert seconds -> milliseconds
# ============================================================

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


# ============================================================
# PRINT VALUES
# ============================================================

print("=" * 70)
print("RETRIEVAL LATENCY")
print("=" * 70)

for i, method in enumerate(METHODS):

    print(
        f"{method:25s} "
        f"Mean = {retrieval_mean_ms[i]:.3f} ms | "
        f"P95 = {retrieval_p95_ms[i]:.3f} ms"
    )


# ============================================================
# SPEEDUP OF PROPOSED METHOD
# ============================================================

standard_speedup = (
    retrieval_mean_ms[0]
    /
    retrieval_mean_ms[1]
)

ifkg_speedup = (
    retrieval_mean_ms[2]
    /
    retrieval_mean_ms[1]
)

standard_reduction = (
    (
        retrieval_mean_ms[0]
        -
        retrieval_mean_ms[1]
    )
    /
    retrieval_mean_ms[0]
    *
    100
)


print("\nPROPOSED METHOD IMPROVEMENT")

print(
    f"Speedup over Standard RAG : "
    f"{standard_speedup:.2f}x"
)

print(
    f"Latency reduction vs Standard RAG : "
    f"{standard_reduction:.2f}%"
)

print(
    f"Speedup over IFKG          : "
    f"{ifkg_speedup:.1f}x"
)


# ============================================================
# PLOT SETTINGS
# Designed for TWO side-by-side subfigures
# ============================================================

LABEL_SIZE = 16
TICK_SIZE = 14
VALUE_SIZE = 12
LEGEND_SIZE = 13

x = np.arange(len(METHODS))
width = 0.34


# ============================================================
# RETRIEVAL LATENCY PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.2, 4.8)
)

bars_mean = ax.bar(
    x - width / 2,
    retrieval_mean_ms,
    width,
    label="Mean"
)

bars_p95 = ax.bar(
    x + width / 2,
    retrieval_p95_ms,
    width,
    label="P95"
)


# ============================================================
# LOG SCALE
# ============================================================

ax.set_yscale("log")

# Explicit upper limit provides sufficient room
# for the IFKG value annotations.
ax.set_ylim(
    0.3,
    7000
)


# ============================================================
# VALUE LABELS
# ============================================================

for bars in [bars_mean, bars_p95]:

    for bar in bars:

        value = bar.get_height()

        if value < 10:
            label = f"{value:.3f}"
        else:
            label = f"{value:.1f}"

        ax.annotate(
            label,
            xy=(
                bar.get_x() + bar.get_width() / 2,
                value
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            fontweight="bold",
            clip_on=False
        )


# ============================================================
# AXES
# ============================================================

ax.set_ylabel(
    "Retrieval Latency (ms, log scale)",
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
    which="both",
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
    "retrieval_latency_log_scale.png"
)

PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "retrieval_latency_log_scale.pdf"
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

print("\nSaved:")
print(PNG_FILE)
print(PDF_FILE)