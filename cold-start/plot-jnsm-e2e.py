import os
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

SCALA_FILE = os.path.join(
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
# LOAD DATA
# ============================================================

standard = pd.read_csv(STANDARD_FILE)
scala = pd.read_csv(SCALA_FILE)
ifkg = pd.read_csv(IFKG_FILE)

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
# END-TO-END LATENCY
# ============================================================

e2e_mean = [
    standard["e2e_latency_sec"].mean(),
    scala["e2e_sec"].mean(),
    ifkg["e2e_sec"].mean()
]

e2e_p95 = [
    standard["e2e_latency_sec"].quantile(0.95),
    scala["e2e_sec"].quantile(0.95),
    ifkg["e2e_sec"].quantile(0.95)
]


# ============================================================
# PRINT VALUES
# ============================================================

print("=" * 70)
print("END-TO-END LATENCY")
print("=" * 70)

for i, method in enumerate(METHODS):

    print(
        f"{method:15s} "
        f"Mean = {e2e_mean[i]:.3f} s | "
        f"P95 = {e2e_p95[i]:.3f} s"
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
# END-TO-END LATENCY PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.2, 4.8)
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


# ============================================================
# VALUE LABELS
# ============================================================

for bars in [bars_mean, bars_p95]:

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
    "End-to-End Latency (s)",
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

# Leave sufficient space above the P95 labels
ax.set_ylim(
    0,
    max(e2e_p95) * 1.15
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
    "e2e_latency.png"
)

PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "e2e_latency.pdf"
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