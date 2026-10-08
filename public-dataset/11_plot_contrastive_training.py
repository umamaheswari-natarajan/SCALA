import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "contrastive_training",
    "training_history.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("CONTRASTIVE TRAINING HISTORY")
print("=" * 80)

print(df)


# ============================================================
# BEST EPOCH
# ============================================================

best_idx = df["mrr"].idxmax()
best_epoch = int(df.loc[best_idx, "epoch"])
best_mrr = float(df.loc[best_idx, "mrr"])

print("\nBest epoch:", best_epoch)
print("Best validation MRR:", best_mrr)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 4.8)
)


# ============================================================
# PANEL A — TRAIN / VALIDATION LOSS
# ============================================================

ax = axes[0]

ax.plot(
    df["epoch"],
    df["train_loss"],
    marker="o",
    label="Training Loss"
)

ax.plot(
    df["epoch"],
    df["val_loss"],
    marker="o",
    label="Validation Loss"
)

ax.axvline(
    best_epoch,
    linestyle="--",
    alpha=0.7,
    label=f"Selected Epoch {best_epoch}"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Contrastive Loss")
ax.set_title("(a) Training and Validation Loss")
ax.grid(alpha=0.25)
ax.legend()


# ============================================================
# PANEL B — RETRIEVAL METRICS
# ============================================================

ax = axes[1]

ax.plot(
    df["epoch"],
    df["recall_at_1"],
    marker="o",
    label="Recall@1"
)

ax.plot(
    df["epoch"],
    df["recall_at_5"],
    marker="o",
    label="Recall@5"
)

ax.plot(
    df["epoch"],
    df["recall_at_10"],
    marker="o",
    label="Recall@10"
)

ax.plot(
    df["epoch"],
    df["mrr"],
    marker="o",
    linewidth=2,
    label="MRR"
)

ax.axvline(
    best_epoch,
    linestyle="--",
    alpha=0.7
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("(b) Validation Retrieval Performance")
ax.grid(alpha=0.25)
ax.legend()


# ============================================================
# PANEL C — POSITIVE / NEGATIVE COSINE
# ============================================================

ax = axes[2]

ax.plot(
    df["epoch"],
    df["positive_cosine"],
    marker="o",
    label="Positive Pair Cosine"
)

ax.plot(
    df["epoch"],
    df["negative_cosine"],
    marker="o",
    label="Negative Pair Cosine"
)

ax.axvline(
    best_epoch,
    linestyle="--",
    alpha=0.7
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Cosine Similarity")
ax.set_title("(c) Contrastive Pair Separation")
ax.grid(alpha=0.25)
ax.legend()


# ============================================================
# FINAL LAYOUT
# ============================================================

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

PNG_FILE = os.path.join(
    OUTPUT_DIR,
    "fig_contrastive_training.png"
)

PDF_FILE = os.path.join(
    OUTPUT_DIR,
    "fig_contrastive_training.pdf"
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

plt.close(fig)


# ============================================================
# PRINT BEST EPOCH DETAILS
# ============================================================

print("\n" + "=" * 80)
print("BEST EPOCH DETAILS")
print("=" * 80)

row = df.loc[best_idx]

print(f"Epoch              : {best_epoch}")
print(f"Train loss         : {row['train_loss']:.4f}")
print(f"Validation loss    : {row['val_loss']:.4f}")
print(f"Recall@1           : {row['recall_at_1']:.4f}")
print(f"Recall@5           : {row['recall_at_5']:.4f}")
print(f"Recall@10          : {row['recall_at_10']:.4f}")
print(f"MRR                : {row['mrr']:.4f}")
print(f"Positive cosine    : {row['positive_cosine']:.4f}")
print(f"Negative cosine    : {row['negative_cosine']:.4f}")

print("\nSaved:")
print(PNG_FILE)
print(PDF_FILE)