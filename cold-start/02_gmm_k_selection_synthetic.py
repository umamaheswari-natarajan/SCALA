import os
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_DIR = "synthetic_contrastive_model"

CHECKPOINT_FILE = os.path.join(
    MODEL_DIR,
    "best_contrastive_model.pt"
)

TRAIN_FILE = os.path.join(
    MODEL_DIR,
    "train_8400.csv"
)

VAL_FILE = os.path.join(
    MODEL_DIR,
    "validation_2100.csv"
)

OUTPUT_DIR = "synthetic_gmm_selection_fine"

FAULT_COL = "Fault Description"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECTION_DIM = 256
MAX_LENGTH = 128

BATCH_SIZE = 128

K_VALUES = list(range(90, 151, 5))

SEED = 42

# Same GMM configuration should later be reused
# for the final 10,500-record KB.
#
# diag is much more practical in a 256-dimensional
# representation than estimating a separate full 256x256
# covariance matrix for every cluster.
COVARIANCE_TYPE = "diag"

GMM_MAX_ITER = 300
GMM_N_INIT = 1
REG_COVAR = 1e-6

# Full cosine silhouette over 8,400 records.
# If this is too slow on your machine, change this to 3000.
SILHOUETTE_SAMPLE_SIZE = None

NUM_WORKERS = 0


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("STAGE 2: GMM K SELECTION")
print("=" * 80)

print("Device:", device)
print("K values:", K_VALUES)
print("GMM covariance:", COVARIANCE_TYPE)


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. LOAD TRAIN / VALIDATION SPLIT
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

val_df = pd.read_csv(
    VAL_FILE
)

print("\nDataset sizes")
print("Train      :", len(train_df))
print("Validation :", len(val_df))

assert len(train_df) == 8400
assert len(val_df) == 2100


# ============================================================
# 5. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 6. DATASET
# ============================================================

class FaultDataset(Dataset):

    def __init__(self, texts):
        self.texts = list(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def collate_faults(batch):

    return tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


train_loader = DataLoader(
    FaultDataset(
        train_df[FAULT_COL].astype(str)
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_faults
)

val_loader = DataLoader(
    FaultDataset(
        val_df[FAULT_COL].astype(str)
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_faults
)


# ============================================================
# 7. MEAN POOLING
# ============================================================

def mean_pooling(
    token_embeddings,
    attention_mask
):

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    summed = torch.sum(
        token_embeddings * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


# ============================================================
# 8. PROJECTION HEAD
#    Must match Stage 1 exactly
# ============================================================

class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim=384,
        output_dim=256
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                input_dim
            ),
            nn.GELU(),
            nn.Linear(
                input_dim,
                output_dim
            )
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 9. MODEL
#    Must match Stage 1 exactly
# ============================================================

class FaultResolutionCLIP(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.fault_projection = ProjectionHead(
            input_dim=384,
            output_dim=PROJECTION_DIM
        )

        self.resolution_projection = ProjectionHead(
            input_dim=384,
            output_dim=PROJECTION_DIM
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(
                np.log(1 / 0.07),
                dtype=torch.float32
            )
        )


    def encode_backbone(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )


    def encode_fault(
        self,
        input_ids,
        attention_mask,
        normalize=False
    ):

        x = self.encode_backbone(
            input_ids,
            attention_mask
        )

        z = self.fault_projection(x)

        if normalize:
            z = F.normalize(
                z,
                p=2,
                dim=1
            )

        return z


# ============================================================
# 10. LOAD BEST CHECKPOINT
# ============================================================

print("\nLoading best contrastive checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device
)

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

if "validation_metrics" in checkpoint:

    print(
        "Checkpoint Recall@1:",
        checkpoint[
            "validation_metrics"
        ]["recall_at_1"]
    )

    print(
        "Checkpoint MRR:",
        checkpoint[
            "validation_metrics"
        ]["mrr"]
    )


model = FaultResolutionCLIP().to(
    device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# 11. EMBEDDING EXTRACTION
# ============================================================

def extract_fault_embeddings(
    loader,
    name
):

    embeddings = []

    start = time.perf_counter()

    with torch.no_grad():

        for batch_idx, tokens in enumerate(
            loader,
            start=1
        ):

            tokens = {
                k: v.to(device)
                for k, v in tokens.items()
            }

            # IMPORTANT:
            # normalize=False
            #
            # These RAW projected vectors are
            # what we use for GMM fitting.
            z = model.encode_fault(
                tokens["input_ids"],
                tokens["attention_mask"],
                normalize=False
            )

            embeddings.append(
                z.cpu().numpy()
            )

            if batch_idx % 20 == 0:

                print(
                    f"{name}: "
                    f"{batch_idx}/{len(loader)} batches"
                )


    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    print(
        f"{name} embeddings: "
        f"{embeddings.shape}"
    )

    print(
        f"{name} encoding time: "
        f"{elapsed:.3f} sec"
    )

    return (
        embeddings.astype(np.float32),
        elapsed
    )


print("\n" + "=" * 80)
print("EXTRACTING FAULT EMBEDDINGS")
print("=" * 80)


train_raw, train_encoding_time = (
    extract_fault_embeddings(
        train_loader,
        "Train"
    )
)

val_raw, val_encoding_time = (
    extract_fault_embeddings(
        val_loader,
        "Validation"
    )
)


# ============================================================
# 12. NORMALIZED COPIES
#
# Raw:
#     GMM
#
# Normalized:
#     cosine diagnostics / later retrieval
# ============================================================

train_norm = train_raw / np.clip(
    np.linalg.norm(
        train_raw,
        axis=1,
        keepdims=True
    ),
    1e-12,
    None
)

val_norm = val_raw / np.clip(
    np.linalg.norm(
        val_raw,
        axis=1,
        keepdims=True
    ),
    1e-12,
    None
)


# ============================================================
# 13. SAVE EMBEDDINGS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_fault_embeddings_raw.npy"
    ),
    train_raw
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_fault_embeddings_normalized.npy"
    ),
    train_norm
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "validation_fault_embeddings_raw.npy"
    ),
    val_raw
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "validation_fault_embeddings_normalized.npy"
    ),
    val_norm
)


# ============================================================
# 14. INTRA-CLUSTER COSINE SIMILARITY
#
# Computes average pairwise cosine similarity within clusters.
#
# This version avoids explicitly constructing every pairwise
# similarity matrix.
# ============================================================

def intra_cluster_cosine(
    normalized_embeddings,
    labels
):

    weighted_similarity_sum = 0.0
    total_pairs = 0


    for cluster_id in np.unique(labels):

        cluster_vectors = (
            normalized_embeddings[
                labels == cluster_id
            ]
        )

        n = len(cluster_vectors)

        if n < 2:
            continue


        # For unit vectors:
        #
        # sum_{i != j} xi.xj
        # =
        # ||sum_i xi||^2 - n
        #
        vector_sum = (
            cluster_vectors.sum(axis=0)
        )

        ordered_pair_sum = (
            np.dot(
                vector_sum,
                vector_sum
            )
            -
            n
        )

        # Unique unordered pairs
        pair_sum = (
            ordered_pair_sum / 2.0
        )

        n_pairs = (
            n * (n - 1) // 2
        )

        weighted_similarity_sum += (
            pair_sum
        )

        total_pairs += (
            n_pairs
        )


    if total_pairs == 0:
        return np.nan


    return float(
        weighted_similarity_sum
        /
        total_pairs
    )


# ============================================================
# 15. GMM K SWEEP
# ============================================================

results = []

print("\n" + "=" * 80)
print("GMM MODEL SELECTION")
print("=" * 80)


for K in K_VALUES:

    print("\n" + "-" * 80)
    print(f"K = {K}")
    print("-" * 80)


    # --------------------------------------------------------
    # Fit GMM using RAW projected embeddings
    # --------------------------------------------------------

    gmm = GaussianMixture(
        n_components=K,
        covariance_type=COVARIANCE_TYPE,
        max_iter=GMM_MAX_ITER,
        n_init=GMM_N_INIT,
        reg_covar=REG_COVAR,
        random_state=SEED
    )


    fit_start = time.perf_counter()

    gmm.fit(
        train_raw
    )

    fit_time = (
        time.perf_counter()
        -
        fit_start
    )


    # --------------------------------------------------------
    # Train labels
    # --------------------------------------------------------

    train_labels = gmm.predict(
        train_raw
    )


    # --------------------------------------------------------
    # BIC
    #
    # Lower = better
    # --------------------------------------------------------

    bic = gmm.bic(
        train_raw
    )


    # --------------------------------------------------------
    # Validation average log likelihood
    #
    # Higher = better
    # --------------------------------------------------------

    val_avg_log_likelihood = (
        gmm.score(
            val_raw
        )
    )


    # --------------------------------------------------------
    # Cosine silhouette
    #
    # Higher = better
    # --------------------------------------------------------

    unique_clusters = np.unique(
        train_labels
    )

    if len(unique_clusters) > 1:

        silhouette_start = (
            time.perf_counter()
        )

        silhouette = silhouette_score(
            train_norm,
            train_labels,
            metric="cosine",
            sample_size=SILHOUETTE_SAMPLE_SIZE,
            random_state=SEED
        )

        silhouette_time = (
            time.perf_counter()
            -
            silhouette_start
        )

    else:

        silhouette = np.nan
        silhouette_time = 0.0


    # --------------------------------------------------------
    # Intra-cluster cosine
    #
    # Higher = tighter clusters
    # --------------------------------------------------------

    intra_cosine = intra_cluster_cosine(
        train_norm,
        train_labels
    )


    # --------------------------------------------------------
    # Cluster size statistics
    # --------------------------------------------------------

    cluster_sizes = np.bincount(
        train_labels,
        minlength=K
    )

    min_cluster_size = int(
        cluster_sizes.min()
    )

    max_cluster_size = int(
        cluster_sizes.max()
    )

    mean_cluster_size = float(
        cluster_sizes.mean()
    )

    median_cluster_size = float(
        np.median(
            cluster_sizes
        )
    )

    empty_clusters = int(
        np.sum(
            cluster_sizes == 0
        )
    )


    print(
        f"BIC                         : "
        f"{bic:.4f}"
    )

    print(
        f"Validation avg log likelihood: "
        f"{val_avg_log_likelihood:.6f}"
    )

    print(
        f"Cosine silhouette           : "
        f"{silhouette:.6f}"
    )

    print(
        f"Intra-cluster cosine        : "
        f"{intra_cosine:.6f}"
    )

    print(
        f"GMM fit time                : "
        f"{fit_time:.3f} sec"
    )

    print(
        f"Silhouette computation time : "
        f"{silhouette_time:.3f} sec"
    )

    print(
        f"Cluster size min/mean/max   : "
        f"{min_cluster_size}/"
        f"{mean_cluster_size:.2f}/"
        f"{max_cluster_size}"
    )


    results.append({

        "K":
            K,

        "bic":
            bic,

        "validation_avg_log_likelihood":
            val_avg_log_likelihood,

        "silhouette_cosine":
            silhouette,

        "intra_cluster_cosine":
            intra_cosine,

        "gmm_fit_time_sec":
            fit_time,

        "silhouette_time_sec":
            silhouette_time,

        "converged":
            bool(
                gmm.converged_
            ),

        "n_iter":
            int(
                gmm.n_iter_
            ),

        "min_cluster_size":
            min_cluster_size,

        "mean_cluster_size":
            mean_cluster_size,

        "median_cluster_size":
            median_cluster_size,

        "max_cluster_size":
            max_cluster_size,

        "empty_clusters":
            empty_clusters
    })


    # --------------------------------------------------------
    # Save each candidate GMM
    #
    # Useful so we don't have to refit it immediately
    # when inspecting the results.
    # --------------------------------------------------------

    import joblib

    joblib.dump(
        gmm,
        os.path.join(
            OUTPUT_DIR,
            f"gmm_K{K}.joblib"
        )
    )


    # Save results after every K so progress is not lost
    pd.DataFrame(
        results
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "gmm_k_sweep_results.csv"
        ),
        index=False
    )


# ============================================================
# 16. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

results_file = os.path.join(
    OUTPUT_DIR,
    "gmm_k_sweep_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# 17. IDENTIFY INDIVIDUAL BEST VALUES
#
# We print these, but do NOT automatically finalize K merely
# because one diagnostic is best.
# ============================================================

best_bic_row = results_df.loc[
    results_df["bic"].idxmin()
]

best_ll_row = results_df.loc[
    results_df[
        "validation_avg_log_likelihood"
    ].idxmax()
]

best_sil_row = results_df.loc[
    results_df[
        "silhouette_cosine"
    ].idxmax()
]

best_intra_row = results_df.loc[
    results_df[
        "intra_cluster_cosine"
    ].idxmax()
]


# ============================================================
# 18. PLOT 1: BIC
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    results_df["K"],
    results_df["bic"],
    marker="o"
)

plt.xlabel(
    "Number of GMM clusters (K)"
)

plt.ylabel(
    "BIC"
)

plt.title(
    "GMM Model Selection: BIC"
)

plt.xticks(
    K_VALUES
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "gmm_bic_vs_k.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 19. PLOT 2: VALIDATION LOG LIKELIHOOD
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    results_df["K"],
    results_df[
        "validation_avg_log_likelihood"
    ],
    marker="o"
)

plt.xlabel(
    "Number of GMM clusters (K)"
)

plt.ylabel(
    "Average validation log likelihood"
)

plt.title(
    "GMM Validation Log Likelihood"
)

plt.xticks(
    K_VALUES
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "gmm_validation_loglikelihood_vs_k.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 20. PLOT 3: COSINE CLUSTER QUALITY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    results_df["K"],
    results_df["silhouette_cosine"],
    marker="o",
    label="Cosine silhouette"
)

plt.plot(
    results_df["K"],
    results_df["intra_cluster_cosine"],
    marker="o",
    label="Intra-cluster cosine"
)

plt.xlabel(
    "Number of GMM clusters (K)"
)

plt.ylabel(
    "Cosine score"
)

plt.title(
    "GMM Cluster Quality"
)

plt.xticks(
    K_VALUES
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "gmm_cluster_quality_vs_k.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 21. SAVE EMBEDDING / TIMING INFORMATION
# ============================================================

timing_df = pd.DataFrame(
    [
        {
            "dataset":
                "train_8400",

            "records":
                len(train_df),

            "embedding_time_sec":
                train_encoding_time
        },

        {
            "dataset":
                "validation_2100",

            "records":
                len(val_df),

            "embedding_time_sec":
                val_encoding_time
        }
    ]
)

timing_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "embedding_times.csv"
    ),
    index=False
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("GMM K SWEEP COMPLETE")
print("=" * 80)

print("\nFull results:\n")

display_cols = [
    "K",
    "bic",
    "validation_avg_log_likelihood",
    "silhouette_cosine",
    "intra_cluster_cosine",
    "gmm_fit_time_sec"
]

print(
    results_df[
        display_cols
    ].to_string(
        index=False
    )
)


print("\n" + "-" * 80)
print("INDIVIDUAL METRIC OPTIMA")
print("-" * 80)

print(
    f"Lowest BIC:"
    f" K={int(best_bic_row['K'])}"
    f" | BIC={best_bic_row['bic']:.4f}"
)

print(
    f"Highest validation log likelihood:"
    f" K={int(best_ll_row['K'])}"
    f" | score="
    f"{best_ll_row['validation_avg_log_likelihood']:.6f}"
)

print(
    f"Highest cosine silhouette:"
    f" K={int(best_sil_row['K'])}"
    f" | score="
    f"{best_sil_row['silhouette_cosine']:.6f}"
)

print(
    f"Highest intra-cluster cosine:"
    f" K={int(best_intra_row['K'])}"
    f" | score="
    f"{best_intra_row['intra_cluster_cosine']:.6f}"
)


print("\nFiles saved in:")
print(OUTPUT_DIR)

print("\nMain CSV:")
print(results_file)

print("\nIMPORTANT:")
print(
    "Do not build the final 10,500-record GMM yet."
)

print(
    "First inspect this K sweep, select K, "
    "then evaluate Top-1 / Top-3 / Top-5 routing "
    "on the 2,100 validation records."
)