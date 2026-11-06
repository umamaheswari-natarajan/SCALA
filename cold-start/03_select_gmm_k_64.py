from pathlib import Path
import os
import time
import random
import joblib

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

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = os.path.join(
    BASE_DIR,
    "real64_contrastive_model"
)

CHECKPOINT_FILE = os.path.join(
    MODEL_DIR,
    "best_contrastive_model.pt"
)

TRAIN_FILE = os.path.join(
    MODEL_DIR,
    "train_64_split.csv"
)

VAL_FILE = os.path.join(
    MODEL_DIR,
    "validation_64_split.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "real64_gmm_selection"
)

FAULT_COL = "Fault Description"

# Candidate GMM component counts for the small 64-record setting.
K_VALUES = [2, 3, 4, 5, 6, 8, 10]

SEED = 42

# Keep the same GMM configuration used in the
# larger SCALA experiment.
COVARIANCE_TYPE = "diag"
GMM_MAX_ITER = 300
GMM_N_INIT = 1
REG_COVAR = 1e-6

BATCH_SIZE = 32
NUM_WORKERS = 0

EXPECTED_TRAIN_SIZE = 51
EXPECTED_VAL_SIZE = 13
EXPECTED_PROJECTED_DIM = 384


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
print("SCALA-64: GMM K SELECTION")
print("=" * 80)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

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
# 4. CHECK INPUT FILES
# ============================================================

for path in [
    CHECKPOINT_FILE,
    TRAIN_FILE,
    VAL_FILE
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


# ============================================================
# 5. LOAD TRAIN / VALIDATION SPLIT
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

if FAULT_COL not in train_df.columns:
    raise ValueError(
        f"Missing column '{FAULT_COL}' in training file."
    )

if FAULT_COL not in val_df.columns:
    raise ValueError(
        f"Missing column '{FAULT_COL}' in validation file."
    )

print("\nDataset sizes")
print("Train      :", len(train_df))
print("Validation :", len(val_df))

if len(train_df) != EXPECTED_TRAIN_SIZE:
    raise ValueError(
        f"Expected {EXPECTED_TRAIN_SIZE} training records, "
        f"found {len(train_df)}."
    )

if len(val_df) != EXPECTED_VAL_SIZE:
    raise ValueError(
        f"Expected {EXPECTED_VAL_SIZE} validation records, "
        f"found {len(val_df)}."
    )


# ============================================================
# 6. LOAD CHECKPOINT METADATA
# ============================================================

print("\nLoading best contrastive checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device
)

required_checkpoint_keys = [
    "epoch",
    "model_state_dict",
    "projection_dim",
    "model_name",
    "max_length"
]

for key in required_checkpoint_keys:
    if key not in checkpoint:
        raise KeyError(
            f"Checkpoint missing required key: {key}"
        )

MODEL_NAME = checkpoint["model_name"]
PROJECTION_DIM = int(
    checkpoint["projection_dim"]
)
MAX_LENGTH = int(
    checkpoint["max_length"]
)

print("Checkpoint epoch      :", checkpoint["epoch"])
print("Model                 :", MODEL_NAME)
print("Projection dimension  :", PROJECTION_DIM)
print("Maximum sequence len. :", MAX_LENGTH)

if PROJECTION_DIM != EXPECTED_PROJECTED_DIM:
    raise ValueError(
        f"Expected {EXPECTED_PROJECTED_DIM}-D checkpoint, "
        f"found {PROJECTION_DIM}-D."
    )

if "validation_metrics" in checkpoint:
    metrics = checkpoint["validation_metrics"]

    if "recall_at_1" in metrics:
        print(
            "Checkpoint Recall@1  :",
            metrics["recall_at_1"]
        )

    if "mrr" in metrics:
        print(
            "Checkpoint MRR       :",
            metrics["mrr"]
        )


# ============================================================
# 7. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 8. DATASET
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
# 9. MEAN POOLING
# ============================================================

def mean_pooling(
    model_output,
    attention_mask
):

    token_embeddings = (
        model_output.last_hidden_state
    )

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
# 10. PROJECTION HEAD
#     Must match contrastive training exactly
# ============================================================

class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim
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
# 11. CONTRASTIVE MODEL
#     Must match contrastive training exactly
# ============================================================

class FaultResolutionCLIP(nn.Module):

    def __init__(
        self,
        model_name,
        projection_dim
    ):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_dim = (
            self.encoder.config.hidden_size
        )

        self.fault_projection = ProjectionHead(
            hidden_dim,
            projection_dim
        )

        self.resolution_projection = ProjectionHead(
            hidden_dim,
            projection_dim
        )

        # Included because checkpoint contains this parameter.
        self.logit_scale = nn.Parameter(
            torch.tensor(0.0)
        )


# ============================================================
# 12. LOAD BEST CHECKPOINT
# ============================================================

model = FaultResolutionCLIP(
    MODEL_NAME,
    PROJECTION_DIM
)

load_result = model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

print("\nCheckpoint loaded with strict=True.")
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)

model = model.to(device)
model.eval()

BACKBONE_DIM = int(
    model.encoder.config.hidden_size
)

print("Backbone dimension:", BACKBONE_DIM)


# ============================================================
# 13. EXTRACT RAW FAULT EMBEDDINGS
#
# GMM uses RAW projected embeddings.
# ============================================================

def extract_fault_embeddings(
    loader,
    name
):

    embeddings = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for tokens in loader:

            tokens = {
                key: value.to(device)
                for key, value in tokens.items()
            }

            output = model.encoder(
                **tokens
            )

            pooled = mean_pooling(
                output,
                tokens["attention_mask"]
            )

            raw = model.fault_projection(
                pooled
            )

            embeddings.append(
                raw.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - start
    )

    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    print(
        f"{name} embeddings:",
        embeddings.shape
    )

    print(
        f"{name} encoding time:",
        f"{elapsed:.4f} sec"
    )

    return embeddings, elapsed


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
# 14. VALIDATE EMBEDDINGS
# ============================================================

expected_train_shape = (
    EXPECTED_TRAIN_SIZE,
    PROJECTION_DIM
)

expected_val_shape = (
    EXPECTED_VAL_SIZE,
    PROJECTION_DIM
)

if train_raw.shape != expected_train_shape:
    raise ValueError(
        f"Unexpected training embedding shape: "
        f"{train_raw.shape}"
    )

if val_raw.shape != expected_val_shape:
    raise ValueError(
        f"Unexpected validation embedding shape: "
        f"{val_raw.shape}"
    )

if not np.isfinite(train_raw).all():
    raise ValueError(
        "Non-finite values in training embeddings."
    )

if not np.isfinite(val_raw).all():
    raise ValueError(
        "Non-finite values in validation embeddings."
    )


# ============================================================
# 15. NORMALIZED COPIES
#
# Raw:
#     GMM fitting and validation likelihood
#
# Normalized:
#     cosine-based cluster diagnostics
# ============================================================

train_norm = (
    train_raw
    /
    np.clip(
        np.linalg.norm(
            train_raw,
            axis=1,
            keepdims=True
        ),
        1e-12,
        None
    )
)

val_norm = (
    val_raw
    /
    np.clip(
        np.linalg.norm(
            val_raw,
            axis=1,
            keepdims=True
        ),
        1e-12,
        None
    )
)


# ============================================================
# 16. SAVE SPLIT EMBEDDINGS
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
# 17. INTRA-CLUSTER COSINE SIMILARITY
#
# Average pairwise cosine similarity among records belonging
# to the same hard-assigned GMM component.
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

        vector_sum = (
            cluster_vectors.sum(axis=0)
        )

        ordered_pair_sum = (
            np.dot(
                vector_sum,
                vector_sum
            )
            - n
        )

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
# 18. GMM K SWEEP
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
    # Fit GMM on RAW projected embeddings
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
    # Hard train assignments
    # --------------------------------------------------------

    train_labels = gmm.predict(
        train_raw
    )

    unique_clusters = np.unique(
        train_labels
    )

    # --------------------------------------------------------
    # BIC
    # Lower = better
    # --------------------------------------------------------

    bic = gmm.bic(
        train_raw
    )

    # --------------------------------------------------------
    # AIC
    # Lower = better
    # --------------------------------------------------------

    aic = gmm.aic(
        train_raw
    )

    # --------------------------------------------------------
    # Validation average log likelihood
    # Higher = better
    # --------------------------------------------------------

    val_avg_log_likelihood = (
        gmm.score(
            val_raw
        )
    )

    # --------------------------------------------------------
    # Cosine silhouette
    # Higher = better
    # --------------------------------------------------------

    if (
        len(unique_clusters) > 1
        and
        len(unique_clusters) < len(train_raw)
    ):

        silhouette = silhouette_score(
            train_norm,
            train_labels,
            metric="cosine"
        )

    else:
        silhouette = np.nan

    # --------------------------------------------------------
    # Intra-cluster cosine
    # Higher = tighter clusters
    # --------------------------------------------------------

    intra_cosine = intra_cluster_cosine(
        train_norm,
        train_labels
    )

    # --------------------------------------------------------
    # Cluster-size statistics
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

    singleton_clusters = int(
        np.sum(
            cluster_sizes == 1
        )
    )

    clusters_lt_2 = int(
        np.sum(
            cluster_sizes < 2
        )
    )

    print(
        f"BIC                         : "
        f"{bic:.4f}"
    )

    print(
        f"AIC                         : "
        f"{aic:.4f}"
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
        f"GMM converged               : "
        f"{gmm.converged_}"
    )

    print(
        f"GMM iterations              : "
        f"{gmm.n_iter_}"
    )

    print(
        f"GMM fit time                : "
        f"{fit_time:.4f} sec"
    )

    print(
        f"Cluster size min/mean/max   : "
        f"{min_cluster_size}/"
        f"{mean_cluster_size:.2f}/"
        f"{max_cluster_size}"
    )

    print(
        f"Cluster size median         : "
        f"{median_cluster_size:.2f}"
    )

    print(
        f"Empty clusters              : "
        f"{empty_clusters}"
    )

    print(
        f"Singleton clusters          : "
        f"{singleton_clusters}"
    )

    print(
        f"Clusters with <2 records    : "
        f"{clusters_lt_2}"
    )

    results.append(
        {
            "K": K,
            "bic": float(bic),
            "aic": float(aic),
            "validation_avg_log_likelihood":
                float(val_avg_log_likelihood),
            "silhouette_cosine":
                float(silhouette),
            "intra_cluster_cosine":
                float(intra_cosine),
            "gmm_fit_time_sec":
                float(fit_time),
            "converged":
                bool(gmm.converged_),
            "n_iter":
                int(gmm.n_iter_),
            "min_cluster_size":
                min_cluster_size,
            "mean_cluster_size":
                mean_cluster_size,
            "median_cluster_size":
                median_cluster_size,
            "max_cluster_size":
                max_cluster_size,
            "empty_clusters":
                empty_clusters,
            "singleton_clusters":
                singleton_clusters,
            "clusters_lt_2":
                clusters_lt_2
        }
    )

    # Save each candidate model.
    joblib.dump(
        gmm,
        os.path.join(
            OUTPUT_DIR,
            f"gmm_K{K}.joblib"
        )
    )

    # Checkpoint sweep results after every K.
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
# 19. RESULTS DATAFRAME
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
# 20. INDIVIDUAL METRIC OPTIMA
#
# These are diagnostics only.
# We do NOT automatically select K from one metric.
# ============================================================

best_bic_row = results_df.loc[
    results_df["bic"].idxmin()
]

best_aic_row = results_df.loc[
    results_df["aic"].idxmin()
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
# 21. PLOT: BIC
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
    "Number of GMM components (K)"
)

plt.ylabel(
    "BIC"
)

plt.title(
    "SCALA-64 GMM Model Selection: BIC"
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

plt.close()


# ============================================================
# 22. PLOT: AIC
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    results_df["K"],
    results_df["aic"],
    marker="o"
)

plt.xlabel(
    "Number of GMM components (K)"
)

plt.ylabel(
    "AIC"
)

plt.title(
    "SCALA-64 GMM Model Selection: AIC"
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
        "gmm_aic_vs_k.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 23. PLOT: VALIDATION LOG LIKELIHOOD
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
    "Number of GMM components (K)"
)

plt.ylabel(
    "Average validation log likelihood"
)

plt.title(
    "SCALA-64 GMM Validation Log Likelihood"
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

plt.close()


# ============================================================
# 24. PLOT: COSINE CLUSTER QUALITY
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
    "Number of GMM components (K)"
)

plt.ylabel(
    "Cosine score"
)

plt.title(
    "SCALA-64 GMM Cluster Quality"
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

plt.close()


# ============================================================
# 25. SAVE EMBEDDING TIMING
# ============================================================

timing_df = pd.DataFrame(
    [
        {
            "dataset": "train_51",
            "records": len(train_df),
            "embedding_time_sec":
                train_encoding_time
        },
        {
            "dataset": "validation_13",
            "records": len(val_df),
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
# 26. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SCALA-64 GMM K SWEEP COMPLETE")
print("=" * 80)

display_cols = [
    "K",
    "bic",
    "aic",
    "validation_avg_log_likelihood",
    "silhouette_cosine",
    "intra_cluster_cosine",
    "min_cluster_size",
    "mean_cluster_size",
    "max_cluster_size",
    "singleton_clusters",
    "clusters_lt_2"
]

print("\nFull results:\n")

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
    f"Lowest AIC:"
    f" K={int(best_aic_row['K'])}"
    f" | AIC={best_aic_row['aic']:.4f}"
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
    "Do not automatically choose K from a single metric."
)

print(
    "Inspect BIC, AIC, validation likelihood, "
    "cosine cluster quality, and cluster sizes."
)

print(
    "Because final retrieval uses K=2 records, "
    "also inspect singleton_clusters and clusters_lt_2."
)