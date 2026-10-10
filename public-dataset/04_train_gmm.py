import os
import math
import time
import json
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "train_fault_resolution.xlsx"
)

VAL_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "val_fault_resolution.xlsx"
)

CONTRASTIVE_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "contrastive_minilm"
)

BACKBONE_DIR = os.path.join(
    CONTRASTIVE_MODEL_DIR,
    "backbone"
)

PROJECTION_FILE = os.path.join(
    CONTRASTIVE_MODEL_DIR,
    "projection_heads.pt"
)

GMM_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gmm"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "gmm"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

os.makedirs(GMM_MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 64

RANDOM_STATE = 42

# Candidate numbers of GMM components
GMM_COMPONENTS = [
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60
]

# Diagonal covariance is much more practical
# than full covariance for 256-D embeddings
COVARIANCE_TYPE = "diag"

N_INIT = 3

MAX_ITER = 500


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("DEVICE")
print("=" * 80)

print("Using:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD SAVED CONTRASTIVE CONFIG
# ============================================================

print("\n" + "=" * 80)
print("LOADING SAVED CONTRASTIVE MODEL")
print("=" * 80)

checkpoint = torch.load(
    PROJECTION_FILE,
    map_location="cpu"
)

BACKBONE_DIM = checkpoint[
    "backbone_dim"
]

PROJECTION_DIM = checkpoint[
    "projection_dim"
]

MAX_LENGTH = checkpoint[
    "max_length"
]

BEST_EPOCH = checkpoint[
    "epoch"
]

print("Best contrastive epoch :", BEST_EPOCH)
print("Backbone dimension     :", BACKBONE_DIM)
print("Projection dimension   :", PROJECTION_DIM)
print("Maximum token length   :", MAX_LENGTH)

print(
    "Saved validation metrics:",
    checkpoint["validation_metrics"]
)


# ============================================================
# LOAD FINE-TUNED MINILM
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    BACKBONE_DIR
)

backbone = AutoModel.from_pretrained(
    BACKBONE_DIR
)


# ============================================================
# FAULT PROJECTION HEAD
#
# We only need the fault projection for GMM.
# ============================================================

fault_projection = nn.Linear(
    BACKBONE_DIM,
    PROJECTION_DIM,
    bias=False
)

fault_projection.load_state_dict(
    checkpoint[
        "fault_projection_state_dict"
    ]
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

backbone = backbone.to(DEVICE)
fault_projection = fault_projection.to(DEVICE)

backbone.eval()
fault_projection.eval()


# ============================================================
# DATA
# ============================================================

train_df = pd.read_excel(
    TRAIN_FILE
)

val_df = pd.read_excel(
    VAL_FILE
)

print("\n" + "=" * 80)
print("DATA")
print("=" * 80)

print("Training faults  :", len(train_df))
print("Validation faults:", len(val_df))


# ============================================================
# DATASET
# ============================================================

class FaultDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        return {
            "id": str(row["id"]),
            "fault_text": str(
                row["fault_text"]
            )
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    ids = [
        x["id"]
        for x in batch
    ]

    texts = [
        x["fault_text"]
        for x in batch
    ]

    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return {
        "ids": ids,
        "texts": texts,
        "tokens": tokens
    }


train_loader = DataLoader(
    FaultDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    FaultDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# MEAN POOLING
#
# Must be identical to contrastive-training code.
# ============================================================

def mean_pooling(
    token_embeddings,
    attention_mask
):

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(
            token_embeddings.size()
        )
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
# GENERATE FAULT EMBEDDINGS
#
# Returns TWO versions:
#
# raw:
#   MiniLM -> fault projection
#   Used for GMM
#
# normalized:
#   L2-normalized projection
#   Used later for cosine retrieval
# ============================================================

@torch.no_grad()
def generate_fault_embeddings(
    loader,
    name
):

    raw_embeddings = []
    normalized_embeddings = []

    all_ids = []
    all_texts = []

    start_time = time.perf_counter()

    for batch in tqdm(
        loader,
        desc=f"Encoding {name}"
    ):

        tokens = {
            key: value.to(DEVICE)
            for key, value
            in batch["tokens"].items()
        }

        outputs = backbone(
            input_ids=tokens[
                "input_ids"
            ],
            attention_mask=tokens[
                "attention_mask"
            ]
        )

        # MiniLM sentence embedding
        h = mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"]
        )

        # Fine-tuned fault projection
        u = fault_projection(h)

        # Normalized form for cosine similarity
        z = F.normalize(
            u,
            p=2,
            dim=1
        )

        raw_embeddings.append(
            u.cpu().numpy()
        )

        normalized_embeddings.append(
            z.cpu().numpy()
        )

        all_ids.extend(
            batch["ids"]
        )

        all_texts.extend(
            batch["texts"]
        )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    raw_embeddings = np.vstack(
        raw_embeddings
    )

    normalized_embeddings = np.vstack(
        normalized_embeddings
    )


    print(
        f"\n{name} raw embedding shape:",
        raw_embeddings.shape
    )

    print(
        f"{name} normalized embedding shape:",
        normalized_embeddings.shape
    )

    print(
        f"{name} embedding generation time:",
        f"{elapsed:.2f} sec"
    )


    return {
        "raw":
            raw_embeddings,

        "normalized":
            normalized_embeddings,

        "ids":
            all_ids,

        "texts":
            all_texts,

        "time":
            elapsed
    }


# ============================================================
# GENERATE TRAINING EMBEDDINGS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING TRAINING FAULT EMBEDDINGS")
print("=" * 80)

train_encoded = generate_fault_embeddings(
    train_loader,
    "training"
)


# ============================================================
# GENERATE VALIDATION EMBEDDINGS
#
# These are NOT used to fit GMM.
#
# They will be useful later for routing /
# threshold tuning.
# ============================================================

print("\n" + "=" * 80)
print("GENERATING VALIDATION FAULT EMBEDDINGS")
print("=" * 80)

val_encoded = generate_fault_embeddings(
    val_loader,
    "validation"
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    os.path.join(
        EMBEDDING_DIR,
        "train_fault_embeddings_raw.npy"
    ),
    train_encoded["raw"]
)

np.save(
    os.path.join(
        EMBEDDING_DIR,
        "train_fault_embeddings_normalized.npy"
    ),
    train_encoded["normalized"]
)

np.save(
    os.path.join(
        EMBEDDING_DIR,
        "val_fault_embeddings_raw.npy"
    ),
    val_encoded["raw"]
)

np.save(
    os.path.join(
        EMBEDDING_DIR,
        "val_fault_embeddings_normalized.npy"
    ),
    val_encoded["normalized"]
)


# ============================================================
# SAVE EMBEDDING METADATA
#
# Keeps row alignment between embeddings and original KB.
# ============================================================

train_metadata = train_df.copy()

train_metadata[
    "embedding_index"
] = np.arange(
    len(train_metadata)
)

train_metadata.to_excel(
    os.path.join(
        EMBEDDING_DIR,
        "train_embedding_metadata.xlsx"
    ),
    index=False
)


val_metadata = val_df.copy()

val_metadata[
    "embedding_index"
] = np.arange(
    len(val_metadata)
)

val_metadata.to_excel(
    os.path.join(
        EMBEDDING_DIR,
        "val_embedding_metadata.xlsx"
    ),
    index=False
)


# ============================================================
# INTRA-CLUSTER COSINE SIMILARITY
#
# Uses normalized embeddings.
#
# For each cluster:
#
#   compute centroid
#   compare each member to centroid
#
# Then average across all records.
# ============================================================

def intra_cluster_cosine(
    normalized_embeddings,
    labels
):

    similarities = []

    unique_labels = np.unique(
        labels
    )

    for cluster_id in unique_labels:

        mask = (
            labels == cluster_id
        )

        cluster_vectors = (
            normalized_embeddings[
                mask
            ]
        )

        if len(cluster_vectors) == 0:
            continue


        centroid = cluster_vectors.mean(
            axis=0
        )

        norm = np.linalg.norm(
            centroid
        )

        if norm == 0:
            continue

        centroid = centroid / norm


        cosine_values = (
            cluster_vectors
            @ centroid
        )

        similarities.extend(
            cosine_values.tolist()
        )


    if len(similarities) == 0:
        return np.nan

    return float(
        np.mean(similarities)
    )


# ============================================================
# TRAIN MULTIPLE GMMs
# ============================================================

X_train = train_encoded["raw"]

X_train_norm = train_encoded[
    "normalized"
]

X_val = val_encoded["raw"]


gmm_results = []

best_gmm = None
best_bic = float("inf")
best_k = None


print("\n" + "=" * 80)
print("GMM MODEL SELECTION")
print("=" * 80)


for k in GMM_COMPONENTS:

    print("\n" + "-" * 80)

    print(
        f"Training GMM with K = {k}"
    )

    print("-" * 80)


    start = time.perf_counter()


    gmm = GaussianMixture(
        n_components=k,
        covariance_type=COVARIANCE_TYPE,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        reg_covar=1e-6
    )


    gmm.fit(
        X_train
    )


    fit_time = (
        time.perf_counter()
        -
        start
    )


    # --------------------------------------------------------
    # Hard labels only for evaluation
    # --------------------------------------------------------

    labels = gmm.predict(
        X_train
    )


    # --------------------------------------------------------
    # BIC
    #
    # LOWER is better.
    # --------------------------------------------------------

    bic = gmm.bic(
        X_train
    )


    # --------------------------------------------------------
    # AIC
    #
    # Also useful as supporting metric.
    # LOWER is better.
    # --------------------------------------------------------

    aic = gmm.aic(
        X_train
    )


    # --------------------------------------------------------
    # Silhouette with cosine distance
    #
    # HIGHER is better.
    # --------------------------------------------------------

    if (
        len(np.unique(labels)) > 1
        and
        len(np.unique(labels))
        < len(X_train)
    ):

        silhouette = silhouette_score(
            X_train_norm,
            labels,
            metric="cosine"
        )

    else:

        silhouette = np.nan


    # --------------------------------------------------------
    # Intra-cluster cosine similarity
    #
    # HIGHER is tighter.
    # --------------------------------------------------------

    intra_cosine = intra_cluster_cosine(
        X_train_norm,
        labels
    )


    # --------------------------------------------------------
    # Validation average log-likelihood
    #
    # Higher is better.
    #
    # Validation is NOT used for fitting.
    # --------------------------------------------------------

    val_log_likelihood = gmm.score(
        X_val
    )


    # --------------------------------------------------------
    # Cluster size statistics
    # --------------------------------------------------------

    cluster_sizes = np.bincount(
        labels,
        minlength=k
    )

    smallest_cluster = int(
        cluster_sizes.min()
    )

    largest_cluster = int(
        cluster_sizes.max()
    )

    mean_cluster_size = float(
        cluster_sizes.mean()
    )


    # --------------------------------------------------------
    # Convergence information
    # --------------------------------------------------------

    converged = bool(
        gmm.converged_
    )

    iterations = int(
        gmm.n_iter_
    )


    result = {
        "k":
            k,

        "bic":
            bic,

        "aic":
            aic,

        "silhouette_cosine":
            silhouette,

        "intra_cluster_cosine":
            intra_cosine,

        "validation_avg_log_likelihood":
            val_log_likelihood,

        "fit_time_seconds":
            fit_time,

        "converged":
            converged,

        "iterations":
            iterations,

        "smallest_cluster":
            smallest_cluster,

        "largest_cluster":
            largest_cluster,

        "mean_cluster_size":
            mean_cluster_size
    }


    gmm_results.append(
        result
    )


    print(
        f"BIC                         : {bic:.2f}"
    )

    print(
        f"AIC                         : {aic:.2f}"
    )

    print(
        f"Silhouette (cosine)         : {silhouette:.4f}"
    )

    print(
        f"Intra-cluster cosine        : {intra_cosine:.4f}"
    )

    print(
        f"Validation avg log-likelihood: {val_log_likelihood:.4f}"
    )

    print(
        f"Fit time                    : {fit_time:.2f} sec"
    )

    print(
        f"Converged                   : {converged}"
    )

    print(
        f"Iterations                  : {iterations}"
    )

    print(
        f"Cluster size range          : "
        f"{smallest_cluster} - {largest_cluster}"
    )


    # --------------------------------------------------------
    # PRIMARY MODEL SELECTION:
    # LOWEST BIC
    # --------------------------------------------------------

    if bic < best_bic:

        best_bic = bic
        best_gmm = gmm
        best_k = k


# ============================================================
# SAVE GMM METRICS
# ============================================================

gmm_results_df = pd.DataFrame(
    gmm_results
)

gmm_results_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "gmm_model_selection.csv"
    ),
    index=False
)


# ============================================================
# SAVE BEST GMM
# ============================================================

BEST_GMM_FILE = os.path.join(
    GMM_MODEL_DIR,
    "best_gmm.joblib"
)

joblib.dump(
    best_gmm,
    BEST_GMM_FILE
)


# ============================================================
# GET TRAINING CLUSTER ASSIGNMENTS
# ============================================================

train_cluster_labels = (
    best_gmm.predict(
        X_train
    )
)

train_probabilities = (
    best_gmm.predict_proba(
        X_train
    )
)


# ============================================================
# ADD CLUSTER INFORMATION TO TRAINING KB
# ============================================================

train_kb = train_df.copy()

train_kb[
    "gmm_cluster"
] = train_cluster_labels

train_kb[
    "gmm_max_probability"
] = train_probabilities.max(
    axis=1
)


# ============================================================
# POSTERIOR ENTROPY
#
# Lower = more confident cluster assignment
# ============================================================

epsilon = 1e-12

entropy = -np.sum(
    train_probabilities
    *
    np.log(
        train_probabilities
        + epsilon
    ),
    axis=1
)

train_kb[
    "gmm_entropy"
] = entropy


TRAIN_KB_FILE = os.path.join(
    RESULT_DIR,
    "training_kb_with_gmm_clusters.xlsx"
)

train_kb.to_excel(
    TRAIN_KB_FILE,
    index=False
)


# ============================================================
# CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    train_kb
    .groupby(
        "gmm_cluster"
    )
    .agg(
        records=(
            "id",
            "count"
        ),

        avg_max_probability=(
            "gmm_max_probability",
            "mean"
        ),

        avg_entropy=(
            "gmm_entropy",
            "mean"
        )
    )
    .reset_index()
)

cluster_summary.to_excel(
    os.path.join(
        RESULT_DIR,
        "gmm_cluster_summary.xlsx"
    ),
    index=False
)


# ============================================================
# SAVE GMM CONFIGURATION
# ============================================================

best_result = (
    gmm_results_df[
        gmm_results_df["k"] == best_k
    ]
    .iloc[0]
    .to_dict()
)


gmm_config = {
    "selected_k":
        int(best_k),

    "selection_criterion":
        "minimum BIC",

    "covariance_type":
        COVARIANCE_TYPE,

    "embedding_dimension":
        int(PROJECTION_DIM),

    "training_records":
        int(len(X_train)),

    "best_metrics":
        {
            key:
                (
                    value.item()
                    if hasattr(
                        value,
                        "item"
                    )
                    else value
                )

            for key, value
            in best_result.items()
        }
}


with open(
    os.path.join(
        GMM_MODEL_DIR,
        "gmm_config.json"
    ),
    "w"
) as f:

    json.dump(
        gmm_config,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("GMM TRAINING COMPLETE")
print("=" * 80)

print(
    "Training embedding shape:",
    X_train.shape
)

print(
    "Selected K:",
    best_k
)

print(
    "Best BIC:",
    f"{best_bic:.2f}"
)

print(
    "\nBest GMM saved to:"
)

print(
    BEST_GMM_FILE
)

print(
    "\nTraining KB with cluster assignments:"
)

print(
    TRAIN_KB_FILE
)

print(
    "\nGMM model-selection metrics:"
)

print(
    os.path.join(
        RESULT_DIR,
        "gmm_model_selection.csv"
    )
)

print(
    "\nEmbedding files:"
)

print(
    EMBEDDING_DIR
)