import os
import json
import time

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "train_fault_resolution.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "plain_gmm"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embeddings",
    "minilm"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    EMBEDDING_DIR,
    exist_ok=True
)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

N_COMPONENTS = 55

RANDOM_STATE = 42

BATCH_SIZE = 64


# ============================================================
# OUTPUT FILES
# ============================================================

RAW_EMBEDDING_FILE = os.path.join(
    EMBEDDING_DIR,
    "minilm_train_fault_embeddings.npy"
)

NORMALIZED_EMBEDDING_FILE = os.path.join(
    EMBEDDING_DIR,
    "minilm_train_fault_embeddings_normalized.npy"
)

TRAIN_METADATA_FILE = os.path.join(
    EMBEDDING_DIR,
    "minilm_train_metadata.xlsx"
)

GMM_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "plain_gmm_55.joblib"
)

ASSIGNMENT_FILE = os.path.join(
    OUTPUT_DIR,
    "training_kb_with_plain_gmm_clusters.xlsx"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "plain_gmm_summary.json"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 90)
print("PLAIN GMM-RAG BASELINE")
print("=" * 90)

train_df = pd.read_excel(
    TRAIN_FILE
)

print("\nTraining records:", len(train_df))
print("Columns:", train_df.columns.tolist())


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "id",
    "fault_text",
    "resolution_text"
]

missing_columns = [
    col
    for col in required_columns
    if col not in train_df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


train_df["fault_text"] = (
    train_df["fault_text"]
    .fillna("")
    .astype(str)
)

train_df["resolution_text"] = (
    train_df["resolution_text"]
    .fillna("")
    .astype(str)
)


empty_faults = (
    train_df["fault_text"]
    .str.strip()
    .eq("")
    .sum()
)

if empty_faults > 0:

    raise ValueError(
        f"Found {empty_faults} empty fault texts."
    )


train_df = train_df.reset_index(
    drop=True
)


# ============================================================
# LOAD ORIGINAL PRETRAINED MINILM
#
# IMPORTANT:
# This is NOT the CLIP-trained encoder.
# No projection head is loaded.
# Resolution text is NOT used for representation learning.
# ============================================================

print("\n" + "=" * 90)
print("LOADING PRETRAINED MINILM")
print("=" * 90)

print("Model:", MODEL_NAME)

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# GENERATE RAW FAULT EMBEDDINGS
#
# fault_text
#     |
#     v
# pretrained MiniLM
#     |
#     v
# 384-D embedding
#
# normalize_embeddings=False is intentional.
# ============================================================

print("\n" + "=" * 90)
print("GENERATING TRAINING FAULT EMBEDDINGS")
print("=" * 90)

fault_texts = (
    train_df["fault_text"]
    .tolist()
)

start_time = time.perf_counter()

raw_embeddings = model.encode(
    fault_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False
)

embedding_time = (
    time.perf_counter()
    -
    start_time
)

raw_embeddings = raw_embeddings.astype(
    np.float32
)


print("\nRaw embedding shape:")
print(raw_embeddings.shape)

print(
    "Embedding generation time:",
    f"{embedding_time:.3f} sec"
)


# ============================================================
# NORMALIZE EMBEDDINGS FOR COSINE RETRIEVAL
#
# z_normalized = z / ||z||
#
# These embeddings will later be used by:
#
#   1. Plain GMM-RAG retrieval
#   2. Standard RAG retrieval
#
# ============================================================

print("\nNormalizing embeddings for cosine retrieval...")

norms = np.linalg.norm(
    raw_embeddings,
    axis=1,
    keepdims=True
)

norms = np.maximum(
    norms,
    1e-12
)

normalized_embeddings = (
    raw_embeddings
    /
    norms
).astype(
    np.float32
)


# ============================================================
# SAVE MINILM EMBEDDINGS
# ============================================================

np.save(
    RAW_EMBEDDING_FILE,
    raw_embeddings
)

np.save(
    NORMALIZED_EMBEDDING_FILE,
    normalized_embeddings
)


# ============================================================
# SAVE METADATA
#
# Row i in this file corresponds exactly to row i in the
# two embedding arrays.
# ============================================================

metadata_columns = [
    col
    for col in [
        "id",
        "fault_text",
        "root_cause",
        "resolution_text",
        "source_dataset",
        "split"
    ]
    if col in train_df.columns
]

metadata_df = train_df[
    metadata_columns
].copy()

metadata_df.insert(
    0,
    "embedding_index",
    np.arange(
        len(metadata_df)
    )
)

metadata_df.to_excel(
    TRAIN_METADATA_FILE,
    index=False
)


# ============================================================
# TRAIN GMM
#
# GMM is fitted on RAW MiniLM embeddings.
#
# It learns:
#
#   p(z | cluster k)
#
# for k = 1,...,55
# ============================================================

print("\n" + "=" * 90)
print("TRAINING PLAIN GMM")
print("=" * 90)

print(
    "Number of clusters:",
    N_COMPONENTS
)

gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    covariance_type="diag",
    random_state=RANDOM_STATE,
    n_init=3,
    reg_covar=1e-6
)

gmm_start = time.perf_counter()

gmm.fit(
    raw_embeddings
)

gmm_training_time = (
    time.perf_counter()
    -
    gmm_start
)

print(
    "GMM training time:",
    f"{gmm_training_time:.3f} sec"
)


# ============================================================
# POSTERIOR PROBABILITIES
#
# For every training fault:
#
# P(C1 | z), P(C2 | z), ..., P(C55 | z)
# ============================================================

posterior_probabilities = (
    gmm.predict_proba(
        raw_embeddings
    )
)

print(
    "\nPosterior matrix shape:",
    posterior_probabilities.shape
)


# ============================================================
# HARD PRIMARY ASSIGNMENT FOR KB ORGANIZATION
#
# cluster(i) =
#
# argmax_k P(C_k | z_i)
#
# Although GMM is probabilistic, each training record needs
# a primary cluster so we can organize the retrieval KB.
# ============================================================

cluster_assignments = np.argmax(
    posterior_probabilities,
    axis=1
)

max_probabilities = np.max(
    posterior_probabilities,
    axis=1
)


train_df[
    "plain_gmm_cluster"
] = cluster_assignments

train_df[
    "plain_gmm_max_probability"
] = max_probabilities


# ============================================================
# CLUSTER SIZE STATISTICS
# ============================================================

cluster_sizes = np.bincount(
    cluster_assignments,
    minlength=N_COMPONENTS
)


print("\nCluster sizes:")
print(cluster_sizes)

print(
    "\nSmallest cluster:",
    int(cluster_sizes.min())
)

print(
    "Largest cluster:",
    int(cluster_sizes.max())
)

print(
    "Mean cluster size:",
    float(cluster_sizes.mean())
)

print(
    "Median cluster size:",
    float(np.median(cluster_sizes))
)

print(
    "Empty clusters:",
    int(
        np.sum(
            cluster_sizes == 0
        )
    )
)


# ============================================================
# COSINE SILHOUETTE
#
# We evaluate semantic separation using normalized embeddings.
# ============================================================

print("\nCalculating cosine silhouette score...")

unique_clusters = np.unique(
    cluster_assignments
)

if len(unique_clusters) > 1:

    cosine_silhouette = silhouette_score(
        normalized_embeddings,
        cluster_assignments,
        metric="cosine"
    )

else:

    cosine_silhouette = None


print(
    "Cosine silhouette:",
    cosine_silhouette
)


# ============================================================
# AVERAGE INTRA-CLUSTER COSINE SIMILARITY
#
# For each cluster:
#   calculate centroid of normalized vectors
#   normalize centroid
#   calculate cosine similarity of members to centroid
#
# Then average across all records.
# ============================================================

intra_similarity_values = []

for cluster_id in range(
    N_COMPONENTS
):

    indices = np.where(
        cluster_assignments
        ==
        cluster_id
    )[0]

    if len(indices) == 0:
        continue

    cluster_vectors = (
        normalized_embeddings[
            indices
        ]
    )

    centroid = cluster_vectors.mean(
        axis=0
    )

    centroid_norm = np.linalg.norm(
        centroid
    )

    if centroid_norm == 0:
        continue

    centroid = (
        centroid
        /
        centroid_norm
    )

    similarities = (
        cluster_vectors
        @
        centroid
    )

    intra_similarity_values.extend(
        similarities.tolist()
    )


if intra_similarity_values:

    avg_intra_cluster_cosine = float(
        np.mean(
            intra_similarity_values
        )
    )

else:

    avg_intra_cluster_cosine = None


print(
    "Average intra-cluster cosine:",
    avg_intra_cluster_cosine
)


# ============================================================
# GMM BIC / AIC FOR THE FITTED MODEL
# ============================================================

bic = float(
    gmm.bic(
        raw_embeddings
    )
)

aic = float(
    gmm.aic(
        raw_embeddings
    )
)


print(
    "BIC:",
    bic
)

print(
    "AIC:",
    aic
)


# ============================================================
# SAVE GMM MODEL
# ============================================================

joblib.dump(
    gmm,
    GMM_MODEL_FILE
)


# ============================================================
# SAVE TRAINING ASSIGNMENTS
# ============================================================

train_df.to_excel(
    ASSIGNMENT_FILE,
    index=False
)


# ============================================================
# BUILD SUMMARY
# ============================================================

summary = {

    "baseline": "Plain GMM-RAG",

    "embedding_model": MODEL_NAME,

    "embedding_type":
        "Original pretrained MiniLM fault embeddings",

    "uses_resolution_for_embedding_learning":
        False,

    "uses_clip_projection_head":
        False,

    "training_records":
        int(
            len(train_df)
        ),

    "embedding_dimension":
        int(
            raw_embeddings.shape[1]
        ),

    "gmm_components":
        int(
            N_COMPONENTS
        ),

    "covariance_type":
        "diag",

    "gmm_n_init":
        3,

    "random_state":
        RANDOM_STATE,

    "embedding_generation_time_sec":
        float(
            embedding_time
        ),

    "gmm_training_time_sec":
        float(
            gmm_training_time
        ),

    "smallest_cluster":
        int(
            cluster_sizes.min()
        ),

    "largest_cluster":
        int(
            cluster_sizes.max()
        ),

    "mean_cluster_size":
        float(
            cluster_sizes.mean()
        ),

    "median_cluster_size":
        float(
            np.median(
                cluster_sizes
            )
        ),

    "empty_clusters":
        int(
            np.sum(
                cluster_sizes == 0
            )
        ),

    "cosine_silhouette":
        (
            float(
                cosine_silhouette
            )
            if cosine_silhouette
            is not None
            else None
        ),

    "avg_intra_cluster_cosine":
        avg_intra_cluster_cosine,

    "bic":
        bic,

    "aic":
        aic,

    "cluster_sizes":
        cluster_sizes
        .astype(int)
        .tolist(),

    "files": {

        "raw_embeddings":
            RAW_EMBEDDING_FILE,

        "normalized_embeddings":
            NORMALIZED_EMBEDDING_FILE,

        "metadata":
            TRAIN_METADATA_FILE,

        "gmm_model":
            GMM_MODEL_FILE,

        "cluster_assignments":
            ASSIGNMENT_FILE
    }
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("PLAIN GMM TRAINING COMPLETE")
print("=" * 90)

print(
    "\nTraining records:",
    len(train_df)
)

print(
    "Embedding dimension:",
    raw_embeddings.shape[1]
)

print(
    "GMM clusters:",
    N_COMPONENTS
)

print(
    "Cosine silhouette:",
    cosine_silhouette
)

print(
    "Average intra-cluster cosine:",
    avg_intra_cluster_cosine
)

print(
    "\nRaw MiniLM embeddings:"
)
print(
    RAW_EMBEDDING_FILE
)

print(
    "\nNormalized MiniLM embeddings:"
)
print(
    NORMALIZED_EMBEDDING_FILE
)

print(
    "\nTraining metadata:"
)
print(
    TRAIN_METADATA_FILE
)

print(
    "\nGMM model:"
)
print(
    GMM_MODEL_FILE
)

print(
    "\nCluster assignments:"
)
print(
    ASSIGNMENT_FILE
)

print(
    "\nSummary:"
)
print(
    SUMMARY_FILE
)

print("\nIMPORTANT:")
print(
    "The saved MiniLM embeddings can be reused "
    "later for Standard RAG."
)