import os
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "train_fault_resolution.xlsx"
)

TRAIN_EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "train_fault_embeddings_normalized.npy"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "kmeans"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "kmeans"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


KMEANS_FILE = os.path.join(
    MODEL_DIR,
    "kmeans_55.joblib"
)

ASSIGNMENT_FILE = os.path.join(
    RESULT_DIR,
    "training_kb_with_kmeans_clusters.xlsx"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "kmeans_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

N_CLUSTERS = 55
RANDOM_STATE = 42
N_INIT = 20


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("LOADING TRAINING DATA")
print("=" * 90)

train_df = pd.read_excel(TRAIN_FILE)

train_embeddings = np.load(
    TRAIN_EMBEDDING_FILE
)

print("Training rows:", len(train_df))
print("Embedding shape:", train_embeddings.shape)

if len(train_df) != len(train_embeddings):
    raise ValueError(
        "Training rows and embedding count do not match."
    )


# ============================================================
# CHECK NORMALIZATION
# ============================================================

norms = np.linalg.norm(
    train_embeddings,
    axis=1
)

print(
    "Mean embedding norm:",
    float(norms.mean())
)

print(
    "Min embedding norm:",
    float(norms.min())
)

print(
    "Max embedding norm:",
    float(norms.max())
)


# ============================================================
# TRAIN K-MEANS
# ============================================================

print("\n" + "=" * 90)
print("TRAINING K-MEANS")
print("=" * 90)

print("Clusters:", N_CLUSTERS)

start = time.perf_counter()

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)

cluster_labels = kmeans.fit_predict(
    train_embeddings
)

training_time = (
    time.perf_counter() - start
)


# ============================================================
# CLUSTER SIZE STATISTICS
# ============================================================

cluster_sizes = np.bincount(
    cluster_labels,
    minlength=N_CLUSTERS
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


# ============================================================
# SILHOUETTE SCORE
#
# We use cosine because downstream retrieval is cosine-based.
# ============================================================

print("\nCalculating cosine silhouette score...")

silhouette_cosine = silhouette_score(
    train_embeddings,
    cluster_labels,
    metric="cosine"
)

print(
    "Cosine silhouette:",
    silhouette_cosine
)


# ============================================================
# AVERAGE INTRA-CLUSTER COSINE SIMILARITY
#
# Because vectors are normalized:
#
# cosine(x, centroid_direction)
# = dot(x, normalized centroid)
# ============================================================

cluster_intra_scores = []

for cluster_id in range(N_CLUSTERS):

    indices = np.where(
        cluster_labels == cluster_id
    )[0]

    cluster_vectors = train_embeddings[
        indices
    ]

    centroid = cluster_vectors.mean(
        axis=0
    )

    centroid_norm = np.linalg.norm(
        centroid
    )

    if centroid_norm == 0:
        continue

    centroid = (
        centroid / centroid_norm
    )

    similarities = np.dot(
        cluster_vectors,
        centroid
    )

    cluster_intra_scores.extend(
        similarities.tolist()
    )


avg_intra_cluster_cosine = float(
    np.mean(cluster_intra_scores)
)

print(
    "Average intra-cluster cosine:",
    avg_intra_cluster_cosine
)


# ============================================================
# K-MEANS INERTIA
# ============================================================

print(
    "K-Means inertia:",
    float(kmeans.inertia_)
)

print(
    "Training time:",
    f"{training_time:.3f} sec"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    kmeans,
    KMEANS_FILE
)


# ============================================================
# SAVE TRAINING ASSIGNMENTS
# ============================================================

assignment_df = train_df.copy()

assignment_df[
    "kmeans_cluster"
] = cluster_labels

assignment_df.to_excel(
    ASSIGNMENT_FILE,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "n_training_records":
        int(len(train_df)),

    "embedding_dimension":
        int(train_embeddings.shape[1]),

    "n_clusters":
        N_CLUSTERS,

    "random_state":
        RANDOM_STATE,

    "n_init":
        N_INIT,

    "training_time_seconds":
        float(training_time),

    "inertia":
        float(kmeans.inertia_),

    "silhouette_cosine":
        float(silhouette_cosine),

    "avg_intra_cluster_cosine":
        avg_intra_cluster_cosine,

    "smallest_cluster":
        int(cluster_sizes.min()),

    "largest_cluster":
        int(cluster_sizes.max()),

    "mean_cluster_size":
        float(cluster_sizes.mean()),

    "median_cluster_size":
        float(np.median(cluster_sizes)),

    "cluster_sizes":
        cluster_sizes.tolist()
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
# FINAL
# ============================================================

print("\n" + "=" * 90)
print("K-MEANS TRAINING COMPLETE")
print("=" * 90)

print("Model:")
print(KMEANS_FILE)

print("\nAssignments:")
print(ASSIGNMENT_FILE)

print("\nSummary:")
print(SUMMARY_FILE)

if cluster_sizes.min() < 5:

    print(
        "\nWARNING:"
        " At least one cluster contains fewer than "
        "5 records."
    )

else:

    print(
        "\nAll clusters contain at least 5 records."
    )