import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Existing CLIP proposed-model artifacts
SOURCE_DIR = os.path.join(
    BASE_DIR,
    "synthetic_proposed_final"
)

# Knowledge base
KB_FILE = os.path.join(
    BASE_DIR,
    "knowledge-base.xlsx"
)

# New PCA experiment directory
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "synthetic_proposed_pca50_gmm_K50"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Existing learned CLIP fault embeddings
KB_RAW_EMBEDDING_FILE = os.path.join(
    SOURCE_DIR,
    "kb_fault_embeddings_raw.npy"
)

PCA_DIM = 50
GMM_K = 50
RANDOM_STATE = 42


# ============================================================
# OUTPUT FILES
# ============================================================

PCA_FILE = os.path.join(
    OUTPUT_DIR,
    "pca_50.joblib"
)

GMM_FILE = os.path.join(
    OUTPUT_DIR,
    "gmm_K50_pca50.joblib"
)

KB_PCA_RAW_FILE = os.path.join(
    OUTPUT_DIR,
    "kb_fault_embeddings_pca50_raw.npy"
)

KB_PCA_NORMALIZED_FILE = os.path.join(
    OUTPUT_DIR,
    "kb_fault_embeddings_pca50_normalized.npy"
)

CLUSTER_ASSIGNMENT_FILE = os.path.join(
    OUTPUT_DIR,
    "cluster_assignments.npy"
)

CLUSTER_LOOKUP_FILE = os.path.join(
    OUTPUT_DIR,
    "cluster_lookup.joblib"
)

CLUSTER_STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "cluster_statistics.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "offline_summary.json"
)


# ============================================================
# LOAD KB
# ============================================================

print("=" * 85)
print("CLIP + PCA(50) + GMM(K=50) OFFLINE BUILD")
print("=" * 85)

print("\nLoading knowledge base...")

kb_df = pd.read_excel(
    KB_FILE
)

print("KB records:", len(kb_df))

if len(kb_df) != 10500:
    print(
        f"WARNING: Expected 10500 KB records, "
        f"but found {len(kb_df)}"
    )


# ============================================================
# LOAD EXISTING CLIP EMBEDDINGS
# ============================================================

print("\nLoading learned CLIP fault embeddings...")

kb_raw = np.load(
    KB_RAW_EMBEDDING_FILE
)

print(
    "Original CLIP embedding shape:",
    kb_raw.shape
)

if kb_raw.shape[0] != len(kb_df):
    raise ValueError(
        "KB row count and embedding row count do not match."
    )

if kb_raw.shape[1] != 256:
    print(
        f"WARNING: Expected 256-D CLIP embeddings, "
        f"but found {kb_raw.shape[1]} dimensions."
    )


# ============================================================
# PCA: 256 -> 50
# ============================================================

print("\n" + "=" * 85)
print("FITTING PCA")
print("=" * 85)

pca_start = time.perf_counter()

pca = PCA(
    n_components=PCA_DIM,
    random_state=RANDOM_STATE
)

kb_pca_raw = pca.fit_transform(
    kb_raw
)

pca_end = time.perf_counter()

pca_fit_transform_sec = (
    pca_end - pca_start
)

explained_variance = float(
    np.sum(
        pca.explained_variance_ratio_
    )
)

print(
    "PCA output shape:",
    kb_pca_raw.shape
)

print(
    f"Explained variance retained: "
    f"{explained_variance:.6f}"
)

print(
    f"PCA fit + KB transform time: "
    f"{pca_fit_transform_sec:.6f} sec"
)


# ============================================================
# NORMALIZE PCA EMBEDDINGS FOR COSINE RETRIEVAL
# ============================================================

print("\nNormalizing PCA embeddings for cosine retrieval...")

norms = np.linalg.norm(
    kb_pca_raw,
    axis=1,
    keepdims=True
)

norms = np.clip(
    norms,
    a_min=1e-12,
    a_max=None
)

kb_pca_normalized = (
    kb_pca_raw / norms
)

print(
    "Normalized PCA embedding shape:",
    kb_pca_normalized.shape
)


# ============================================================
# FIT GMM ON RAW PCA EMBEDDINGS
# ============================================================

print("\n" + "=" * 85)
print("FITTING GMM")
print("=" * 85)

gmm_start = time.perf_counter()

gmm = GaussianMixture(
    n_components=GMM_K,
    covariance_type="diag",
    random_state=RANDOM_STATE
)

gmm.fit(
    kb_pca_raw
)

gmm_end = time.perf_counter()

gmm_fit_sec = (
    gmm_end - gmm_start
)

print(
    f"GMM fitting time: "
    f"{gmm_fit_sec:.6f} sec"
)


# ============================================================
# ASSIGN KB RECORDS TO GMM CLUSTERS
# ============================================================

print("\nAssigning KB records to clusters...")

assignment_start = time.perf_counter()

cluster_assignments = gmm.predict(
    kb_pca_raw
)

assignment_end = time.perf_counter()

cluster_assignment_sec = (
    assignment_end -
    assignment_start
)

print(
    f"Cluster assignment time: "
    f"{cluster_assignment_sec:.6f} sec"
)


# ============================================================
# BUILD CLUSTER LOOKUP
# ============================================================

cluster_lookup = {}

cluster_stats = []

for cluster_id in range(GMM_K):

    indices = np.where(
        cluster_assignments == cluster_id
    )[0].astype(np.int32)

    cluster_lookup[
        cluster_id
    ] = indices

    cluster_stats.append(
        {
            "cluster_id": cluster_id,
            "candidate_count": len(indices),
            "candidate_fraction": (
                len(indices) / len(kb_df)
            ),
            "candidate_reduction": (
                1.0 -
                len(indices) / len(kb_df)
            )
        }
    )


cluster_stats_df = pd.DataFrame(
    cluster_stats
)

cluster_sizes = cluster_stats_df[
    "candidate_count"
].to_numpy()


# ============================================================
# CLUSTER STATISTICS
# ============================================================

print("\n" + "=" * 85)
print("CLUSTER STATISTICS")
print("=" * 85)

print(
    "Number of clusters:",
    GMM_K
)

print(
    f"Average cluster size: "
    f"{np.mean(cluster_sizes):.2f}"
)

print(
    "Minimum cluster size:",
    int(np.min(cluster_sizes))
)

print(
    "Maximum cluster size:",
    int(np.max(cluster_sizes))
)

print(
    f"Median cluster size: "
    f"{np.median(cluster_sizes):.2f}"
)

empty_clusters = int(
    np.sum(cluster_sizes == 0)
)

print(
    "Empty clusters:",
    empty_clusters
)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

print("\nSaving PCA/GMM artifacts...")

joblib.dump(
    pca,
    PCA_FILE
)

joblib.dump(
    gmm,
    GMM_FILE
)

np.save(
    KB_PCA_RAW_FILE,
    kb_pca_raw
)

np.save(
    KB_PCA_NORMALIZED_FILE,
    kb_pca_normalized
)

np.save(
    CLUSTER_ASSIGNMENT_FILE,
    cluster_assignments
)

joblib.dump(
    cluster_lookup,
    CLUSTER_LOOKUP_FILE
)

cluster_stats_df.to_csv(
    CLUSTER_STATS_FILE,
    index=False
)


# ============================================================
# OFFLINE SUMMARY
# ============================================================

summary = {

    "kb_records": int(
        len(kb_df)
    ),

    "original_embedding_dimension": int(
        kb_raw.shape[1]
    ),

    "pca_dimension": PCA_DIM,

    "pca_explained_variance_ratio_sum": (
        explained_variance
    ),

    "gmm_k": GMM_K,

    "gmm_covariance_type": "diag",

    "random_state": RANDOM_STATE,

    "pca_fit_and_kb_transform_sec": (
        pca_fit_transform_sec
    ),

    "gmm_fit_sec": (
        gmm_fit_sec
    ),

    "cluster_assignment_sec": (
        cluster_assignment_sec
    ),

    "average_cluster_size": float(
        np.mean(cluster_sizes)
    ),

    "median_cluster_size": float(
        np.median(cluster_sizes)
    ),

    "minimum_cluster_size": int(
        np.min(cluster_sizes)
    ),

    "maximum_cluster_size": int(
        np.max(cluster_sizes)
    ),

    "empty_clusters": empty_clusters
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
# FINAL REPORT
# ============================================================

print("\n" + "=" * 85)
print("PCA + GMM OFFLINE BUILD COMPLETE")
print("=" * 85)

print(
    f"\nKB records             : "
    f"{len(kb_df)}"
)

print(
    f"Original dimension     : "
    f"{kb_raw.shape[1]}"
)

print(
    f"PCA dimension          : "
    f"{PCA_DIM}"
)

print(
    f"Explained variance     : "
    f"{explained_variance:.6f}"
)

print(
    f"GMM K                  : "
    f"{GMM_K}"
)

print(
    f"Average cluster size   : "
    f"{np.mean(cluster_sizes):.2f}"
)

print(
    f"Min cluster size       : "
    f"{np.min(cluster_sizes)}"
)

print(
    f"Max cluster size       : "
    f"{np.max(cluster_sizes)}"
)

print(
    f"\nPCA fit + transform    : "
    f"{pca_fit_transform_sec:.6f} sec"
)

print(
    f"GMM fitting            : "
    f"{gmm_fit_sec:.6f} sec"
)

print(
    f"Cluster assignment     : "
    f"{cluster_assignment_sec:.6f} sec"
)

print(
    "\nArtifacts saved in:"
)

print(
    OUTPUT_DIR
)