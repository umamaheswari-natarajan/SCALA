from pathlib import Path
import os
import json
import time
import joblib
import numpy as np

from sklearn.mixture import GaussianMixture


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "synthetic_proposed_final"
)

GMM_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gmm_k55"
)

os.makedirs(
    GMM_MODEL_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILE
#
# Existing raw 256-D learned fault embeddings.
#
# These were already generated using:
#
# trained contrastive encoder
#        +
# learned fault projection
#
# No need to regenerate them here.
# ============================================================

TRAIN_RAW_FILE = os.path.join(
    EMBEDDING_DIR,
    "kb_fault_embeddings_raw.npy"
)


# ============================================================
# OUTPUT FILES
# ============================================================

GMM_FILE = os.path.join(
    GMM_MODEL_DIR,
    "gmm_K55_final.joblib"
)

MEANS_FILE = os.path.join(
    GMM_MODEL_DIR,
    "gmm_means.npy"
)

LABELS_FILE = os.path.join(
    GMM_MODEL_DIR,
    "train_cluster_labels.npy"
)

CLUSTER_LOOKUP_FILE = os.path.join(
    GMM_MODEL_DIR,
    "cluster_to_indices.joblib"
)

SUMMARY_FILE = os.path.join(
    GMM_MODEL_DIR,
    "gmm_K55_summary.json"
)


# ============================================================
# FROZEN SETTINGS
# ============================================================

SEED = 42

EXPECTED_KB_SIZE = 10500

EXPECTED_EMBEDDING_DIM = 384

GMM_K = 55

GMM_COVARIANCE_TYPE = "diag"

GMM_MAX_ITER = 300

GMM_N_INIT = 1

GMM_REG_COVAR = 1e-6


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(
    TRAIN_RAW_FILE
):

    raise FileNotFoundError(
        f"Missing raw learned embeddings:\n"
        f"{TRAIN_RAW_FILE}"
    )


# ============================================================
# LOAD EXISTING RAW LEARNED KB EMBEDDINGS
# ============================================================

print("=" * 80)
print("COLD-START SCALA OFFLINE PREPARATION")
print("GMM K=55 | RAW 256-D LEARNED EMBEDDINGS | NO PCA")
print("=" * 80)


train_raw = np.load(
    TRAIN_RAW_FILE
)

train_raw = np.ascontiguousarray(
    train_raw,
    dtype=np.float32
)


print(
    "\nRaw KB embedding shape:",
    train_raw.shape
)


# ============================================================
# SANITY CHECK EMBEDDINGS
# ============================================================

expected_shape = (
    EXPECTED_KB_SIZE,
    EXPECTED_EMBEDDING_DIM
)


if train_raw.shape != expected_shape:

    raise ValueError(
        f"Expected raw embedding shape "
        f"{expected_shape}, "
        f"found {train_raw.shape}."
    )


if not np.isfinite(
    train_raw
).all():

    raise ValueError(
        "Non-finite value detected in raw embeddings."
    )


# ============================================================
# FIT GMM
#
# IMPORTANT:
#
# GMM is fitted directly on:
#
#     RAW learned 256-D fault embeddings
#
# No PCA.
# No L2 normalization before GMM fitting.
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "FITTING GMM"
)

print(
    "=" * 80
)


print(
    "Components:",
    GMM_K
)

print(
    "Covariance type:",
    GMM_COVARIANCE_TYPE
)

print(
    "Input dimension:",
    train_raw.shape[1]
)


gmm = GaussianMixture(

    n_components=
        GMM_K,

    covariance_type=
        GMM_COVARIANCE_TYPE,

    max_iter=
        GMM_MAX_ITER,

    n_init=
        GMM_N_INIT,

    reg_covar=
        GMM_REG_COVAR,

    random_state=
        SEED
)


gmm_start = (
    time.perf_counter()
)


gmm.fit(
    train_raw
)


gmm_fit_time_sec = (
    time.perf_counter()
    -
    gmm_start
)


print(
    "\nGMM converged:",
    gmm.converged_
)

print(
    "GMM iterations:",
    gmm.n_iter_
)

print(
    "GMM fit time:",
    f"{gmm_fit_time_sec:.4f} sec"
)

print(
    "GMM means shape:",
    gmm.means_.shape
)


# ============================================================
# VERIFY GMM
# ============================================================

expected_mean_shape = (
    GMM_K,
    EXPECTED_EMBEDDING_DIM
)


if gmm.means_.shape != expected_mean_shape:

    raise ValueError(
        f"Expected GMM means shape "
        f"{expected_mean_shape}, "
        f"found {gmm.means_.shape}."
    )


if not np.isfinite(
    gmm.means_
).all():

    raise ValueError(
        "Non-finite value detected in GMM means."
    )


# ============================================================
# SAVE COMPLETE GMM
# ============================================================

joblib.dump(
    gmm,
    GMM_FILE
)


# ============================================================
# The complete GMM is used during inference:
#
# raw query 384-D
#       ↓
# GMM predict_proba()
#       ↓
# posterior probabilities over 55 components
#       ↓
# Top-15 components
# ============================================================

gmm_means = np.ascontiguousarray(
    gmm.means_,
    dtype=np.float32
)


np.save(
    MEANS_FILE,
    gmm_means
)


# ============================================================
# ASSIGN ALL KB RECORDS TO THEIR GMM COMPONENT
#
# This is an OFFLINE operation.
#
# The resulting labels remain fixed during inference.
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "ASSIGNING KB RECORDS TO GMM CLUSTERS"
)

print(
    "=" * 80
)


assignment_start = (
    time.perf_counter()
)


train_cluster_labels = gmm.predict(
    train_raw
)


assignment_time_sec = (
    time.perf_counter()
    -
    assignment_start
)


train_cluster_labels = (
    train_cluster_labels
    .astype(
        np.int32
    )
)


print(
    "Cluster assignment time:",
    f"{assignment_time_sec:.4f} sec"
)


if train_cluster_labels.shape != (
    EXPECTED_KB_SIZE,
):

    raise ValueError(
        f"Expected cluster label shape "
        f"({EXPECTED_KB_SIZE},), "
        f"found {train_cluster_labels.shape}."
    )


np.save(
    LABELS_FILE,
    train_cluster_labels
)


# ============================================================
# CREATE CLUSTER -> KB INDEX LOOKUP
#
# Example:
#
# cluster 0 -> [4, 27, 81, ...]
# cluster 1 -> [2, 19, 65, ...]
# ...
# cluster 54 -> [...]
#
# During inference:
#
# nearest 15 clusters
#        ↓
# fetch their KB indices
#        ↓
# candidate pool
# ============================================================

cluster_to_indices = {}


for cluster_id in range(
    GMM_K
):

    indices = np.where(
        train_cluster_labels
        ==
        cluster_id
    )[0]

    cluster_to_indices[
        cluster_id
    ] = np.ascontiguousarray(
        indices,
        dtype=np.int32
    )


joblib.dump(
    cluster_to_indices,
    CLUSTER_LOOKUP_FILE
)


# ============================================================
# CLUSTER SIZE STATISTICS
# ============================================================

cluster_sizes = np.array(
    [
        len(
            cluster_to_indices[
                cluster_id
            ]
        )

        for cluster_id in range(
            GMM_K
        )
    ],
    dtype=np.int32
)


print(
    "\nCluster size statistics"
)

print(
    "Number of clusters:",
    len(
        cluster_sizes
    )
)

print(
    "Minimum:",
    int(
        cluster_sizes.min()
    )
)

print(
    "Maximum:",
    int(
        cluster_sizes.max()
    )
)

print(
    "Mean:",
    float(
        cluster_sizes.mean()
    )
)

print(
    "Median:",
    float(
        np.median(
            cluster_sizes
        )
    )
)

print(
    "Total:",
    int(
        cluster_sizes.sum()
    )
)


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

if cluster_sizes.sum() != EXPECTED_KB_SIZE:

    raise ValueError(
        "Cluster membership count does not match "
        "the KB size."
    )


if len(
    cluster_to_indices
) != GMM_K:

    raise ValueError(
        "Number of cluster mappings does not "
        "match GMM components."
    )


if len(
    train_cluster_labels
) != EXPECTED_KB_SIZE:

    raise ValueError(
        "Number of cluster labels does not "
        "match KB size."
    )


if gmm_means.shape != (
    GMM_K,
    EXPECTED_EMBEDDING_DIM
):

    raise ValueError(
        "Saved GMM means have unexpected shape."
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "method":
        "SCALA",

    "kb_size":
        EXPECTED_KB_SIZE,

    "embedding_type":
        "contrastive learned fault embeddings",

    "embedding_dimension":
        EXPECTED_EMBEDDING_DIM,

    "embedding_space_for_gmm":
        "raw",

    "pca_used":
        False,

    "gmm_components":
        GMM_K,

    "gmm_covariance_type":
        GMM_COVARIANCE_TYPE,

    "gmm_max_iter":
        GMM_MAX_ITER,

    "gmm_n_init":
        GMM_N_INIT,

    "gmm_reg_covar":
        GMM_REG_COVAR,

    "random_seed":
        SEED,

    "gmm_converged":
        bool(
            gmm.converged_
        ),

    "gmm_iterations":
        int(
            gmm.n_iter_
        ),

    "gmm_fit_time_sec":
        float(
            gmm_fit_time_sec
        ),

    "cluster_assignment_time_sec":
        float(
            assignment_time_sec
        ),

    "minimum_cluster_size":
        int(
            cluster_sizes.min()
        ),

    "maximum_cluster_size":
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

    "total_cluster_members":
        int(
            cluster_sizes.sum()
        ),

    "routing_method":
        "GMM posterior probability (predict_proba)",

    "inference_top_clusters":
        15,

    "inference_top_k_records":
        2
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
# FINAL VERIFICATION BY RELOADING SAVED ARTIFACTS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "VERIFYING SAVED ARTIFACTS"
)

print(
    "=" * 80
)


saved_gmm = joblib.load(
    GMM_FILE
)

saved_means = np.load(
    MEANS_FILE
)

saved_labels = np.load(
    LABELS_FILE
)

saved_lookup = joblib.load(
    CLUSTER_LOOKUP_FILE
)


if saved_gmm.n_components != GMM_K:

    raise ValueError(
        "Saved GMM has incorrect number of components."
    )


if saved_means.shape != (
    GMM_K,
    EXPECTED_EMBEDDING_DIM
):

    raise ValueError(
        "Saved GMM means have incorrect shape."
    )


if saved_labels.shape != (
    EXPECTED_KB_SIZE,
):

    raise ValueError(
        "Saved cluster labels have incorrect shape."
    )


if len(
    saved_lookup
) != GMM_K:

    raise ValueError(
        "Saved cluster lookup has incorrect "
        "number of clusters."
    )


saved_total = sum(
    len(
        indices
    )

    for indices in saved_lookup.values()
)


if saved_total != EXPECTED_KB_SIZE:

    raise ValueError(
        "Saved cluster lookup does not contain "
        "all KB records."
    )


# ============================================================
# FINAL PRINT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "COLD-START SCALA OFFLINE PREPARATION COMPLETE"
)

print(
    "=" * 80
)


print(
    "\nConfiguration:"
)

print(
    "KB records              :",
    EXPECTED_KB_SIZE
)

print(
    "Embedding dimension     :",
    EXPECTED_EMBEDDING_DIM
)

print(
    "PCA                     : NO"
)

print(
    "GMM components          :",
    GMM_K
)

print(
    "GMM covariance          :",
    GMM_COVARIANCE_TYPE
)

print(
    "Routing metric          : GMM posterior probability"
)

print(
    "Inference Top clusters  : 15"
)

print(
    "Final retrieval K       : 2"
)


print(
    "\nSaved files:"
)

print(
    "GMM                     :",
    GMM_FILE
)

print(
    "Raw GMM means           :",
    MEANS_FILE
)

print(
    "Cluster labels          :",
    LABELS_FILE
)

print(
    "Cluster -> indices      :",
    CLUSTER_LOOKUP_FILE
)

print(
    "Summary                 :",
    SUMMARY_FILE
)


print(
    "\nCluster statistics:"
)

print(
    "Minimum                 :",
    int(
        cluster_sizes.min()
    )
)

print(
    "Maximum                 :",
    int(
        cluster_sizes.max()
    )
)

print(
    "Mean                    :",
    f"{cluster_sizes.mean():.4f}"
)

print(
    "Median                  :",
    f"{np.median(cluster_sizes):.4f}"
)

print(
    "Total                   :",
    int(
        cluster_sizes.sum()
    )
)


print(
    "\nOffline timing:"
)

print(
    "GMM fitting             :",
    f"{gmm_fit_time_sec:.4f} sec"
)

print(
    "Cluster assignment      :",
    f"{assignment_time_sec:.4f} sec"
)


print(
    "\nNext inference pipeline:"
)

print(
    "query -> trained contrastive encoder -> raw 384-D"
)

print(
    "-> GMM predict_proba over 55 GMM components"
)

print(
    "->  Top-15 posterior-probability components"
)

print(
    "-> candidate KB pool"
)

print(
    "-> normalized 384-D cosine similarity"
)

print(
    "-> Top-2 fault-resolution pairs"
)

print(
    "-> GPT-4.1-mini"
)