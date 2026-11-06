from pathlib import Path
import os
import json
import joblib
import numpy as np

from sklearn.mixture import GaussianMixture


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Raw 384-D embeddings from the ORIGINAL pretrained MiniLM.
# These are the same representation used by Standard RAG.
EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "minilm",
    "minilm_train_fault_embeddings.npy"
)

# IMPORTANT:
# Keep this GMM separate from SCALA's GMM because this GMM
# is fitted in the pretrained MiniLM representation space.
GMM_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gmm_no_clip"
)

os.makedirs(
    GMM_MODEL_DIR,
    exist_ok=True
)

K = 55


# ============================================================
# LOAD RAW PRETRAINED MINILM EMBEDDINGS
#
# These embeddings are:
#
# fault text
#     ->
# pretrained all-MiniLM-L6-v2
#     ->
# raw 384-D embedding
#
# No CLIP fine-tuning.
# No projection head.
# ============================================================

X_train = np.load(
    EMBEDDING_FILE
).astype(np.float32)

X_train = np.ascontiguousarray(
    X_train,
    dtype=np.float32
)

print("=" * 80)
print("LOADING PRETRAINED MINILM EMBEDDINGS")
print("=" * 80)

print(
    "Embedding file:",
    EMBEDDING_FILE
)

print(
    "Training embeddings:",
    X_train.shape
)

if X_train.shape != (3049, 384):

    raise ValueError(
        f"Unexpected embedding shape: {X_train.shape}"
    )


# ============================================================
# FIT K=55 GMM
#
# GMM is fitted on RAW pretrained MiniLM embeddings.
#
# This is the GMM + No CLIP ablation:
#
# pretrained MiniLM representation
#             +
#          GMM K=55
# ============================================================

print("\n" + "=" * 80)
print("FITTING GMM + NO CLIP")
print("=" * 80)

print(
    "Fitting GMM with K=55..."
)

gmm = GaussianMixture(
    n_components=K,
    covariance_type="diag",
    random_state=42,
    n_init=3,
    max_iter=500,
    reg_covar=1e-6
)

gmm.fit(
    X_train
)

print(
    "Converged :",
    gmm.converged_
)

print(
    "Iterations:",
    gmm.n_iter_
)

print(
    "BIC       :",
    gmm.bic(
        X_train
    )
)


# ============================================================
# SAVE FITTED GMM
# ============================================================

GMM_FILE = os.path.join(
    GMM_MODEL_DIR,
    "best_gmm.joblib"
)

joblib.dump(
    gmm,
    GMM_FILE
)


# ============================================================
# OFFLINE KB CLUSTER MEMBERSHIP
#
# Each training fault is assigned to its GMM component using:
#
# gmm.predict(X_train)
#
# This assignment is performed OFFLINE only.
#
# Online query routing will NOT use GMM likelihood or
# posterior probability.
# ============================================================

labels = gmm.predict(
    X_train
).astype(np.int32)

LABEL_FILE = os.path.join(
    GMM_MODEL_DIR,
    "train_cluster_labels.npy"
)

np.save(
    LABEL_FILE,
    labels
)


# ============================================================
# CLUSTER -> TRAINING KB INDICES
# ============================================================

cluster_to_indices = {}

for cluster_id in range(K):

    cluster_to_indices[
        cluster_id
    ] = (
        np.where(
            labels == cluster_id
        )[0]
        .astype(np.int32)
    )


# ============================================================
# VERIFY NO EMPTY GMM COMPONENTS
# ============================================================

empty_clusters = [
    cluster_id
    for cluster_id in range(K)
    if len(
        cluster_to_indices[
            cluster_id
        ]
    ) == 0
]

if empty_clusters:

    raise RuntimeError(
        "Empty GMM clusters found: "
        + str(
            empty_clusters
        )
    )


# ============================================================
# SAVE CLUSTER LOOKUP
# ============================================================

LOOKUP_FILE = os.path.join(
    GMM_MODEL_DIR,
    "cluster_to_indices.joblib"
)

joblib.dump(
    cluster_to_indices,
    LOOKUP_FILE
)


# ============================================================
# SAVE RAW GMM COMPONENT MEANS
#
# Shape:
#
# (55, 384)
#
# These means remain in the original pretrained MiniLM
# representation space.
# ============================================================

gmm_means = gmm.means_.astype(
    np.float32
)

gmm_means = np.ascontiguousarray(
    gmm_means,
    dtype=np.float32
)

MEANS_FILE = os.path.join(
    GMM_MODEL_DIR,
    "gmm_means.npy"
)

np.save(
    MEANS_FILE,
    gmm_means
)


# ============================================================
# NORMALIZE GMM MEANS FOR ONLINE COSINE ROUTING
#
# During inference:
#
# normalized query
#       dot
# normalized GMM mean
#
# gives cosine similarity.
# ============================================================

mean_norms = np.linalg.norm(
    gmm_means,
    axis=1,
    keepdims=True
)

if np.any(
    mean_norms == 0
):

    raise ValueError(
        "A GMM component mean has zero norm."
    )

gmm_means_normalized = (
    gmm_means
    /
    np.clip(
        mean_norms,
        1e-12,
        None
    )
).astype(np.float32)

gmm_means_normalized = np.ascontiguousarray(
    gmm_means_normalized,
    dtype=np.float32
)

NORMALIZED_MEANS_FILE = os.path.join(
    GMM_MODEL_DIR,
    "gmm_means_normalized.npy"
)

np.save(
    NORMALIZED_MEANS_FILE,
    gmm_means_normalized
)


# ============================================================
# CLUSTER SIZE STATISTICS
# ============================================================

cluster_sizes = np.bincount(
    labels,
    minlength=K
)


# ============================================================
# VERIFY ALL TRAINING RECORDS ARE REPRESENTED
# ============================================================

total_cluster_records = sum(
    len(indices)
    for indices
    in cluster_to_indices.values()
)

if total_cluster_records != len(
    X_train
):

    raise ValueError(
        "Cluster lookup does not contain "
        "all training records."
    )


# ============================================================
# SAVE CONFIG
# ============================================================

config = {

    "method":
        "GMM + No CLIP",

    "embedding_model":
        "sentence-transformers/all-MiniLM-L6-v2",

    "uses_clip_representation":
        False,

    "uses_projection_head":
        False,

    "uses_gmm":
        True,

    "selected_k":
        K,

    "selection_criterion":
        (
            "fixed K=55 for consistency with "
            "the SCALA experimental configuration"
        ),

    "covariance_type":
        "diag",

    "random_state":
        42,

    "n_init":
        3,

    "max_iter":
        500,

    "reg_covar":
        1e-6,

    "embedding_dimension":
        int(
            X_train.shape[1]
        ),

    "training_records":
        int(
            X_train.shape[0]
        ),

    "gmm_fit_representation":
        (
            "raw pretrained MiniLM "
            "fault embeddings"
        ),

    "offline_cluster_assignment":
        "gmm.predict(X_train)",

    "online_routing":
        (
            "cosine similarity to L2-normalized "
            "GMM component means"
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
        )
}


CONFIG_FILE = os.path.join(
    GMM_MODEL_DIR,
    "gmm_config.json"
)

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

normalized_mean_norms = np.linalg.norm(
    gmm_means_normalized,
    axis=1
)

print("\n" + "=" * 80)
print("GMM + NO CLIP: K=55 GMM READY")
print("=" * 80)

print(
    "Representation              :",
    "Pretrained all-MiniLM-L6-v2"
)

print(
    "CLIP representation         :",
    "No"
)

print(
    "Projection head             :",
    "No"
)

print(
    "GMM                         :",
    "Yes"
)

print(
    "Training records            :",
    X_train.shape[0]
)

print(
    "Embedding dimension         :",
    X_train.shape[1]
)

print(
    "GMM means shape             :",
    gmm_means.shape
)

print(
    "Normalized means shape      :",
    gmm_means_normalized.shape
)

print(
    "Cluster labels shape        :",
    labels.shape
)

print(
    "Number of clusters          :",
    len(
        cluster_to_indices
    )
)

print(
    "Cluster size range          :",
    int(
        cluster_sizes.min()
    ),
    "-",
    int(
        cluster_sizes.max()
    )
)

print(
    "Mean cluster size           :",
    f"{cluster_sizes.mean():.2f}"
)

print(
    "Total KB records            :",
    int(
        cluster_sizes.sum()
    )
)

print(
    "Normalized mean norm range  :",
    f"{normalized_mean_norms.min():.6f}",
    "to",
    f"{normalized_mean_norms.max():.6f}"
)

print(
    "\nOnline inference will use cosine "
    "nearest-mean routing over 55 GMM means."
)


# ============================================================
# SAVED FILES
# ============================================================

print("\nSaved:")

print(
    GMM_FILE
)

print(
    LABEL_FILE
)

print(
    MEANS_FILE
)

print(
    NORMALIZED_MEANS_FILE
)

print(
    LOOKUP_FILE
)

print(
    CONFIG_FILE
)