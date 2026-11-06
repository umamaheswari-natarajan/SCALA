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

EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "train_fault_embeddings_raw.npy"
)

GMM_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gmm"
)

os.makedirs(
    GMM_MODEL_DIR,
    exist_ok=True
)

K = 55


# ============================================================
# LOAD RAW 384-D EMBEDDINGS
# ============================================================

X_train = np.load(
    EMBEDDING_FILE
).astype(np.float32)

print("Training embeddings:", X_train.shape)

if X_train.shape != (3049, 384):
    raise ValueError(
        f"Unexpected embedding shape: {X_train.shape}"
    )


# ============================================================
# FIT FINAL K=55 GMM
# ============================================================

print("\nFitting final GMM with K=55...")

gmm = GaussianMixture(
    n_components=K,
    covariance_type="diag",
    random_state=42,
    n_init=3,
    max_iter=500,
    reg_covar=1e-6
)

gmm.fit(X_train)

print("Converged :", gmm.converged_)
print("Iterations:", gmm.n_iter_)
print("BIC       :", gmm.bic(X_train))


# ============================================================
# SAVE FINAL GMM
# ============================================================

joblib.dump(
    gmm,
    os.path.join(
        GMM_MODEL_DIR,
        "best_gmm.joblib"
    )
)


# ============================================================
# OFFLINE KB CLUSTER MEMBERSHIP
# ============================================================

labels = gmm.predict(
    X_train
).astype(np.int32)

np.save(
    os.path.join(
        GMM_MODEL_DIR,
        "train_cluster_labels.npy"
    ),
    labels
)


# ============================================================
# CLUSTER -> KB INDICES
# ============================================================

cluster_to_indices = {}

for cluster_id in range(K):

    cluster_to_indices[cluster_id] = (
        np.where(
            labels == cluster_id
        )[0]
        .astype(np.int32)
    )

joblib.dump(
    cluster_to_indices,
    os.path.join(
        GMM_MODEL_DIR,
        "cluster_to_indices.joblib"
    )
)


# ============================================================
# SAVE RAW GMM MEANS
# ============================================================

gmm_means = gmm.means_.astype(
    np.float32
)

np.save(
    os.path.join(
        GMM_MODEL_DIR,
        "gmm_means.npy"
    ),
    gmm_means
)


# ============================================================
# NORMALIZE GMM MEANS FOR COSINE ROUTING
# ============================================================

mean_norms = np.linalg.norm(
    gmm_means,
    axis=1,
    keepdims=True
)

if np.any(mean_norms == 0):
    raise ValueError(
        "A GMM component mean has zero norm."
    )

gmm_means_normalized = (
    gmm_means / mean_norms
).astype(np.float32)

np.save(
    os.path.join(
        GMM_MODEL_DIR,
        "gmm_means_normalized.npy"
    ),
    gmm_means_normalized
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "selected_k": 55,
    "selection_criterion": (
        "fixed K=55 for consistency with "
        "the SCALA experimental configuration"
    ),
    "covariance_type": "diag",
    "embedding_dimension": 384,
    "training_records": 3049,
    "online_routing": (
        "cosine similarity to L2-normalized "
        "GMM component means"
    )
}

with open(
    os.path.join(
        GMM_MODEL_DIR,
        "gmm_config.json"
    ),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# VERIFY
# ============================================================

cluster_sizes = np.bincount(
    labels,
    minlength=K
)

print("\n" + "=" * 80)
print("FINAL K=55 GMM READY")
print("=" * 80)

print(
    "GMM means shape            :",
    gmm_means.shape
)

print(
    "Normalized means shape     :",
    gmm_means_normalized.shape
)

print(
    "Cluster labels shape       :",
    labels.shape
)

print(
    "Number of clusters         :",
    len(cluster_to_indices)
)

print(
    "Cluster size range         :",
    cluster_sizes.min(),
    "-",
    cluster_sizes.max()
)

print(
    "Total KB records           :",
    cluster_sizes.sum()
)

print(
    "Normalized mean norm range :",
    np.linalg.norm(
        gmm_means_normalized,
        axis=1
    ).min(),
    "to",
    np.linalg.norm(
        gmm_means_normalized,
        axis=1
    ).max()
)

print(
    "\nOnline inference will use cosine "
    "nearest-mean routing over 55 GMM means."
)