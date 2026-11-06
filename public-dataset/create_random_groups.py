from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "train_fault_resolution.xlsx"
)

RAW_EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "train_fault_embeddings_raw.npy"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "random_groups"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "random_groups"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

NUM_GROUPS = 55
RANDOM_STATE = 42


# ============================================================
# LOAD TRAINING DATA AND LEARNED EMBEDDINGS
# ============================================================

print("=" * 80)
print("LOADING TRAINING DATA AND LEARNED EMBEDDINGS")
print("=" * 80)

train_df = pd.read_excel(
    TRAIN_FILE
)

train_raw = np.load(
    RAW_EMBEDDING_FILE
).astype(np.float32)

train_raw = np.ascontiguousarray(
    train_raw,
    dtype=np.float32
)

N_TRAIN = len(train_df)

print(
    "Training records :",
    N_TRAIN
)

print(
    "Embedding shape  :",
    train_raw.shape
)

if len(train_raw) != N_TRAIN:

    raise ValueError(
        "Training dataframe and embedding count do not match."
    )


# ============================================================
# RANDOM GROUP ASSIGNMENT
#
# Each training record is independently assigned to one of
# the 55 groups.
#
# No balancing is performed.
#
# This removes GMM-based semantic organization while keeping
# the learned contrastive representation unchanged.
# ============================================================

print("\n" + "=" * 80)
print("CREATING RANDOM GROUPS")
print("=" * 80)

rng = np.random.default_rng(
    RANDOM_STATE
)

random_labels = rng.integers(
    low=0,
    high=NUM_GROUPS,
    size=N_TRAIN
).astype(np.int32)


# ============================================================
# CHECK GROUP SIZES
# ============================================================

group_sizes = np.bincount(
    random_labels,
    minlength=NUM_GROUPS
)

print(
    "Number of groups :",
    NUM_GROUPS
)

print(
    "Smallest group   :",
    int(group_sizes.min())
)

print(
    "Largest group    :",
    int(group_sizes.max())
)

print(
    "Mean group size  :",
    float(group_sizes.mean())
)

empty_groups = np.where(
    group_sizes == 0
)[0]

if len(empty_groups) > 0:

    raise RuntimeError(
        "Empty random groups found: "
        + str(
            empty_groups.tolist()
        )
    )


# ============================================================
# COMPUTE RANDOM-GROUP MEANS
#
# The means are computed from the RAW learned embeddings.
#
# This parallels the offline organization used for SCALA:
#
# learned raw embeddings
#       ->
# group organization
#       ->
# group means
#
# Here the organization is random rather than GMM-based.
# ============================================================

print("\n" + "=" * 80)
print("COMPUTING RANDOM-GROUP MEANS")
print("=" * 80)

random_means = np.zeros(
    (
        NUM_GROUPS,
        train_raw.shape[1]
    ),
    dtype=np.float32
)

for group_id in range(
    NUM_GROUPS
):

    indices = np.where(
        random_labels == group_id
    )[0]

    random_means[group_id] = (
        train_raw[
            indices
        ].mean(
            axis=0
        )
    )

random_means = np.ascontiguousarray(
    random_means,
    dtype=np.float32
)

print(
    "Raw random means shape:",
    random_means.shape
)


# ============================================================
# NORMALIZE RANDOM-GROUP MEANS
#
# Online routing will use cosine similarity between:
#
# normalized query embedding
#             and
# normalized random-group means
#
# This keeps the routing metric consistent with final SCALA.
# ============================================================

print("\n" + "=" * 80)
print("NORMALIZING RANDOM-GROUP MEANS")
print("=" * 80)

mean_norms = np.linalg.norm(
    random_means,
    axis=1,
    keepdims=True
)

random_means_normalized = (
    random_means
    /
    np.clip(
        mean_norms,
        1e-12,
        None
    )
).astype(np.float32)

random_means_normalized = np.ascontiguousarray(
    random_means_normalized,
    dtype=np.float32
)

print(
    "Normalized random means shape:",
    random_means_normalized.shape
)


# ============================================================
# SANITY CHECK NORMALIZATION
# ============================================================

normalized_norms = np.linalg.norm(
    random_means_normalized,
    axis=1
)

print(
    "Normalized mean norm range:",
    f"{normalized_norms.min():.6f}",
    "to",
    f"{normalized_norms.max():.6f}"
)


# ============================================================
# CREATE GROUP -> TRAINING INDICES LOOKUP
# ============================================================

group_to_indices = {}

for group_id in range(
    NUM_GROUPS
):

    indices = np.where(
        random_labels == group_id
    )[0]

    group_to_indices[
        int(group_id)
    ] = indices.astype(
        np.int32
    )


# ============================================================
# SANITY CHECK LOOKUP
# ============================================================

total_lookup_records = sum(
    len(indices)
    for indices
    in group_to_indices.values()
)

if total_lookup_records != N_TRAIN:

    raise ValueError(
        "Random-group lookup does not contain all training records."
    )

print(
    "Records in group lookup:",
    total_lookup_records
)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

LABEL_FILE = os.path.join(
    MODEL_DIR,
    "random_group_labels.npy"
)

MEAN_FILE = os.path.join(
    MODEL_DIR,
    "random_group_means.npy"
)

NORMALIZED_MEAN_FILE = os.path.join(
    MODEL_DIR,
    "random_group_means_normalized.npy"
)

LOOKUP_FILE = os.path.join(
    MODEL_DIR,
    "random_group_to_indices.joblib"
)


np.save(
    LABEL_FILE,
    random_labels
)

np.save(
    MEAN_FILE,
    random_means
)

np.save(
    NORMALIZED_MEAN_FILE,
    random_means_normalized
)

joblib.dump(
    group_to_indices,
    LOOKUP_FILE
)


# ============================================================
# SAVE TRAINING KB WITH RANDOM GROUP
# ============================================================

training_kb = train_df.copy()

training_kb[
    "random_group"
] = random_labels

KB_FILE = os.path.join(
    RESULT_DIR,
    "training_kb_with_random_groups.xlsx"
)

training_kb.to_excel(
    KB_FILE,
    index=False
)


# ============================================================
# SAVE GROUP SUMMARY
# ============================================================

group_summary = pd.DataFrame({

    "random_group":
        np.arange(
            NUM_GROUPS
        ),

    "records":
        group_sizes
})

GROUP_SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "random_group_summary.xlsx"
)

group_summary.to_excel(
    GROUP_SUMMARY_FILE,
    index=False
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {

    "num_groups":
        NUM_GROUPS,

    "random_state":
        RANDOM_STATE,

    "training_records":
        N_TRAIN,

    "embedding_dimension":
        int(
            train_raw.shape[1]
        ),

    "assignment":
        "independent uniform random assignment",

    "balanced":
        False,

    "smallest_group":
        int(
            group_sizes.min()
        ),

    "largest_group":
        int(
            group_sizes.max()
        ),

    "mean_group_size":
        float(
            group_sizes.mean()
        ),

    "group_mean_source":
        "raw learned fault embeddings",

    "online_routing":
        (
            "cosine similarity to L2-normalized "
            "random-group means"
        )
}

CONFIG_FILE = os.path.join(
    MODEL_DIR,
    "random_group_config.json"
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
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("RANDOM GROUP ORGANIZATION COMPLETE")
print("=" * 80)

print(
    "Training records :",
    N_TRAIN
)

print(
    "Number of groups :",
    NUM_GROUPS
)

print(
    "Embedding dim    :",
    train_raw.shape[1]
)

print(
    "Smallest group   :",
    int(
        group_sizes.min()
    )
)

print(
    "Largest group    :",
    int(
        group_sizes.max()
    )
)

print(
    "Mean group size  :",
    f"{group_sizes.mean():.2f}"
)

print(
    "Routing          :",
    "cosine similarity to normalized random-group means"
)


print("\nSaved:")

print(
    LABEL_FILE
)

print(
    MEAN_FILE
)

print(
    NORMALIZED_MEAN_FILE
)

print(
    LOOKUP_FILE
)

print(
    KB_FILE
)

print(
    GROUP_SUMMARY_FILE
)

print(
    CONFIG_FILE
)