import os
import time
import json
import random

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from sklearn.mixture import GaussianMixture


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = (
    r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC"
    r"\SCALA\synthetic-dataset"
)

KB_FILE = os.path.join(
    BASE_DIR,
    "knowledge-base.xlsx"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "synthetic_contrastive_model"
)

CHECKPOINT_FILE = os.path.join(
    MODEL_DIR,
    "best_contrastive_model.pt"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "synthetic_proposed_final"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

RAW_EMBEDDING_FILE = os.path.join(
    OUTPUT_DIR,
    "kb_fault_embeddings_raw.npy"
)

NORM_EMBEDDING_FILE = os.path.join(
    OUTPUT_DIR,
    "kb_fault_embeddings_normalized.npy"
)

METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "kb_embedding_metadata.csv"
)

GMM_FILE = os.path.join(
    OUTPUT_DIR,
    "gmm_K50_final.joblib"
)

CLUSTER_ASSIGNMENT_FILE = os.path.join(
    OUTPUT_DIR,
    "cluster_assignments.npy"
)

CLUSTER_LOOKUP_FILE = os.path.join(
    OUTPUT_DIR,
    "cluster_lookup.joblib"
)

SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "offline_summary.json"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "offline_summary.csv"
)


# ============================================================
# FROZEN EXPERIMENT SETTINGS
# ============================================================

SEED = 42

GMM_K = 50

GMM_COVARIANCE_TYPE = "diag"

GMM_MAX_ITER = 300

GMM_N_INIT = 1

GMM_REG_COVAR = 1e-6

BATCH_SIZE = 128

# Already measured from final synthetic training
CONTRASTIVE_TRAINING_TIME_SEC = 288.96958


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 90)
print("FINAL SYNTHETIC PROPOSED KB BUILD")
print("=" * 90)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# CHECK INPUT FILES
# ============================================================

for file_path in [
    KB_FILE,
    CHECKPOINT_FILE
]:

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            f"Missing required file:\n{file_path}"
        )


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "LOADING 10,500-RECORD KNOWLEDGE BASE"
)

print(
    "=" * 90
)


kb_df = pd.read_excel(
    KB_FILE
)


print(
    "KB records:",
    len(kb_df)
)

print(
    "Columns:",
    list(kb_df.columns)
)


required_columns = [
    "Fault Description",
    "Resolution"
]


for column in required_columns:

    if column not in kb_df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


if len(kb_df) != 10500:

    raise ValueError(
        f"Expected exactly 10500 records, "
        f"found {len(kb_df)}"
    )


if kb_df[
    required_columns
].isna().any().any():

    raise ValueError(
        "Missing Fault Description or Resolution values."
    )


kb_df = (
    kb_df
    .reset_index(drop=True)
)


fault_texts = (
    kb_df[
        "Fault Description"
    ]
    .astype(str)
    .tolist()
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "LOADING FROZEN EPOCH-9 CONTRASTIVE CHECKPOINT"
)

print(
    "=" * 90
)


checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE
)


print(
    "Epoch:",
    checkpoint["epoch"]
)

print(
    "Model:",
    checkpoint["model_name"]
)

print(
    "Projection dimension:",
    checkpoint["projection_dim"]
)

print(
    "Maximum sequence length:",
    checkpoint["max_length"]
)

print(
    "Stored temperature:",
    checkpoint["temperature"]
)


if int(
    checkpoint["epoch"]
) != 9:

    raise ValueError(
        "Expected frozen selected checkpoint epoch 9."
    )


MODEL_NAME = checkpoint[
    "model_name"
]

PROJECTED_DIM = int(
    checkpoint[
        "projection_dim"
    ]
)

MAX_LENGTH = int(
    checkpoint[
        "max_length"
    ]
)


# ============================================================
# PROJECTION HEAD
#
# Exact architecture from training:
#
# 384 -> 384 -> GELU -> 256
# ============================================================

class ProjectionHead(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        projection_dim
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
                projection_dim
            )
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# ============================================================
# FULL CONTRASTIVE MODEL
#
# Names MUST match checkpoint:
#
# encoder.*
# fault_projection.*
# resolution_projection.*
# logit_scale
# ============================================================

class ContrastiveModel(
    nn.Module
):

    def __init__(
        self,
        model_name,
        projection_dim
    ):

        super().__init__()


        self.encoder = (
            AutoModel
            .from_pretrained(
                model_name
            )
        )


        hidden_dim = (
            self.encoder
            .config
            .hidden_size
        )


        self.fault_projection = (
            ProjectionHead(
                hidden_dim,
                projection_dim
            )
        )


        self.resolution_projection = (
            ProjectionHead(
                hidden_dim,
                projection_dim
            )
        )


        # Scalar parameter.
        # Its initialization does not matter because the
        # checkpoint overwrites it immediately.
        self.logit_scale = nn.Parameter(
            torch.tensor(
                0.0
            )
        )


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)


# ============================================================
# RECONSTRUCT MODEL
# ============================================================

print(
    "\nReconstructing contrastive model..."
)


model = ContrastiveModel(

    MODEL_NAME,

    PROJECTED_DIM
)


# ============================================================
# LOAD COMPLETE TRAINED STATE
# ============================================================

load_result = model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ],

    strict=True
)


print(
    "Checkpoint loaded with strict=True."
)

print(
    "Missing keys:",
    load_result.missing_keys
)

print(
    "Unexpected keys:",
    load_result.unexpected_keys
)


model = model.to(
    DEVICE
)

model.eval()


BACKBONE_DIM = int(
    model.encoder
    .config
    .hidden_size
)


print(
    "Backbone dimension:",
    BACKBONE_DIM
)

print(
    "Projected dimension:",
    PROJECTED_DIM
)


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pooling(
    model_output,
    attention_mask
):

    token_embeddings = (
        model_output
        .last_hidden_state
    )


    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(
            token_embeddings.size()
        )
        .float()
    )


    summed_embeddings = torch.sum(

        token_embeddings
        *
        mask,

        dim=1
    )


    summed_mask = torch.clamp(

        mask.sum(
            dim=1
        ),

        min=1e-9
    )


    return (
        summed_embeddings
        /
        summed_mask
    )


# ============================================================
# ENCODE ALL 10,500 FAULTS
#
# IMPORTANT:
#
# We use ONLY:
#
#   fine-tuned encoder
#       +
#   learned fault projection
#
# Resolution projection is not needed for KB fault indexing.
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "ENCODING ALL 10,500 FAULT DESCRIPTIONS"
)

print(
    "=" * 90
)


embedding_batches = []


if torch.cuda.is_available():

    torch.cuda.synchronize()


embedding_start = (
    time.perf_counter()
)


with torch.no_grad():

    for start_index in range(
        0,
        len(fault_texts),
        BATCH_SIZE
    ):


        end_index = min(

            start_index
            +
            BATCH_SIZE,

            len(fault_texts)
        )


        batch_text = (
            fault_texts[
                start_index:end_index
            ]
        )


        encoded = tokenizer(

            batch_text,

            padding=True,

            truncation=True,

            max_length=MAX_LENGTH,

            return_tensors="pt"
        )


        encoded = {

            key:
                value.to(
                    DEVICE
                )

            for key, value
            in encoded.items()
        }


        encoder_output = (
            model.encoder(
                **encoded
            )
        )


        pooled = mean_pooling(

            encoder_output,

            encoded[
                "attention_mask"
            ]
        )


        raw_projection = (
            model
            .fault_projection(
                pooled
            )
        )


        embedding_batches.append(

            raw_projection
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )


        if (
            end_index % 1000 < BATCH_SIZE
            or
            end_index == 10500
        ):

            print(
                f"Encoded "
                f"{end_index}/10500"
            )


if torch.cuda.is_available():

    torch.cuda.synchronize()


embedding_time_sec = (

    time.perf_counter()
    -
    embedding_start
)


kb_raw = np.vstack(
    embedding_batches
)


print(
    "\nRaw embedding shape:",
    kb_raw.shape
)


# ============================================================
# NORMALIZE FOR COSINE RETRIEVAL
# ============================================================

norms = np.linalg.norm(

    kb_raw,

    axis=1,

    keepdims=True
)


norms = np.maximum(

    norms,

    1e-12
)


kb_normalized = (

    kb_raw
    /
    norms

).astype(
    np.float32
)


print(
    "Normalized embedding shape:",
    kb_normalized.shape
)


print(
    "10,500 KB embedding time:",
    f"{embedding_time_sec:.4f} sec"
)


# ============================================================
# SANITY CHECK EMBEDDINGS
# ============================================================

expected_shape = (

    10500,

    PROJECTED_DIM
)


if kb_raw.shape != expected_shape:

    raise ValueError(
        f"Expected embedding shape "
        f"{expected_shape}, "
        f"got {kb_raw.shape}"
    )


if np.isnan(
    kb_raw
).any():

    raise ValueError(
        "NaN detected in raw embeddings."
    )


if np.isnan(
    kb_normalized
).any():

    raise ValueError(
        "NaN detected in normalized embeddings."
    )


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(

    RAW_EMBEDDING_FILE,

    kb_raw
)


np.save(

    NORM_EMBEDDING_FILE,

    kb_normalized
)


print(
    "\nKB embeddings saved."
)


# ============================================================
# FIT FINAL GMM
#
# Frozen development choice:
#
# K = 50
# covariance = diag
#
# GMM uses RAW projected fault embeddings.
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FITTING FINAL GMM ON ALL 10,500 KB RECORDS"
)

print(
    "=" * 90
)


print(
    "K:",
    GMM_K
)

print(
    "Covariance:",
    GMM_COVARIANCE_TYPE
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
    kb_raw
)


gmm_fit_time_sec = (

    time.perf_counter()
    -
    gmm_start
)


print(
    "\nGMM fit time:",
    f"{gmm_fit_time_sec:.4f} sec"
)

print(
    "GMM converged:",
    gmm.converged_
)

print(
    "GMM iterations:",
    gmm.n_iter_
)


# ============================================================
# ASSIGN ALL KB RECORDS TO GMM CLUSTERS
# ============================================================

assignment_start = (
    time.perf_counter()
)


cluster_assignments = (
    gmm.predict(
        kb_raw
    )
)


cluster_assignment_time_sec = (

    time.perf_counter()
    -
    assignment_start
)


print(
    "Cluster assignment time:",
    f"{cluster_assignment_time_sec:.4f} sec"
)


# ============================================================
# PRECOMPUTE CLUSTER -> KB INDICES
#
# This is OFFLINE.
#
# During inference we simply fetch stored indices.
# ============================================================

cluster_lookup = {}


for cluster_id in range(
    GMM_K
):

    cluster_lookup[
        cluster_id
    ] = np.where(

        cluster_assignments
        ==
        cluster_id

    )[0].astype(
        np.int32
    )


cluster_sizes = np.array([

    len(
        cluster_lookup[
            cluster_id
        ]
    )

    for cluster_id
    in range(
        GMM_K
    )

])


print(
    "\nCluster statistics:"
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


# ============================================================
# SAVE FINAL GMM
# ============================================================

joblib.dump(

    gmm,

    GMM_FILE
)


# ============================================================
# SAVE CLUSTER ASSIGNMENTS
# ============================================================

np.save(

    CLUSTER_ASSIGNMENT_FILE,

    cluster_assignments
)


# ============================================================
# SAVE CLUSTER LOOKUP
# ============================================================

joblib.dump(

    cluster_lookup,

    CLUSTER_LOOKUP_FILE
)


# ============================================================
# SAVE STABLE KB METADATA
# ============================================================

metadata_df = pd.DataFrame({

    "kb_index":
        np.arange(
            len(kb_df)
        ),

    "fault_text":
        kb_df[
            "Fault Description"
        ].astype(str),

    "resolution_text":
        kb_df[
            "Resolution"
        ].astype(str),

    "gmm_cluster":
        cluster_assignments
})


metadata_df.to_csv(

    METADATA_FILE,

    index=False
)


# ============================================================
# OFFLINE COST
#
# Development K sweep and routing validation are NOT counted
# as deployment-time offline build.
# ============================================================

final_build_time_sec = (

    embedding_time_sec
    +
    gmm_fit_time_sec
    +
    cluster_assignment_time_sec
)


deployed_offline_cost_sec = (

    CONTRASTIVE_TRAINING_TIME_SEC
    +
    final_build_time_sec
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "kb_size":
        10500,

    "selected_epoch":
        int(
            checkpoint["epoch"]
        ),

    "model_name":
        MODEL_NAME,

    "backbone_dimension":
        BACKBONE_DIM,

    "projection_dimension":
        PROJECTED_DIM,

    "max_length":
        MAX_LENGTH,

    "batch_size":
        BATCH_SIZE,

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

    "contrastive_training_time_sec":
        CONTRASTIVE_TRAINING_TIME_SEC,

    "kb_embedding_time_sec":
        float(
            embedding_time_sec
        ),

    "gmm_fit_time_sec":
        float(
            gmm_fit_time_sec
        ),

    "cluster_assignment_time_sec":
        float(
            cluster_assignment_time_sec
        ),

    "final_kb_build_time_sec":
        float(
            final_build_time_sec
        ),

    "deployed_offline_cost_sec":
        float(
            deployed_offline_cost_sec
        ),

    "gmm_converged":
        bool(
            gmm.converged_
        ),

    "gmm_iterations":
        int(
            gmm.n_iter_
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
        )
}


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(

    SUMMARY_JSON,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        summary,

        f,

        indent=4
    )


pd.DataFrame(
    [summary]
).to_csv(

    SUMMARY_CSV,

    index=False
)


# ============================================================
# FINAL CHECKS
# ============================================================

assert kb_raw.shape == (
    10500,
    PROJECTED_DIM
)

assert kb_normalized.shape == (
    10500,
    PROJECTED_DIM
)

assert len(
    cluster_assignments
) == 10500

assert sum(

    len(indices)

    for indices
    in cluster_lookup.values()

) == 10500


# ============================================================
# FINAL PRINT
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FINAL PROPOSED SYNTHETIC KB BUILD COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nFrozen checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "KB records:",
    len(kb_df)
)

print(
    "Embedding dimension:",
    PROJECTED_DIM
)

print(
    "Final GMM K:",
    GMM_K
)


print(
    "\nOFFLINE COST"
)


print(
    "Contrastive training:",
    f"{CONTRASTIVE_TRAINING_TIME_SEC:.3f} sec"
)

print(
    "10,500 KB embedding:",
    f"{embedding_time_sec:.3f} sec"
)

print(
    "Final GMM fitting:",
    f"{gmm_fit_time_sec:.3f} sec"
)

print(
    "Cluster assignment:",
    f"{cluster_assignment_time_sec:.3f} sec"
)

print(
    "Final KB build:",
    f"{final_build_time_sec:.3f} sec"
)

print(
    "Total deployed offline cost:",
    f"{deployed_offline_cost_sec:.3f} sec"
)


print(
    "\nOUTPUT FILES"
)

print(
    RAW_EMBEDDING_FILE
)

print(
    NORM_EMBEDDING_FILE
)

print(
    GMM_FILE
)

print(
    CLUSTER_ASSIGNMENT_FILE
)

print(
    CLUSTER_LOOKUP_FILE
)

print(
    METADATA_FILE
)

print(
    SUMMARY_JSON
)