from pathlib import Path
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "train_fault_resolution.xlsx"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "contrastive_minilm"
)

BACKBONE_DIR = os.path.join(
    MODEL_DIR,
    "backbone"
)

PROJECTION_FILE = os.path.join(
    MODEL_DIR,
    "projection_heads.pt"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

os.makedirs(
    EMBEDDING_DIR,
    exist_ok=True
)

RAW_OUTPUT_FILE = os.path.join(
    EMBEDDING_DIR,
    "train_fault_embeddings_raw.npy"
)

NORMALIZED_OUTPUT_FILE = os.path.join(
    EMBEDDING_DIR,
    "train_fault_embeddings_normalized.npy"
)

BATCH_SIZE = 64


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    TRAIN_FILE,
    PROJECTION_FILE
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

if not os.path.isdir(BACKBONE_DIR):

    raise FileNotFoundError(
        f"Backbone directory not found:\n{BACKBONE_DIR}"
    )


# ============================================================
# LOAD TRAINING KB
# ============================================================

print("\n" + "=" * 80)
print("LOADING TRAINING KB")
print("=" * 80)

train_df = pd.read_excel(
    TRAIN_FILE
)

required_columns = [
    "id",
    "fault_text",
    "resolution_text",
    "source_dataset"
]

for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )

print(
    "Training KB records:",
    len(train_df)
)


# ============================================================
# LOAD CONTRASTIVE CHECKPOINT
# ============================================================

print("\n" + "=" * 80)
print("LOADING CONTRASTIVE MODEL")
print("=" * 80)

checkpoint = torch.load(
    PROJECTION_FILE,
    map_location=DEVICE
)

BACKBONE_DIM = int(
    checkpoint["backbone_dim"]
)

PROJECTION_DIM = int(
    checkpoint["projection_dim"]
)

MAX_LENGTH = int(
    checkpoint["max_length"]
)

print(
    "Backbone dimension :",
    BACKBONE_DIM
)

print(
    "Projection dimension:",
    PROJECTION_DIM
)

print(
    "Max sequence length :",
    MAX_LENGTH
)


# ============================================================
# IMPORTANT CHECK
#
# For the new experiment we expect:
#
# MiniLM output      = 384
# learned projection = 384
# ============================================================

if BACKBONE_DIM != 384:

    raise ValueError(
        "Expected backbone dimension 384, "
        f"but found {BACKBONE_DIM}."
    )

if PROJECTION_DIM != 384:

    raise ValueError(
        "Expected projection dimension 384, "
        f"but found {PROJECTION_DIM}. "
        "Check that the new contrastive model was loaded."
    )


# ============================================================
# LOAD TOKENIZER + FINE-TUNED BACKBONE
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    BACKBONE_DIR
)

backbone = AutoModel.from_pretrained(
    BACKBONE_DIR
)

backbone = backbone.to(
    DEVICE
)

backbone.eval()


# ============================================================
# LOAD LEARNED FAULT PROJECTION HEAD
#
# IMPORTANT:
# We use the FAULT projection because the KB representation
# used by SCALA is based on fault_text.
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

fault_projection = fault_projection.to(
    DEVICE
)

fault_projection.eval()


# ============================================================
# MEAN POOLING
#
# This is identical to the pooling used during contrastive
# training.
# ============================================================

def mean_pooling(
    token_embeddings,
    attention_mask
):

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
# FAULT TEXTS
# ============================================================

fault_texts = (
    train_df["fault_text"]
    .fillna("")
    .astype(str)
    .tolist()
)


# ============================================================
# GENERATE EMBEDDINGS
#
# fault_text
#     -> fine-tuned MiniLM
#     -> mean pooling
#     -> learned FAULT projection
#
# We save:
#
# 1. raw projected embeddings
# 2. L2-normalized projected embeddings
# ============================================================

print("\n" + "=" * 80)
print("GENERATING TRAINING FAULT EMBEDDINGS")
print("=" * 80)

all_raw_embeddings = []
all_normalized_embeddings = []


with torch.no_grad():

    for start in tqdm(
        range(
            0,
            len(fault_texts),
            BATCH_SIZE
        ),
        desc="Encoding KB faults"
    ):

        batch_texts = fault_texts[
            start:
            start + BATCH_SIZE
        ]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        tokens = {
            key: value.to(DEVICE)
            for key, value
            in tokens.items()
        }

        outputs = backbone(
            input_ids=tokens["input_ids"],
            attention_mask=tokens[
                "attention_mask"
            ]
        )

        pooled = mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"]
        )

        raw_projected = fault_projection(
            pooled
        )

        normalized = F.normalize(
            raw_projected,
            p=2,
            dim=1
        )

        all_raw_embeddings.append(
            raw_projected
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        all_normalized_embeddings.append(
            normalized
            .cpu()
            .numpy()
            .astype(np.float32)
        )


# ============================================================
# CONCATENATE
# ============================================================

train_raw = np.concatenate(
    all_raw_embeddings,
    axis=0
)

train_normalized = np.concatenate(
    all_normalized_embeddings,
    axis=0
)

train_raw = np.ascontiguousarray(
    train_raw,
    dtype=np.float32
)

train_normalized = np.ascontiguousarray(
    train_normalized,
    dtype=np.float32
)


# ============================================================
# SANITY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING EMBEDDINGS")
print("=" * 80)

print(
    "Raw embedding shape       :",
    train_raw.shape
)

print(
    "Normalized embedding shape:",
    train_normalized.shape
)


expected_shape = (
    len(train_df),
    PROJECTION_DIM
)

if train_raw.shape != expected_shape:

    raise ValueError(
        "Unexpected raw embedding shape. "
        f"Expected {expected_shape}, "
        f"got {train_raw.shape}."
    )

if train_normalized.shape != expected_shape:

    raise ValueError(
        "Unexpected normalized embedding shape. "
        f"Expected {expected_shape}, "
        f"got {train_normalized.shape}."
    )


# ============================================================
# CHECK NORMALIZATION
# ============================================================

norms = np.linalg.norm(
    train_normalized,
    axis=1
)

print(
    "Normalized vector norm range:",
    norms.min(),
    "to",
    norms.max()
)

if not np.allclose(
    norms,
    1.0,
    atol=1e-5
):

    raise ValueError(
        "Normalized embeddings are not unit length."
    )


# ============================================================
# CHECK FINITE VALUES
# ============================================================

if not np.all(
    np.isfinite(train_raw)
):

    raise ValueError(
        "Raw embeddings contain NaN or Inf."
    )

if not np.all(
    np.isfinite(train_normalized)
):

    raise ValueError(
        "Normalized embeddings contain NaN or Inf."
    )


# ============================================================
# SAVE
# ============================================================

np.save(
    RAW_OUTPUT_FILE,
    train_raw
)

np.save(
    NORMALIZED_OUTPUT_FILE,
    train_normalized
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EMBEDDING GENERATION COMPLETE")
print("=" * 80)

print(
    "\nNumber of KB records:",
    len(train_df)
)

print(
    "Embedding dimension:",
    PROJECTION_DIM
)

print(
    "\nRaw embeddings saved to:"
)

print(
    RAW_OUTPUT_FILE
)

print(
    "\nNormalized embeddings saved to:"
)

print(
    NORMALIZED_OUTPUT_FILE
)