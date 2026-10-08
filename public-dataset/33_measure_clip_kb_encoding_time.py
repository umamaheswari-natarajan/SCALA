import os
import time
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "train_fault_resolution.xlsx"
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

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "offline_timing"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "clip_training_kb_encoding_time.json"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 64

# Warm-up is NOT included in the reported timing.
WARMUP_BATCHES = 2

# We repeat the full 3049-fault encoding 3 times.
NUM_RUNS = 3


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 90)
print("DEVICE")
print("=" * 90)

print(
    "Using:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD CONTRASTIVE CHECKPOINT
# ============================================================

print("\n" + "=" * 90)
print("LOADING TRAINED CLIP / CONTRASTIVE MODEL")
print("=" * 90)

checkpoint = torch.load(
    PROJECTION_FILE,
    map_location="cpu"
)

BACKBONE_DIM = int(
    checkpoint[
        "backbone_dim"
    ]
)

PROJECTION_DIM = int(
    checkpoint[
        "projection_dim"
    ]
)

MAX_LENGTH = int(
    checkpoint[
        "max_length"
    ]
)

BEST_EPOCH = int(
    checkpoint[
        "epoch"
    ]
)

print(
    "Best epoch:",
    BEST_EPOCH
)

print(
    "Backbone dimension:",
    BACKBONE_DIM
)

print(
    "Projection dimension:",
    PROJECTION_DIM
)

print(
    "Maximum token length:",
    MAX_LENGTH
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
# LOAD FAULT PROJECTION HEAD
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
# LOAD TRAINING DATA
# ============================================================

train_df = pd.read_excel(
    TRAIN_FILE
)

if "fault_text" not in train_df.columns:

    raise ValueError(
        "fault_text column is missing."
    )

train_df[
    "fault_text"
] = (
    train_df[
        "fault_text"
    ]
    .fillna("")
    .astype(str)
)

print(
    "\nTraining faults:",
    len(train_df)
)


# ============================================================
# DATASET
# ============================================================

class FaultDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

    def __len__(
        self
    ):

        return len(
            self.df
        )

    def __getitem__(
        self,
        idx
    ):

        return str(
            self.df
            .iloc[idx][
                "fault_text"
            ]
        )


# ============================================================
# COLLATE
# ============================================================

def collate_fn(
    texts
):

    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return tokens


loader = DataLoader(
    FaultDataset(
        train_df
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# MEAN POOLING
#
# Same logic as training/GMM construction code.
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
        token_embeddings
        *
        mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(
            dim=1
        ),
        min=1e-9
    )

    return (
        summed
        /
        counts
    )


# ============================================================
# ENCODE ONE BATCH
# ============================================================

@torch.no_grad()
def encode_batch(
    tokens
):

    tokens = {
        key:
            value.to(
                DEVICE
            )
        for key, value
        in tokens.items()
    }

    outputs = backbone(
        input_ids=
            tokens[
                "input_ids"
            ],

        attention_mask=
            tokens[
                "attention_mask"
            ]
    )

    h = mean_pooling(
        outputs.last_hidden_state,
        tokens[
            "attention_mask"
        ]
    )

    raw = fault_projection(
        h
    )

    normalized = F.normalize(
        raw,
        p=2,
        dim=1
    )

    return (
        raw,
        normalized
    )


# ============================================================
# GPU SYNCHRONIZATION
# ============================================================

def sync_device():

    if torch.cuda.is_available():

        torch.cuda.synchronize()


# ============================================================
# WARM-UP
#
# Not counted in timing.
# ============================================================

print("\n" + "=" * 90)
print("WARM-UP")
print("=" * 90)

for batch_index, tokens in enumerate(
    loader
):

    encode_batch(
        tokens
    )

    if (
        batch_index
        +
        1
        >=
        WARMUP_BATCHES
    ):

        break


sync_device()

print(
    "Warm-up complete."
)


# ============================================================
# FULL ENCODING RUNS
# ============================================================

print("\n" + "=" * 90)
print("MEASURING TRAINING-KB CLIP ENCODING")
print("=" * 90)

run_times = []


for run in range(
    1,
    NUM_RUNS + 1
):

    print(
        f"\nRun {run}/{NUM_RUNS}"
    )

    raw_embeddings = []
    normalized_embeddings = []

    sync_device()

    start = time.perf_counter()


    for tokens in tqdm(
        loader,
        desc=f"Encoding run {run}"
    ):

        raw, normalized = (
            encode_batch(
                tokens
            )
        )

        raw_embeddings.append(
            raw.cpu().numpy()
        )

        normalized_embeddings.append(
            normalized.cpu().numpy()
        )


    sync_device()


    elapsed = (
        time.perf_counter()
        -
        start
    )


    raw_embeddings = np.vstack(
        raw_embeddings
    )

    normalized_embeddings = np.vstack(
        normalized_embeddings
    )


    if (
        len(
            raw_embeddings
        )
        !=
        len(
            train_df
        )
    ):

        raise ValueError(
            "Encoded row count does not match "
            "training-data row count."
        )


    run_times.append(
        elapsed
    )


    print(
        f"Run {run} time:",
        f"{elapsed:.4f} sec"
    )


# ============================================================
# SUMMARY
# ============================================================

mean_time = float(
    np.mean(
        run_times
    )
)

median_time = float(
    np.median(
        run_times
    )
)

std_time = float(
    np.std(
        run_times
    )
)

min_time = float(
    np.min(
        run_times
    )
)

max_time = float(
    np.max(
        run_times
    )
)


summary = {

    "measurement":
        "CLIP training-KB fault embedding generation",

    "training_records":
        int(
            len(
                train_df
            )
        ),

    "batch_size":
        int(
            BATCH_SIZE
        ),

    "num_runs":
        int(
            NUM_RUNS
        ),

    "warmup_batches":
        int(
            WARMUP_BATCHES
        ),

    "device":
        str(
            DEVICE
        ),

    "backbone_dimension":
        int(
            BACKBONE_DIM
        ),

    "projection_dimension":
        int(
            PROJECTION_DIM
        ),

    "run_times_seconds":
        [
            float(x)
            for x in run_times
        ],

    "mean_time_seconds":
        mean_time,

    "median_time_seconds":
        median_time,

    "std_time_seconds":
        std_time,

    "min_time_seconds":
        min_time,

    "max_time_seconds":
        max_time
}


# ============================================================
# SAVE
# ============================================================

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
print("CLIP TRAINING-KB ENCODING TIME")
print("=" * 90)

print(
    "Training faults:",
    len(
        train_df
    )
)

print(
    "Runs:",
    NUM_RUNS
)

print(
    "\nIndividual runs:"
)

for i, value in enumerate(
    run_times,
    start=1
):

    print(
        f"Run {i}:",
        f"{value:.4f} sec"
    )


print(
    "\nMean:",
    f"{mean_time:.4f} sec"
)

print(
    "Median:",
    f"{median_time:.4f} sec"
)

print(
    "Std:",
    f"{std_time:.4f} sec"
)

print(
    "\nSaved:"
)

print(
    SUMMARY_FILE
)