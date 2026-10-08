import os
import re
import json
import math
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Sampler

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "train_fault_resolution.xlsx"
)

VAL_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "val_fault_resolution.xlsx"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "contrastive_minilm"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "contrastive_training"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# MiniLM sentence embedding dimension
BACKBONE_DIM = 384

# Learned fault-resolution joint space
PROJECTION_DIM = 256

MAX_LENGTH = 256


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 10

# Small learning rate for pretrained MiniLM
BACKBONE_LR = 2e-5

# Larger LR for newly initialized projection heads
HEAD_LR = 1e-3

WEIGHT_DECAY = 1e-4

# CLIP commonly starts near temperature 0.07
INITIAL_TEMPERATURE = 0.07

RANDOM_SEED = 42

# Early stopping
PATIENCE = 3


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


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
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TEXT NORMALIZATION
#
# Used only for detecting identical texts in a minibatch
# and for validation metrics.
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_excel(TRAIN_FILE)
val_df = pd.read_excel(VAL_FILE)


print("\n" + "=" * 80)
print("DATA")
print("=" * 80)

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))


required_columns = [
    "id",
    "fault_text",
    "resolution_text",
    "source_dataset"
]

for col in required_columns:

    if col not in train_df.columns:
        raise ValueError(
            f"Missing train column: {col}"
        )

    if col not in val_df.columns:
        raise ValueError(
            f"Missing validation column: {col}"
        )


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer/model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATASET
# ============================================================

class FaultResolutionDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        return {
            "index": index,
            "id": str(row["id"]),
            "fault_text": str(row["fault_text"]),
            "resolution_text": str(
                row["resolution_text"]
            )
        }


train_dataset = FaultResolutionDataset(train_df)
val_dataset = FaultResolutionDataset(val_df)


# ============================================================
# NO-DUPLICATE BATCH SAMPLER
#
# Why?
#
# Some faults can share the exact same resolution.
#
# Standard CLIP assumes:
#
#    pair 1 -> positive
#    every other pair in batch -> negative
#
# If an identical resolution occurs twice in one batch,
# CLIP would incorrectly treat one identical resolution
# as a negative.
#
# This sampler tries to ensure that exact fault/resolution
# texts do not repeat within the same minibatch.
# ============================================================

class NoDuplicateBatchSampler(Sampler):

    def __init__(
        self,
        dataframe,
        batch_size,
        shuffle=True
    ):

        self.df = dataframe.reset_index(drop=True)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.fault_norm = [
            normalize_text(x)
            for x in self.df["fault_text"]
        ]

        self.res_norm = [
            normalize_text(x)
            for x in self.df["resolution_text"]
        ]

    def __iter__(self):

        indices = list(range(len(self.df)))

        if self.shuffle:
            random.shuffle(indices)

        remaining = indices.copy()

        while len(remaining) > 0:

            batch = []

            batch_faults = set()
            batch_resolutions = set()

            next_remaining = []

            for idx in remaining:

                f = self.fault_norm[idx]
                r = self.res_norm[idx]

                if (
                    len(batch) < self.batch_size
                    and f not in batch_faults
                    and r not in batch_resolutions
                ):

                    batch.append(idx)

                    batch_faults.add(f)
                    batch_resolutions.add(r)

                else:

                    next_remaining.append(idx)

            if len(batch) > 0:
                yield batch

            remaining = next_remaining

    def __len__(self):

        return math.ceil(
            len(self.df) / self.batch_size
        )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    fault_texts = [
        item["fault_text"]
        for item in batch
    ]

    resolution_texts = [
        item["resolution_text"]
        for item in batch
    ]

    ids = [
        item["id"]
        for item in batch
    ]


    fault_tokens = tokenizer(
        fault_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    resolution_tokens = tokenizer(
        resolution_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return {
        "ids": ids,
        "fault_texts": fault_texts,
        "resolution_texts": resolution_texts,
        "fault_tokens": fault_tokens,
        "resolution_tokens": resolution_tokens
    }


# ============================================================
# DATALOADERS
# ============================================================

train_batch_sampler = NoDuplicateBatchSampler(
    train_df,
    BATCH_SIZE,
    shuffle=True
)


train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=collate_fn
)


# Validation does not require special batches because
# evaluation compares the whole validation set.

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# MODEL
# ============================================================

class FaultResolutionContrastiveModel(nn.Module):

    def __init__(
        self,
        model_name,
        backbone_dim,
        projection_dim,
        initial_temperature
    ):

        super().__init__()


        # ----------------------------------------------------
        # SHARED MiniLM
        #
        # SAME encoder is used for:
        #
        # fault text
        # resolution text
        #
        # Its weights ARE updated during training.
        # ----------------------------------------------------

        self.backbone = AutoModel.from_pretrained(
            model_name
        )


        # ----------------------------------------------------
        # SEPARATE projection heads
        #
        # fault representation:
        #     384 -> 256
        #
        # resolution representation:
        #     384 -> 256
        # ----------------------------------------------------

        self.fault_projection = nn.Linear(
            backbone_dim,
            projection_dim,
            bias=False
        )

        self.resolution_projection = nn.Linear(
            backbone_dim,
            projection_dim,
            bias=False
        )


        # ----------------------------------------------------
        # Learnable CLIP temperature / logit scale
        #
        # scale = exp(logit_scale)
        # ----------------------------------------------------

        self.logit_scale = nn.Parameter(
            torch.tensor(
                math.log(
                    1.0 / initial_temperature
                ),
                dtype=torch.float
            )
        )


    # --------------------------------------------------------
    # MEAN POOLING
    #
    # Transformer gives one vector per token.
    #
    # We combine them into one sentence vector.
    # --------------------------------------------------------

    def mean_pooling(
        self,
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


    # --------------------------------------------------------
    # SHARED TEXT ENCODER
    # --------------------------------------------------------

    def encode_backbone(self, tokens):

        outputs = self.backbone(
            input_ids=tokens["input_ids"],
            attention_mask=tokens[
                "attention_mask"
            ]
        )

        sentence_embedding = self.mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"]
        )

        return sentence_embedding


    # --------------------------------------------------------
    # FAULT ENCODER
    #
    # Returns:
    #
    # raw_projected = u_F
    #
    # normalized    = z_F
    # --------------------------------------------------------

    def encode_fault(self, tokens):

        h = self.encode_backbone(tokens)

        raw_projected = self.fault_projection(h)

        normalized = F.normalize(
            raw_projected,
            p=2,
            dim=1
        )

        return raw_projected, normalized


    # --------------------------------------------------------
    # RESOLUTION ENCODER
    # --------------------------------------------------------

    def encode_resolution(self, tokens):

        h = self.encode_backbone(tokens)

        raw_projected = (
            self.resolution_projection(h)
        )

        normalized = F.normalize(
            raw_projected,
            p=2,
            dim=1
        )

        return raw_projected, normalized


    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(
        self,
        fault_tokens,
        resolution_tokens
    ):

        fault_raw, fault_norm = (
            self.encode_fault(
                fault_tokens
            )
        )

        resolution_raw, resolution_norm = (
            self.encode_resolution(
                resolution_tokens
            )
        )


        # Prevent temperature from becoming extreme
        with torch.no_grad():

            self.logit_scale.clamp_(
                max=math.log(100.0)
            )


        scale = self.logit_scale.exp()


        # ----------------------------------------------------
        # CLIP similarity matrix
        #
        # If batch size = 32:
        #
        # logits shape = [32, 32]
        #
        # row i:
        # fault_i vs every resolution
        # ----------------------------------------------------

        logits = (
            scale
            *
            torch.matmul(
                fault_norm,
                resolution_norm.T
            )
        )


        return {
            "logits": logits,
            "fault_raw": fault_raw,
            "fault_norm": fault_norm,
            "resolution_raw": resolution_raw,
            "resolution_norm": resolution_norm
        }


# ============================================================
# CREATE MODEL
# ============================================================

model = FaultResolutionContrastiveModel(
    MODEL_NAME,
    BACKBONE_DIM,
    PROJECTION_DIM,
    INITIAL_TEMPERATURE
)

model = model.to(DEVICE)


# ============================================================
# OPTIMIZER
#
# MiniLM:
# small LR
#
# Projection heads + temperature:
# larger LR
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params":
                model.backbone.parameters(),

            "lr":
                BACKBONE_LR
        },

        {
            "params":
                model.fault_projection.parameters(),

            "lr":
                HEAD_LR
        },

        {
            "params":
                model.resolution_projection.parameters(),

            "lr":
                HEAD_LR
        },

        {
            "params":
                [model.logit_scale],

            "lr":
                HEAD_LR
        }
    ],

    weight_decay=WEIGHT_DECAY
)

# AdamW is an appropriate standard optimizer for fine-tuning.
# PyTorch provides it directly.


# ============================================================
# CLIP SYMMETRIC LOSS
# ============================================================

def clip_loss(logits):

    n = logits.shape[0]

    labels = torch.arange(
        n,
        device=logits.device
    )


    # --------------------------------------------
    # Fault -> Resolution
    #
    # row i should select resolution i
    # --------------------------------------------

    loss_f2r = F.cross_entropy(
        logits,
        labels
    )


    # --------------------------------------------
    # Resolution -> Fault
    #
    # column i should select fault i
    # --------------------------------------------

    loss_r2f = F.cross_entropy(
        logits.T,
        labels
    )


    loss = (
        loss_f2r
        +
        loss_r2f
    ) / 2.0


    return (
        loss,
        loss_f2r,
        loss_r2f
    )


# ============================================================
# MOVE TOKENS TO DEVICE
# ============================================================

def move_tokens(tokens):

    return {
        key: value.to(DEVICE)
        for key, value in tokens.items()
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_f2r = 0.0
    total_r2f = 0.0

    batches = 0


    progress = tqdm(
        train_loader,
        desc="Training"
    )


    for batch in progress:

        fault_tokens = move_tokens(
            batch["fault_tokens"]
        )

        resolution_tokens = move_tokens(
            batch["resolution_tokens"]
        )


        optimizer.zero_grad()


        outputs = model(
            fault_tokens,
            resolution_tokens
        )


        loss, loss_f2r, loss_r2f = (
            clip_loss(
                outputs["logits"]
            )
        )


        loss.backward()


        # Helps avoid unstable updates
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        total_loss += loss.item()
        total_f2r += loss_f2r.item()
        total_r2f += loss_r2f.item()

        batches += 1


        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    return {
        "loss":
            total_loss / batches,

        "f2r_loss":
            total_f2r / batches,

        "r2f_loss":
            total_r2f / batches
    }


# ============================================================
# ENCODE COMPLETE DATASET
#
# Used for validation.
# ============================================================

@torch.no_grad()
def encode_dataset(loader):

    model.eval()


    all_fault_norm = []
    all_resolution_norm = []

    all_fault_raw = []
    all_resolution_raw = []

    all_fault_texts = []
    all_resolution_texts = []


    for batch in tqdm(
        loader,
        desc="Encoding validation"
    ):

        fault_tokens = move_tokens(
            batch["fault_tokens"]
        )

        resolution_tokens = move_tokens(
            batch["resolution_tokens"]
        )


        fault_raw, fault_norm = (
            model.encode_fault(
                fault_tokens
            )
        )

        resolution_raw, resolution_norm = (
            model.encode_resolution(
                resolution_tokens
            )
        )


        all_fault_raw.append(
            fault_raw.cpu()
        )

        all_resolution_raw.append(
            resolution_raw.cpu()
        )

        all_fault_norm.append(
            fault_norm.cpu()
        )

        all_resolution_norm.append(
            resolution_norm.cpu()
        )


        all_fault_texts.extend(
            batch["fault_texts"]
        )

        all_resolution_texts.extend(
            batch["resolution_texts"]
        )


    return {
        "fault_raw":
            torch.cat(
                all_fault_raw,
                dim=0
            ),

        "resolution_raw":
            torch.cat(
                all_resolution_raw,
                dim=0
            ),

        "fault_norm":
            torch.cat(
                all_fault_norm,
                dim=0
            ),

        "resolution_norm":
            torch.cat(
                all_resolution_norm,
                dim=0
            ),

        "fault_texts":
            all_fault_texts,

        "resolution_texts":
            all_resolution_texts
    }


# ============================================================
# VALIDATION METRICS
#
# Fault -> resolution retrieval
#
# Important:
#
# If two validation records have EXACTLY the same
# resolution text, either is considered correct.
# ============================================================

@torch.no_grad()
def evaluate_validation():

    encoded = encode_dataset(
        val_loader
    )


    fault_embeddings = (
        encoded["fault_norm"]
    )

    resolution_embeddings = (
        encoded["resolution_norm"]
    )


    # --------------------------------------------
    # cosine similarity matrix
    #
    # [436, 436]
    # --------------------------------------------

    similarity_matrix = torch.matmul(
        fault_embeddings,
        resolution_embeddings.T
    )


    resolution_texts_norm = [
        normalize_text(x)
        for x in encoded[
            "resolution_texts"
        ]
    ]


    ranks = []

    top1_hits = 0
    top5_hits = 0
    top10_hits = 0


    positive_cosines = []


    for i in range(
        len(fault_embeddings)
    ):

        scores = similarity_matrix[i]


        sorted_indices = torch.argsort(
            scores,
            descending=True
        )


        # ----------------------------------------
        # All exact-equivalent resolution texts
        # are valid positives.
        # ----------------------------------------

        target_resolution = (
            resolution_texts_norm[i]
        )


        valid_indices = {
            j
            for j, text
            in enumerate(
                resolution_texts_norm
            )
            if text == target_resolution
        }


        rank = None


        for position, idx in enumerate(
            sorted_indices.tolist(),
            start=1
        ):

            if idx in valid_indices:

                rank = position
                break


        ranks.append(rank)


        if rank <= 1:
            top1_hits += 1

        if rank <= 5:
            top5_hits += 1

        if rank <= 10:
            top10_hits += 1


        positive_cosines.append(
            similarity_matrix[i, i].item()
        )


    n = len(ranks)


    recall1 = top1_hits / n
    recall5 = top5_hits / n
    recall10 = top10_hits / n


    mrr = np.mean(
        [
            1.0 / rank
            for rank in ranks
        ]
    )


    positive_cosine = float(
        np.mean(
            positive_cosines
        )
    )


    # --------------------------------------------
    # Mean negative cosine similarity
    #
    # Everything off the diagonal.
    # --------------------------------------------

    sim_np = similarity_matrix.numpy()

    mask = ~np.eye(
        sim_np.shape[0],
        dtype=bool
    )

    negative_cosine = float(
        sim_np[mask].mean()
    )


    # --------------------------------------------
    # Validation CLIP loss
    #
    # Entire validation set can fit in memory
    # because it is only 436 x 436.
    # --------------------------------------------

    scale = model.logit_scale.exp().cpu()

    logits = (
        similarity_matrix
        *
        scale
    )


    labels = torch.arange(
        len(logits)
    )


    val_f2r = F.cross_entropy(
        logits,
        labels
    )

    val_r2f = F.cross_entropy(
        logits.T,
        labels
    )


    val_loss = (
        val_f2r
        +
        val_r2f
    ) / 2.0


    return {
        "val_loss":
            float(val_loss.item()),

        "recall_at_1":
            recall1,

        "recall_at_5":
            recall5,

        "recall_at_10":
            recall10,

        "mrr":
            float(mrr),

        "positive_cosine":
            positive_cosine,

        "negative_cosine":
            negative_cosine,

        "temperature":
            float(
                1.0 /
                model.logit_scale
                .exp()
                .item()
            )
    }


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_model(
    epoch,
    metrics
):

    # --------------------------------------------------------
    # Save MiniLM backbone
    # --------------------------------------------------------

    backbone_dir = os.path.join(
        MODEL_DIR,
        "backbone"
    )

    os.makedirs(
        backbone_dir,
        exist_ok=True
    )


    model.backbone.save_pretrained(
        backbone_dir
    )

    tokenizer.save_pretrained(
        backbone_dir
    )


    # --------------------------------------------------------
    # Save projection heads + temperature
    # --------------------------------------------------------

    torch.save(
        {
            "epoch":
                epoch,

            "fault_projection_state_dict":
                model
                .fault_projection
                .state_dict(),

            "resolution_projection_state_dict":
                model
                .resolution_projection
                .state_dict(),

            "logit_scale":
                model
                .logit_scale
                .detach()
                .cpu(),

            "backbone_dim":
                BACKBONE_DIM,

            "projection_dim":
                PROJECTION_DIM,

            "model_name":
                MODEL_NAME,

            "max_length":
                MAX_LENGTH,

            "validation_metrics":
                metrics
        },

        os.path.join(
            MODEL_DIR,
            "projection_heads.pt"
        )
    )


    # --------------------------------------------------------
    # Human-readable config
    # --------------------------------------------------------

    config = {
        "model_name":
            MODEL_NAME,

        "backbone_dim":
            BACKBONE_DIM,

        "projection_dim":
            PROJECTION_DIM,

        "max_length":
            MAX_LENGTH,

        "batch_size":
            BATCH_SIZE,

        "backbone_lr":
            BACKBONE_LR,

        "head_lr":
            HEAD_LR,

        "best_epoch":
            epoch,

        "validation_metrics":
            metrics
    }


    with open(
        os.path.join(
            MODEL_DIR,
            "training_config.json"
        ),
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )


# ============================================================
# TRAINING LOOP
# ============================================================

history = []

best_mrr = -1.0
epochs_without_improvement = 0


training_start = time.perf_counter()


for epoch in range(
    1,
    EPOCHS + 1
):

    print("\n" + "=" * 80)
    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )
    print("=" * 80)


    epoch_start = time.perf_counter()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_metrics = train_one_epoch()


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    val_metrics = evaluate_validation()


    epoch_time = (
        time.perf_counter()
        -
        epoch_start
    )


    row = {
        "epoch": epoch,

        "train_loss":
            train_metrics["loss"],

        "train_f2r_loss":
            train_metrics[
                "f2r_loss"
            ],

        "train_r2f_loss":
            train_metrics[
                "r2f_loss"
            ],

        **val_metrics,

        "epoch_time_seconds":
            epoch_time
    }


    history.append(row)


    print("\nRESULTS")

    print(
        f"Train loss     : "
        f"{row['train_loss']:.4f}"
    )

    print(
        f"Validation loss: "
        f"{row['val_loss']:.4f}"
    )

    print(
        f"Recall@1       : "
        f"{row['recall_at_1']:.4f}"
    )

    print(
        f"Recall@5       : "
        f"{row['recall_at_5']:.4f}"
    )

    print(
        f"Recall@10      : "
        f"{row['recall_at_10']:.4f}"
    )

    print(
        f"MRR            : "
        f"{row['mrr']:.4f}"
    )

    print(
        f"Positive cosine: "
        f"{row['positive_cosine']:.4f}"
    )

    print(
        f"Negative cosine: "
        f"{row['negative_cosine']:.4f}"
    )

    print(
        f"Temperature    : "
        f"{row['temperature']:.4f}"
    )

    print(
        f"Epoch time     : "
        f"{epoch_time:.2f} sec"
    )


    # --------------------------------------------------------
    # SAVE HISTORY AFTER EVERY EPOCH
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "training_history.csv"
        ),
        index=False
    )


    # --------------------------------------------------------
    # BEST CHECKPOINT BY VALIDATION MRR
    # --------------------------------------------------------

    if row["mrr"] > best_mrr:

        best_mrr = row["mrr"]

        epochs_without_improvement = 0

        print(
            "\nNew best validation MRR."
        )

        print(
            "Saving model..."
        )


        save_model(
            epoch,
            val_metrics
        )


    else:

        epochs_without_improvement += 1


        print(
            "\nNo MRR improvement for",
            epochs_without_improvement,
            "epoch(s)."
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# TOTAL TRAINING TIME
# ============================================================

total_training_time = (
    time.perf_counter()
    -
    training_start
)


print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(
    "Total training time:",
    f"{total_training_time:.2f} sec"
)

print(
    "Best validation MRR:",
    f"{best_mrr:.4f}"
)


# ============================================================
# SAVE TOTAL TIME
# ============================================================

with open(
    os.path.join(
        RESULT_DIR,
        "training_time.txt"
    ),
    "w"
) as f:

    f.write(
        f"Total training time seconds: "
        f"{total_training_time:.4f}\n"
    )

    f.write(
        f"Best validation MRR: "
        f"{best_mrr:.6f}\n"
    )


print("\nSaved trained model to:")

print(MODEL_DIR)

print("\nSaved training results to:")

print(RESULT_DIR)