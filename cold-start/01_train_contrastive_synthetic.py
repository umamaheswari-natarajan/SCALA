import os
import random
import time
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "knowledge-base.xlsx"

FAULT_COL = "Fault Description"
RESOLUTION_COL = "Resolution"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR = "synthetic_contrastive_model"

SEED = 42

TRAIN_RATIO = 0.80

MAX_LENGTH = 128

BATCH_SIZE = 64

EPOCHS = 10

# Backbone learning rate
BACKBONE_LR = 2e-5

# Projection-head + temperature learning rate
HEAD_LR = 1e-4

WEIGHT_DECAY = 1e-4

PROJECTION_DIM = 256

NUM_WORKERS = 0


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 75)
print("SYNTHETIC CLIP-STYLE CONTRASTIVE TRAINING")
print("=" * 75)

print("Device:", device)


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. LOAD DATASET
# ============================================================

df = pd.read_excel(DATA_FILE)

print("\nOriginal records:", len(df))
print("Columns:", list(df.columns))


if FAULT_COL not in df.columns:
    raise ValueError(
        f"Missing column: {FAULT_COL}"
    )

if RESOLUTION_COL not in df.columns:
    raise ValueError(
        f"Missing column: {RESOLUTION_COL}"
    )


# Remove missing/empty rows only if present
df = df[
    df[FAULT_COL].notna()
    &
    df[RESOLUTION_COL].notna()
].copy()

df[FAULT_COL] = (
    df[FAULT_COL]
    .astype(str)
    .str.strip()
)

df[RESOLUTION_COL] = (
    df[RESOLUTION_COL]
    .astype(str)
    .str.strip()
)

df = df[
    (df[FAULT_COL] != "")
    &
    (df[RESOLUTION_COL] != "")
].reset_index(drop=True)


print("Usable records:", len(df))

if len(df) != 10500:
    print(
        "WARNING: Expected 10500 records, "
        f"but found {len(df)} usable records."
    )


# ============================================================
# 5. FIXED TRAIN / VALIDATION SPLIT
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=1.0 - TRAIN_RATIO,
    random_state=SEED,
    shuffle=True
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


print("\n" + "=" * 75)
print("DEVELOPMENT SPLIT")
print("=" * 75)

print("Train      :", len(train_df))
print("Validation :", len(val_df))


# Save the exact split
train_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train_8400.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_2100.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 7. DATASET
# ============================================================

class FaultResolutionDataset(Dataset):

    def __init__(self, dataframe):

        self.faults = (
            dataframe[FAULT_COL]
            .astype(str)
            .tolist()
        )

        self.resolutions = (
            dataframe[RESOLUTION_COL]
            .astype(str)
            .tolist()
        )


    def __len__(self):

        return len(self.faults)


    def __getitem__(self, idx):

        return {
            "fault": self.faults[idx],
            "resolution": self.resolutions[idx]
        }


# ============================================================
# 8. COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    faults = [
        item["fault"]
        for item in batch
    ]

    resolutions = [
        item["resolution"]
        for item in batch
    ]


    fault_tokens = tokenizer(
        faults,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


    resolution_tokens = tokenizer(
        resolutions,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


    return (
        fault_tokens,
        resolution_tokens
    )


# ============================================================
# 9. DATALOADERS
# ============================================================

train_dataset = FaultResolutionDataset(
    train_df
)

val_dataset = FaultResolutionDataset(
    val_df
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)


# ============================================================
# 10. MEAN POOLING
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
# 11. PROJECTION HEAD
# ============================================================

class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim=384,
        output_dim=256
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
                output_dim
            )
        )


    def forward(self, x):

        return self.net(x)


# ============================================================
# 12. CLIP-STYLE MODEL
# ============================================================

class FaultResolutionCLIP(nn.Module):

    def __init__(self):

        super().__init__()


        # Shared MiniLM backbone
        self.encoder = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
        )


        # Separate projection spaces
        self.fault_projection = (
            ProjectionHead(
                input_dim=384,
                output_dim=PROJECTION_DIM
            )
        )

        self.resolution_projection = (
            ProjectionHead(
                input_dim=384,
                output_dim=PROJECTION_DIM
            )
        )


        # CLIP-style learnable logit scale.
        #
        # Initial temperature approximately:
        #
        # 1 / exp(log(1/0.07)) = 0.07
        #
        self.logit_scale = nn.Parameter(
            torch.tensor(
                np.log(1 / 0.07),
                dtype=torch.float32
            )
        )


    def encode_backbone(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        return pooled


    def encode_fault(
        self,
        input_ids,
        attention_mask,
        normalize=True
    ):

        x = self.encode_backbone(
            input_ids,
            attention_mask
        )

        z = self.fault_projection(x)

        if normalize:
            z = F.normalize(
                z,
                p=2,
                dim=1
            )

        return z


    def encode_resolution(
        self,
        input_ids,
        attention_mask,
        normalize=True
    ):

        x = self.encode_backbone(
            input_ids,
            attention_mask
        )

        z = self.resolution_projection(x)

        if normalize:
            z = F.normalize(
                z,
                p=2,
                dim=1
            )

        return z


    def get_temperature(self):

        scale = (
            self.logit_scale
            .exp()
            .clamp(
                max=100.0
            )
        )

        return 1.0 / scale


# ============================================================
# 13. MODEL
# ============================================================

model = FaultResolutionCLIP().to(
    device
)

print("\nModel loaded.")

print(
    "Projection dimension:",
    PROJECTION_DIM
)

print(
    "Initial temperature:",
    round(
        model.get_temperature().item(),
        6
    )
)


# ============================================================
# 14. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    [
        {
            "params":
                model.encoder.parameters(),

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


# ============================================================
# 15. SYMMETRIC CONTRASTIVE LOSS
# ============================================================

def contrastive_loss(
    fault_embeddings,
    resolution_embeddings,
    model
):

    logit_scale = (
        model.logit_scale
        .exp()
        .clamp(
            max=100.0
        )
    )


    logits = (

        logit_scale

        *

        (
            fault_embeddings
            @
            resolution_embeddings.T
        )
    )


    batch_size = (
        logits.size(0)
    )


    labels = torch.arange(
        batch_size,
        device=logits.device
    )


    # Fault -> Resolution
    loss_f2r = F.cross_entropy(
        logits,
        labels
    )


    # Resolution -> Fault
    loss_r2f = F.cross_entropy(
        logits.T,
        labels
    )


    loss = (
        loss_f2r
        +
        loss_r2f
    ) / 2.0


    return loss


# ============================================================
# 16. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    start_time = time.perf_counter()


    for batch_idx, (
        fault_tokens,
        resolution_tokens
    ) in enumerate(train_loader):


        fault_tokens = {
            k: v.to(device)
            for k, v in fault_tokens.items()
        }


        resolution_tokens = {
            k: v.to(device)
            for k, v in resolution_tokens.items()
        }


        optimizer.zero_grad()


        fault_emb = model.encode_fault(
            fault_tokens["input_ids"],
            fault_tokens["attention_mask"]
        )


        resolution_emb = (
            model.encode_resolution(
                resolution_tokens["input_ids"],
                resolution_tokens[
                    "attention_mask"
                ]
            )
        )


        loss = contrastive_loss(
            fault_emb,
            resolution_emb,
            model
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        total_loss += (
            loss.item()
        )


        # Optional progress display
        if (
            (batch_idx + 1) % 25
            == 0
        ):

            print(
                f"    Batch "
                f"{batch_idx + 1}/"
                f"{len(train_loader)}"
                f" | loss "
                f"{loss.item():.4f}"
            )


    avg_loss = (
        total_loss
        /
        len(train_loader)
    )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    return avg_loss, elapsed


# ============================================================
# 17. VALIDATION LOSS
# ============================================================

def validation_loss():

    model.eval()

    total_loss = 0.0


    with torch.no_grad():

        for (
            fault_tokens,
            resolution_tokens
        ) in val_loader:


            fault_tokens = {
                k: v.to(device)
                for k, v
                in fault_tokens.items()
            }


            resolution_tokens = {
                k: v.to(device)
                for k, v
                in resolution_tokens.items()
            }


            fault_emb = (
                model.encode_fault(
                    fault_tokens["input_ids"],
                    fault_tokens[
                        "attention_mask"
                    ]
                )
            )


            resolution_emb = (
                model.encode_resolution(
                    resolution_tokens[
                        "input_ids"
                    ],
                    resolution_tokens[
                        "attention_mask"
                    ]
                )
            )


            loss = contrastive_loss(
                fault_emb,
                resolution_emb,
                model
            )


            total_loss += (
                loss.item()
            )


    return (
        total_loss
        /
        len(val_loader)
    )


# ============================================================
# 18. ENCODE COMPLETE VALIDATION SET
# ============================================================

def encode_validation():

    model.eval()

    fault_embeddings = []
    resolution_embeddings = []


    with torch.no_grad():

        for (
            fault_tokens,
            resolution_tokens
        ) in val_loader:


            fault_tokens = {
                k: v.to(device)
                for k, v
                in fault_tokens.items()
            }


            resolution_tokens = {
                k: v.to(device)
                for k, v
                in resolution_tokens.items()
            }


            fault_emb = (
                model.encode_fault(
                    fault_tokens["input_ids"],
                    fault_tokens[
                        "attention_mask"
                    ]
                )
            )


            resolution_emb = (
                model.encode_resolution(
                    resolution_tokens[
                        "input_ids"
                    ],
                    resolution_tokens[
                        "attention_mask"
                    ]
                )
            )


            fault_embeddings.append(
                fault_emb.cpu()
            )

            resolution_embeddings.append(
                resolution_emb.cpu()
            )


    fault_embeddings = (
        torch.cat(
            fault_embeddings,
            dim=0
        )
    )


    resolution_embeddings = (
        torch.cat(
            resolution_embeddings,
            dim=0
        )
    )


    return (
        fault_embeddings,
        resolution_embeddings
    )


# ============================================================
# 19. VALIDATION RETRIEVAL METRICS
# ============================================================

def retrieval_metrics():

    fault_emb, resolution_emb = (
        encode_validation()
    )


    # Since embeddings already normalized,
    # dot product = cosine similarity
    similarity = (
        fault_emb
        @
        resolution_emb.T
    )


    N = similarity.shape[0]


    # --------------------------------------------------------
    # Positive cosine
    # --------------------------------------------------------

    positive_cosine = (
        similarity.diag()
        .mean()
        .item()
    )


    # --------------------------------------------------------
    # Negative cosine
    #
    # Average off-diagonal similarity
    # --------------------------------------------------------

    total_sum = (
        similarity.sum()
    )

    positive_sum = (
        similarity.diag().sum()
    )

    negative_cosine = (

        (
            total_sum
            -
            positive_sum
        )

        /

        (
            N * N
            -
            N
        )
    ).item()


    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranked_indices = (
        torch.argsort(
            similarity,
            dim=1,
            descending=True
        )
    )


    target = torch.arange(
        N
    ).unsqueeze(1)


    matches = (
        ranked_indices
        ==
        target
    )


    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    recall_1 = (
        matches[:, :1]
        .any(dim=1)
        .float()
        .mean()
        .item()
    )


    recall_5 = (
        matches[:, :5]
        .any(dim=1)
        .float()
        .mean()
        .item()
    )


    recall_10 = (
        matches[:, :10]
        .any(dim=1)
        .float()
        .mean()
        .item()
    )


    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    ranks = (
        matches
        .float()
        .argmax(dim=1)
        +
        1
    )


    mrr = (
        (
            1.0
            /
            ranks.float()
        )
        .mean()
        .item()
    )


    return {
        "recall_at_1":
            recall_1,

        "recall_at_5":
            recall_5,

        "recall_at_10":
            recall_10,

        "mrr":
            mrr,

        "positive_cosine":
            positive_cosine,

        "negative_cosine":
            negative_cosine
    }


# ============================================================
# 20. TRAINING HISTORY
# ============================================================

history = []

best_recall1 = -1.0
best_mrr = -1.0

best_epoch = None

total_training_start = (
    time.perf_counter()
)


# ============================================================
# 21. TRAIN
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):


    print("\n" + "=" * 75)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("=" * 75)


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss, epoch_time = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # Validation loss
    # --------------------------------------------------------

    val_loss = (
        validation_loss()
    )


    # --------------------------------------------------------
    # Retrieval metrics
    # --------------------------------------------------------

    metrics = (
        retrieval_metrics()
    )


    temperature = (
        model.get_temperature()
        .item()
    )


    print("\nRESULTS")

    print(
        f"Train Loss      : "
        f"{train_loss:.6f}"
    )

    print(
        f"Validation Loss : "
        f"{val_loss:.6f}"
    )

    print(
        f"Recall@1        : "
        f"{metrics['recall_at_1']:.6f}"
    )

    print(
        f"Recall@5        : "
        f"{metrics['recall_at_5']:.6f}"
    )

    print(
        f"Recall@10       : "
        f"{metrics['recall_at_10']:.6f}"
    )

    print(
        f"MRR             : "
        f"{metrics['mrr']:.6f}"
    )

    print(
        f"Positive cosine : "
        f"{metrics['positive_cosine']:.6f}"
    )

    print(
        f"Negative cosine : "
        f"{metrics['negative_cosine']:.6f}"
    )

    print(
        f"Temperature     : "
        f"{temperature:.6f}"
    )

    print(
        f"Epoch time      : "
        f"{epoch_time:.2f} sec"
    )


    epoch_result = {

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "validation_loss":
            val_loss,

        "recall_at_1":
            metrics[
                "recall_at_1"
            ],

        "recall_at_5":
            metrics[
                "recall_at_5"
            ],

        "recall_at_10":
            metrics[
                "recall_at_10"
            ],

        "mrr":
            metrics["mrr"],

        "positive_cosine":
            metrics[
                "positive_cosine"
            ],

        "negative_cosine":
            metrics[
                "negative_cosine"
            ],

        "temperature":
            temperature,

        "epoch_time_sec":
            epoch_time
    }


    history.append(
        epoch_result
    )


    # Save history after every epoch
    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "training_history.csv"
        ),
        index=False
    )


    # --------------------------------------------------------
    # Best model selection
    #
    # Primary: Recall@1
    # Tie-breaker: MRR
    # --------------------------------------------------------

    current_r1 = (
        metrics["recall_at_1"]
    )

    current_mrr = (
        metrics["mrr"]
    )


    is_better = False


    if current_r1 > best_recall1:

        is_better = True


    elif (
        current_r1
        ==
        best_recall1
        and
        current_mrr
        >
        best_mrr
    ):

        is_better = True


    if is_better:

        best_recall1 = (
            current_r1
        )

        best_mrr = (
            current_mrr
        )

        best_epoch = (
            epoch
        )


        checkpoint = {

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "projection_dim":
                PROJECTION_DIM,

            "model_name":
                MODEL_NAME,

            "max_length":
                MAX_LENGTH,

            "seed":
                SEED,

            "validation_metrics":
                metrics,

            "temperature":
                temperature
        }


        torch.save(
            checkpoint,

            os.path.join(
                OUTPUT_DIR,
                "best_contrastive_model.pt"
            )
        )


        print(
            "\n*** NEW BEST CHECKPOINT SAVED ***"
        )


# ============================================================
# 22. TOTAL TRAINING TIME
# ============================================================

total_training_time = (
    time.perf_counter()
    -
    total_training_start
)


# ============================================================
# 23. SAVE FINAL HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    ),
    index=False
)


# ============================================================
# 24. LOSS PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    marker="o",
    label="Training loss"
)

plt.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    marker="o",
    label="Validation loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Contrastive loss"
)

plt.title(
    "Contrastive Training and Validation Loss"
)

plt.xticks(
    history_df["epoch"]
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "contrastive_loss_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 25. RETRIEVAL METRIC PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["recall_at_1"],
    marker="o",
    label="Recall@1"
)

plt.plot(
    history_df["epoch"],
    history_df["recall_at_5"],
    marker="o",
    label="Recall@5"
)

plt.plot(
    history_df["epoch"],
    history_df["recall_at_10"],
    marker="o",
    label="Recall@10"
)

plt.plot(
    history_df["epoch"],
    history_df["mrr"],
    marker="o",
    label="MRR"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Validation Retrieval Performance"
)

plt.xticks(
    history_df["epoch"]
)

plt.ylim(
    0,
    1
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "validation_retrieval_metrics.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print(
    f"Total contrastive training time: "
    f"{total_training_time:.2f} sec"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best Recall@1: "
    f"{best_recall1:.6f}"
)

print(
    f"Best MRR: "
    f"{best_mrr:.6f}"
)

print(
    "\nBest checkpoint:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "best_contrastive_model.pt"
    )
)

print(
    "\nTraining history:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    )
)

print(
    "\nLoss plot:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "contrastive_loss_curve.png"
    )
)

print(
    "\nValidation metrics plot:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "validation_retrieval_metrics.png"
    )
)