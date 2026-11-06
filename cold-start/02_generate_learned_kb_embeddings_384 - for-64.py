from pathlib import Path
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

KB_FILE = os.path.join(
    BASE_DIR,
    "fault-logs-o-ran-64-with-resolution.xls"
)

CHECKPOINT_FILE = os.path.join(
    BASE_DIR,
    "real64_contrastive_model",
    "best_contrastive_model.pt"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "real64_proposed_final"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

SEED = 42

EXPECTED_KB_SIZE = 64
EXPECTED_PROJECTED_DIM = 384

BATCH_SIZE = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("COLD-START LEARNED KB EMBEDDING GENERATION")
print("=" * 80)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# CHECK INPUTS AND LOAD KB
# ============================================================
for p in [KB_FILE, CHECKPOINT_FILE]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file:\n{p}")

kb_df = pd.read_excel(KB_FILE)
required = ["Fault Description", "Resolution"]
for col in required:
    if col not in kb_df.columns:
        raise ValueError(f"Missing column: {col}")
if len(kb_df) != EXPECTED_KB_SIZE:
    raise ValueError(
        f"Expected {EXPECTED_KB_SIZE} KB records, found {len(kb_df)}."
    )
if kb_df[required].isna().any().any():
    raise ValueError("Missing Fault Description or Resolution values.")

kb_df = kb_df.reset_index(drop=True)
fault_texts = kb_df["Fault Description"].astype(str).tolist()
print("KB records:", len(kb_df))

# ============================================================
# LOAD NEW CONTRASTIVE CHECKPOINT
#
# The checkpoint supplies the selected epoch, MiniLM model,
# projection dimension, and maximum token length. No epoch
# number is hard-coded here.
# ============================================================
checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)

for key in [
    "epoch", "model_state_dict", "projection_dim",
    "model_name", "max_length"
]:
    if key not in checkpoint:
        raise KeyError(f"Checkpoint missing required key: {key}")

MODEL_NAME = checkpoint["model_name"]
PROJECTED_DIM = int(checkpoint["projection_dim"])
MAX_LENGTH = int(checkpoint["max_length"])

print("\nCheckpoint epoch:", checkpoint["epoch"])
print("Model:", MODEL_NAME)
print("Projection dimension:", PROJECTED_DIM)
print("Maximum sequence length:", MAX_LENGTH)

# Prevent accidental reuse of the old 256-D checkpoint.
if PROJECTED_DIM != EXPECTED_PROJECTED_DIM:
    raise ValueError(
        f"Expected the new {EXPECTED_PROJECTED_DIM}-D checkpoint, "
        f"but found projection_dim={PROJECTED_DIM}."
    )

# ============================================================
# MODEL ARCHITECTURE
#
# This exactly matches contrastive training:
# MiniLM pooled representation -> Linear -> GELU -> Linear.
# For the new experiment the final projection is 384-D.
# ============================================================
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class FaultResolutionCLIP(nn.Module):
    def __init__(self, model_name, projection_dim):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size
        self.fault_projection = ProjectionHead(
            hidden_dim, projection_dim
        )
        self.resolution_projection = ProjectionHead(
            hidden_dim, projection_dim
        )
        # Included because the checkpoint contains this parameter.
        self.logit_scale = nn.Parameter(torch.tensor(0.0))


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = FaultResolutionCLIP(MODEL_NAME, PROJECTED_DIM)

load_result = model.load_state_dict(
    checkpoint["model_state_dict"], strict=True
)
print("\nCheckpoint loaded with strict=True.")
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)

model = model.to(DEVICE)
model.eval()
BACKBONE_DIM = int(model.encoder.config.hidden_size)

print("Backbone dimension:", BACKBONE_DIM)
print("Learned projection dimension:", PROJECTED_DIM)

# ============================================================
# ENCODE ALL 64 KB FAULTS
#
# We use the fine-tuned MiniLM encoder + learned fault
# projection. The resolution projection is not used for KB
# fault indexing.
#
# RAW embeddings:
#   later used for Random+CLIP group means and SCALA GMM.
#
# NORMALIZED embeddings:
#   later used for cosine retrieval.
# ============================================================
print("\n" + "=" * 80)
print("ENCODING ALL 64 FAULT DESCRIPTIONS")
print("=" * 80)

batches = []
if torch.cuda.is_available():
    torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.no_grad():
    for start in range(0, len(fault_texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(fault_texts))
        texts = fault_texts[start:end]

        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        output = model.encoder(**tokens)
        pooled = mean_pooling(output, tokens["attention_mask"])
        raw = model.fault_projection(pooled)

        batches.append(
            raw.detach().cpu().numpy().astype(np.float32)
        )

        if end % 1000 < BATCH_SIZE or end == EXPECTED_KB_SIZE:
            print(f"Encoded {end}/{EXPECTED_KB_SIZE}")

if torch.cuda.is_available():
    torch.cuda.synchronize()
embedding_time_sec = time.perf_counter() - start_time

kb_raw = np.vstack(batches)

# ============================================================
# NORMALIZE FOR COSINE RETRIEVAL
# ============================================================
norms = np.linalg.norm(kb_raw, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-12)
kb_normalized = (kb_raw / norms).astype(np.float32)

expected_shape = (EXPECTED_KB_SIZE, PROJECTED_DIM)
if kb_raw.shape != expected_shape:
    raise ValueError(
        f"Expected raw shape {expected_shape}, got {kb_raw.shape}."
    )
if kb_normalized.shape != expected_shape:
    raise ValueError(
        f"Expected normalized shape {expected_shape}, "
        f"got {kb_normalized.shape}."
    )
if not np.isfinite(kb_raw).all():
    raise ValueError("Non-finite value in raw embeddings.")
if not np.isfinite(kb_normalized).all():
    raise ValueError("Non-finite value in normalized embeddings.")

unit_norms = np.linalg.norm(kb_normalized, axis=1)
if not np.allclose(unit_norms, 1.0, atol=1e-5):
    raise ValueError("Normalized embeddings are not unit length.")

# ============================================================
# SAVE
#
# These intentionally overwrite the previous 256-D learned
# embedding files at the same paths.
# ============================================================
np.save(RAW_EMBEDDING_FILE, kb_raw)
np.save(NORM_EMBEDDING_FILE, kb_normalized)

# Stable metadata preserves KB row/index correspondence.
metadata_df = pd.DataFrame({
    "kb_index": np.arange(len(kb_df), dtype=np.int32),
    "fault_text": kb_df["Fault Description"].astype(str),
    "resolution_text": kb_df["Resolution"].astype(str),
})
metadata_df.to_csv(
    METADATA_FILE, index=False, encoding="utf-8-sig"
)

# ============================================================
# FINAL RELOAD CHECK
# ============================================================
saved_raw = np.load(RAW_EMBEDDING_FILE)
saved_norm = np.load(NORM_EMBEDDING_FILE)

assert saved_raw.shape == expected_shape
assert saved_norm.shape == expected_shape
assert len(metadata_df) == EXPECTED_KB_SIZE

print("\n" + "=" * 80)
print("LEARNED KB EMBEDDING GENERATION COMPLETE")
print("=" * 80)
print("Checkpoint epoch       :", checkpoint["epoch"])
print("KB records             :", len(kb_df))
print("Backbone dimension     :", BACKBONE_DIM)
print("Projection dimension   :", PROJECTED_DIM)
print("Raw embedding shape    :", saved_raw.shape)
print("Normalized shape       :", saved_norm.shape)
print("Embedding time         :", f"{embedding_time_sec:.4f} sec")
print("\nSaved raw embeddings   :", RAW_EMBEDDING_FILE)
print("Saved normalized       :", NORM_EMBEDDING_FILE)
print("Saved metadata         :", METADATA_FILE)
print("\nNext:")
print("\nNext:")
print("Fit the SCALA GMM on the 64 raw 384-D embeddings.")
