from pathlib import Path
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
from openai import OpenAI

from rouge_score import rouge_scorer
from bert_score import score as bert_score


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "dataset_splits"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

CONTRASTIVE_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "contrastive_minilm"
)

GMM_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gmm"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_nearest_mean_cosine_top15"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# DATA FILES
# ============================================================

TRAIN_FILE = os.path.join(
    SPLIT_DIR,
    "train_fault_resolution.xlsx"
)

TEST_FILE = os.path.join(
    SPLIT_DIR,
    "test_fault_resolution.xlsx"
)

TRAIN_NORM_EMBEDDING_FILE = os.path.join(
    EMBEDDING_DIR,
    "train_fault_embeddings_normalized.npy"
)

GMM_MEANS_FILE = os.path.join(
    GMM_MODEL_DIR,
    "gmm_means_normalized.npy"
)

CLUSTER_LOOKUP_FILE = os.path.join(
    GMM_MODEL_DIR,
    "cluster_to_indices.joblib"
)

TRAIN_CLUSTER_LABELS_FILE = os.path.join(
    GMM_MODEL_DIR,
    "train_cluster_labels.npy"
)


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

# Select the 10 closest cluster means from the 55 clusters
TOP_CLUSTERS = 15

# Retrieve the final Top-5 KB records
RETRIEVAL_K = 5


# ============================================================
# LLM SETTINGS
# ============================================================

LLM_MODEL = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 500


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42

# Set to a small number for testing.
# Set to None for all test queries.
MAX_TEST_QUERIES = None

BERTSCORE_LANG = "en"


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
    "PyTorch device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version:",
        torch.version.cuda
    )


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_vector(vector):

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def percentile(values, p):

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            p
        )
    )


def top_k(scores, k):

    k = min(
        k,
        len(scores)
    )

    return np.argsort(
        -scores
    )[:k]


# ============================================================
# CHECK INPUT FILES
# ============================================================

required_files = [
    TRAIN_FILE,
    TEST_FILE,
    TRAIN_NORM_EMBEDDING_FILE,
    GMM_MEANS_FILE,
    CLUSTER_LOOKUP_FILE,
    TRAIN_CLUSTER_LABELS_FILE
]

missing = [
    f
    for f in required_files
    if not os.path.exists(f)
]

if missing:

    print("\nMissing required files:")

    for f in missing:
        print(f)

    raise FileNotFoundError(
        "Required files are missing."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("LOADING DATASETS")
print("=" * 90)

train_df = pd.read_excel(
    TRAIN_FILE
)

test_df = pd.read_excel(
    TEST_FILE
)

print(
    "Training KB records:",
    len(train_df)
)

print(
    "Test records:",
    len(test_df)
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
            f"Missing train column: {column}"
        )

    if column not in test_df.columns:

        raise ValueError(
            f"Missing test column: {column}"
        )


# ============================================================
# OPTIONAL TEST LIMIT
# ============================================================

if MAX_TEST_QUERIES is not None:

    test_df = (
        test_df
        .iloc[:MAX_TEST_QUERIES]
        .copy()
        .reset_index(drop=True)
    )

N_TRAIN = len(train_df)
N_TEST = len(test_df)

print(
    "Queries being processed:",
    N_TEST
)


# ============================================================
# LOAD NORMALIZED TRAINING EMBEDDINGS
#
# These are used for cosine retrieval after cluster routing.
# ============================================================

print("\n" + "=" * 90)
print("LOADING NORMALIZED TRAINING EMBEDDINGS")
print("=" * 90)

train_norm = np.load(
    TRAIN_NORM_EMBEDDING_FILE
)

train_norm = np.ascontiguousarray(
    train_norm,
    dtype=np.float32
)

print(
    "Normalized shape:",
    train_norm.shape
)

if len(train_norm) != N_TRAIN:

    raise ValueError(
        "Training embedding count does not match training dataset."
    )


# ============================================================
# LOAD OFFLINE ROUTING ARTIFACTS
# ============================================================

print("\n" + "=" * 90)
print("LOADING COSINE NEAREST-MEAN ROUTING ARTIFACTS")
print("=" * 90)

gmm_means = np.load(
    GMM_MEANS_FILE
)

gmm_means = np.ascontiguousarray(
    gmm_means,
    dtype=np.float32
)

cluster_to_indices = joblib.load(
    CLUSTER_LOOKUP_FILE
)

train_cluster_labels = np.load(
    TRAIN_CLUSTER_LABELS_FILE
).astype(np.int32)

N_CLUSTERS = len(
    cluster_to_indices
)

print(
    "GMM means shape:",
    gmm_means.shape
)

print(
    "Number of clusters:",
    N_CLUSTERS
)

print(
    "Cluster labels shape:",
    train_cluster_labels.shape
)

print(
    "Cluster lookup loaded successfully."
)


# ============================================================
# ROUTING ARTIFACT SANITY CHECKS
# ============================================================

if gmm_means.shape[0] != N_CLUSTERS:

    raise ValueError(
        "Number of GMM means does not match cluster lookup."
    )

if gmm_means.shape[1] != train_norm.shape[1]:

    raise ValueError(
        "GMM mean dimension does not match embedding dimension."
    )

if len(train_cluster_labels) != N_TRAIN:

    raise ValueError(
        "Cluster label count does not match training KB size."
    )

if TOP_CLUSTERS > N_CLUSTERS:

    raise ValueError(
        "TOP_CLUSTERS cannot exceed the number of available clusters."
    )

total_cluster_records = sum(
    len(indices)
    for indices in cluster_to_indices.values()
)

if total_cluster_records != N_TRAIN:

    raise ValueError(
        "Cluster lookup does not contain all training KB records."
    )

print(
    "Total records in cluster lookup:",
    total_cluster_records
)


# ============================================================
# LOAD CONTRASTIVE MODEL
# ============================================================

HF_MODEL_DIR = os.path.join(
    CONTRASTIVE_MODEL_DIR,
    "backbone"
)

PROJECTION_FILE = os.path.join(
    CONTRASTIVE_MODEL_DIR,
    "projection_heads.pt"
)

print("\n" + "=" * 90)
print("LOADING TRAINED CONTRASTIVE MODEL")
print("=" * 90)

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_DIR
)

encoder = AutoModel.from_pretrained(
    HF_MODEL_DIR
)

encoder = encoder.to(
    DEVICE
)

encoder.eval()

checkpoint = torch.load(
    PROJECTION_FILE,
    map_location=DEVICE
)

MAX_LENGTH = int(
    checkpoint[
        "max_length"
    ]
)


# ============================================================
# FAULT PROJECTION HEAD
# ============================================================

fault_projection = nn.Linear(

    int(
        checkpoint[
            "backbone_dim"
        ]
    ),

    int(
        checkpoint[
            "projection_dim"
        ]
    ),

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

print(
    "Fault projection loaded successfully."
)

print(
    f"Query embedding: "
    f"{checkpoint['backbone_dim']}D -> "
    f"{checkpoint['projection_dim']}D"
)


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pooling(
    model_output,
    attention_mask
):

    token_embeddings = (
        model_output.last_hidden_state
    )

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

    denominator = torch.clamp(
        mask.sum(
            dim=1
        ),
        min=1e-9
    )

    return (
        summed
        /
        denominator
    )


# ============================================================
# ONLINE QUERY ENCODING
#
# Encoding timer includes:
#
# tokenizer
# encoder
# mean pooling
# fault projection
# conversion to NumPy
# query normalization
#
# Routing/retrieval is NOT included.
# ============================================================

def encode_fault_online(
    fault_text
):

    start = time.perf_counter()

    encoded = tokenizer(
        fault_text,
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

    with torch.no_grad():

        output = encoder(
            **encoded
        )

        pooled = mean_pooling(
            output,
            encoded[
                "attention_mask"
            ]
        )

        raw_embedding = fault_projection(
            pooled
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    raw_np = (
        raw_embedding
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    raw_np = np.ascontiguousarray(
        raw_np,
        dtype=np.float32
    )

    norm_np = normalize_vector(
        raw_np
    )

    norm_np = np.ascontiguousarray(
        norm_np,
        dtype=np.float32
    )

    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000

    return (
        raw_np,
        norm_np,
        elapsed_ms
    )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    retrieved_indices,
    similarities
):

    blocks = []

    for rank, (
        train_index,
        similarity
    ) in enumerate(
        zip(
            retrieved_indices,
            similarities
        ),
        start=1
    ):

        row = train_df.iloc[
            int(train_index)
        ]

        fault = safe_text(
            row[
                "fault_text"
            ]
        )

        resolution = safe_text(
            row[
                "resolution_text"
            ]
        )

        block = (
            f"Historical Example {rank}\n"
            f"Fault: {fault}\n"
            f"Resolution: {resolution}\n"
            f"Similarity: {float(similarity):.4f}"
        )

        blocks.append(
            block
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a network and IT fault-resolution assistant.

Given a new fault and several historical fault-resolution
examples, produce a concise and technically appropriate
resolution for the new fault.

Use the historical examples only as supporting evidence.
Do not mention that you were given examples.
Do not copy irrelevant details.
Return only the recommended resolution.
""".strip()


def make_user_prompt(
    query_fault,
    context
):

    return f"""
NEW FAULT
---------

{query_fault}

RELEVANT HISTORICAL FAULT-RESOLUTION EXAMPLES
----------------------------------------------

{context}

TASK
----

Provide the most appropriate resolution for the new fault.
""".strip()


# ============================================================
# OPENAI
# ============================================================

if not os.getenv(
    "OPENAI_API_KEY"
):

    raise EnvironmentError(
        "\nOPENAI_API_KEY is not set.\n"
        "In PowerShell use:\n"
        '$env:OPENAI_API_KEY="your-key"'
    )

client = OpenAI(
    timeout=60.0
)


# ============================================================
# GENERATION
# ============================================================

def generate_resolution(
    query_fault,
    context
):

    user_prompt = make_user_prompt(
        query_fault,
        context
    )

    start = time.perf_counter()

    response = client.responses.create(
        model=LLM_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0
    )

    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000

    return (
        response.output_text.strip(),
        elapsed_ms
    )


# ============================================================
# RESULT CONTAINERS
# ============================================================

query_results = []
retrieval_records = []


# ============================================================
# MAIN INFERENCE
# ============================================================

print("\n" + "=" * 90)
print("SCALA COSINE NEAREST-MEAN TOP-15 CLUSTER INFERENCE")
print("=" * 90)

print(
    "Test queries:",
    N_TEST
)

print(
    "Training KB:",
    N_TRAIN
)

print(
    "Total routing clusters:",
    N_CLUSTERS
)

print(
    "Selected nearest clusters:",
    TOP_CLUSTERS
)

print(
    "Routing method:",
    "Cosine similarity to normalized GMM means"
)

print(
    "Final retrieval Top-K:",
    RETRIEVAL_K
)

print(
    "LLM:",
    LLM_MODEL
)


for q in range(
    N_TEST
):

    print(
        "\n" + "-" * 90
    )

    print(
        f"QUERY {q + 1}/{N_TEST}"
    )

    print(
        "-" * 90
    )

    row = test_df.iloc[
        q
    ]

    query_id = safe_text(
        row[
            "id"
        ]
    )

    query_fault = safe_text(
        row[
            "fault_text"
        ]
    )

    ground_truth = safe_text(
        row[
            "resolution_text"
        ]
    )

    source_dataset = safe_text(
        row[
            "source_dataset"
        ]
    )


    # ========================================================
    # 1. ONLINE QUERY ENCODING
    # ========================================================

    (
        query_raw,
        query_norm,
        encoding_ms
    ) = encode_fault_online(
        query_fault
    )


    # ========================================================
    # 2. RETRIEVAL TIMER START
    #
    # INCLUDED:
    #
    # - query-to-55-mean cosine similarity computation
    # - Top-15 cluster selection
    # - cluster -> KB lookup
    # - pooling candidate indices
    # - cosine similarity over pooled records
    # - final Top-5 selection
    #
    # NOT INCLUDED:
    #
    # - candidate count calculation
    # - candidate reduction calculation
    # - diagnostics/reporting
    # ========================================================

    retrieval_start = time.perf_counter()


    # --------------------------------------------------------
    # 2A. COSINE SIMILARITY TO ALL 55 GMM MEANS
    # --------------------------------------------------------

    cluster_scores = np.dot(
    gmm_means,
    query_norm
)

    selected_clusters = np.argpartition(
        -cluster_scores,
        TOP_CLUSTERS - 1
    )[:TOP_CLUSTERS]

    selected_clusters = np.ascontiguousarray(
        selected_clusters,
        dtype=np.int32
    )


    # --------------------------------------------------------
    # 2C. CLUSTER -> KB LOOKUP AND CANDIDATE POOLING
    # --------------------------------------------------------

    candidate_groups = [
        cluster_to_indices[
            int(cluster_id)
        ]
        for cluster_id
        in selected_clusters
    ]

    candidates = np.concatenate(
        candidate_groups
    )

    candidates = np.ascontiguousarray(
        candidates,
        dtype=np.int32
    )


    # --------------------------------------------------------
    # 2D. COSINE SIMILARITY OVER POOLED CANDIDATES
    # --------------------------------------------------------

    candidate_scores = np.dot(
        train_norm[
            candidates
        ],
        query_norm
    )


    # --------------------------------------------------------
    # 2E. GLOBAL TOP-5 RECORD SELECTION
    # --------------------------------------------------------

    k = min(
        RETRIEVAL_K,
        len(candidate_scores)
    )

    local_top = np.argpartition(
        -candidate_scores,
        k - 1
    )[:k]

    local_top = local_top[
        np.argsort(
            -candidate_scores[local_top]
        )
    ]

    retrieved_indices = candidates[
        local_top
    ]

    similarities = [
        float(
            candidate_scores[
                local_position
            ]
        )
        for local_position
        in local_top
    ]


    # ========================================================
    # RETRIEVAL TIMER END
    # ========================================================

    retrieval_ms = (
        time.perf_counter()
        -
        retrieval_start
    ) * 1000


    # ========================================================
    # RETRIEVAL DIAGNOSTICS
    #
    # IMPORTANT:
    # These calculations occur AFTER the retrieval timer.
    # They therefore do NOT contribute to retrieval_ms
    # and do NOT contribute to e2e_ms.
    # ========================================================

    selected_cluster_scores = [
        float(
            cluster_scores[
                int(cluster_id)
            ]
        )
        for cluster_id
        in selected_clusters
    ]

    candidate_count = int(
        len(candidates)
    )

    candidate_reduction = (
        1.0
        -
        (
            candidate_count
            /
            N_TRAIN
        )
    )

    selected_clusters_string = ",".join(
        str(int(cluster_id))
        for cluster_id
        in selected_clusters
    )

    selected_scores_string = ",".join(
        f"{score:.6f}"
        for score
        in selected_cluster_scores
    )

    # selected_distances_string = ",".join(
    #     f"{distance:.6f}"
    #     for distance
    #     in selected_cluster_distances
    # )


    # ========================================================
    # 3. BUILD CONTEXT / PROMPT
    # ========================================================

    prompt_start = time.perf_counter()

    context = build_context(
        retrieved_indices,
        similarities
    )

    # _ = make_user_prompt(
    #     query_fault,
    #     context
    # )

    prompt_ms = (
        time.perf_counter()
        -
        prompt_start
    ) * 1000


    # ========================================================
    # 4. GPT GENERATION
    # ========================================================

    print(
        "Calling LLM..."
    )

    (
        generated_resolution,
        llm_ms
    ) = generate_resolution(
        query_fault,
        context
    )


    # ========================================================
    # 5. END-TO-END
    #
    # Candidate diagnostics are NOT included.
    # ========================================================

    e2e_ms = (
        encoding_ms
        +
        retrieval_ms
        +
        prompt_ms
        +
        llm_ms
    )


    # ========================================================
    # 6. SAVE QUERY RESULT
    # ========================================================

    query_results.append({

        "query_index":
            q,

        "query_id":
            query_id,

        "source_dataset":
            source_dataset,

        "query_fault":
            query_fault,

        "num_selected_clusters":
            int(
                len(selected_clusters)
            ),

        "selected_clusters":
            selected_clusters_string,

        "selected_cluster_scores":
            selected_scores_string,

        "candidate_count":
            candidate_count,

        "candidate_reduction":
            float(
                candidate_reduction
            ),

        "ground_truth_resolution":
            ground_truth,

        "generated_resolution":
            generated_resolution,

        "encoding_ms":
            encoding_ms,

        "retrieval_ms":
            retrieval_ms,

        "prompt_ms":
            prompt_ms,

        "llm_ms":
            llm_ms,

        "e2e_ms":
            e2e_ms
    })


    # ========================================================
    # 7. SAVE FINAL TOP-5 RETRIEVED RECORDS
    # ========================================================

    for rank, (
        train_index,
        similarity
    ) in enumerate(
        zip(
            retrieved_indices,
            similarities
        ),
        start=1
    ):

        retrieved = train_df.iloc[
            int(train_index)
        ]

        retrieved_record_cluster = int(
            train_cluster_labels[
                int(train_index)
            ]
        )

        cluster_routing_rank = int(
            np.where(
                selected_clusters
                ==
                retrieved_record_cluster
            )[0][0]
        ) + 1

        retrieval_records.append({

            "query_index":
                q,

            "query_id":
                query_id,

            "selected_clusters":
                selected_clusters_string,

            "rank":
                rank,

            "retrieved_train_index":
                int(train_index),

            "retrieved_id":
                safe_text(
                    retrieved[
                        "id"
                    ]
                ),

            "retrieved_record_cluster":
                retrieved_record_cluster,

            "cluster_routing_rank":
                cluster_routing_rank,

            "retrieved_fault":
                safe_text(
                    retrieved[
                        "fault_text"
                    ]
                ),

            "retrieved_resolution":
                safe_text(
                    retrieved[
                        "resolution_text"
                    ]
                ),

            "retrieved_source_dataset":
                safe_text(
                    retrieved[
                        "source_dataset"
                    ]
                ),

            "cosine_similarity":
                similarity
        })


    # ========================================================
    # QUERY DIAGNOSTICS
    #
    # Printing is also outside all latency measurements.
    # ========================================================

    print(
        "Selected clusters:",
        selected_clusters.tolist()
    )

    print(
        "Cluster cosine scores:",
        [
            round(score, 6)
            for score
            in selected_cluster_scores
        ]
    )

    print(
        "Candidate records searched:",
        candidate_count
    )

    print(
        "Candidate reduction:",
        f"{candidate_reduction * 100:.2f}%"
    )

    print(
        "Top-5 cosine similarities:",
        [
            round(
                value,
                4
            )
            for value
            in similarities
        ]
    )

    print(
        "Top-5 source cluster ranks:",
        [
            int(
                np.where(
                    selected_clusters
                    ==
                    train_cluster_labels[
                        int(train_index)
                    ]
                )[0][0]
            ) + 1
            for train_index
            in retrieved_indices
        ]
    )

    print(
        f"Encoding latency: "
        f"{encoding_ms:.3f} ms"
    )

    print(
        f"Retrieval latency: "
        f"{retrieval_ms:.3f} ms"
    )

    print(
        f"Prompt latency: "
        f"{prompt_ms:.3f} ms"
    )

    print(
        f"LLM latency: "
        f"{llm_ms:.3f} ms"
    )

    print(
        f"E2E: "
        f"{e2e_ms / 1000:.3f} sec"
    )


    # ========================================================
    # CHECKPOINT
    #
    # Outside latency measurements.
    # ========================================================

    pd.DataFrame(
        query_results
    ).to_csv(

        os.path.join(
            RESULT_DIR,
            "scala_query_results_checkpoint.csv"
        ),

        index=False
    )


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    query_results
)

retrieval_df = pd.DataFrame(
    retrieval_records
)


# ============================================================
# QUALITY METRICS
# ============================================================

print("\n" + "=" * 90)
print("COMPUTING QUALITY METRICS")
print("=" * 90)

references = (
    results_df[
        "ground_truth_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)

predictions = (
    results_df[
        "generated_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


# ============================================================
# ROUGE-L
# ============================================================

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

rouge_scores = []

for reference, prediction in zip(
    references,
    predictions
):

    score = rouge.score(
        reference,
        prediction
    )

    rouge_scores.append(
        score[
            "rougeL"
        ].fmeasure
    )

results_df[
    "rougeL"
] = rouge_scores


# ============================================================
# BERTSCORE
# ============================================================

print(
    "Calculating BERTScore..."
)

P, R, F1 = bert_score(
    predictions,
    references,
    lang=BERTSCORE_LANG,
    verbose=True,
    device=str(DEVICE)
)

results_df[
    "bertscore_precision"
] = P.cpu().numpy()

results_df[
    "bertscore_recall"
] = R.cpu().numpy()

results_df[
    "bertscore_f1"
] = F1.cpu().numpy()


# ============================================================
# LATENCY SUMMARY
# ============================================================

def latency_summary(
    series
):

    values = (
        series
        .dropna()
        .astype(float)
        .tolist()
    )

    return {

        "mean_ms":
            float(
                np.mean(values)
            ),

        "median_ms":
            float(
                np.median(values)
            ),

        "std_ms":
            float(
                np.std(values)
            ),

        "p95_ms":
            percentile(
                values,
                95
            )
    }


# ============================================================
# CANDIDATE-SPACE SUMMARY
# ============================================================

avg_candidate_count = float(
    results_df[
        "candidate_count"
    ].mean()
)

avg_candidate_reduction = float(
    results_df[
        "candidate_reduction"
    ].mean()
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "method":
        "SCALA Cosine Nearest-Mean Top-15 Clusters",

    "routing_method":
        "Cosine similarity to normalized GMM component means",

    "num_test_queries":
        N_TEST,

    "training_kb_size":
        N_TRAIN,

    "gmm_components":
        int(
            N_CLUSTERS
        ),

    "top_clusters":
        TOP_CLUSTERS,

    "retrieval_k":
        RETRIEVAL_K,

    "avg_candidate_count":
        avg_candidate_count,

    "avg_candidate_reduction":
        avg_candidate_reduction,

    "llm_model":
        LLM_MODEL,

    "temperature":
        0,

    "avg_rougeL":
        float(
            results_df[
                "rougeL"
            ].mean()
        ),

    "avg_bertscore_f1":
        float(
            results_df[
                "bertscore_f1"
            ].mean()
        ),

    "encoding_latency":
        latency_summary(
            results_df[
                "encoding_ms"
            ]
        ),

    "retrieval_latency":
        latency_summary(
            results_df[
                "retrieval_ms"
            ]
        ),

    "prompt_latency":
        latency_summary(
            results_df[
                "prompt_ms"
            ]
        ),

    "llm_latency":
        latency_summary(
            results_df[
                "llm_ms"
            ]
        ),

    "e2e_latency":
        latency_summary(
            results_df[
                "e2e_ms"
            ]
        )
}


# ============================================================
# SAVE RESULTS
# ============================================================

QUERY_RESULTS_CSV = os.path.join(
    RESULT_DIR,
    "scala_query_results.csv"
)

QUERY_RESULTS_XLSX = os.path.join(
    RESULT_DIR,
    "scala_query_results.xlsx"
)

RETRIEVAL_CSV = os.path.join(
    RESULT_DIR,
    "scala_retrieved_records.csv"
)

RETRIEVAL_XLSX = os.path.join(
    RESULT_DIR,
    "scala_retrieved_records.xlsx"
)

SUMMARY_JSON = os.path.join(
    RESULT_DIR,
    "scala_overall_summary.json"
)


results_df.to_csv(
    QUERY_RESULTS_CSV,
    index=False
)

results_df.to_excel(
    QUERY_RESULTS_XLSX,
    index=False
)

retrieval_df.to_csv(
    RETRIEVAL_CSV,
    index=False
)

retrieval_df.to_excel(
    RETRIEVAL_XLSX,
    index=False
)

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


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("SCALA COSINE NEAREST-MEAN TOP-15 EXPERIMENT COMPLETE")
print("=" * 90)


print("\nCONFIGURATION")

print(
    "Total clusters:",
    N_CLUSTERS
)

print(
    "Selected nearest clusters:",
    TOP_CLUSTERS
)

print(
    "Final retrieved records:",
    RETRIEVAL_K
)


print("\nQUALITY")

print(
    "ROUGE-L:",
    f"{summary['avg_rougeL']:.6f}"
)

print(
    "BERTScore F1:",
    f"{summary['avg_bertscore_f1']:.6f}"
)


print("\nCANDIDATE SPACE")

print(
    "Average candidate records:",
    f"{summary['avg_candidate_count']:.2f}"
)

print(
    "Average candidate reduction:",
    f"{summary['avg_candidate_reduction'] * 100:.2f}%"
)


print("\nMEAN LATENCY")

print(
    "Encoding:",
    f"{summary['encoding_latency']['mean_ms']:.6f}",
    "ms"
)

print(
    "Retrieval:",
    f"{summary['retrieval_latency']['mean_ms']:.6f}",
    "ms"
)

print(
    "Prompt:",
    f"{summary['prompt_latency']['mean_ms']:.6f}",
    "ms"
)

print(
    "LLM:",
    f"{summary['llm_latency']['mean_ms']:.6f}",
    "ms"
)


print("\nP95 LATENCY")

print(
    "Encoding:",
    f"{summary['encoding_latency']['p95_ms']:.6f}",
    "ms"
)

print(
    "Retrieval:",
    f"{summary['retrieval_latency']['p95_ms']:.6f}",
    "ms"
)

print(
    "Prompt:",
    f"{summary['prompt_latency']['p95_ms']:.6f}",
    "ms"
)

print(
    "LLM:",
    f"{summary['llm_latency']['p95_ms']:.6f}",
    "ms"
)

print(
    "E2E:",
    f"{summary['e2e_latency']['p95_ms'] / 1000:.6f}",
    "sec"
)


print("\nEND-TO-END")

print(
    "Mean:",
    f"{summary['e2e_latency']['mean_ms'] / 1000:.6f}",
    "sec"
)

print(
    "P95:",
    f"{summary['e2e_latency']['p95_ms'] / 1000:.6f}",
    "sec"
)


print(
    "\nResults saved in:",
    RESULT_DIR
)