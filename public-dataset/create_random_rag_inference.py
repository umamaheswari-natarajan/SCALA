from pathlib import Path
import os
import json
import time
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

from openai import OpenAI

from rouge_score import rouge_scorer
from bert_score import score as bert_score

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "train_fault_resolution.xlsx"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "test_fault_resolution.xlsx"
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

NORMALIZED_EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "train_fault_embeddings_normalized.npy"
)

RANDOM_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "random_groups"
)

# IMPORTANT:
# Use the newly generated L2-normalized random-group means.
RANDOM_MEANS_FILE = os.path.join(
    RANDOM_MODEL_DIR,
    "random_group_means_normalized.npy"
)

RANDOM_LABELS_FILE = os.path.join(
    RANDOM_MODEL_DIR,
    "random_group_labels.npy"
)

RANDOM_LOOKUP_FILE = os.path.join(
    RANDOM_MODEL_DIR,
    "random_group_to_indices.joblib"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "random_group_nearest_mean_top15_rag"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

NUM_GROUPS = 55
TOP_GROUPS = 15
RETRIEVAL_K = 5

LLM_MODEL = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 500

MAX_TEST_QUERIES = None


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("DEVICE")
print("=" * 80)

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
# OPENAI API
# ============================================================

if not os.getenv(
    "OPENAI_API_KEY"
):

    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

client = OpenAI()


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

train_df = pd.read_excel(
    TRAIN_FILE
).reset_index(
    drop=True
)

test_df = pd.read_excel(
    TEST_FILE
).reset_index(
    drop=True
)

if MAX_TEST_QUERIES is not None:

    test_df = test_df.iloc[
        :MAX_TEST_QUERIES
    ].copy()

N_TRAIN = len(
    train_df
)

print(
    "Training records :",
    N_TRAIN
)

print(
    "Test queries     :",
    len(test_df)
)


# ============================================================
# LOAD SAVED NORMALIZED LEARNED KB EMBEDDINGS
#
# Fine-tuned MiniLM
#       ->
# learned fault projection
#       ->
# L2 normalization
#
# These are used for candidate-record cosine retrieval.
# ============================================================

print("\n" + "=" * 80)
print("LOADING NORMALIZED LEARNED KB EMBEDDINGS")
print("=" * 80)

train_norm = np.load(
    NORMALIZED_EMBEDDING_FILE
).astype(
    np.float32
)

train_norm = np.ascontiguousarray(
    train_norm,
    dtype=np.float32
)

print(
    "Normalized KB embedding shape:",
    train_norm.shape
)

if len(train_norm) != N_TRAIN:

    raise ValueError(
        "Training KB and normalized embedding "
        "counts do not match."
    )


# ============================================================
# LOAD RANDOM GROUP ORGANIZATION
# ============================================================

print("\n" + "=" * 80)
print("LOADING RANDOM GROUP ORGANIZATION")
print("=" * 80)

random_means = np.load(
    RANDOM_MEANS_FILE
).astype(
    np.float32
)

random_means = np.ascontiguousarray(
    random_means,
    dtype=np.float32
)

random_labels = np.load(
    RANDOM_LABELS_FILE
).astype(
    np.int32
)

group_to_indices = joblib.load(
    RANDOM_LOOKUP_FILE
)

print(
    "Normalized random means shape:",
    random_means.shape
)

print(
    "Random labels shape:",
    random_labels.shape
)

if random_means.shape[0] != NUM_GROUPS:

    raise ValueError(
        "Unexpected number of random groups."
    )

if len(random_labels) != N_TRAIN:

    raise ValueError(
        "Random labels and training KB "
        "counts do not match."
    )


# ============================================================
# LOAD TRAINED CONTRASTIVE MODEL
# ============================================================

print("\n" + "=" * 80)
print("LOADING TRAINED CONTRASTIVE MODEL")
print("=" * 80)

checkpoint = torch.load(
    PROJECTION_FILE,
    map_location="cpu"
)

BACKBONE_DIM = checkpoint[
    "backbone_dim"
]

PROJECTION_DIM = checkpoint[
    "projection_dim"
]

MAX_LENGTH = checkpoint[
    "max_length"
]

print(
    "Backbone dimension  :",
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
# TOKENIZER + FINE-TUNED BACKBONE
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    BACKBONE_DIR
)

backbone = AutoModel.from_pretrained(
    BACKBONE_DIR
)


# ============================================================
# LOAD TRAINED FAULT PROJECTION HEAD
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

backbone = backbone.to(
    DEVICE
)

fault_projection = fault_projection.to(
    DEVICE
)

backbone.eval()
fault_projection.eval()


# ============================================================
# VALIDATE DIMENSIONS
# ============================================================

if random_means.shape[1] != PROJECTION_DIM:

    raise ValueError(
        "Random means dimension does not match "
        "learned projection dimension."
    )

if train_norm.shape[1] != PROJECTION_DIM:

    raise ValueError(
        "Normalized KB embedding dimension "
        "does not match learned projection dimension."
    )


# ============================================================
# MEAN POOLING
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
        token_embeddings * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(
            dim=1
        ),
        min=1e-9
    )

    return summed / counts


# ============================================================
# SAFE TEXT
# ============================================================

def safe_text(
    value
):

    if pd.isna(
        value
    ):
        return ""

    return str(
        value
    ).strip()


# ============================================================
# QUERY ENCODING
#
# Query:
# fault text
#   ->
# fine-tuned MiniLM
#   ->
# learned fault projection
#   ->
# 384-D embedding
#   ->
# L2 normalization
#
# Only the normalized representation is required online:
#
# 1. cosine routing against normalized random-group means
# 2. cosine retrieval against normalized KB embeddings
# ============================================================

@torch.no_grad()
def encode_query(
    text
):

    tokens = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    tokens = {
        key: value.to(
            DEVICE
        )
        for key, value
        in tokens.items()
    }

    outputs = backbone(
        input_ids=tokens[
            "input_ids"
        ],
        attention_mask=tokens[
            "attention_mask"
        ]
    )

    h = mean_pooling(
        outputs.last_hidden_state,
        tokens[
            "attention_mask"
        ]
    )

    u = fault_projection(
        h
    )

    z = F.normalize(
        u,
        p=2,
        dim=1
    )

    norm_np = (
        z[0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    norm_np = np.ascontiguousarray(
        norm_np,
        dtype=np.float32
    )

    return norm_np


# ============================================================
# BUILD RETRIEVAL CONTEXT
# ============================================================

def build_context(
    retrieved_indices,
    similarities
):

    blocks = []

    for rank, (
        kb_index,
        similarity
    ) in enumerate(
        zip(
            retrieved_indices,
            similarities
        ),
        start=1
    ):

        row = train_df.iloc[
            int(kb_index)
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
            f"Fault:\n"
            f"{fault}\n\n"
            f"Resolution:\n"
            f"{resolution}\n\n"
            f"Similarity Score: "
            f"{float(similarity):.6f}"
        )

        blocks.append(
            block
        )

    return "\n\n---\n\n".join(
        blocks
    )


# ============================================================
# LLM INSTRUCTIONS
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


# ============================================================
# BUILD USER PROMPT
# ============================================================

def make_user_prompt(
    query_fault,
    context
):

    return f"""
NEW FAULT:

{query_fault}

RELEVANT HISTORICAL FAULT-RESOLUTION EXAMPLES:

{context}

Provide the recommended resolution for the new fault.
""".strip()


# ============================================================
# GENERATE RESOLUTION
#
# LLM timing measures only the API generation call.
# Ground truth is NEVER passed to the generator.
# ============================================================

def generate_resolution(
    query_fault,
    context
):

    user_prompt = make_user_prompt(
        query_fault,
        context
    )

    llm_start = time.perf_counter()

    response = client.responses.create(
        model=LLM_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0
    )

    llm_ms = (
        time.perf_counter()
        -
        llm_start
    ) * 1000

    generated_resolution = (
        response.output_text.strip()
    )

    return (
        generated_resolution,
        llm_ms
    )


# ============================================================
# ROUGE
# ============================================================

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)


# ============================================================
# RESULTS
# ============================================================

query_results = []
retrieved_records = []


# ============================================================
# RUN INFERENCE
# ============================================================

print("\n" + "=" * 80)

print(
    "NO GMM + CLIP: RANDOM GROUP "
    "COSINE NEAREST-MEAN TOP-15 / TOP-5 INFERENCE"
)

print("=" * 80)


for idx, row in tqdm(
    test_df.iterrows(),
    total=len(test_df),
    desc="Queries"
):

    # ========================================================
    # QUERY INFORMATION
    # ========================================================

    query_id = (
        row["id"]
        if "id" in test_df.columns
        else idx
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


    # ========================================================
    # 1. QUERY ENCODING
    # ========================================================

    encoding_start = time.perf_counter()

    query_norm = encode_query(
        query_fault
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    encoding_ms = (
        time.perf_counter()
        -
        encoding_start
    ) * 1000


    # ========================================================
    # 2. COSINE ROUTING
    #
    # Compare normalized query against all 55 normalized
    # random-group means.
    #
    # Since both are normalized:
    #
    # cosine similarity = dot product
    #
    # Select Top-15 groups using argpartition.
    #
    # No sorting of the selected 15 groups is required because
    # their ordering does not affect candidate pooling.
    # ========================================================

    routing_start = time.perf_counter()

    group_scores = np.dot(
        random_means,
        query_norm
    )

    selected_groups = np.argpartition(
        -group_scores,
        TOP_GROUPS - 1
    )[:TOP_GROUPS]

    selected_groups = np.ascontiguousarray(
        selected_groups,
        dtype=np.int32
    )

    routing_ms = (
        time.perf_counter()
        -
        routing_start
    ) * 1000


    # ========================================================
    # 3. POOL RECORDS FROM SELECTED 15 GROUPS
    #
    # 4. COSINE SIMILARITY OVER CANDIDATES
    #
    # 5. SELECT FINAL TOP-5
    #
    # These are measured together as retrieval latency.
    # ========================================================

    retrieval_start = time.perf_counter()

    candidate_parts = []

    for group_id in selected_groups:

        indices = np.asarray(
            group_to_indices[
                int(group_id)
            ],
            dtype=np.int64
        )

        candidate_parts.append(
            indices
        )

    candidates = np.concatenate(
        candidate_parts
    )

    candidates = np.ascontiguousarray(
        candidates,
        dtype=np.int64
    )

    candidate_count = int(
        len(candidates)
    )


    # --------------------------------------------------------
    # COSINE SIMILARITY OVER CANDIDATE RECORDS
    #
    # train_norm and query_norm are already normalized.
    # --------------------------------------------------------

    candidate_scores = np.dot(
        train_norm[
            candidates
        ],
        query_norm
    )


    # --------------------------------------------------------
    # FINAL TOP-5
    #
    # argpartition identifies the five largest cosine scores.
    #
    # Only these final five are sorted because they are passed
    # to the LLM as ranked Historical Examples 1-5.
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
            -candidate_scores[
                local_top
            ]
        )
    ]

    retrieved_indices = candidates[
        local_top
    ]

    retrieved_similarities = candidate_scores[
        local_top
    ]

    retrieval_ms = (
        time.perf_counter()
        -
        retrieval_start
    ) * 1000


    # ========================================================
    # CANDIDATE / COMPARISON DIAGNOSTICS
    #
    # Outside retrieval timing.
    # ========================================================

    candidate_fraction = (
        candidate_count
        /
        N_TRAIN
    )

    candidate_reduction = (
        1.0
        -
        candidate_fraction
    )

    routing_comparisons = (
        NUM_GROUPS
    )

    cosine_comparisons = (
        candidate_count
    )

    total_comparisons = (
        routing_comparisons
        +
        cosine_comparisons
    )


    # ========================================================
    # SELECTED GROUP SCORES
    #
    # Diagnostic only and outside routing/retrieval timing.
    # Selected groups are not ordered by these scores.
    # ========================================================

    selected_group_scores = group_scores[
        selected_groups
    ]


    # ========================================================
    # 6. BUILD CONTEXT
    # ========================================================

    prompt_start = time.perf_counter()

    context = build_context(
        retrieved_indices,
        retrieved_similarities
    )

    prompt_ms = (
        time.perf_counter()
        -
        prompt_start
    ) * 1000


    # ========================================================
    # 7. LLM GENERATION
    # ========================================================

    generated_resolution, llm_ms = (
        generate_resolution(
            query_fault,
            context
        )
    )


    # ========================================================
    # 8. E2E LATENCY
    #
    # Same component-based convention:
    #
    # encoding + routing + retrieval + prompt + LLM
    # ========================================================

    e2e_ms = (
        encoding_ms
        +
        routing_ms
        +
        retrieval_ms
        +
        prompt_ms
        +
        llm_ms
    )


    # ========================================================
    # SAVE QUERY RESULT
    # ========================================================

    query_results.append({

        "query_id":
            query_id,

        "query_fault":
            query_fault,

        "ground_truth_resolution":
            ground_truth,

        "generated_resolution":
            generated_resolution,

        "selected_random_groups":
            json.dumps(
                [
                    int(x)
                    for x
                    in selected_groups
                ]
            ),

        "selected_group_cosine_scores":
            json.dumps(
                [
                    float(x)
                    for x
                    in selected_group_scores
                ]
            ),

        "num_selected_groups":
            int(
                len(
                    selected_groups
                )
            ),

        "candidate_count":
            candidate_count,

        "candidate_fraction":
            float(
                candidate_fraction
            ),

        "candidate_reduction":
            float(
                candidate_reduction
            ),

        "routing_comparisons":
            int(
                routing_comparisons
            ),

        "cosine_comparisons":
            int(
                cosine_comparisons
            ),

        "total_similarity_comparisons":
            int(
                total_comparisons
            ),

        "encoding_ms":
            float(
                encoding_ms
            ),

        "routing_ms":
            float(
                routing_ms
            ),

        "retrieval_ms":
            float(
                retrieval_ms
            ),

        "routing_plus_retrieval_ms":
            float(
                routing_ms
                +
                retrieval_ms
            ),

        "prompt_ms":
            float(
                prompt_ms
            ),

        "llm_ms":
            float(
                llm_ms
            ),

        "e2e_ms":
            float(
                e2e_ms
            )
    })


    # ========================================================
    # SAVE TOP-5 RETRIEVED RECORDS
    # ========================================================

    for rank, (
        kb_index,
        similarity
    ) in enumerate(
        zip(
            retrieved_indices,
            retrieved_similarities
        ),
        start=1
    ):

        kb_index = int(
            kb_index
        )

        kb_row = train_df.iloc[
            kb_index
        ]

        retrieved_records.append({

            "query_id":
                query_id,

            "query_fault":
                query_fault,

            "rank":
                rank,

            "training_index":
                kb_index,

            "training_id":
                (
                    kb_row["id"]
                    if "id"
                    in train_df.columns
                    else kb_index
                ),

            "random_group":
                int(
                    random_labels[
                        kb_index
                    ]
                ),

            "retrieved_fault":
                safe_text(
                    kb_row[
                        "fault_text"
                    ]
                ),

            "retrieved_resolution":
                safe_text(
                    kb_row[
                        "resolution_text"
                    ]
                ),

            "cosine_similarity":
                float(
                    similarity
                ),

            "selected_random_groups":
                json.dumps(
                    [
                        int(x)
                        for x
                        in selected_groups
                    ]
                )
        })


# ============================================================
# CREATE DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    query_results
)

retrieved_df = pd.DataFrame(
    retrieved_records
)


# ============================================================
# SAVE CHECKPOINT RESULTS BEFORE QUALITY METRICS
# ============================================================

CHECKPOINT_CSV = os.path.join(
    RESULT_DIR,
    "random_group_top15_query_results_checkpoint.csv"
)

results_df.to_csv(
    CHECKPOINT_CSV,
    index=False
)


# ============================================================
# ROUGE-L
# ============================================================

print("\n" + "=" * 80)
print("COMPUTING ROUGE-L")
print("=" * 80)

rouge_l_scores = []

for _, row in tqdm(
    results_df.iterrows(),
    total=len(results_df),
    desc="ROUGE-L"
):

    reference = safe_text(
        row[
            "ground_truth_resolution"
        ]
    )

    generated = safe_text(
        row[
            "generated_resolution"
        ]
    )

    rouge_result = rouge.score(
        reference,
        generated
    )

    rouge_l_scores.append(
        rouge_result[
            "rougeL"
        ].fmeasure
    )

results_df[
    "rouge_l"
] = rouge_l_scores


# ============================================================
# BERTSCORE
# ============================================================

print("\n" + "=" * 80)
print("COMPUTING BERTSCORE")
print("=" * 80)

references = (
    results_df[
        "ground_truth_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)

generated_resolutions = (
    results_df[
        "generated_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)

P, R, F1 = bert_score(
    generated_resolutions,
    references,
    lang="en",
    verbose=True,
    device=str(
        DEVICE
    )
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
# SAVE FINAL QUERY RESULTS
# ============================================================

QUERY_XLSX = os.path.join(
    RESULT_DIR,
    "random_group_top15_query_results.xlsx"
)

QUERY_CSV = os.path.join(
    RESULT_DIR,
    "random_group_top15_query_results.csv"
)

results_df.to_excel(
    QUERY_XLSX,
    index=False
)

results_df.to_csv(
    QUERY_CSV,
    index=False
)


# ============================================================
# SAVE RETRIEVED RECORDS
# ============================================================

RETRIEVED_XLSX = os.path.join(
    RESULT_DIR,
    "random_group_top15_retrieved_records.xlsx"
)

RETRIEVED_CSV = os.path.join(
    RESULT_DIR,
    "random_group_top15_retrieved_records.csv"
)

retrieved_df.to_excel(
    RETRIEVED_XLSX,
    index=False
)

retrieved_df.to_csv(
    RETRIEVED_CSV,
    index=False
)


# ============================================================
# LATENCY HELPER
# ============================================================

def latency_summary(
    series
):

    values = np.asarray(
        series,
        dtype=np.float64
    )

    return {

        "mean_ms":
            float(
                np.mean(
                    values
                )
            ),

        "median_ms":
            float(
                np.median(
                    values
                )
            ),

        "std_ms":
            float(
                np.std(
                    values
                )
            ),

        "p95_ms":
            float(
                np.percentile(
                    values,
                    95
                )
            )
    }


# ============================================================
# OVERALL SUMMARY
# ============================================================

summary = {

    "method":
        (
            "Learned Embeddings + Random Groups + "
            "Cosine Nearest-Mean Routing"
        ),

    "organization":
        "Random assignment",

    "balanced_random_groups":
        False,

    "num_random_groups":
        NUM_GROUPS,

    "num_routed_groups":
        TOP_GROUPS,

    "retrieval_k":
        RETRIEVAL_K,

    "routing_metric":
        (
            "Cosine similarity to L2-normalized "
            "random-group means"
        ),

    "record_retrieval_metric":
        "Cosine similarity",

    "top_k_implementation":
        (
            "argpartition for Top-15 routing and "
            "Top-5 candidate selection; final Top-5 "
            "sorted by cosine similarity"
        ),

    "training_records":
        N_TRAIN,

    "test_queries":
        int(
            len(
                results_df
            )
        ),

    "embedding_dimension":
        int(
            PROJECTION_DIM
        ),

    "generator":
        LLM_MODEL,

    "avg_candidate_count":
        float(
            results_df[
                "candidate_count"
            ].mean()
        ),

    "avg_cosine_comparisons":
        float(
            results_df[
                "cosine_comparisons"
            ].mean()
        ),

    "routing_comparisons_per_query":
        NUM_GROUPS,

    "avg_total_similarity_comparisons":
        float(
            results_df[
                "total_similarity_comparisons"
            ].mean()
        ),

    "avg_candidate_fraction":
        float(
            results_df[
                "candidate_fraction"
            ].mean()
        ),

    "avg_search_space_reduction":
        float(
            results_df[
                "candidate_reduction"
            ].mean()
        ),

    "rouge_l":
        float(
            results_df[
                "rouge_l"
            ].mean()
        ),

    "bertscore_f1":
        float(
            results_df[
                "bertscore_f1"
            ].mean()
        ),

    "latency": {

        "encoding":
            latency_summary(
                results_df[
                    "encoding_ms"
                ]
            ),

        "routing":
            latency_summary(
                results_df[
                    "routing_ms"
                ]
            ),

        "retrieval":
            latency_summary(
                results_df[
                    "retrieval_ms"
                ]
            ),

        "routing_plus_retrieval":
            latency_summary(
                results_df[
                    "routing_plus_retrieval_ms"
                ]
            ),

        "prompt":
            latency_summary(
                results_df[
                    "prompt_ms"
                ]
            ),

        "llm":
            latency_summary(
                results_df[
                    "llm_ms"
                ]
            ),

        "e2e":
            latency_summary(
                results_df[
                    "e2e_ms"
                ]
            )
    }
}


# ============================================================
# SAVE SUMMARY
# ============================================================

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "random_group_top15_overall_summary.json"
)

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
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)

print(
    "NO GMM + CLIP: RANDOM GROUP COSINE "
    "NEAREST-MEAN TOP-15 / TOP-5 EXPERIMENT COMPLETE"
)

print("=" * 80)

print(
    "Queries:",
    len(
        results_df
    )
)

print(
    "Random groups:",
    NUM_GROUPS
)

print(
    "Selected groups:",
    TOP_GROUPS
)

print(
    "Final retrieved pairs:",
    RETRIEVAL_K
)

print(
    "\nAverage candidate count:",
    f"{summary['avg_candidate_count']:.2f}"
)

print(
    "Average cosine comparisons:",
    f"{summary['avg_cosine_comparisons']:.2f}"
)

print(
    "Routing comparisons/query:",
    summary[
        "routing_comparisons_per_query"
    ]
)

print(
    "Average total similarity comparisons:",
    f"{summary['avg_total_similarity_comparisons']:.2f}"
)

print(
    "Average search-space reduction:",
    f"{summary['avg_search_space_reduction'] * 100:.2f}%"
)

print(
    "\nROUGE-L:",
    f"{summary['rouge_l']:.6f}"
)

print(
    "BERTScore F1:",
    f"{summary['bertscore_f1']:.6f}"
)

print(
    "\nMean Encoding:",
    f"{summary['latency']['encoding']['mean_ms']:.3f} ms"
)

print(
    "Mean Routing:",
    f"{summary['latency']['routing']['mean_ms']:.3f} ms"
)

print(
    "Mean Retrieval:",
    f"{summary['latency']['retrieval']['mean_ms']:.3f} ms"
)

print(
    "Mean Routing + Retrieval:",
    f"{summary['latency']['routing_plus_retrieval']['mean_ms']:.3f} ms"
)

print(
    "Mean Prompt:",
    f"{summary['latency']['prompt']['mean_ms']:.3f} ms"
)

print(
    "Mean LLM:",
    f"{summary['latency']['llm']['mean_ms'] / 1000:.3f} sec"
)

print(
    "Mean E2E:",
    f"{summary['latency']['e2e']['mean_ms'] / 1000:.3f} sec"
)

print(
    "P95 E2E:",
    f"{summary['latency']['e2e']['p95_ms'] / 1000:.3f} sec"
)

print("\nFiles saved:")

print(
    QUERY_XLSX
)

print(
    QUERY_CSV
)

print(
    RETRIEVED_XLSX
)

print(
    RETRIEVED_CSV
)

print(
    SUMMARY_FILE
)