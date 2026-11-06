from pathlib import Path
import os
import time
import json
import random
import joblib

import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer

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
    "embeddings",
    "minilm"
)

GMM_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gmm_no_clip"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "gmm_no_clip_top15_rag"
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

# Same normalized pretrained MiniLM embeddings used by
# Standard RAG.
TRAIN_NORM_EMBEDDING_FILE = os.path.join(
    EMBEDDING_DIR,
    "minilm_train_fault_embeddings_normalized.npy"
)

# GMM fitted on RAW pretrained MiniLM embeddings.
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

TOP_CLUSTERS = 15
RETRIEVAL_K = 5


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

LLM_MODEL = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 500


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42

# None = all 872 test queries
MAX_TEST_QUERIES = None

BERTSCORE_LANG = "en"


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 90)
print("DEVICE")
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

    norm = np.linalg.norm(
        vector
    )

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

    print(
        "\nMissing required files:"
    )

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

N_TRAIN = len(
    train_df
)

N_TEST = len(
    test_df
)

print(
    "Queries being processed:",
    N_TEST
)


# ============================================================
# LOAD NORMALIZED PRETRAINED MINILM KB EMBEDDINGS
#
# These are the SAME representations used by Standard RAG.
#
# No CLIP.
# No projection head.
# ============================================================

print("\n" + "=" * 90)
print("LOADING PRETRAINED MINILM KB EMBEDDINGS")
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
        "Training embedding count does not "
        "match training dataset."
    )


# ============================================================
# LOAD GMM + NO CLIP ROUTING ARTIFACTS
# ============================================================

print("\n" + "=" * 90)
print("LOADING GMM + NO CLIP ROUTING ARTIFACTS")
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
    "Normalized GMM means shape:",
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


# ============================================================
# ROUTING ARTIFACT SANITY CHECKS
# ============================================================

if gmm_means.shape[0] != N_CLUSTERS:

    raise ValueError(
        "Number of GMM means does not "
        "match cluster lookup."
    )

if gmm_means.shape[1] != train_norm.shape[1]:

    raise ValueError(
        "GMM mean dimension does not "
        "match embedding dimension."
    )

if len(train_cluster_labels) != N_TRAIN:

    raise ValueError(
        "Cluster label count does not "
        "match training KB size."
    )

if TOP_CLUSTERS > N_CLUSTERS:

    raise ValueError(
        "TOP_CLUSTERS cannot exceed "
        "the number of clusters."
    )

total_cluster_records = sum(
    len(indices)
    for indices
    in cluster_to_indices.values()
)

if total_cluster_records != N_TRAIN:

    raise ValueError(
        "Cluster lookup does not contain "
        "all training KB records."
    )

print(
    "Total records in cluster lookup:",
    total_cluster_records
)


# ============================================================
# LOAD ORIGINAL PRETRAINED MINILM
#
# IMPORTANT:
#
# This is the same query representation used by Standard RAG.
#
# No CLIP fine-tuning.
# No contrastive backbone.
# No fault projection head.
# ============================================================

print("\n" + "=" * 90)
print("LOADING ORIGINAL PRETRAINED MINILM")
print("=" * 90)

encoder = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

print(
    "Model:",
    MODEL_NAME
)

print(
    "CLIP representation: No"
)

print(
    "Projection head: No"
)


# ============================================================
# ONLINE QUERY ENCODING
#
# fault text
#      ↓
# original pretrained all-MiniLM-L6-v2
#      ↓
# raw 384-D embedding
#      ↓
# L2 normalization
#
# This is identical to the representation used in
# Standard RAG.
# ============================================================

def encode_fault_online(
    fault_text
):

    start = time.perf_counter()

    raw = encoder.encode(
        [fault_text],
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False
    )[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    raw = raw.astype(
        np.float32
    )

    raw = np.ascontiguousarray(
        raw,
        dtype=np.float32
    )

    norm = normalize_vector(
        raw
    ).astype(
        np.float32
    )

    norm = np.ascontiguousarray(
        norm,
        dtype=np.float32
    )

    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000

    return (
        norm,
        elapsed_ms
    )


# ============================================================
# BUILD CONTEXT
#
# Kept identical to final SCALA.
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
print("GMM + NO CLIP: COSINE NEAREST-MEAN TOP-15 INFERENCE")
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
    "Representation:",
    "Pretrained all-MiniLM-L6-v2"
)

print(
    "Total GMM clusters:",
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
        query_norm,
        encoding_ms
    ) = encode_fault_online(
        query_fault
    )


    # ========================================================
    # 2. RETRIEVAL TIMER START
    #
    # Includes:
    #
    # - cosine against all 55 GMM means
    # - Top-15 selection
    # - cluster -> KB lookup
    # - candidate pooling
    # - candidate cosine similarities
    # - final Top-5 selection
    # ========================================================

    retrieval_start = time.perf_counter()


    # --------------------------------------------------------
    # 2A. COSINE AGAINST ALL 55 NORMALIZED GMM MEANS
    # --------------------------------------------------------

    cluster_scores = np.dot(
        gmm_means,
        query_norm
    )


    # --------------------------------------------------------
    # 2B. TOP-15 GMM CLUSTERS
    #
    # No sorting is needed because cluster order does not
    # affect candidate pooling.
    # --------------------------------------------------------

    selected_clusters = np.argpartition(
        -cluster_scores,
        TOP_CLUSTERS - 1
    )[:TOP_CLUSTERS]

    selected_clusters = np.ascontiguousarray(
        selected_clusters,
        dtype=np.int32
    )


    # --------------------------------------------------------
    # 2C. POOL RECORDS FROM SELECTED CLUSTERS
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
    # 2D. COSINE SIMILARITY OVER CANDIDATE RECORDS
    # --------------------------------------------------------

    candidate_scores = np.dot(
        train_norm[
            candidates
        ],
        query_norm
    )


    # --------------------------------------------------------
    # 2E. FINAL TOP-5
    # --------------------------------------------------------

    k = min(
        RETRIEVAL_K,
        len(candidate_scores)
    )

    local_top = np.argpartition(
        -candidate_scores,
        k - 1
    )[:k]

    # Sort ONLY the final selected Top-5.
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
    # Outside retrieval timing.
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
        str(
            int(cluster_id)
        )
        for cluster_id
        in selected_clusters
    )

    selected_scores_string = ",".join(
        f"{score:.6f}"
        for score
        in selected_cluster_scores
    )


    # ========================================================
    # 3. BUILD CONTEXT
    # ========================================================

    prompt_start = time.perf_counter()

    context = build_context(
        retrieved_indices,
        similarities
    )

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
                len(
                    selected_clusters
                )
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
                int(
                    train_index
                ),

            "retrieved_id":
                safe_text(
                    retrieved[
                        "id"
                    ]
                ),

            "retrieved_record_cluster":
                retrieved_record_cluster,

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
    # ========================================================

    print(
        "Selected clusters:",
        selected_clusters.tolist()
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
    # ========================================================

    pd.DataFrame(
        query_results
    ).to_csv(
        os.path.join(
            RESULT_DIR,
            "gmm_no_clip_query_results_checkpoint.csv"
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
            percentile(
                values,
                95
            )
    }


# ============================================================
# SUMMARY
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

summary = {

    "method":
        "GMM + No CLIP",

    "embedding_model":
        MODEL_NAME,

    "uses_clip_representation":
        False,

    "uses_gmm":
        True,

    "routing_method":
        (
            "Cosine similarity to normalized "
            "GMM component means"
        ),

    "num_test_queries":
        N_TEST,

    "training_kb_size":
        N_TRAIN,

    "embedding_dimension":
        int(
            train_norm.shape[1]
        ),

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
    "gmm_no_clip_query_results.csv"
)

QUERY_RESULTS_XLSX = os.path.join(
    RESULT_DIR,
    "gmm_no_clip_query_results.xlsx"
)

RETRIEVAL_CSV = os.path.join(
    RESULT_DIR,
    "gmm_no_clip_retrieved_records.csv"
)

RETRIEVAL_XLSX = os.path.join(
    RESULT_DIR,
    "gmm_no_clip_retrieved_records.xlsx"
)

SUMMARY_JSON = os.path.join(
    RESULT_DIR,
    "gmm_no_clip_overall_summary.json"
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
print("GMM + NO CLIP EXPERIMENT COMPLETE")
print("=" * 90)

print("\nCONFIGURATION")

print(
    "Representation:",
    "Pretrained all-MiniLM-L6-v2"
)

print(
    "Total GMM clusters:",
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