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
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "dataset_splits"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embeddings",
    "minilm"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "plain_gmm"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_rag"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


TRAIN_FILE = os.path.join(
    SPLIT_DIR,
    "train_fault_resolution.xlsx"
)

TEST_FILE = os.path.join(
    SPLIT_DIR,
    "test_fault_resolution.xlsx"
)

TRAIN_RAW_FILE = os.path.join(
    EMBEDDING_DIR,
    "minilm_train_fault_embeddings.npy"
)

TRAIN_NORM_FILE = os.path.join(
    EMBEDDING_DIR,
    "minilm_train_fault_embeddings_normalized.npy"
)

GMM_FILE = os.path.join(
    MODEL_DIR,
    "plain_gmm_55.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_K = 5

LLM_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 500

# First test with 5.
# Then change to None for all 872 queries.
MAX_TEST_QUERIES = None

BERTSCORE_LANG = "en"

SEED = 42


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

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


def top_k(scores, k):

    k = min(
        k,
        len(scores)
    )

    return np.argsort(
        -scores
    )[:k]


def percentile(values, p):

    return float(
        np.percentile(
            values,
            p
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("LOADING DATA")
print("=" * 90)

train_df = pd.read_excel(
    TRAIN_FILE
)

test_df = pd.read_excel(
    TEST_FILE
)

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
    "Training KB records:",
    N_TRAIN
)

print(
    "Test records:",
    N_TEST
)


# ============================================================
# LOAD SAVED MINILM TRAINING EMBEDDINGS
# ============================================================

print("\n" + "=" * 90)
print("LOADING TRAINING EMBEDDINGS")
print("=" * 90)

train_raw = np.load(
    TRAIN_RAW_FILE
)

train_norm = np.load(
    TRAIN_NORM_FILE
)

print(
    "Raw shape:",
    train_raw.shape
)

print(
    "Normalized shape:",
    train_norm.shape
)


if (
    len(train_raw)
    !=
    N_TRAIN
):

    raise ValueError(
        "Training embedding count does not match training KB."
    )


# ============================================================
# LOAD PLAIN GMM
# ============================================================

print("\n" + "=" * 90)
print("LOADING PLAIN GMM")
print("=" * 90)

gmm = joblib.load(
    GMM_FILE
)

print(
    "GMM components:",
    gmm.n_components
)


# ============================================================
# BUILD TRAINING CLUSTER MEMBERSHIP
#
# For each training fault:
#
#   cluster =
#   argmax_k P(C_k | z_i)
# ============================================================

train_probabilities = (
    gmm.predict_proba(
        train_raw
    )
)

train_cluster_labels = np.argmax(
    train_probabilities,
    axis=1
)


cluster_to_indices = {}

for cluster_id in range(
    gmm.n_components
):

    cluster_to_indices[
        cluster_id
    ] = np.where(
        train_cluster_labels
        ==
        cluster_id
    )[0]


cluster_sizes = [
    len(
        cluster_to_indices[c]
    )
    for c in range(
        gmm.n_components
    )
]


print(
    "Smallest cluster:",
    min(cluster_sizes)
)

print(
    "Largest cluster:",
    max(cluster_sizes)
)

print(
    "Mean cluster size:",
    np.mean(cluster_sizes)
)


# ============================================================
# LOAD ORIGINAL PRETRAINED MINILM
#
# IMPORTANT:
# No CLIP-trained backbone.
# No projection head.
# ============================================================

print("\n" + "=" * 90)
print("LOADING PRETRAINED MINILM")
print("=" * 90)

encoder = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

print(
    "Model:",
    MODEL_NAME
)


# ============================================================
# ONLINE QUERY ENCODING
#
# fault_text
#    |
#    v
# MiniLM
#    |
#    v
# raw 384-D embedding
#    |
#    +----> raw for GMM routing
#    |
#    +----> normalized for cosine retrieval
# ============================================================

def encode_query(
    fault_text
):

    start = time.perf_counter()

    raw = encoder.encode(
        [fault_text],
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False
    )[0]

    raw = raw.astype(
        np.float32
    )

    norm = normalize_vector(
        raw
    ).astype(
        np.float32
    )

    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000

    return (
        raw,
        norm,
        elapsed_ms
    )


# ============================================================
# CONTEXT BUILDING
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

        block = (
            f"Historical Example {rank}\n"
            f"Fault: {safe_text(row['fault_text'])}\n"
            f"Resolution: "
            f"{safe_text(row['resolution_text'])}\n"
            f"Similarity: {float(similarity):.4f}"
        )

        blocks.append(
            block
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# LLM PROMPT
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
        "OPENAI_API_KEY is not set."
    )


client = OpenAI()


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
# MAIN LOOP
# ============================================================

query_results = []

retrieval_records = []


print("\n" + "=" * 90)
print("PLAIN GMM-RAG INFERENCE")
print("=" * 90)


for q in range(
    N_TEST
):

    row = test_df.iloc[
        q
    ]

    query_id = safe_text(
        row["id"]
    )

    query_fault = safe_text(
        row["fault_text"]
    )

    ground_truth = safe_text(
        row["resolution_text"]
    )


    # ========================================================
    # 1. ENCODING
    # ========================================================

    (
        query_raw,
        query_norm,
        encoding_ms
    ) = encode_query(
        query_fault
    )


    # ========================================================
    # 2. UNRESTRICTED MINILM TOP-5
    #
    # Reference only for overlap.
    # NO GPT CALL.
    # ========================================================

    flat_scores = np.dot(
        train_norm,
        query_norm
    )

    flat_indices = top_k(
        flat_scores,
        RETRIEVAL_K
    )


    # ========================================================
    # 3. GMM ROUTING
    #
    # Query probabilities:
    #
    # P(C1|q), ..., P(C55|q)
    #
    # Standard baseline chooses ONLY:
    #
    # argmax_k P(C_k | q)
    # ========================================================

    routing_start = (
        time.perf_counter()
    )

    probabilities = (
        gmm.predict_proba(
            query_raw.reshape(
                1,
                -1
            )
        )[0]
    )

    selected_cluster = int(
        np.argmax(
            probabilities
        )
    )

    selected_probability = float(
        probabilities[
            selected_cluster
        ]
    )

    candidates = cluster_to_indices[
        selected_cluster
    ]

    routing_ms = (
        time.perf_counter()
        -
        routing_start
    ) * 1000


    # ========================================================
    # 4. RETRIEVAL INSIDE SELECTED CLUSTER
    #
    # cosine(query, candidate faults)
    # ========================================================

    retrieval_start = (
        time.perf_counter()
    )

    candidate_scores = np.dot(
        train_norm[
            candidates
        ],
        query_norm
    )

    local_top = top_k(
        candidate_scores,
        RETRIEVAL_K
    )

    gmm_indices = candidates[
        local_top
    ]

    gmm_similarities = [
        float(
            candidate_scores[i]
        )
        for i in local_top
    ]

    retrieval_ms = (
        time.perf_counter()
        -
        retrieval_start
    ) * 1000


    # ========================================================
    # 5. OVERLAP WITH UNRESTRICTED MINILM TOP-5
    # ========================================================

    overlap_count = len(
        set(
            flat_indices.tolist()
        )
        &
        set(
            gmm_indices.tolist()
        )
    )

    overlap_rate = (
        overlap_count
        /
        RETRIEVAL_K
    )


    # ========================================================
    # 6. CANDIDATE REDUCTION
    # ========================================================

    candidate_count = len(
        candidates
    )

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


    # ========================================================
    # 7. BUILD PROMPT
    # ========================================================

    prompt_start = (
        time.perf_counter()
    )

    context = build_context(
        gmm_indices,
        gmm_similarities
    )

    prompt_ms = (
        time.perf_counter()
        -
        prompt_start
    ) * 1000


    # ========================================================
    # 8. GPT GENERATION
    # ========================================================

    (
        generated_resolution,
        llm_ms
    ) = generate_resolution(
        query_fault,
        context
    )


    # ========================================================
    # 9. E2E
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

        "query_index":
            q,

        "query_id":
            query_id,

        "query_fault":
            query_fault,

        "ground_truth_resolution":
            ground_truth,

        "selected_gmm_cluster":
            selected_cluster,

        "selected_gmm_probability":
            selected_probability,

        "candidate_count":
            candidate_count,

        "candidate_fraction":
            candidate_fraction,

        "candidate_reduction":
            candidate_reduction,

        "minilm_gmm_overlap_count":
            overlap_count,

        "minilm_gmm_overlap_rate":
            overlap_rate,

        "plain_gmm_generated_resolution":
            generated_resolution,

        "encoding_ms":
            encoding_ms,

        "plain_gmm_routing_ms":
            routing_ms,

        "plain_gmm_retrieval_ms":
            retrieval_ms,

        "plain_gmm_prompt_ms":
            prompt_ms,

        "plain_gmm_llm_ms":
            llm_ms,

        "plain_gmm_e2e_ms":
            e2e_ms
    })


    # ========================================================
    # SAVE RETRIEVED RECORDS
    # ========================================================

    for rank, (
        train_index,
        similarity
    ) in enumerate(
        zip(
            gmm_indices,
            gmm_similarities
        ),
        start=1
    ):

        retrieved = train_df.iloc[
            int(train_index)
        ]

        retrieval_records.append({

            "query_index":
                q,

            "query_id":
                query_id,

            "selected_cluster":
                selected_cluster,

            "rank":
                rank,

            "retrieved_train_index":
                int(train_index),

            "retrieved_id":
                safe_text(
                    retrieved["id"]
                ),

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

            "cosine_similarity":
                similarity
        })


    print(
        f"{q + 1}/{N_TEST} | "
        f"Cluster={selected_cluster} | "
        f"P={selected_probability:.4f} | "
        f"Candidates={candidate_count} | "
        f"Reduction={candidate_reduction:.4f} | "
        f"Top-{RETRIEVAL_K} overlap="
        f"{overlap_rate:.4f} | "
        f"E2E={e2e_ms / 1000:.3f}s"
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    pd.DataFrame(
        query_results
    ).to_csv(

        os.path.join(
            RESULT_DIR,
            "plain_gmm_query_results_checkpoint.csv"
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
# QUALITY REFERENCES
# ============================================================

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
        "plain_gmm_generated_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


# ============================================================
# ROUGE-L
# ============================================================

print("\nCalculating ROUGE-L...")

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
    "plain_gmm_rougeL"
] = rouge_scores


# ============================================================
# BERTSCORE
# ============================================================

print(
    "\nCalculating Plain GMM BERTScore..."
)


P, R, F1 = bert_score(
    predictions,
    references,
    lang=BERTSCORE_LANG,
    verbose=True,
    device=DEVICE
)


results_df[
    "plain_gmm_bertscore_precision"
] = P.cpu().numpy()


results_df[
    "plain_gmm_bertscore_recall"
] = R.cpu().numpy()


results_df[
    "plain_gmm_bertscore_f1"
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
# SUMMARY
# ============================================================

summary = {

    "baseline":
        "Plain GMM-RAG",

    "num_test_queries":
        N_TEST,

    "training_kb_size":
        N_TRAIN,

    "embedding_model":
        MODEL_NAME,

    "embedding_dimension":
        int(
            train_raw.shape[1]
        ),

    "uses_clip_representation":
        False,

    "uses_resolution_for_representation_learning":
        False,

    "gmm_clusters":
        int(
            gmm.n_components
        ),

    "cluster_routing":
        "Top-1 posterior cluster",

    "retrieval_k":
        RETRIEVAL_K,

    "llm_model":
        LLM_MODEL,

    "temperature":
        0,

    "avg_candidate_count":
        float(
            results_df[
                "candidate_count"
            ].mean()
        ),

    "avg_candidate_reduction":
        float(
            results_df[
                "candidate_reduction"
            ].mean()
        ),

    "avg_minilm_gmm_topk_overlap":
        float(
            results_df[
                "minilm_gmm_overlap_rate"
            ].mean()
        ),

    "avg_selected_cluster_probability":
        float(
            results_df[
                "selected_gmm_probability"
            ].mean()
        ),

    "avg_rougeL":
        float(
            results_df[
                "plain_gmm_rougeL"
            ].mean()
        ),

    "avg_bertscore_f1":
        float(
            results_df[
                "plain_gmm_bertscore_f1"
            ].mean()
        ),

    "encoding_latency":
        latency_summary(
            results_df[
                "encoding_ms"
            ]
        ),

    "routing_latency":
        latency_summary(
            results_df[
                "plain_gmm_routing_ms"
            ]
        ),

    "retrieval_latency":
        latency_summary(
            results_df[
                "plain_gmm_retrieval_ms"
            ]
        ),

    "prompt_latency":
        latency_summary(
            results_df[
                "plain_gmm_prompt_ms"
            ]
        ),

    "llm_latency":
        latency_summary(
            results_df[
                "plain_gmm_llm_ms"
            ]
        ),

    "e2e_latency":
        latency_summary(
            results_df[
                "plain_gmm_e2e_ms"
            ]
        )
}


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_XLSX = os.path.join(
    RESULT_DIR,
    "plain_gmm_query_results.xlsx"
)


RESULT_CSV = os.path.join(
    RESULT_DIR,
    "plain_gmm_query_results.csv"
)


RETRIEVAL_XLSX = os.path.join(
    RESULT_DIR,
    "plain_gmm_retrieved_records.xlsx"
)


SUMMARY_JSON = os.path.join(
    RESULT_DIR,
    "plain_gmm_overall_summary.json"
)


results_df.to_excel(
    RESULT_XLSX,
    index=False
)


results_df.to_csv(
    RESULT_CSV,
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
# FINAL
# ============================================================

print("\n" + "=" * 90)
print("PLAIN GMM-RAG EXPERIMENT COMPLETE")
print("=" * 90)


print(
    "Average candidate count:",
    f"{summary['avg_candidate_count']:.2f}"
)

print(
    "Average candidate reduction:",
    f"{summary['avg_candidate_reduction']:.4f}"
)

print(
    f"MiniLM/GMM Top-{RETRIEVAL_K} overlap:",
    f"{summary['avg_minilm_gmm_topk_overlap']:.4f}"
)


print("\nQUALITY")


print(
    "ROUGE-L:",
    f"{summary['avg_rougeL']:.4f}"
)


print(
    "BERTScore F1:",
    f"{summary['avg_bertscore_f1']:.4f}"
)


print("\nMEAN LATENCY")


print(
    "Encoding:",
    f"{summary['encoding_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "Routing:",
    f"{summary['routing_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "Retrieval:",
    f"{summary['retrieval_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "Prompt:",
    f"{summary['prompt_latency']['mean_ms']:.3f}",
    "ms"
)


print("\nEND-TO-END")


print(
    "Mean E2E:",
    f"{summary['e2e_latency']['mean_ms'] / 1000:.3f}",
    "sec"
)


print(
    "P95 E2E:",
    f"{summary['e2e_latency']['p95_ms'] / 1000:.3f}",
    "sec"
)


print("\nResults saved in:")
print(
    RESULT_DIR
)