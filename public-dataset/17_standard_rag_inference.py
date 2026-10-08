import os
import time
import json
import random

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

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "dataset_splits"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embeddings",
    "minilm"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "standard_rag"
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

TRAIN_NORM_FILE = os.path.join(
    EMBEDDING_DIR,
    "minilm_train_fault_embeddings_normalized.npy"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_K = 5

LLM_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 500

# First use 5.
# Then change to None for the full 872-query experiment.
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
    "Test queries:",
    N_TEST
)


# ============================================================
# LOAD SAVED NORMALIZED MINILM TRAINING EMBEDDINGS
#
# These are the exact same embeddings created before fitting
# the plain GMM baseline.
# ============================================================

print("\n" + "=" * 90)
print("LOADING MINILM TRAINING EMBEDDINGS")
print("=" * 90)


train_norm = np.load(
    TRAIN_NORM_FILE
)


print(
    "Training embedding shape:",
    train_norm.shape
)


if len(train_norm) != N_TRAIN:

    raise ValueError(
        "Training embedding count does not match training KB."
    )


# ============================================================
# LOAD ORIGINAL PRETRAINED MINILM
#
# No CLIP fine-tuning.
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
# fault text
#    |
#    v
# pretrained MiniLM
#    |
#    v
# 384-D embedding
#    |
#    v
# L2 normalization
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
        norm,
        elapsed_ms
    )


# ============================================================
# BUILD RETRIEVED CONTEXT
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
print("STANDARD RAG INFERENCE")
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
    # 1. QUERY ENCODING
    # ========================================================

    (
        query_norm,
        encoding_ms
    ) = encode_query(
        query_fault
    )


    # ========================================================
    # 2. FULL-KB COSINE RETRIEVAL
    #
    # No clustering.
    # No routing.
    #
    # Compare the query against ALL 3049 records.
    # ========================================================

    retrieval_start = (
        time.perf_counter()
    )


    scores = np.dot(
        train_norm,
        query_norm
    )


    retrieved_indices = top_k(
        scores,
        RETRIEVAL_K
    )


    similarities = [
        float(
            scores[i]
        )
        for i in retrieved_indices
    ]


    retrieval_ms = (
        time.perf_counter()
        -
        retrieval_start
    ) * 1000


    # ========================================================
    # 3. BUILD PROMPT
    # ========================================================

    prompt_start = (
        time.perf_counter()
    )


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
    # 4. LLM GENERATION
    # ========================================================

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
    # Standard RAG has NO routing term.
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

        "candidate_count":
            N_TRAIN,

        "candidate_fraction":
            1.0,

        "candidate_reduction":
            0.0,

        "standard_rag_generated_resolution":
            generated_resolution,

        "encoding_ms":
            encoding_ms,

        "standard_rag_retrieval_ms":
            retrieval_ms,

        "standard_rag_prompt_ms":
            prompt_ms,

        "standard_rag_llm_ms":
            llm_ms,

        "standard_rag_e2e_ms":
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
            retrieved_indices,
            similarities
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
        f"Candidates={N_TRAIN} | "
        f"Top-{RETRIEVAL_K} retrieved | "
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
            "standard_rag_query_results_checkpoint.csv"
        ),

        index=False
    )


# ============================================================
# BUILD DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    query_results
)


retrieval_df = pd.DataFrame(
    retrieval_records
)


# ============================================================
# QUALITY INPUTS
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
        "standard_rag_generated_resolution"
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
    "standard_rag_rougeL"
] = rouge_scores


# ============================================================
# BERTSCORE
# ============================================================

print(
    "\nCalculating Standard RAG BERTScore..."
)


P, R, F1 = bert_score(
    predictions,
    references,
    lang=BERTSCORE_LANG,
    verbose=True,
    device=DEVICE
)


results_df[
    "standard_rag_bertscore_precision"
] = P.cpu().numpy()


results_df[
    "standard_rag_bertscore_recall"
] = R.cpu().numpy()


results_df[
    "standard_rag_bertscore_f1"
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
# FINAL SUMMARY
# ============================================================

summary = {

    "baseline":
        "Standard RAG",

    "num_test_queries":
        N_TEST,

    "training_kb_size":
        N_TRAIN,

    "embedding_model":
        MODEL_NAME,

    "embedding_dimension":
        int(
            train_norm.shape[1]
        ),

    "uses_clip_representation":
        False,

    "uses_clustering":
        False,

    "uses_routing":
        False,

    "retrieval_scope":
        "Full training KB",

    "retrieval_k":
        RETRIEVAL_K,

    "llm_model":
        LLM_MODEL,

    "temperature":
        0,

    "avg_candidate_count":
        float(
            N_TRAIN
        ),

    "avg_candidate_reduction":
        0.0,

    "avg_rougeL":
        float(
            results_df[
                "standard_rag_rougeL"
            ].mean()
        ),

    "avg_bertscore_f1":
        float(
            results_df[
                "standard_rag_bertscore_f1"
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
                "standard_rag_retrieval_ms"
            ]
        ),

    "prompt_latency":
        latency_summary(
            results_df[
                "standard_rag_prompt_ms"
            ]
        ),

    "llm_latency":
        latency_summary(
            results_df[
                "standard_rag_llm_ms"
            ]
        ),

    "e2e_latency":
        latency_summary(
            results_df[
                "standard_rag_e2e_ms"
            ]
        )
}


# ============================================================
# SAVE
# ============================================================

RESULT_XLSX = os.path.join(
    RESULT_DIR,
    "standard_rag_query_results.xlsx"
)


RESULT_CSV = os.path.join(
    RESULT_DIR,
    "standard_rag_query_results.csv"
)


RETRIEVAL_XLSX = os.path.join(
    RESULT_DIR,
    "standard_rag_retrieved_records.xlsx"
)


SUMMARY_JSON = os.path.join(
    RESULT_DIR,
    "standard_rag_overall_summary.json"
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
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("STANDARD RAG EXPERIMENT COMPLETE")
print("=" * 90)


print(
    "Average candidate count:",
    f"{summary['avg_candidate_count']:.2f}"
)


print(
    "Average candidate reduction:",
    f"{summary['avg_candidate_reduction']:.4f}"
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