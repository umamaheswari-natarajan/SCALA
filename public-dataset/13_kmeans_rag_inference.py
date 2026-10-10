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
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

KMEANS_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "kmeans"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_rag"
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
    "train_fault_embeddings_normalized.npy"
)

KMEANS_FILE = os.path.join(
    KMEANS_MODEL_DIR,
    "kmeans_55.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

RETRIEVAL_K = 5

LLM_MODEL = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 500

MAX_TEST_QUERIES = None

BERTSCORE_LANG = "en"

SEED = 42


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

print("\nTraining KB:", N_TRAIN)
print("Test queries:", N_TEST)


# ============================================================
# LOAD TRAINING EMBEDDINGS
# ============================================================

train_norm = np.load(
    TRAIN_NORM_FILE
)

print(
    "Training embedding shape:",
    train_norm.shape
)


# ============================================================
# LOAD K-MEANS
# ============================================================

kmeans = joblib.load(
    KMEANS_FILE
)

print(
    "K-Means clusters:",
    kmeans.n_clusters
)


train_cluster_labels = kmeans.labels_


cluster_to_indices = {}

for cluster_id in range(
    kmeans.n_clusters
):

    cluster_to_indices[
        cluster_id
    ] = np.where(
        train_cluster_labels
        ==
        cluster_id
    )[0]


cluster_sizes = [
    len(cluster_to_indices[c])
    for c in range(kmeans.n_clusters)
]

print(
    "Smallest cluster:",
    min(cluster_sizes)
)

print(
    "Largest cluster:",
    max(cluster_sizes)
)


# ============================================================
# LOAD CONTRASTIVE ENCODER
# ============================================================

HF_MODEL_DIR = os.path.join(
    CONTRASTIVE_MODEL_DIR,
    "backbone"
)

PROJECTION_FILE = os.path.join(
    CONTRASTIVE_MODEL_DIR,
    "projection_heads.pt"
)


tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_DIR
)

encoder = AutoModel.from_pretrained(
    HF_MODEL_DIR
)

encoder = encoder.to(DEVICE)
encoder.eval()


checkpoint = torch.load(
    PROJECTION_FILE,
    map_location=DEVICE
)

MAX_LENGTH = int(
    checkpoint["max_length"]
)


fault_projection = nn.Linear(
    int(checkpoint["backbone_dim"]),
    int(checkpoint["projection_dim"]),
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
        token_embeddings * mask,
        dim=1
    )

    denominator = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / denominator


# ============================================================
# ONLINE QUERY ENCODING
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
        key: value.to(DEVICE)
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
        .astype(np.float32)
    )


    norm_np = normalize_vector(
        raw_np
    )


    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    return (
        norm_np,
        elapsed_ms
    )


# ============================================================
# CONTEXT
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

        blocks.append(block)

    return "\n\n".join(
        blocks
    )


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

if not os.getenv("OPENAI_API_KEY"):

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
print("K-MEANS RAG INFERENCE")
print("=" * 90)


for q in range(N_TEST):

    row = test_df.iloc[q]

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
        query_norm,
        encoding_ms
    ) = encode_fault_online(
        query_fault
    )


    # ========================================================
    # 2. FLAT CLIP TOP-5
    #
    # Used only as retrieval reference.
    # No GPT call.
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
    # 3. K-MEANS ROUTING
    # ========================================================

    routing_start = time.perf_counter()

    selected_cluster = int(
        kmeans.predict(
            query_norm.reshape(
                1,
                -1
            )
        )[0]
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
    # 4. RETRIEVAL INSIDE ONE CLUSTER
    # ========================================================

    retrieval_start = time.perf_counter()

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

    kmeans_indices = candidates[
        local_top
    ]

    kmeans_similarities = [
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
    # 5. OVERLAP WITH FLAT CLIP TOP-5
    # ========================================================

    overlap_count = len(
        set(flat_indices.tolist())
        &
        set(kmeans_indices.tolist())
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

    prompt_start = time.perf_counter()

    context = build_context(
        kmeans_indices,
        kmeans_similarities
    )

    prompt_ms = (
        time.perf_counter()
        -
        prompt_start
    ) * 1000


    # ========================================================
    # 8. GPT GENERATION
    # ========================================================

    generated_resolution, llm_ms = (
        generate_resolution(
            query_fault,
            context
        )
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

        "selected_kmeans_cluster":
            selected_cluster,

        "candidate_count":
            candidate_count,

        "candidate_fraction":
            candidate_fraction,

        "candidate_reduction":
            candidate_reduction,

        "clip_kmeans_overlap_count":
            overlap_count,

        "clip_kmeans_overlap_rate":
            overlap_rate,

        "kmeans_generated_resolution":
            generated_resolution,

        "encoding_ms":
            encoding_ms,

        "kmeans_routing_ms":
            routing_ms,

        "kmeans_retrieval_ms":
            retrieval_ms,

        "kmeans_prompt_ms":
            prompt_ms,

        "kmeans_llm_ms":
            llm_ms,

        "kmeans_e2e_ms":
            e2e_ms
    })


    # ========================================================
    # RETRIEVED RECORD DETAILS
    # ========================================================

    for rank, (
        train_index,
        similarity
    ) in enumerate(
        zip(
            kmeans_indices,
            kmeans_similarities
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

            "kmeans_cluster":
                selected_cluster,

            "cosine_similarity":
                similarity
        })


    print(
        f"{q + 1}/{N_TEST} | "
        f"Cluster={selected_cluster} | "
        f"Candidates={candidate_count} | "
        f"Reduction={candidate_reduction:.4f} | "
        f"Top-{RETRIEVAL_K} overlap={overlap_rate:.4f} | "
        f"E2E={e2e_ms / 1000:.3f}s"
    )


    pd.DataFrame(
        query_results
    ).to_csv(
        os.path.join(
            RESULT_DIR,
            "kmeans_query_results_checkpoint.csv"
        ),
        index=False
    )


# ============================================================
# QUALITY METRICS
# ============================================================

results_df = pd.DataFrame(
    query_results
)

retrieval_df = pd.DataFrame(
    retrieval_records
)


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
        "kmeans_generated_resolution"
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
        score["rougeL"].fmeasure
    )


results_df[
    "kmeans_rougeL"
] = rouge_scores


# ============================================================
# BERTSCORE
# ============================================================

print("\nCalculating K-Means BERTScore...")


P, R, F1 = bert_score(
    predictions,
    references,
    lang=BERTSCORE_LANG,
    verbose=True,
    device=str(DEVICE)
)


results_df[
    "kmeans_bertscore_precision"
] = P.cpu().numpy()

results_df[
    "kmeans_bertscore_recall"
] = R.cpu().numpy()

results_df[
    "kmeans_bertscore_f1"
] = F1.cpu().numpy()


# ============================================================
# LATENCY SUMMARY
# ============================================================

def latency_summary(series):

    values = (
        series
        .dropna()
        .astype(float)
        .tolist()
    )

    return {
        "mean_ms":
            float(np.mean(values)),

        "median_ms":
            float(np.median(values)),

        "std_ms":
            float(np.std(values)),

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

    "num_test_queries":
        N_TEST,

    "training_kb_size":
        N_TRAIN,

    "kmeans_clusters":
        int(kmeans.n_clusters),

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

    "avg_clip_kmeans_topk_overlap":
        float(
            results_df[
                "clip_kmeans_overlap_rate"
            ].mean()
        ),

    "avg_rougeL":
        float(
            results_df[
                "kmeans_rougeL"
            ].mean()
        ),

    "avg_bertscore_f1":
        float(
            results_df[
                "kmeans_bertscore_f1"
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
                "kmeans_routing_ms"
            ]
        ),

    "retrieval_latency":
        latency_summary(
            results_df[
                "kmeans_retrieval_ms"
            ]
        ),

    "llm_latency":
        latency_summary(
            results_df[
                "kmeans_llm_ms"
            ]
        ),

    "e2e_latency":
        latency_summary(
            results_df[
                "kmeans_e2e_ms"
            ]
        )
}


# ============================================================
# SAVE
# ============================================================

results_df.to_excel(
    os.path.join(
        RESULT_DIR,
        "kmeans_query_results.xlsx"
    ),
    index=False
)

results_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "kmeans_query_results.csv"
    ),
    index=False
)

retrieval_df.to_excel(
    os.path.join(
        RESULT_DIR,
        "kmeans_retrieved_records.xlsx"
    ),
    index=False
)

with open(
    os.path.join(
        RESULT_DIR,
        "kmeans_overall_summary.json"
    ),
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
print("K-MEANS RAG EXPERIMENT COMPLETE")
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
    f"CLIP/K-Means Top-{RETRIEVAL_K} overlap:",
    f"{summary['avg_clip_kmeans_topk_overlap']:.4f}"
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

print("\nLATENCY")

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
print(RESULT_DIR)