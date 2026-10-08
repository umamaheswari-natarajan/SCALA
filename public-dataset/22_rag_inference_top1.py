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

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

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


# ============================================================
# SEPARATE OUTPUT DIRECTORY FOR TOP-1
# ============================================================

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top1"
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


# ============================================================
# TRAIN EMBEDDINGS FROM CONTRASTIVE MODEL
# ============================================================

TRAIN_RAW_EMBEDDING_FILE = os.path.join(
    EMBEDDING_DIR,
    "train_fault_embeddings_raw.npy"
)

TRAIN_NORM_EMBEDDING_FILE = os.path.join(
    EMBEDDING_DIR,
    "train_fault_embeddings_normalized.npy"
)


# ============================================================
# OPTIONAL TEST EMBEDDINGS
# ============================================================

TEST_RAW_EMBEDDING_FILE = os.path.join(
    EMBEDDING_DIR,
    "test_fault_embeddings_raw.npy"
)

TEST_NORM_EMBEDDING_FILE = os.path.join(
    EMBEDDING_DIR,
    "test_fault_embeddings_normalized.npy"
)


# ============================================================
# SAVED GMM
# ============================================================

GMM_FILE = os.path.join(
    GMM_MODEL_DIR,
    "best_gmm.joblib"
)


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

# ------------------------------------------------------------
# ONLY ALGORITHMIC CHANGE
# ------------------------------------------------------------

NUM_GMM_CLUSTERS = 1


# ------------------------------------------------------------
# Final historical examples sent to GPT.
#
# IMPORTANT:
# Top-1 cluster does NOT mean Top-1 record.
# We still retrieve the final Top-5 records.
# ------------------------------------------------------------

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

# First use 5 for sanity testing.
# Then change to None for all 872.
MAX_TEST_QUERIES = None


# ============================================================
# BERTSCORE
# ============================================================

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

    torch.cuda.manual_seed_all(
        SEED
    )


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


def percentile(
    values,
    p
):

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            p
        )
    )


def top_k(
    scores,
    k
):

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
    TRAIN_RAW_EMBEDDING_FILE,
    TRAIN_NORM_EMBEDDING_FILE,
    GMM_FILE
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
# TEST EMBEDDING CHECK
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "TEST EMBEDDING CHECK"
)

print(
    "=" * 90
)


test_embeddings_exist = (
    os.path.exists(
        TEST_RAW_EMBEDDING_FILE
    )
    and
    os.path.exists(
        TEST_NORM_EMBEDDING_FILE
    )
)


if test_embeddings_exist:

    print(
        "Saved test embeddings FOUND."
    )

    print(
        "They will NOT be used for latency."
    )

    print(
        "Every test fault will still be encoded online."
    )

else:

    print(
        "Saved test embeddings NOT found."
    )

    print(
        "This is fine."
    )


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "LOADING DATASETS"
)

print(
    "=" * 90
)


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
# LOAD TRAIN EMBEDDINGS
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "LOADING TRAINING EMBEDDINGS"
)

print(
    "=" * 90
)


train_raw = np.load(
    TRAIN_RAW_EMBEDDING_FILE
)

train_norm = np.load(
    TRAIN_NORM_EMBEDDING_FILE
)


print(
    "Raw shape:",
    train_raw.shape
)

print(
    "Normalized shape:",
    train_norm.shape
)


if len(train_raw) != N_TRAIN:

    raise ValueError(
        "Training embedding count does not match training dataset."
    )


# ============================================================
# LOAD GMM
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "LOADING GMM"
)

print(
    "=" * 90
)


gmm = joblib.load(
    GMM_FILE
)


print(
    "GMM components:",
    gmm.n_components
)


# ============================================================
# TRAINING CLUSTER ASSIGNMENTS
# ============================================================

train_cluster_labels = gmm.predict(
    train_raw
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


print(
    "\n" + "=" * 90
)

print(
    "LOADING TRAINED CONTRASTIVE MODEL"
)

print(
    "=" * 90
)


print(
    "Backbone:",
    HF_MODEL_DIR
)

print(
    "Projection checkpoint:",
    PROJECTION_FILE
)


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


print(
    "Saved epoch:",
    checkpoint["epoch"]
)

print(
    "Saved backbone dimension:",
    checkpoint["backbone_dim"]
)

print(
    "Saved projection dimension:",
    checkpoint["projection_dim"]
)

print(
    "Original model:",
    checkpoint["model_name"]
)

print(
    "Maximum sequence length:",
    checkpoint["max_length"]
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


    norm_np = normalize_vector(
        raw_np
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
# CANDIDATE INDICES
# ============================================================

def get_candidate_indices(
    selected_clusters
):

    arrays = [
        cluster_to_indices[
            int(cluster_id)
        ]
        for cluster_id
        in selected_clusters
    ]


    if len(arrays) == 0:

        return np.array(
            [],
            dtype=int
        )


    return np.unique(
        np.concatenate(
            arrays
        )
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

print(
    "\n" + "=" * 90
)

print(
    "RETRIEVAL-AWARE GMM TOP-1 INFERENCE"
)

print(
    "=" * 90
)


print(
    "Test queries:",
    N_TEST
)

print(
    "Training KB:",
    N_TRAIN
)

print(
    "GMM clusters/query:",
    NUM_GMM_CLUSTERS
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
    # 1. ONLINE ENCODING
    # ========================================================

    (
        query_raw,
        query_norm,
        encoding_ms
    ) = encode_fault_online(
        query_fault
    )


    # ========================================================
    # 2. UNRESTRICTED CLIP TOP-5
    #
    # Reference only for retrieval-overlap calculation.
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
    # 3. GMM TOP-1 ROUTING
    # ========================================================

    routing_start = time.perf_counter()


    probabilities = gmm.predict_proba(
        query_raw.reshape(
            1,
            -1
        )
    )[0]


    cluster_order = np.argsort(
        -probabilities
    )


    selected_clusters = cluster_order[
        :NUM_GMM_CLUSTERS
    ]


    selected_probabilities = [
        float(
            probabilities[c]
        )
        for c
        in selected_clusters
    ]


    candidates = get_candidate_indices(
        selected_clusters
    )


    routing_ms = (
        time.perf_counter()
        -
        routing_start
    ) * 1000


    # ========================================================
    # 4. RETRIEVE FINAL TOP-5 FROM SELECTED CLUSTER
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


    gmm_indices = candidates[
        local_top
    ]


    gmm_similarities = [
        float(
            candidate_scores[
                local_position
            ]
        )
        for local_position
        in local_top
    ]


    retrieval_ms = (
        time.perf_counter()
        -
        retrieval_start
    ) * 1000


    # ========================================================
    # 5. CANDIDATE METRICS
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
    # 6. OVERLAP WITH UNRESTRICTED CLIP TOP-5
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
    # 7. BUILD CONTEXT
    # ========================================================

    prompt_start = time.perf_counter()


    context = build_context(
        gmm_indices,
        gmm_similarities
    )


    _ = make_user_prompt(
        query_fault,
        context
    )


    prompt_ms = (
        time.perf_counter()
        -
        prompt_start
    ) * 1000


    # ========================================================
    # 8. GPT GENERATION
    #
    # ONLY ONE GPT CALL.
    # ========================================================

    print(
        "Calling LLM for Retrieval-Aware GMM Top-1..."
    )


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
    # 10. POSTERIOR ENTROPY
    # ========================================================

    posterior_entropy = float(
        -np.sum(
            probabilities
            *
            np.log(
                probabilities
                +
                1e-12
            )
        )
    )


    # ========================================================
    # SAVE QUERY RESULT
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

        "ground_truth_resolution":
            ground_truth,


        "selected_clusters":
            ",".join(
                map(
                    str,
                    selected_clusters
                )
            ),


        "selected_cluster_probabilities":
            ",".join(
                f"{p:.8f}"
                for p
                in selected_probabilities
            ),


        "max_gmm_probability":
            float(
                probabilities.max()
            ),


        "posterior_entropy":
            posterior_entropy,


        "candidate_count":
            candidate_count,

        "candidate_fraction":
            candidate_fraction,

        "candidate_reduction":
            candidate_reduction,


        "flat_gmm_overlap_count":
            overlap_count,

        "flat_gmm_overlap_rate":
            overlap_rate,


        "gmm_top1_generated_resolution":
            generated_resolution,


        "encoding_ms":
            encoding_ms,

        "gmm_top1_routing_ms":
            routing_ms,

        "gmm_top1_retrieval_ms":
            retrieval_ms,

        "gmm_top1_prompt_ms":
            prompt_ms,

        "gmm_top1_llm_ms":
            llm_ms,

        "gmm_top1_e2e_ms":
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

            "method":
                "retrieval_aware_gmm_top1",

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

            "retrieved_fault":
                safe_text(
                    retrieved[
                        "fault_text"
                    ]
                ),

            "retrieved_root_cause":
                safe_text(
                    retrieved.get(
                        "root_cause",
                        ""
                    )
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

            "retrieved_cluster":
                int(
                    train_cluster_labels[
                        train_index
                    ]
                ),

            "cosine_similarity":
                similarity
        })


    print(
        f"Selected cluster: "
        f"{selected_clusters.tolist()}"
    )


    print(
        f"Candidates: "
        f"{candidate_count}/{N_TRAIN} "
        f"| Reduction: "
        f"{candidate_reduction:.4f}"
    )


    print(
        f"Flat/Top-1 GMM "
        f"Top-{RETRIEVAL_K} overlap: "
        f"{overlap_rate:.4f}"
    )


    print(
        f"Top-1 GMM E2E: "
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
            "rag_top1_query_results_checkpoint.csv"
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

print(
    "\n" + "=" * 90
)

print(
    "COMPUTING TOP-1 QUALITY METRICS"
)

print(
    "=" * 90
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
        "gmm_top1_generated_resolution"
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
    "gmm_top1_rougeL"
] = rouge_scores


# ============================================================
# BERTSCORE
# ============================================================

print(
    "Calculating Top-1 GMM BERTScore..."
)


P, R, F1 = bert_score(
    predictions,
    references,
    lang=BERTSCORE_LANG,
    verbose=True,
    device=str(DEVICE)
)


results_df[
    "gmm_top1_bertscore_precision"
] = P.cpu().numpy()


results_df[
    "gmm_top1_bertscore_recall"
] = R.cpu().numpy()


results_df[
    "gmm_top1_bertscore_f1"
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

    "method":
        "Retrieval-Aware GMM Top-1",

    "num_test_queries":
        N_TEST,

    "training_kb_size":
        N_TRAIN,

    "gmm_components":
        int(
            gmm.n_components
        ),

    "gmm_clusters_selected":
        NUM_GMM_CLUSTERS,

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


    "avg_flat_gmm_topk_overlap":
        float(
            results_df[
                "flat_gmm_overlap_rate"
            ].mean()
        ),


    "gmm_avg_rougeL":
        float(
            results_df[
                "gmm_top1_rougeL"
            ].mean()
        ),


    "gmm_avg_bertscore_f1":
        float(
            results_df[
                "gmm_top1_bertscore_f1"
            ].mean()
        ),


    "encoding_latency":
        latency_summary(
            results_df[
                "encoding_ms"
            ]
        ),


    "gmm_routing_latency":
        latency_summary(
            results_df[
                "gmm_top1_routing_ms"
            ]
        ),


    "gmm_retrieval_latency":
        latency_summary(
            results_df[
                "gmm_top1_retrieval_ms"
            ]
        ),


    "gmm_prompt_latency":
        latency_summary(
            results_df[
                "gmm_top1_prompt_ms"
            ]
        ),


    "gmm_llm_latency":
        latency_summary(
            results_df[
                "gmm_top1_llm_ms"
            ]
        ),


    "gmm_e2e_latency":
        latency_summary(
            results_df[
                "gmm_top1_e2e_ms"
            ]
        )
}


# ============================================================
# SAVE RESULTS
# ============================================================

QUERY_RESULTS_CSV = os.path.join(
    RESULT_DIR,
    "rag_top1_query_results.csv"
)


QUERY_RESULTS_XLSX = os.path.join(
    RESULT_DIR,
    "rag_top1_query_results.xlsx"
)


RETRIEVAL_CSV = os.path.join(
    RESULT_DIR,
    "rag_top1_retrieved_records.csv"
)


RETRIEVAL_XLSX = os.path.join(
    RESULT_DIR,
    "rag_top1_retrieved_records.xlsx"
)


SUMMARY_JSON = os.path.join(
    RESULT_DIR,
    "rag_top1_overall_summary.json"
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

print(
    "\n" + "=" * 90
)

print(
    "RETRIEVAL-AWARE GMM TOP-1 EXPERIMENT COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nAverage candidate count:",
    f"{summary['avg_candidate_count']:.2f}"
)


print(
    "Average candidate reduction:",
    f"{summary['avg_candidate_reduction']:.4f}"
)


print(
    f"Flat/Top-1 GMM "
    f"Top-{RETRIEVAL_K} overlap:",
    f"{summary['avg_flat_gmm_topk_overlap']:.4f}"
)


print(
    "\nQUALITY"
)


print(
    "Top-1 ROUGE-L:",
    f"{summary['gmm_avg_rougeL']:.4f}"
)


print(
    "Top-1 BERTScore F1:",
    f"{summary['gmm_avg_bertscore_f1']:.4f}"
)


print(
    "\nMEAN LATENCY"
)


print(
    "Encoding:",
    f"{summary['encoding_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "GMM routing:",
    f"{summary['gmm_routing_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "GMM candidate retrieval:",
    f"{summary['gmm_retrieval_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "Prompt:",
    f"{summary['gmm_prompt_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "\nEND-TO-END"
)


print(
    "Top-1 GMM mean:",
    f"{summary['gmm_e2e_latency']['mean_ms'] / 1000:.3f}",
    "sec"
)


print(
    "Top-1 GMM P95:",
    f"{summary['gmm_e2e_latency']['p95_ms'] / 1000:.3f}",
    "sec"
)


print(
    "\nResults saved in:"
)

print(
    RESULT_DIR
)