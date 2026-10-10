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

GMM_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gmm"
)


# ============================================================
# IMPORTANT:
# Separate folder for Top-3 results.
# Existing Top-5 results will NOT be overwritten.
# ============================================================

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top3"
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
# SAVED TRAIN EMBEDDINGS
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
#
# These are checked only.
# Test queries are encoded online for latency measurement.
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
# ONLY ALGORITHMIC CHANGE FROM THE EXISTING TOP-5 EXPERIMENT
# ------------------------------------------------------------

NUM_GMM_CLUSTERS = 3


# ------------------------------------------------------------
# Final number of historical records sent to GPT.
#
# IMPORTANT:
# This remains 5.
# We route to 3 clusters but retrieve Top-5 records.
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


# ------------------------------------------------------------
# First run with 5.
# After sanity checking, change to None.
# ------------------------------------------------------------

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

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


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

    return str(
        value
    ).strip()


def normalize_vector(
    vector
):

    norm = np.linalg.norm(
        vector
    )

    if norm == 0:
        return vector

    return (
        vector
        /
        norm
    )


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
# CHECK OPTIONAL SAVED TEST EMBEDDINGS
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
        "Each test query will be encoded online."
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
        .iloc[
            :MAX_TEST_QUERIES
        ]
        .copy()
        .reset_index(
            drop=True
        )
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
# LOAD TRAINING EMBEDDINGS
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
        "Training embedding count does not match "
        "training dataset."
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
# TRAINING KB CLUSTER ASSIGNMENTS
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
        cluster_to_indices[
            c
        ]
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
# LOAD TRAINED CONTRASTIVE MODEL
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


# ------------------------------------------------------------
# Fine-tuned MiniLM backbone
# ------------------------------------------------------------

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


hidden_size = (
    encoder.config.hidden_size
)


print(
    "Backbone dimension:",
    hidden_size
)


# ------------------------------------------------------------
# Projection checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    PROJECTION_FILE,
    map_location=DEVICE
)


print(
    "Saved epoch:",
    checkpoint[
        "epoch"
    ]
)

print(
    "Saved backbone dimension:",
    checkpoint[
        "backbone_dim"
    ]
)

print(
    "Saved projection dimension:",
    checkpoint[
        "projection_dim"
    ]
)

print(
    "Original model:",
    checkpoint[
        "model_name"
    ]
)

print(
    "Maximum sequence length:",
    checkpoint[
        "max_length"
    ]
)


PROJECTED_DIM = int(
    checkpoint[
        "projection_dim"
    ]
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
        model_output
        .last_hidden_state
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

    start = (
        time.perf_counter()
    )


    encoded = tokenizer(

        fault_text,

        padding=True,

        truncation=True,

        max_length=
            MAX_LENGTH,

        return_tensors=
            "pt"
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


        raw_embedding = (
            fault_projection(
                pooled
            )
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
# RETRIEVAL HELPERS
# ============================================================

def get_candidate_indices(
    selected_clusters
):

    arrays = [

        cluster_to_indices[
            int(
                cluster_id
            )
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
# BUILD RAG CONTEXT
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
            int(
                train_index
            )
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
            f"Similarity: "
            f"{float(similarity):.4f}"
        )


        blocks.append(
            block
        )


    return "\n\n".join(
        blocks
    )


# ============================================================
# RAG PROMPT
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
# OPENAI CLIENT
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
# GENERATE RESOLUTION
# ============================================================

def generate_resolution(
    query_fault,
    context
):

    user_prompt = make_user_prompt(
        query_fault,
        context
    )


    start = (
        time.perf_counter()
    )


    response = client.responses.create(

        model=
            LLM_MODEL,

        instructions=
            SYSTEM_PROMPT,

        input=
            user_prompt,

        max_output_tokens=
            MAX_OUTPUT_TOKENS,

        temperature=0
    )


    elapsed_ms = (

        time.perf_counter()
        -
        start

    ) * 1000


    generated_text = (
        response
        .output_text
        .strip()
    )


    return (
        generated_text,
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
    "RETRIEVAL-AWARE GMM TOP-3 INFERENCE"
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
    # 1. ONLINE QUERY ENCODING
    # ========================================================

    (
        query_raw,
        query_norm,
        encoding_ms
    ) = encode_fault_online(
        query_fault
    )


    print(
        f"Encoding: "
        f"{encoding_ms:.3f} ms"
    )


    # ========================================================
    # 2. UNRESTRICTED CLIP RETRIEVAL
    #
    # IMPORTANT:
    # This is used ONLY as a reference for Top-5 overlap.
    #
    # There is NO Flat-RAG GPT call in this Top-3 experiment.
    # ========================================================

    flat_start = (
        time.perf_counter()
    )


    flat_scores = np.dot(
        train_norm,
        query_norm
    )


    flat_indices = top_k(
        flat_scores,
        RETRIEVAL_K
    )


    flat_retrieval_ms = (

        time.perf_counter()
        -
        flat_start

    ) * 1000


    # ========================================================
    # 3. GMM ROUTING
    #
    # Calculate posterior probabilities for all GMM clusters.
    #
    # Select ONLY the 3 highest-probability clusters.
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


    cluster_order = np.argsort(
        -probabilities
    )


    selected_clusters = (
        cluster_order[
            :NUM_GMM_CLUSTERS
        ]
    )


    selected_probabilities = [

        float(
            probabilities[
                c
            ]
        )

        for c in selected_clusters
    ]


    candidates = (
        get_candidate_indices(
            selected_clusters
        )
    )


    routing_ms = (

        time.perf_counter()
        -
        routing_start

    ) * 1000


    # ========================================================
    # 4. RETRIEVE TOP-5 FROM TOP-3 CLUSTER CANDIDATES
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


    gmm_indices = (
        candidates[
            local_top
        ]
    )


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
    # 6. TOP-5 OVERLAP WITH UNRESTRICTED CLIP RETRIEVAL
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
    # 7. BUILD TOP-3 GMM RAG CONTEXT
    # ========================================================

    prompt_start = (
        time.perf_counter()
    )


    gmm_context = build_context(
        gmm_indices,
        gmm_similarities
    )


    # Build prompt here so prompt-building latency is measured.
    _ = make_user_prompt(
        query_fault,
        gmm_context
    )


    prompt_ms = (

        time.perf_counter()
        -
        prompt_start

    ) * 1000


    # ========================================================
    # 8. GPT GENERATION
    #
    # ONLY ONE GPT CALL PER QUERY.
    # ========================================================

    print(
        "Calling LLM for "
        "Retrieval-Aware GMM Top-3..."
    )


    (
        gmm_generated,
        gmm_llm_ms
    ) = generate_resolution(
        query_fault,
        gmm_context
    )


    # ========================================================
    # 9. END-TO-END LATENCY
    # ========================================================

    gmm_e2e_ms = (

        encoding_ms
        +
        routing_ms
        +
        retrieval_ms
        +
        prompt_ms
        +
        gmm_llm_ms
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
    # 11. SAVE QUERY RESULT
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


        # ----------------------------------------------------
        # Routing
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Candidate pool
        # ----------------------------------------------------

        "candidate_count":
            candidate_count,

        "candidate_fraction":
            candidate_fraction,

        "candidate_reduction":
            candidate_reduction,


        # ----------------------------------------------------
        # Retrieval overlap
        # ----------------------------------------------------

        "flat_gmm_overlap_count":
            overlap_count,

        "flat_gmm_overlap_rate":
            overlap_rate,


        # ----------------------------------------------------
        # Generated output
        # ----------------------------------------------------

        "gmm_top3_generated_resolution":
            gmm_generated,


        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        "encoding_ms":
            encoding_ms,

        "flat_reference_retrieval_ms":
            flat_retrieval_ms,

        "gmm_top3_routing_ms":
            routing_ms,

        "gmm_top3_retrieval_ms":
            retrieval_ms,

        "gmm_top3_prompt_ms":
            prompt_ms,

        "gmm_top3_llm_ms":
            gmm_llm_ms,

        "gmm_top3_e2e_ms":
            gmm_e2e_ms
    })


    # ========================================================
    # 12. SAVE RETRIEVED TOP-5 RECORDS
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
            int(
                train_index
            )
        ]


        retrieval_records.append({

            "query_index":
                q,

            "query_id":
                query_id,

            "method":
                "retrieval_aware_gmm_top3",

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


    # ========================================================
    # PRINT QUERY SUMMARY
    # ========================================================

    print(
        f"Selected clusters: "
        f"{selected_clusters.tolist()}"
    )


    print(
        f"Candidates: "
        f"{candidate_count}/{N_TRAIN} "
        f"| Reduction: "
        f"{candidate_reduction:.4f}"
    )


    print(
        f"Flat/Top-3 GMM "
        f"Top-{RETRIEVAL_K} overlap: "
        f"{overlap_rate:.4f}"
    )


    print(
        f"Top-3 GMM E2E: "
        f"{gmm_e2e_ms / 1000:.3f} sec"
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    pd.DataFrame(
        query_results
    ).to_csv(

        os.path.join(
            RESULT_DIR,
            "rag_top3_query_results_checkpoint.csv"
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
    "COMPUTING TOP-3 RESOLUTION QUALITY METRICS"
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


gmm_predictions = (

    results_df[
        "gmm_top3_generated_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


# ============================================================
# ROUGE-L
# ============================================================

rouge = rouge_scorer.RougeScorer(
    [
        "rougeL"
    ],
    use_stemmer=True
)


gmm_rouge = []


for reference, prediction in zip(
    references,
    gmm_predictions
):

    score = rouge.score(
        reference,
        prediction
    )


    gmm_rouge.append(
        score[
            "rougeL"
        ].fmeasure
    )


results_df[
    "gmm_top3_rougeL"
] = gmm_rouge


# ============================================================
# BERTSCORE
# ============================================================

print(
    "Calculating Top-3 GMM BERTScore..."
)


gmm_P, gmm_R, gmm_F1 = bert_score(

    gmm_predictions,
    references,

    lang=
        BERTSCORE_LANG,

    verbose=True,

    device=
        str(DEVICE)
)


results_df[
    "gmm_top3_bertscore_precision"
] = (
    gmm_P
    .cpu()
    .numpy()
)


results_df[
    "gmm_top3_bertscore_recall"
] = (
    gmm_R
    .cpu()
    .numpy()
)


results_df[
    "gmm_top3_bertscore_f1"
] = (
    gmm_F1
    .cpu()
    .numpy()
)


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
# OVERALL SUMMARY
# ============================================================

summary = {

    "method":
        "Retrieval-Aware GMM Top-3",

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


    # --------------------------------------------------------
    # Candidate / retrieval
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    "gmm_avg_rougeL":

        float(
            results_df[
                "gmm_top3_rougeL"
            ].mean()
        ),


    "gmm_avg_bertscore_f1":

        float(
            results_df[
                "gmm_top3_bertscore_f1"
            ].mean()
        ),


    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    "encoding_latency":

        latency_summary(
            results_df[
                "encoding_ms"
            ]
        ),


    "gmm_routing_latency":

        latency_summary(
            results_df[
                "gmm_top3_routing_ms"
            ]
        ),


    "gmm_retrieval_latency":

        latency_summary(
            results_df[
                "gmm_top3_retrieval_ms"
            ]
        ),


    "gmm_prompt_latency":

        latency_summary(
            results_df[
                "gmm_top3_prompt_ms"
            ]
        ),


    "gmm_llm_latency":

        latency_summary(
            results_df[
                "gmm_top3_llm_ms"
            ]
        ),


    "gmm_e2e_latency":

        latency_summary(
            results_df[
                "gmm_top3_e2e_ms"
            ]
        )
}


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

QUERY_RESULTS_CSV = os.path.join(
    RESULT_DIR,
    "rag_top3_query_results.csv"
)


QUERY_RESULTS_XLSX = os.path.join(
    RESULT_DIR,
    "rag_top3_query_results.xlsx"
)


RETRIEVAL_CSV = os.path.join(
    RESULT_DIR,
    "rag_top3_retrieved_records.csv"
)


RETRIEVAL_XLSX = os.path.join(
    RESULT_DIR,
    "rag_top3_retrieved_records.xlsx"
)


SUMMARY_JSON = os.path.join(
    RESULT_DIR,
    "rag_top3_overall_summary.json"
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
# FINAL PRINT
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "RETRIEVAL-AWARE GMM TOP-3 EXPERIMENT COMPLETE"
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
    f"Flat/Top-3 GMM "
    f"Top-{RETRIEVAL_K} overlap:",
    f"{summary['avg_flat_gmm_topk_overlap']:.4f}"
)


print(
    "\nQUALITY"
)


print(
    "Top-3 ROUGE-L:",
    f"{summary['gmm_avg_rougeL']:.4f}"
)


print(
    "Top-3 BERTScore F1:",
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
    "Top-3 GMM mean:",
    f"{summary['gmm_e2e_latency']['mean_ms'] / 1000:.3f}",
    "sec"
)


print(
    "Top-3 GMM P95:",
    f"{summary['gmm_e2e_latency']['p95_ms'] / 1000:.3f}",
    "sec"
)


print(
    "\nResults saved in:"
)

print(
    RESULT_DIR
)