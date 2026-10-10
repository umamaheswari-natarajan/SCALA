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

KB_FILE = os.path.join(
    BASE_DIR,
    "knowledge-base.xlsx"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "synthetic_inference_FINAL_1050_with_reference.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "synthetic_contrastive_model"
)

CHECKPOINT_FILE = os.path.join(
    MODEL_DIR,
    "best_contrastive_model.pt"
)

PROPOSED_DIR = os.path.join(
    BASE_DIR,
    "synthetic_proposed_final"
)

KB_RAW_FILE = os.path.join(
    PROPOSED_DIR,
    "kb_fault_embeddings_raw.npy"
)

KB_NORM_FILE = os.path.join(
    PROPOSED_DIR,
    "kb_fault_embeddings_normalized.npy"
)

GMM_FILE = os.path.join(
    PROPOSED_DIR,
    "gmm_K50_final.joblib"
)

CLUSTER_LOOKUP_FILE = os.path.join(
    PROPOSED_DIR,
    "cluster_lookup.joblib"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

# New folder so previous GPT-4.1-mini / argpartition results
# are not mixed with this experiment.

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "synthetic_clip_gmm_K50_gpt5mini_argsort"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "clip_gmm_K50_query_results.csv"
)

RESULT_XLSX = os.path.join(
    OUTPUT_DIR,
    "clip_gmm_K50_query_results.xlsx"
)

RETRIEVED_FILE = os.path.join(
    OUTPUT_DIR,
    "clip_gmm_K50_retrieved_records.csv"
)

RETRIEVED_XLSX = os.path.join(
    OUTPUT_DIR,
    "clip_gmm_K50_retrieved_records.xlsx"
)

SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "clip_gmm_K50_summary.json"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "clip_gmm_K50_summary.csv"
)

CHECKPOINT_RESULTS = os.path.join(
    OUTPUT_DIR,
    "clip_gmm_K50_checkpoint.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

EXPECTED_GMM_K = 50

TOP_K_RECORDS = 5

# Same generator as new Standard RAG
LLM_MODEL = "gpt-5-mini"

# Increased for GPT-5-mini
MAX_OUTPUT_TOKENS = 500

BERTSCORE_LANG = "en"

# None = all 1050 queries
MAX_TEST_QUERIES = None


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 90)
print("CLIP + GMM(K=50) + TOP-1 CLUSTER + TOP-5 RAG")
print("=" * 90)

print("Device:", DEVICE)

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

    if norm <= 1e-12:
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


def latency_summary(series):

    values = (
        series
        .dropna()
        .astype(float)
        .tolist()
    )

    if len(values) == 0:

        return {
            "mean_sec": np.nan,
            "median_sec": np.nan,
            "std_sec": np.nan,
            "p95_sec": np.nan
        }

    return {

        "mean_sec":
            float(
                np.mean(values)
            ),

        "median_sec":
            float(
                np.median(values)
            ),

        "std_sec":
            float(
                np.std(values)
            ),

        "p95_sec":
            percentile(
                values,
                95
            )
    }


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [

    KB_FILE,
    TEST_FILE,
    CHECKPOINT_FILE,
    KB_RAW_FILE,
    KB_NORM_FILE,
    GMM_FILE,
    CLUSTER_LOOKUP_FILE
]


for file_path in required_files:

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            f"Missing required file:\n{file_path}"
        )


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

print(
    "\nLoading knowledge base..."
)

kb_df = pd.read_excel(
    KB_FILE
)

if len(kb_df) != 10500:

    raise ValueError(
        f"Expected 10500 KB records, "
        f"found {len(kb_df)}"
    )


required_kb_columns = [

    "Fault Description",
    "Resolution"
]


for col in required_kb_columns:

    if col not in kb_df.columns:

        raise ValueError(
            f"Missing KB column: {col}"
        )


kb_df = kb_df.reset_index(
    drop=True
)


print(
    "KB records:",
    len(kb_df)
)


# ============================================================
# LOAD INFERENCE SET
# ============================================================

print(
    "Loading held-out inference set..."
)

test_df = pd.read_csv(
    TEST_FILE
)


QUERY_COLUMN = (
    "Generated Fault Description"
)

REFERENCE_COLUMN = (
    "reference_resolution"
)


if QUERY_COLUMN not in test_df.columns:

    raise ValueError(
        f"Missing query column: "
        f"{QUERY_COLUMN}"
    )


if REFERENCE_COLUMN not in test_df.columns:

    raise ValueError(
        f"Missing reference column: "
        f"{REFERENCE_COLUMN}"
    )


if MAX_TEST_QUERIES is not None:

    test_df = (
        test_df
        .iloc[:MAX_TEST_QUERIES]
        .copy()
        .reset_index(drop=True)
    )


N_TEST = len(
    test_df
)


print(
    "Test queries:",
    N_TEST
)


# ============================================================
# LOAD OFFLINE KB EMBEDDINGS
# ============================================================

kb_raw = np.load(
    KB_RAW_FILE
)

kb_norm = np.load(
    KB_NORM_FILE
)


kb_raw = np.ascontiguousarray(
    kb_raw,
    dtype=np.float32
)

kb_norm = np.ascontiguousarray(
    kb_norm,
    dtype=np.float32
)


print(
    "Raw KB embeddings:",
    kb_raw.shape
)

print(
    "Normalized KB embeddings:",
    kb_norm.shape
)


if kb_raw.shape != (
    10500,
    256
):

    raise ValueError(
        f"Unexpected raw KB shape: "
        f"{kb_raw.shape}"
    )


if kb_norm.shape != (
    10500,
    256
):

    raise ValueError(
        f"Unexpected normalized KB shape: "
        f"{kb_norm.shape}"
    )


# ============================================================
# LOAD GMM K=50
# ============================================================

gmm = joblib.load(
    GMM_FILE
)

cluster_lookup = joblib.load(
    CLUSTER_LOOKUP_FILE
)


print(
    "GMM components:",
    gmm.n_components
)


if gmm.n_components != EXPECTED_GMM_K:

    raise ValueError(
        f"Expected GMM K={EXPECTED_GMM_K}, "
        f"found K={gmm.n_components}"
    )


# ============================================================
# LOAD CONTRASTIVE CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE
)


print(
    "Contrastive checkpoint epoch:",
    checkpoint["epoch"]
)


if int(
    checkpoint["epoch"]
) != 9:

    raise ValueError(
        "Expected frozen epoch 9 checkpoint."
    )


MODEL_NAME = checkpoint[
    "model_name"
]

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
# PROJECTION HEAD
# ============================================================

class ProjectionHead(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        projection_dim
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
                projection_dim
            )
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# ============================================================
# CONTRASTIVE MODEL
# ============================================================

class ContrastiveModel(
    nn.Module
):

    def __init__(
        self,
        model_name,
        projection_dim
    ):

        super().__init__()


        self.encoder = (
            AutoModel
            .from_pretrained(
                model_name
            )
        )


        hidden_dim = (
            self.encoder
            .config
            .hidden_size
        )


        self.fault_projection = (
            ProjectionHead(
                hidden_dim,
                projection_dim
            )
        )


        self.resolution_projection = (
            ProjectionHead(
                hidden_dim,
                projection_dim
            )
        )


        self.logit_scale = nn.Parameter(
            torch.tensor(
                0.0
            )
        )


# ============================================================
# LOAD TOKENIZER + TRAINED MODEL
# ============================================================

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)


model = ContrastiveModel(

    MODEL_NAME,

    PROJECTED_DIM
)


load_result = model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ],

    strict=True
)


print(
    "Missing keys:",
    load_result.missing_keys
)

print(
    "Unexpected keys:",
    load_result.unexpected_keys
)


model = model.to(
    DEVICE
)

model.eval()


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
# QUERY ENCODING
# ============================================================

def encode_query(
    fault_text
):

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

        output = model.encoder(
            **encoded
        )


        pooled = mean_pooling(

            output,

            encoded[
                "attention_mask"
            ]
        )


        raw = model.fault_projection(
            pooled
        )


    raw_np = (

        raw
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
    ).astype(
        np.float32
    )


    return (
        raw_np,
        norm_np
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert in 5G, O-RAN, OpenAirInterface, telecom
network faults, configuration errors, protocol failures,
and operational troubleshooting.

Given a target network fault and retrieved examples from
a fault-resolution knowledge base, provide a concise,
technically actionable resolution.

Use the retrieved examples only as supporting evidence.
Adapt the resolution to the target fault rather than
copying unrelated details.

Return only the recommended resolution in 1 to 3 sentences.
""".strip()


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    query,
    retrieved
):

    context_parts = []


    for rank, (
        fault,
        resolution
    ) in enumerate(
        retrieved,
        start=1
    ):

        context_parts.append(
            f"""
Retrieved Example {rank}

Fault:
{fault}

Resolution:
{resolution}
""".strip()
        )


    context_text = "\n\n".join(
        context_parts
    )


    prompt = f"""
Target Fault:
{query}

Retrieved Fault-Resolution Examples:
{context_text}

Provide the most appropriate resolution for the target fault.
""".strip()


    return prompt


# ============================================================
# OPENAI
# ============================================================

if not os.getenv(
    "OPENAI_API_KEY"
):

    raise EnvironmentError(
        "OPENAI_API_KEY is not set."
    )


client = OpenAI(
    timeout=60.0
)


# ============================================================
# GPT-5-MINI GENERATION
# ============================================================

def generate_resolution(
    user_prompt
):

    response = client.responses.create(

        model=LLM_MODEL,

        instructions=SYSTEM_PROMPT,

        input=user_prompt,

        reasoning={
            "effort": "minimal"
        },

        max_output_tokens=
            MAX_OUTPUT_TOKENS
    )


    return (
        response
        .output_text
        .strip()
    )


# ============================================================
# RESULT CONTAINERS
# ============================================================

query_results = []

retrieval_records = []


# ============================================================
# TOTAL ACCUMULATED TIMES
# ============================================================

total_encoding_time = 0.0

total_retrieval_time = 0.0

total_gpt_time = 0.0

total_e2e_time = 0.0


# ============================================================
# MAIN INFERENCE LOOP
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "RUNNING CLIP + GMM(K=50) + TOP-1 + FULL ARGSORT + TOP-5"
)

print(
    "=" * 90
)


for i in range(
    N_TEST
):

    print(
        f"\nQuery {i + 1}/{N_TEST}"
    )


    row = test_df.iloc[
        i
    ]


    fault = safe_text(
        row[
            QUERY_COLUMN
        ]
    )


    reference_resolution = safe_text(
        row[
            REFERENCE_COLUMN
        ]
    )


    # ========================================================
    # TOTAL TIMER START
    # ========================================================

    total_start = time.perf_counter()


    # ========================================================
    # 1. QUERY ENCODING TIMER
    # ========================================================

    if torch.cuda.is_available():

        torch.cuda.synchronize()


    encoding_start = (
        time.perf_counter()
    )


    (
        query_raw,
        query_norm

    ) = encode_query(
        fault
    )


    if torch.cuda.is_available():

        torch.cuda.synchronize()


    encoding_end = (
        time.perf_counter()
    )


    encoding_sec = (
        encoding_end
        -
        encoding_start
    )


    # ========================================================
    # 2. RETRIEVAL TIMER START
    #
    # Includes:
    #
    # - GMM Top-1 prediction
    # - selected-cluster lookup
    # - candidate-space calculation
    # - cosine similarity
    # - FULL argsort over selected cluster
    # - Top-5 selection
    # - retrieved fault-resolution pair construction
    # ========================================================

    retrieval_start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # 2A. GMM TOP-1 CLUSTER
    #
    # Uses raw 256-D contrastively learned fault embedding.
    # --------------------------------------------------------

    selected_cluster = int(

        gmm.predict(

            query_raw.reshape(
                1,
                -1
            )

        )[0]
    )


    # --------------------------------------------------------
    # 2B. GET RECORDS IN SELECTED CLUSTER
    # --------------------------------------------------------

    candidates = np.asarray(

        cluster_lookup[
            selected_cluster
        ],

        dtype=np.int32
    )


    candidate_count = len(
        candidates
    )


    if candidate_count == 0:

        raise ValueError(
            f"Cluster {selected_cluster} "
            f"contains no KB records."
        )


    candidate_fraction = (

        candidate_count
        /
        len(kb_df)
    )


    candidate_reduction = (

        1.0
        -
        candidate_fraction
    )


    # --------------------------------------------------------
    # 2C. COSINE SIMILARITY WITHIN SELECTED CLUSTER
    #
    # Both KB and query embeddings are normalized.
    # Therefore dot product = cosine similarity.
    # --------------------------------------------------------

    candidate_scores = np.dot(

        kb_norm[
            candidates
        ],

        query_norm
    )


    # --------------------------------------------------------
    # 2D. FULL ARGSORT WITHIN SELECTED CLUSTER
    #
    # IMPORTANT:
    #
    # Same ranking methodology as Standard RAG.
    #
    # Standard:
    #   cosine over 10,500
    #   -> full argsort
    #   -> Top-5
    #
    # Proposed:
    #   GMM Top-1 cluster
    #   -> cosine over selected cluster
    #   -> full argsort
    #   -> Top-5
    #
    # NO argpartition is used here.
    # --------------------------------------------------------

    k = min(

        TOP_K_RECORDS,

        candidate_count
    )


    local_top = np.argsort(

        candidate_scores

    )[-k:][::-1]


    retrieved_indices = (

        candidates[
            local_top
        ]
    )


    retrieved_similarities = (

        candidate_scores[
            local_top
        ]
    )


    # --------------------------------------------------------
    # 2E. CONSTRUCT RETRIEVED FAULT-RESOLUTION PAIRS
    #
    # Deliberately inside retrieval timing to match
    # Standard RAG / historical timing methodology.
    # --------------------------------------------------------

    retrieved = []


    for kb_index in retrieved_indices:

        retrieved_row = (
            kb_df.iloc[
                int(kb_index)
            ]
        )


        retrieved.append(

            (
                safe_text(
                    retrieved_row[
                        "Fault Description"
                    ]
                ),

                safe_text(
                    retrieved_row[
                        "Resolution"
                    ]
                )
            )
        )


    retrieval_end = (
        time.perf_counter()
    )


    retrieval_sec = (

        retrieval_end
        -
        retrieval_start
    )


    # ========================================================
    # 3. BUILD PROMPT
    #
    # Outside retrieval timer, same as Standard RAG.
    # ========================================================

    prompt_start = (
        time.perf_counter()
    )


    user_prompt = build_prompt(

        fault,

        retrieved
    )


    prompt_end = (
        time.perf_counter()
    )


    prompt_sec = (

        prompt_end
        -
        prompt_start
    )


    # ========================================================
    # 4. GPT TIMER
    # ========================================================

    print(
        "  Calling GPT-5-mini..."
    )


    try:

        gpt_start = (
            time.perf_counter()
        )


        generated_resolution = (
            generate_resolution(
                user_prompt
            )
        )


        gpt_end = (
            time.perf_counter()
        )


        gpt_sec = (

            gpt_end
            -
            gpt_start
        )


        status = "success"


    except Exception as e:

        print(
            f"  ERROR: {e}"
        )


        generated_resolution = ""

        gpt_sec = np.nan

        status = (
            "error: "
            +
            str(e)
        )


    # ========================================================
    # 5. TOTAL TIMER END
    #
    # Direct wall-clock E2E measurement.
    # ========================================================

    total_end = (
        time.perf_counter()
    )


    if np.isnan(
        gpt_sec
    ):

        e2e_sec = np.nan

    else:

        e2e_sec = (

            total_end
            -
            total_start
        )


    # ========================================================
    # ACCUMULATE TIMES
    # ========================================================

    total_encoding_time += (
        encoding_sec
    )


    total_retrieval_time += (
        retrieval_sec
    )


    if not np.isnan(
        gpt_sec
    ):

        total_gpt_time += (
            gpt_sec
        )


    if not np.isnan(
        e2e_sec
    ):

        total_e2e_time += (
            e2e_sec
        )


    # ========================================================
    # SAVE QUERY RESULT
    # ========================================================

    query_results.append({

        "query_index":
            i,

        "query_fault":
            fault,

        "reference_resolution":
            reference_resolution,

        "selected_cluster":
            selected_cluster,

        "candidate_count":
            candidate_count,

        "candidate_fraction":
            candidate_fraction,

        "candidate_reduction":
            candidate_reduction,

        "encoding_sec":
            encoding_sec,

        "retrieval_sec":
            retrieval_sec,

        "prompt_sec":
            prompt_sec,

        "gpt_sec":
            gpt_sec,

        "e2e_sec":
            e2e_sec,

        "generated_resolution":
            generated_resolution,

        "status":
            status
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


        retrieved_row = (
            kb_df.iloc[
                int(kb_index)
            ]
        )


        retrieval_records.append({

            "query_index":
                i,

            "selected_cluster":
                selected_cluster,

            "candidate_count":
                candidate_count,

            "rank":
                rank,

            "kb_index":
                int(
                    kb_index
                ),

            "retrieved_fault":
                safe_text(
                    retrieved_row[
                        "Fault Description"
                    ]
                ),

            "retrieved_resolution":
                safe_text(
                    retrieved_row[
                        "Resolution"
                    ]
                ),

            "cosine_similarity":
                float(
                    similarity
                )
        })


    # ========================================================
    # CHECKPOINT AFTER EACH QUERY
    # ========================================================

    pd.DataFrame(
        query_results
    ).to_csv(

        CHECKPOINT_RESULTS,

        index=False
    )


    # ========================================================
    # PRINT CURRENT QUERY
    # ========================================================

    if not np.isnan(
        e2e_sec
    ):

        print(
            f"  Cluster: "
            f"{selected_cluster}"
        )

        print(
            f"  Candidates: "
            f"{candidate_count}"
        )

        print(
            f"  Candidate reduction: "
            f"{candidate_reduction:.4f}"
        )

        print(
            f"  Encoding: "
            f"{encoding_sec:.6f} sec"
        )

        print(
            f"  Retrieval: "
            f"{retrieval_sec:.6f} sec"
        )

        print(
            f"  GPT: "
            f"{gpt_sec:.3f} sec"
        )

        print(
            f"  Total: "
            f"{e2e_sec:.3f} sec"
        )


# ============================================================
# CREATE DATAFRAMES
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
    "COMPUTING QUALITY METRICS"
)

print(
    "=" * 90
)


rouge = rouge_scorer.RougeScorer(

    ["rougeL"],

    use_stemmer=True
)


references = (

    results_df[
        "reference_resolution"
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

    lang=
        BERTSCORE_LANG,

    verbose=
        True,

    device=
        str(
            DEVICE
        )
)


results_df[
    "bertscore_precision"
] = (
    P.cpu().numpy()
)

results_df[
    "bertscore_recall"
] = (
    R.cpu().numpy()
)

results_df[
    "bertscore_f1"
] = (
    F1.cpu().numpy()
)


# ============================================================
# SUCCESSFUL QUERY COUNTS
# ============================================================

successful_mask = (

    results_df[
        "status"
    ]
    ==
    "success"
)


successful_queries = int(
    successful_mask.sum()
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "num_test_queries":
        N_TEST,

    "successful_queries":
        successful_queries,

    "kb_size":
        len(kb_df),

    "checkpoint_epoch":
        int(
            checkpoint[
                "epoch"
            ]
        ),

    "gmm_components":
        int(
            gmm.n_components
        ),

    "clusters_selected":
        1,

    "retrieval_k":
        TOP_K_RECORDS,

    "retrieval_method":
        (
            "GMM_top1_cluster_"
            "cosine_full_argsort_top5"
        ),

    "topk_method":
        "full_argsort_then_top5",

    "llm_model":
        LLM_MODEL,

    "reasoning_effort":
        "minimal",

    "max_output_tokens":
        MAX_OUTPUT_TOKENS,

    "avg_candidate_count":
        float(
            results_df[
                "candidate_count"
            ].mean()
        ),

    "avg_candidate_fraction":
        float(
            results_df[
                "candidate_fraction"
            ].mean()
        ),

    "avg_candidate_reduction":
        float(
            results_df[
                "candidate_reduction"
            ].mean()
        ),

    "mean_rougeL":
        float(
            results_df[
                "rougeL"
            ].mean()
        ),

    "mean_bertscore_precision":
        float(
            results_df[
                "bertscore_precision"
            ].mean()
        ),

    "mean_bertscore_recall":
        float(
            results_df[
                "bertscore_recall"
            ].mean()
        ),

    "mean_bertscore_f1":
        float(
            results_df[
                "bertscore_f1"
            ].mean()
        ),

    "encoding_latency":
        latency_summary(
            results_df[
                "encoding_sec"
            ]
        ),

    "retrieval_latency":
        latency_summary(
            results_df[
                "retrieval_sec"
            ]
        ),

    "prompt_latency":
        latency_summary(
            results_df[
                "prompt_sec"
            ]
        ),

    "gpt_latency":
        latency_summary(
            results_df[
                "gpt_sec"
            ]
        ),

    "e2e_latency":
        latency_summary(
            results_df[
                "e2e_sec"
            ]
        ),

    "total_encoding_time_sec":
        float(
            total_encoding_time
        ),

    "total_retrieval_time_sec":
        float(
            total_retrieval_time
        ),

    "total_gpt_time_sec":
        float(
            total_gpt_time
        ),

    "total_e2e_time_sec":
        float(
            total_e2e_time
        )
}


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(

    RESULT_FILE,

    index=False
)


results_df.to_excel(

    RESULT_XLSX,

    index=False
)


retrieval_df.to_csv(

    RETRIEVED_FILE,

    index=False
)


retrieval_df.to_excel(

    RETRIEVED_XLSX,

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
# FLAT SUMMARY CSV
# ============================================================

flat_summary = {

    "num_test_queries":
        N_TEST,

    "successful_queries":
        successful_queries,

    "kb_size":
        len(kb_df),

    "gmm_components":
        int(
            gmm.n_components
        ),

    "clusters_selected":
        1,

    "retrieval_k":
        TOP_K_RECORDS,

    "retrieval_method":
        summary[
            "retrieval_method"
        ],

    "topk_method":
        summary[
            "topk_method"
        ],

    "llm_model":
        LLM_MODEL,

    "max_output_tokens":
        MAX_OUTPUT_TOKENS,

    "avg_candidate_count":
        summary[
            "avg_candidate_count"
        ],

    "avg_candidate_fraction":
        summary[
            "avg_candidate_fraction"
        ],

    "avg_candidate_reduction":
        summary[
            "avg_candidate_reduction"
        ],

    "mean_rougeL":
        summary[
            "mean_rougeL"
        ],

    "mean_bertscore_f1":
        summary[
            "mean_bertscore_f1"
        ],

    "total_encoding_time_sec":
        total_encoding_time,

    "total_retrieval_time_sec":
        total_retrieval_time,

    "total_gpt_time_sec":
        total_gpt_time,

    "total_e2e_time_sec":
        total_e2e_time,

    "avg_encoding_sec":
        summary[
            "encoding_latency"
        ][
            "mean_sec"
        ],

    "median_encoding_sec":
        summary[
            "encoding_latency"
        ][
            "median_sec"
        ],

    "p95_encoding_sec":
        summary[
            "encoding_latency"
        ][
            "p95_sec"
        ],

    "avg_retrieval_sec":
        summary[
            "retrieval_latency"
        ][
            "mean_sec"
        ],

    "median_retrieval_sec":
        summary[
            "retrieval_latency"
        ][
            "median_sec"
        ],

    "p95_retrieval_sec":
        summary[
            "retrieval_latency"
        ][
            "p95_sec"
        ],

    "avg_gpt_sec":
        summary[
            "gpt_latency"
        ][
            "mean_sec"
        ],

    "median_gpt_sec":
        summary[
            "gpt_latency"
        ][
            "median_sec"
        ],

    "p95_gpt_sec":
        summary[
            "gpt_latency"
        ][
            "p95_sec"
        ],

    "avg_e2e_sec":
        summary[
            "e2e_latency"
        ][
            "mean_sec"
        ],

    "median_e2e_sec":
        summary[
            "e2e_latency"
        ][
            "median_sec"
        ],

    "p95_e2e_sec":
        summary[
            "e2e_latency"
        ][
            "p95_sec"
        ]
}


pd.DataFrame(
    [flat_summary]
).to_csv(

    SUMMARY_CSV,

    index=False
)


# ============================================================
# FINAL PRINT
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "CLIP + GMM(K=50) INFERENCE COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nQueries:",
    N_TEST
)

print(
    "Successful queries:",
    successful_queries
)

print(
    "KB size:",
    len(kb_df)
)

print(
    "GMM K:",
    gmm.n_components
)

print(
    "Clusters selected:",
    1
)

print(
    "Top-K records:",
    TOP_K_RECORDS
)


print(
    "\nRETRIEVAL SPACE"
)

print(
    "Average candidate count:",
    f"{summary['avg_candidate_count']:.2f}"
)

print(
    "Average candidate reduction:",
    f"{summary['avg_candidate_reduction']:.4f}"
)


print(
    "\nQUALITY"
)

print(
    "ROUGE-L:",
    f"{summary['mean_rougeL']:.6f}"
)

print(
    "BERTScore F1:",
    f"{summary['mean_bertscore_f1']:.6f}"
)


print(
    "\nAVERAGE LATENCY"
)

print(
    "Encoding:",
    f"{summary['encoding_latency']['mean_sec']:.6f} sec"
)

print(
    "Retrieval:",
    f"{summary['retrieval_latency']['mean_sec']:.6f} sec"
)

print(
    "GPT:",
    f"{summary['gpt_latency']['mean_sec']:.6f} sec"
)

print(
    "E2E:",
    f"{summary['e2e_latency']['mean_sec']:.6f} sec"
)


print(
    "\nP95 LATENCY"
)

print(
    "Encoding:",
    f"{summary['encoding_latency']['p95_sec']:.6f} sec"
)

print(
    "Retrieval:",
    f"{summary['retrieval_latency']['p95_sec']:.6f} sec"
)

print(
    "GPT:",
    f"{summary['gpt_latency']['p95_sec']:.6f} sec"
)

print(
    "E2E:",
    f"{summary['e2e_latency']['p95_sec']:.6f} sec"
)


print(
    "\nTOTAL ACCUMULATED TIME"
)

print(
    "Encoding:",
    f"{total_encoding_time:.4f} sec"
)

print(
    "Retrieval:",
    f"{total_retrieval_time:.4f} sec"
)

print(
    "GPT:",
    f"{total_gpt_time:.4f} sec"
)

print(
    "E2E:",
    f"{total_e2e_time:.4f} sec"
)


print(
    "\nResults saved in:"
)

print(
    OUTPUT_DIR
)