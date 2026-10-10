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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KB_FILE = os.path.join(
    BASE_DIR,
    "knowledge-base.xlsx"
)

INFERENCE_FILE = os.path.join(
    BASE_DIR,
    "synthetic_inference_FINAL_1050_with_reference.csv"
)

# ------------------------------------------------------------
# Contrastive model
# ------------------------------------------------------------

CONTRASTIVE_DIR = os.path.join(
    BASE_DIR,
    "synthetic_contrastive_model"
)

CHECKPOINT_FILE = os.path.join(
    CONTRASTIVE_DIR,
    "best_contrastive_model.pt"
)

# ------------------------------------------------------------
# PCA + GMM artifacts built in previous step
# ------------------------------------------------------------

PCA_GMM_DIR = os.path.join(
    BASE_DIR,
    "synthetic_proposed_pca50_gmm_K50"
)

PCA_FILE = os.path.join(
    PCA_GMM_DIR,
    "pca_50.joblib"
)

GMM_FILE = os.path.join(
    PCA_GMM_DIR,
    "gmm_K50_pca50.joblib"
)

KB_PCA_NORMALIZED_FILE = os.path.join(
    PCA_GMM_DIR,
    "kb_fault_embeddings_pca50_normalized.npy"
)

CLUSTER_LOOKUP_FILE = os.path.join(
    PCA_GMM_DIR,
    "cluster_lookup.joblib"
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

RESULT_DIR = os.path.join(
    BASE_DIR,
    "synthetic_clip_pca50_gmm_K50_gpt5mini_argsort"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "clip_pca50_gmm_results.csv"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "clip_pca50_gmm_summary.csv"
)

# ------------------------------------------------------------
# Model settings
# ------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECTION_DIM = 256

MAX_LENGTH = 128

TOP_K_RECORDS = 5

GMM_K = 50

PCA_DIM = 50

GPT_MODEL = "gpt-5-mini"

MAX_OUTPUT_TOKENS = 500


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 90)
print("CLIP + PCA(50) + GMM(K=50) INFERENCE")
print("=" * 90)

print("Device:", device)


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI()


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading knowledge base...")

kb_df = pd.read_excel(
    KB_FILE
)

print(
    "KB records:",
    len(kb_df)
)


print("\nLoading inference dataset...")

inf_df = pd.read_csv(
    INFERENCE_FILE
)

print(
    "Inference queries:",
    len(inf_df)
)


# ============================================================
# COLUMN CHECKS
# ============================================================

required_kb_columns = [
    "Fault Description",
    "Resolution"
]

required_inf_columns = [
    "Generated Fault Description",
    "reference_resolution"
]


for col in required_kb_columns:

    if col not in kb_df.columns:

        raise ValueError(
            f"Missing KB column: {col}"
        )


for col in required_inf_columns:

    if col not in inf_df.columns:

        raise ValueError(
            f"Missing inference column: {col}"
        )


# ============================================================
# SAFE TEXT
# ============================================================

def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


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
        .expand(token_embeddings.size())
        .float()
    )

    summed = torch.sum(
        token_embeddings * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


# ============================================================
# PROJECTION HEAD
#
# Must match the contrastive-training architecture exactly.
# ============================================================

class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim=384,
        output_dim=256
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
                output_dim
            )
        )


    def forward(self, x):

        return self.net(x)


# ============================================================
# CONTRASTIVE MODEL
# ============================================================

class FaultResolutionCLIP(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
        )

        self.fault_projection = (
            ProjectionHead(
                input_dim=384,
                output_dim=PROJECTION_DIM
            )
        )

        self.resolution_projection = (
            ProjectionHead(
                input_dim=384,
                output_dim=PROJECTION_DIM
            )
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(
                np.log(1 / 0.07),
                dtype=torch.float32
            )
        )


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD BEST CONTRASTIVE CHECKPOINT
# ============================================================

print("\nLoading best contrastive checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device
)

model = FaultResolutionCLIP().to(
    device
)

load_result = model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=False
)

print(
    "Missing keys:",
    load_result.missing_keys
)

print(
    "Unexpected keys:",
    load_result.unexpected_keys
)

allowed_missing = {
    "encoder.embeddings.position_ids"
}

if set(load_result.missing_keys) != allowed_missing:
    raise RuntimeError(
        f"Unexpected missing keys: "
        f"{load_result.missing_keys}"
    )

if load_result.unexpected_keys:
    raise RuntimeError(
        f"Unexpected checkpoint keys: "
        f"{load_result.unexpected_keys}"
    )

model.eval()

checkpoint_epoch = checkpoint.get(
    "epoch",
    None
)

print(
    "Checkpoint epoch:",
    checkpoint_epoch
)

print(
    "Projection dimension:",
    PROJECTION_DIM
)


# ============================================================
# LOAD PCA
# ============================================================

print("\nLoading PCA...")

pca = joblib.load(
    PCA_FILE
)

print(
    "PCA input dimension:",
    pca.n_features_in_
)

print(
    "PCA output dimension:",
    pca.n_components_
)


# ============================================================
# LOAD GMM
# ============================================================

print("\nLoading GMM...")

gmm = joblib.load(
    GMM_FILE
)

print(
    "GMM components:",
    gmm.n_components
)

print(
    "GMM covariance:",
    gmm.covariance_type
)


# ============================================================
# LOAD PCA KB EMBEDDINGS
# ============================================================

print("\nLoading normalized PCA KB embeddings...")

kb_pca_norm = np.load(
    KB_PCA_NORMALIZED_FILE
)

print(
    "KB PCA embedding shape:",
    kb_pca_norm.shape
)


if kb_pca_norm.shape[0] != len(kb_df):

    raise ValueError(
        "KB row count does not match PCA embedding count."
    )


if kb_pca_norm.shape[1] != PCA_DIM:

    raise ValueError(
        "Unexpected PCA embedding dimension."
    )


# ============================================================
# LOAD CLUSTER LOOKUP
# ============================================================

print("\nLoading cluster lookup...")

cluster_lookup = joblib.load(
    CLUSTER_LOOKUP_FILE
)


# ============================================================
# QUERY ENCODING
#
# Timing includes:
#
# tokenization
# + fine-tuned MiniLM
# + mean pooling
# + learned fault projection 384 -> 256
# + PCA transform 256 -> 50
# + L2 normalization
#
# Returns:
#
# raw PCA 50-D -> GMM
# normalized PCA 50-D -> cosine
# ============================================================

def encode_query_with_pca(
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
        k: v.to(device)
        for k, v in encoded.items()
    }


    with torch.no_grad():

        output = model.encoder(
            **encoded
        )

        pooled = mean_pooling(
            output.last_hidden_state,
            encoded["attention_mask"]
        )

        projected = model.fault_projection(
            pooled
        )


    # 256-D learned fault representation
    query_256 = (
        projected
        .cpu()
        .numpy()
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # PCA: 256 -> 50
    # --------------------------------------------------------

    query_pca_raw = pca.transform(
        query_256
    )[0]


    # --------------------------------------------------------
    # Normalize PCA representation for cosine
    # --------------------------------------------------------

    norm = np.linalg.norm(
        query_pca_raw
    )

    if norm < 1e-12:

        query_pca_norm = (
            query_pca_raw.copy()
        )

    else:

        query_pca_norm = (
            query_pca_raw
            /
            norm
        )


    return (
        query_pca_raw,
        query_pca_norm
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
# GPT-5-MINI GENERATION
# ============================================================

def generate_resolution(
    prompt
):

    response = client.responses.create(

        model=GPT_MODEL,

        instructions=SYSTEM_PROMPT,

        input=prompt,

        reasoning={
            "effort": "minimal"
        },

        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    return (
        response.output_text
        .strip()
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

results = []


# ============================================================
# INFERENCE LOOP
# ============================================================

print("\n" + "=" * 90)
print("STARTING INFERENCE")
print("=" * 90)


for i in tqdm(
    range(len(inf_df))
):

    row = inf_df.iloc[i]

    fault = safe_text(
        row[
            "Generated Fault Description"
        ]
    )

    reference = safe_text(
        row[
            "reference_resolution"
        ]
    )


    # ========================================================
    # TRUE END-TO-END TIMER
    # ========================================================

    total_start = time.perf_counter()


    # ========================================================
    # QUERY ENCODING
    #
    # Includes PCA transform.
    # ========================================================

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    encoding_start = time.perf_counter()

    (
        query_pca_raw,
        query_pca_norm
    ) = encode_query_with_pca(
        fault
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    encoding_end = time.perf_counter()

    encoding_sec = (
        encoding_end
        -
        encoding_start
            )


    # ========================================================
    # RETRIEVAL
    #
    # Same timing definition as current CLIP+GMM:
    #
    # GMM prediction
    # + cluster lookup
    # + candidate-space bookkeeping
    # + cosine similarity
    # + full argsort
    # + Top-5
    # + retrieved record construction
    # ========================================================

    retrieval_start = time.perf_counter()


    # --------------------------------------------------------
    # GMM Top-1 cluster
    # --------------------------------------------------------

    selected_cluster = int(

        gmm.predict(
            query_pca_raw.reshape(
                1,
                -1
            )
        )[0]

    )


    # --------------------------------------------------------
    # Candidate KB records
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
    # Cosine similarity
    #
    # Both query and KB vectors are L2-normalized.
    # Therefore dot product = cosine.
    # --------------------------------------------------------

    candidate_scores = np.dot(
        kb_pca_norm[
            candidates
        ],
        query_pca_norm
    )


    # --------------------------------------------------------
    # Full argsort
    #
    # Same methodology as final Standard and CLIP+GMM.
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
    # Construct retrieved fault-resolution records
    #
    # Kept inside retrieval timer.
    # --------------------------------------------------------

    retrieved = []


    for kb_index in retrieved_indices:

        retrieved_row = kb_df.iloc[
            int(kb_index)
        ]

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


    retrieval_end = time.perf_counter()

    retrieval_sec = (
        retrieval_end
        -
        retrieval_start
    )


    # ========================================================
    # PROMPT BUILD
    # ========================================================

    prompt_start = time.perf_counter()

    prompt = build_prompt(
        fault,
        retrieved
    )

    prompt_end = time.perf_counter()

    prompt_sec = (
        prompt_end
        -
        prompt_start
    )


    # ========================================================
    # GPT GENERATION
    # ========================================================

    gpt_start = time.perf_counter()

    prediction = generate_resolution(
        prompt
    )

    gpt_end = time.perf_counter()

    gpt_sec = (
        gpt_end
        -
        gpt_start
    )


    # ========================================================
    # END-TO-END
    # ========================================================

    total_end = time.perf_counter()

    e2e_sec = (
        total_end
        -
        total_start
    )


    # ========================================================
    # ROUGE-L
    # ========================================================

    rouge_result = rouge.score(
        reference,
        prediction
    )

    rouge_l = (
        rouge_result[
            "rougeL"
        ].fmeasure
    )


    # ========================================================
    # SAVE RETRIEVED DETAILS
    # ========================================================

    retrieved_faults = [
        item[0]
        for item in retrieved
    ]

    retrieved_resolutions = [
        item[1]
        for item in retrieved
    ]


    results.append(
        {
            "query_id":
                i,

            "fault":
                fault,

            "reference_resolution":
                reference,

            "predicted_resolution":
                prediction,

            "selected_cluster":
                selected_cluster,

            "candidate_count":
                candidate_count,

            "candidate_fraction":
                candidate_fraction,

            "candidate_reduction":
                candidate_reduction,

            "retrieved_indices":
                json.dumps(
                    [
                        int(x)
                        for x
                        in retrieved_indices
                    ]
                ),

            "retrieved_similarities":
                json.dumps(
                    [
                        float(x)
                        for x
                        in retrieved_similarities
                    ]
                ),

            "retrieved_faults":
                json.dumps(
                    retrieved_faults,
                    ensure_ascii=False
                ),

            "retrieved_resolutions":
                json.dumps(
                    retrieved_resolutions,
                    ensure_ascii=False
                ),

            "rouge_l":
                rouge_l,

            "encoding_sec":
                encoding_sec,

            "retrieval_sec":
                retrieval_sec,

            "prompt_sec":
                prompt_sec,

            "gpt_sec":
                gpt_sec,

            "e2e_sec":
                e2e_sec
        }
    )


    # --------------------------------------------------------
    # Save continuously
    # --------------------------------------------------------

    pd.DataFrame(
        results
    ).to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # First few queries: useful sanity check
    # --------------------------------------------------------

    

    print("\n" + "-" * 85)

    print(
            f"Query {i + 1}"
        )

    print(
            "Selected cluster:",
            selected_cluster
        )

    print(
            "Candidate count:",
            candidate_count
        )

    print(
            f"Candidate reduction: "
            f"{candidate_reduction:.4f}"
        )

    print(
            "Prediction:",
            prediction
        )

    print(
            f"Encoding: "
            f"{encoding_sec:.6f} sec"
        )

    print(
            f"Retrieval: "
            f"{retrieval_sec:.6f} sec"
        )

    print(
            f"GPT: "
            f"{gpt_sec:.3f} sec"
        )

    print(
            f"E2E: "
            f"{e2e_sec:.3f} sec"
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# BERTSCORE
#
# Compute after all generation is complete.
# ============================================================

print("\n" + "=" * 90)
print("COMPUTING BERTSCORE")
print("=" * 90)


predictions = (
    results_df[
        "predicted_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


references = (
    results_df[
        "reference_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


bert_p, bert_r, bert_f1 = bert_score(

    predictions,

    references,

    lang="en",

    verbose=True
)


results_df[
    "bertscore_precision"
] = bert_p.cpu().numpy()


results_df[
    "bertscore_recall"
] = bert_r.cpu().numpy()


results_df[
    "bertscore_f1"
] = bert_f1.cpu().numpy()


# Save final detailed results
results_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

successful_queries = int(
    results_df[
        "predicted_resolution"
    ]
    .astype(str)
    .str.strip()
    .ne("")
    .sum()
)


summary = {

    "method":
        "CLIP_PCA50_GMM",

    "checkpoint_epoch":
        checkpoint_epoch,

    "kb_size":
        len(kb_df),

    "queries":
        len(results_df),

    "successful_queries":
        successful_queries,

    "original_clip_dimension":
        PROJECTION_DIM,

    "pca_dimension":
        PCA_DIM,

    "gmm_k":
        GMM_K,

    "clusters_selected":
        1,

    "top_k_records":
        TOP_K_RECORDS,

    "retrieval_method":
        "GMM_top1_cluster_cosine_full_argsort_top5",

    "topk_method":
        "full_argsort_then_top5",

    "llm_model":
        GPT_MODEL,

    "reasoning_effort":
        "minimal",

    "mean_candidate_count":
        results_df[
            "candidate_count"
        ].mean(),

    "mean_candidate_fraction":
        results_df[
            "candidate_fraction"
        ].mean(),

    "mean_candidate_reduction":
        results_df[
            "candidate_reduction"
        ].mean(),

    "mean_rouge_l":
        results_df[
            "rouge_l"
        ].mean(),

    "mean_bertscore_precision":
        results_df[
            "bertscore_precision"
        ].mean(),

    "mean_bertscore_recall":
        results_df[
            "bertscore_recall"
        ].mean(),

    "mean_bertscore_f1":
        results_df[
            "bertscore_f1"
        ].mean(),

    "avg_encoding_sec":
        results_df[
            "encoding_sec"
        ].mean(),

    "avg_retrieval_sec":
        results_df[
            "retrieval_sec"
        ].mean(),

    "avg_prompt_sec":
        results_df[
            "prompt_sec"
        ].mean(),

    "avg_gpt_sec":
        results_df[
            "gpt_sec"
        ].mean(),

    "avg_e2e_sec":
        results_df[
            "e2e_sec"
        ].mean(),

    "p95_encoding_sec":
        results_df[
            "encoding_sec"
        ].quantile(0.95),

    "p95_retrieval_sec":
        results_df[
            "retrieval_sec"
        ].quantile(0.95),

    "p95_gpt_sec":
        results_df[
            "gpt_sec"
        ].quantile(0.95),

    "p95_e2e_sec":
        results_df[
            "e2e_sec"
        ].quantile(0.95),

    "total_encoding_sec":
        results_df[
            "encoding_sec"
        ].sum(),

    "total_retrieval_sec":
        results_df[
            "retrieval_sec"
        ].sum(),

    "total_gpt_sec":
        results_df[
            "gpt_sec"
        ].sum(),

    "total_e2e_sec":
        results_df[
            "e2e_sec"
        ].sum()
}


summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 90)

print(
    "CLIP + PCA(50) + GMM(K=50) "
    "INFERENCE COMPLETE"
)

print("=" * 90)


print(
    f"\nQueries: "
    f"{len(results_df)}"
)

print(
    f"Successful queries: "
    f"{successful_queries}"
)

print(
    f"KB size: "
    f"{len(kb_df)}"
)

print(
    f"CLIP dimension: "
    f"{PROJECTION_DIM}"
)

print(
    f"PCA dimension: "
    f"{PCA_DIM}"
)

print(
    f"GMM K: "
    f"{GMM_K}"
)

print(
    f"Clusters selected: "
    f"1"
)

print(
    f"Top-K records: "
    f"{TOP_K_RECORDS}"
)


print("\nRETRIEVAL SPACE")

print(
    f"Average candidate count: "
    f"{summary['mean_candidate_count']:.2f}"
)

print(
    f"Average candidate reduction: "
    f"{summary['mean_candidate_reduction']:.4f}"
)


print("\nQUALITY")

print(
    f"ROUGE-L: "
    f"{summary['mean_rouge_l']:.6f}"
)

print(
    f"BERTScore F1: "
    f"{summary['mean_bertscore_f1']:.6f}"
)


print("\nAVERAGE LATENCY")

print(
    f"Encoding: "
    f"{summary['avg_encoding_sec']:.6f} sec"
)

print(
    f"Retrieval: "
    f"{summary['avg_retrieval_sec']:.6f} sec"
)

print(
    f"GPT: "
    f"{summary['avg_gpt_sec']:.6f} sec"
)

print(
    f"E2E: "
    f"{summary['avg_e2e_sec']:.6f} sec"
)


print("\nP95 LATENCY")

print(
    f"Encoding: "
    f"{summary['p95_encoding_sec']:.6f} sec"
)

print(
    f"Retrieval: "
    f"{summary['p95_retrieval_sec']:.6f} sec"
)

print(
    f"GPT: "
    f"{summary['p95_gpt_sec']:.6f} sec"
)

print(
    f"E2E: "
    f"{summary['p95_e2e_sec']:.6f} sec"
)


print("\nTOTAL ACCUMULATED TIME")

print(
    f"Encoding: "
    f"{summary['total_encoding_sec']:.4f} sec"
)

print(
    f"Retrieval: "
    f"{summary['total_retrieval_sec']:.4f} sec"
)

print(
    f"GPT: "
    f"{summary['total_gpt_sec']:.4f} sec"
)

print(
    f"E2E: "
    f"{summary['total_e2e_sec']:.4f} sec"
)


print("\nResults saved in:")

print(
    RESULT_DIR
)