import os
import re
import json
import time
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

from rouge_score import rouge_scorer
from bert_score import score as bert_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Pilot configuration
# ------------------------------------------------------------

TOP_L = 1
RETRIEVAL_K = 2
TEST_QUERIES = 1050
SEED = 42

RUN_GENERATION = True   # False = reuse existing generated answers
RUN_JUDGE = False

GENERATOR_MODEL = "gpt-4.1-mini"
JUDGE_MODEL = "gpt-4.1-mini"

GENERATOR_MAX_TOKENS = 250
JUDGE_MAX_TOKENS = 120

MAX_RETRIES = 3


# ============================================================
# 2. COMMON TEST SET
# ============================================================

TEST_FILE = (
    BASE_DIR
    / "synthetic_inference_FINAL_1050_with_reference.csv"
)

TEST_FAULT_COL = "Generated Fault Description"
TEST_REFERENCE_COL = "reference_resolution"

EXPECTED_FULL_TEST = 1050


# ============================================================
# 3. SCALA-64 ARTIFACTS
# ============================================================

SCALA64_KB_FILE = (
    BASE_DIR
    / "fault-logs-o-ran-64-with-resolution.xls"
)

SCALA64_CKPT = (
    BASE_DIR
    / "real64_contrastive_model"
    / "best_contrastive_model.pt"
)

SCALA64_RAW = (
    BASE_DIR
    / "real64_proposed_final"
    / "kb_fault_embeddings_raw.npy"
)

SCALA64_NORM = (
    BASE_DIR
    / "real64_proposed_final"
    / "kb_fault_embeddings_normalized.npy"
)

SCALA64_GMM = (
    BASE_DIR
    / "models"
    / "real64_gmm_k6"
    / "gmm_K6_final.joblib"
)

SCALA64_LOOKUP = (
    BASE_DIR
    / "models"
    / "real64_gmm_k6"
    / "cluster_to_indices.joblib"
)
SCALA64_GMM_MEANS_NORM = (
    BASE_DIR
    / "models"
    / "real64_gmm_k6"
    / "gmm_means_normalized.npy"
)
SCALA64_EXPECTED_KB = 64
SCALA64_EXPECTED_GMM_K = 6


# ============================================================
# 4. SCALA-10,500 ARTIFACTS
# ============================================================

SCALA10500_KB_FILE = (
    BASE_DIR
    / "knowledge-base.xlsx"
)

SCALA10500_CKPT = (
    BASE_DIR
    / "synthetic_contrastive_model"
    / "best_contrastive_model.pt"
)

SCALA10500_RAW = (
    BASE_DIR
    / "synthetic_proposed_final"
    / "kb_fault_embeddings_raw.npy"
)

SCALA10500_NORM = (
    BASE_DIR
    / "synthetic_proposed_final"
    / "kb_fault_embeddings_normalized.npy"
)

SCALA10500_GMM = (
    BASE_DIR
    / "models"
    / "gmm_k55"
    / "gmm_K55_final.joblib"
)

SCALA10500_LOOKUP = (
    BASE_DIR
    / "models"
    / "gmm_k55"
    / "cluster_to_indices.joblib"
)
SCALA10500_GMM_MEANS_NORM = (
    BASE_DIR
    / "models"
    / "gmm_k55"
    / "gmm_means_normalized.npy"
)
SCALA10500_EXPECTED_KB = 10500
SCALA10500_EXPECTED_GMM_K = 55


# ============================================================
# 5. COMMON KB COLUMNS
# ============================================================

KB_FAULT_COL = "Fault Description"
KB_RESOLUTION_COL = "Resolution"


# ============================================================
# 6. OUTPUT
# ============================================================

RESULT_DIR = (
    BASE_DIR
    / "results"
    / f"scala64_vs_10500_cosine_top{TOP_L}_k{RETRIEVAL_K}_final{TEST_QUERIES}"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

QUERY_CSV = RESULT_DIR / "query_results.csv"
QUERY_XLSX = RESULT_DIR / "query_results.xlsx"

RETRIEVED_CSV = RESULT_DIR / "retrieved_records.csv"
RETRIEVED_XLSX = RESULT_DIR / "retrieved_records.xlsx"

SUMMARY_CSV = RESULT_DIR / "comparison_summary_adequacy.csv"
SUMMARY_XLSX = RESULT_DIR / "comparison_summary_adequacy.xlsx"

FINAL_CSV = RESULT_DIR / "final_comparison_adequacy.csv"
FINAL_XLSX = RESULT_DIR / "final_comparison_adequacy.xlsx"

DELTA_CSV = RESULT_DIR / "scala10500_minus_scala64_adequacy.csv"
DELTA_XLSX = RESULT_DIR / "scala10500_minus_scala64_adequacy.xlsx"

# JUDGE_CSV = RESULT_DIR / "llm_judge_strict_results.csv"
# JUDGE_XLSX = RESULT_DIR / "llm_judge_strict_results.xlsx"

JUDGE_CSV = RESULT_DIR / "llm_judge_adequacy_results.csv"
JUDGE_XLSX = RESULT_DIR / "llm_judge_adequacy_results.xlsx"

# FINAL_CSV = RESULT_DIR / "final_comparison.csv"
# FINAL_XLSX = RESULT_DIR / "final_comparison.xlsx"

# DELTA_CSV = RESULT_DIR / "scala10500_minus_scala64.csv"
# DELTA_XLSX = RESULT_DIR / "scala10500_minus_scala64.xlsx"


# ============================================================
# 7. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 8. GENERATOR PROMPT
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
# 9. LLM JUDGE PROMPT
#
# IMPORTANT:
# Judge is blind to whether answer came from 64 or 10,500.
# ============================================================
JUDGE_INSTRUCTIONS = """
You are an expert evaluator of fault-resolution recommendations.

You will receive:

1. A fault description.
2. A reference ground-truth resolution.
3. A generated resolution.

Evaluate ONLY whether the generated resolution appropriately
addresses the given fault when compared with the reference resolution.

Assign exactly ONE of the following labels:

RELEVANT:
The generated resolution is technically appropriate for the fault
and is substantially consistent with the reference resolution.
It does not need to use the same wording as the reference.
Equivalent or technically sufficient remediation should be considered
RELEVANT, even when it does not reproduce every secondary step in the
reference resolution.

CLOSER:
The generated resolution is related to the correct remediation and
contains useful or technically appropriate actions, but it is incomplete,
too general, only partially addresses the fault, or misses an important
part of the reference resolution.

INADEQUATE:
The generated resolution does not adequately address the fault,
suggests materially incorrect or unrelated remediation, contradicts
the reference, or fails to provide a useful resolution.

Important rules:

- Judge semantic and technical correctness, NOT lexical overlap.
- Do not penalize different wording when the meaning is equivalent.
- Do not reward an answer merely because it contains similar keywords.
- Do not assume information that is absent from the generated resolution.
- Do not compare this answer with any other system's answer.
- Evaluate this generated resolution independently.

Return exactly:

LABEL: <RELEVANT, CLOSER, or INADEQUATE>
REASON: <one short sentence>
"""


# ============================================================
# 10. REPRODUCIBILITY
# ============================================================

def set_seed():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# 11. MODEL ARCHITECTURE
#
# Must exactly match contrastive training.
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

    return (
        torch.sum(
            token_embeddings * mask,
            dim=1
        )
        /
        torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )
    )


class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim
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


class FaultResolutionCLIP(nn.Module):

    def __init__(
        self,
        model_name,
        projection_dim
    ):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden = (
            self.encoder.config.hidden_size
        )

        self.fault_projection = ProjectionHead(
            hidden,
            projection_dim
        )

        self.resolution_projection = ProjectionHead(
            hidden,
            projection_dim
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(
                np.log(1 / 0.07),
                dtype=torch.float32
            )
        )


# ============================================================
# 12. LOAD KB
# ============================================================

def load_kb(
    path,
    expected_size,
    name
):

    if not path.exists():
        raise FileNotFoundError(
            f"{name} KB not found:\n{path}"
        )

    if path.suffix.lower() in {
        ".xls",
        ".xlsx"
    }:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df = df.reset_index(drop=True)

    for col in [
        KB_FAULT_COL,
        KB_RESOLUTION_COL
    ]:

        if col not in df.columns:
            raise ValueError(
                f"{name}: missing column "
                f"'{col}'"
            )

    if len(df) != expected_size:
        raise ValueError(
            f"{name}: expected "
            f"{expected_size} KB rows; "
            f"found {len(df)}."
        )

    return df


# ============================================================
# 13. LOAD ONE SCALA SYSTEM
# ============================================================

def load_scala_system(
    name,
    kb_file,
    ckpt_file,
    raw_file,
    norm_file,
    gmm_file,
    gmm_means_norm_file,
    lookup_file,
    expected_kb,
    expected_gmm_k
):

    print("\n" + "=" * 80)
    print(f"LOADING {name}")
    print("=" * 80)

    required = [
        kb_file,
        ckpt_file,
        raw_file,
        norm_file,
        gmm_file,
        gmm_means_norm_file,
        lookup_file
    ]

    for path in required:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    kb = load_kb(
        kb_file,
        expected_kb,
        name
    )

    raw = np.load(
        raw_file
    ).astype(np.float32)

    norm = np.load(
        norm_file
    ).astype(np.float32)

    checkpoint = torch.load(
        ckpt_file,
        map_location=DEVICE
    )

    model_name = checkpoint.get(
        "model_name",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    projection_dim = int(
        checkpoint["projection_dim"]
    )

    max_length = int(
        checkpoint.get(
            "max_length",
            128
        )
    )

    if projection_dim != 384:
        raise ValueError(
            f"{name}: expected 384-D "
            f"projection, found "
            f"{projection_dim}."
        )

    if raw.shape != (
        expected_kb,
        projection_dim
    ):
        raise ValueError(
            f"{name}: raw embedding "
            f"shape mismatch: {raw.shape}"
        )

    if norm.shape != (
        expected_kb,
        projection_dim
    ):
        raise ValueError(
            f"{name}: normalized embedding "
            f"shape mismatch: {norm.shape}"
        )

    norms = np.linalg.norm(
        norm,
        axis=1
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3
    ):
        raise ValueError(
            f"{name}: KB normalized "
            f"embeddings are not unit norm."
        )

    gmm = joblib.load(
        gmm_file
    )

    gmm_means_norm = np.load(
        gmm_means_norm_file
    ).astype(np.float32)

    gmm_means_norm = np.ascontiguousarray(
        gmm_means_norm,
        dtype=np.float32
    )

    lookup = joblib.load(
        lookup_file
    )

    if gmm.n_components != expected_gmm_k:
        raise ValueError(
            f"{name}: expected "
            f"{expected_gmm_k} GMM components; "
            f"found {gmm.n_components}."
        )

    if gmm_means_norm.shape != (
        expected_gmm_k,
        projection_dim
    ):
        raise ValueError(
            f"{name}: normalized GMM means "
            f"shape mismatch: {gmm_means_norm.shape}"
        )


    mean_norms = np.linalg.norm(
        gmm_means_norm,
        axis=1
    )

    if not np.allclose(
        mean_norms,
        1.0,
        atol=1e-5
    ):
        raise ValueError(
            f"{name}: normalized GMM means "
            f"are not unit norm."
        )

    if TOP_L > expected_gmm_k:
        raise ValueError(
            f"{name}: TOP_L={TOP_L} exceeds "
            f"GMM K={expected_gmm_k}."
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name
        )
    )

    model = FaultResolutionCLIP(
        model_name,
        projection_dim
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print(
        "KB records          :",
        len(kb)
    )

    print(
        "Projection dimension:",
        projection_dim
    )

    print(
        "GMM components      :",
        gmm.n_components
    )

    print(
        "Checkpoint epoch    :",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    return {
        "name": name,
        "kb": kb,
        "raw": raw,
        "norm": norm,
        "gmm": gmm,
        "gmm_means_norm": gmm_means_norm,
        "lookup": lookup,
        "tokenizer": tokenizer,
        "model": model,
        "max_length": max_length,
        "projection_dim": projection_dim,
        "kb_size": expected_kb,
        "gmm_k": expected_gmm_k
    }


# ============================================================
# 14. QUERY ENCODING
# ============================================================

def normalize(v):

    n = np.linalg.norm(v)

    if n <= 1e-12:
        raise ValueError(
            "Near-zero query embedding."
        )

    return (
        v / n
    ).astype(np.float32)


def encode_query(
    text,
    system
):

    t0 = time.perf_counter()

    tokenizer = system["tokenizer"]
    model = system["model"]

    tokens = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=system["max_length"],
        return_tensors="pt"
    )

    tokens = {
        k: v.to(DEVICE)
        for k, v in tokens.items()
    }

    with torch.no_grad():

        output = model.encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens[
                "attention_mask"
            ]
        )

        pooled = mean_pooling(
            output.last_hidden_state,
            tokens["attention_mask"]
        )

        raw_t = (
            model
            .fault_projection(
                pooled
            )
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    raw = (
        raw_t[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    norm = normalize(raw)

    elapsed_ms = (
        time.perf_counter() - t0
    ) * 1000

    return (
        raw,
        norm,
        elapsed_ms
    )


# ============================================================
# 15. GMM ROUTING
# ============================================================

# ============================================================
# 15. COSINE ROUTING TO NORMALIZED GMM MEANS
# ============================================================

def gmm_top_units(
    query_norm,
    gmm_means_norm
):

    # Both query and GMM means are L2-normalized,
    # therefore dot product = cosine similarity.
    cluster_scores = np.dot(
        gmm_means_norm,
        query_norm
    )

    selected = (
        np.argsort(
            -cluster_scores
        )[:TOP_L]
        .astype(int)
    )

    return (
        selected,
        cluster_scores.astype(np.float32)
    )

def candidate_pool(
    units,
    lookup
):

    parts = []

    for unit in units:

        values = np.asarray(
            lookup[int(unit)],
            dtype=np.int64
        )

        if len(values):
            parts.append(values)

    if not parts:
        raise RuntimeError(
            "Empty routed candidate pool."
        )

    return np.unique(
        np.concatenate(parts)
    )


# ============================================================
# 16. COSINE RETRIEVAL
# ============================================================

def cosine_topk(
    query_norm,
    kb_norm,
    candidates
):

    scores = np.dot(
        kb_norm[candidates],
        query_norm
    )

    order = np.argsort(
        -scores
    )[:RETRIEVAL_K]

    return (
        candidates[order].astype(int),
        scores[order].astype(np.float32)
    )


def global_topk(
    query_norm,
    kb_norm,
    k=5
):

    scores = np.dot(
        kb_norm,
        query_norm
    )

    order = np.argsort(
        -scores
    )[:k]

    return order.astype(int)


# ============================================================
# 17. ROUTING RECALL
# ============================================================

def routing_recall(
    global_indices,
    candidates
):

    global_set = set(
        map(
            int,
            global_indices
        )
    )

    candidate_set = set(
        map(
            int,
            candidates
        )
    )

    return (
        len(
            global_set
            &
            candidate_set
        )
        /
        len(global_set)
    )


# ============================================================
# 18. COMPLETE RETRIEVAL
# ============================================================

# ============================================================
# 18. COMPLETE RETRIEVAL
# ============================================================

def retrieve(
    query,
    system
):

    # --------------------------------------------------------
    # Query encoding
    # --------------------------------------------------------

    raw, norm, encoding_ms = (
        encode_query(
            query,
            system
        )
    )

    # --------------------------------------------------------
    # COMPLETE RETRIEVAL TIMER
    #
    # Includes:
    #   1. cosine routing to normalized GMM means
    #   2. Top-L cluster selection
    #   3. candidate-pool construction
    #   4. cosine similarity over candidate records
    #   5. final Top-K selection
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    # Cosine routing using normalized query
    units, cluster_scores = (
        gmm_top_units(
            norm,
            system["gmm_means_norm"]
        )
    )

    # Pool records belonging to selected cluster(s)
    candidates = candidate_pool(
        units,
        system["lookup"]
    )

    # Final cosine retrieval within candidate pool
    indices, scores = (
        cosine_topk(
            norm,
            system["norm"],
            candidates
        )
    )

    retrieval_ms = (
        time.perf_counter()
        - retrieval_start
    ) * 1000

    # --------------------------------------------------------
    # Full-space diagnostic only
    #
    # IMPORTANT:
    # This is intentionally outside retrieval_ms.
    # It is only used to calculate routing recall.
    # --------------------------------------------------------

    global5 = global_topk(
        norm,
        system["norm"],
        k=5
    )

    r1 = routing_recall(
        global5[:1],
        candidates
    )

    r5 = routing_recall(
        global5[:5],
        candidates
    )

    return {
        "selected_units":
            units,

        "selected_unit_scores":
            [
                float(
                    cluster_scores[u]
                )
                for u in units
            ],

        "candidates":
            candidates,

        "candidate_count":
            int(len(candidates)),

        "indices":
            indices,

        "scores":
            scores,

        "routing_recall_at_1":
            float(r1),

        "routing_recall_at_5":
            float(r5),

        "encoding_ms":
            float(encoding_ms),

        "retrieval_ms":
            float(retrieval_ms)
    }

# ============================================================
# 19. MAP INDICES TO KB RECORDS
# ============================================================

def records_from_indices(
    ret,
    system
):

    rows = []

    kb = system["kb"]

    for rank, (
        idx,
        score
    ) in enumerate(
        zip(
            ret["indices"],
            ret["scores"]
        ),
        start=1
    ):

        row = kb.iloc[
            int(idx)
        ]

        rows.append(
            {
                "rank":
                    rank,

                "kb_index":
                    int(idx),

                "fault":
                    str(
                        row[
                            KB_FAULT_COL
                        ]
                    ),

                "resolution":
                    str(
                        row[
                            KB_RESOLUTION_COL
                        ]
                    ),

                "cosine_similarity":
                    float(score)
            }
        )

    return rows


# ============================================================
# 20. GENERATOR
# ============================================================

def build_generator_prompt(
    query,
    records
):

    parts = []

    for record in records:

        parts.append(
            f"Historical Example "
            f"{record['rank']}:\n"
            f"Fault: "
            f"{record['fault']}\n"
            f"Resolution: "
            f"{record['resolution']}"
        )

    context = "\n\n".join(
        parts
    )

    return (
        f"New Fault:\n"
        f"{query}\n\n"

        f"Historical Fault-Resolution "
        f"Examples:\n"
        f"{context}\n\n"

        f"Recommended Resolution:"
    )


def generate_resolution(
    client,
    query,
    records
):

    prompt_start = (
        time.perf_counter()
    )

    prompt = (
        build_generator_prompt(
            query,
            records
        )
    )

    prompt_ms = (
        time.perf_counter()
        -
        prompt_start
    ) * 1000

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            t0 = (
                time.perf_counter()
            )

            response = (
                client
                .responses
                .create(
                    model=
                        GENERATOR_MODEL,

                    instructions=
                        SYSTEM_PROMPT,

                    input=
                        prompt,

                    temperature=0,

                    max_output_tokens=
                        GENERATOR_MAX_TOKENS
                )
            )

            llm_ms = (
                time.perf_counter()
                -
                t0
            ) * 1000

            text = (
                response
                .output_text
                .strip()
            )

            if not text:
                raise RuntimeError(
                    "Empty generator output."
                )

            return (
                text,
                prompt_ms,
                llm_ms
            )

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:
                time.sleep(
                    2 * attempt
                )

    raise RuntimeError(
        "Generation failed after retries."
    ) from last_error


# ============================================================
# 21. ROUGE-L
# ============================================================

ROUGE = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)


def rouge_l(
    reference,
    generated
):

    return float(
        ROUGE.score(
            str(reference),
            str(generated)
        )[
            "rougeL"
        ].fmeasure
    )


# ============================================================
# 22. BERTSCORE
# ============================================================

def add_bertscore(df):

    print(
        "\nComputing BERTScore-F1 "
        "for all generated answers..."
    )

    _, _, f1 = bert_score(
        cands=
            df[
                "generated_resolution"
            ]
            .astype(str)
            .tolist(),

        refs=
            df[
                "ground_truth_resolution"
            ]
            .astype(str)
            .tolist(),

        lang="en",

        verbose=True,

        device=str(
            DEVICE
        )
    )

    out = df.copy()

    out[
        "bertscore_f1"
    ] = (
        f1
        .detach()
        .cpu()
        .numpy()
    )

    return out


# ============================================================
# 23. BLIND LLM JUDGE
# ============================================================

def build_judge_prompt(
    fault,
    ground_truth,
    generated
):

    return f"""
FAULT:

{fault}


REFERENCE GROUND-TRUTH RESOLUTION:

{ground_truth}


GENERATED RESOLUTION:

{generated}


Classify the generated resolution according to the provided criteria.
""".strip()


def parse_judge_output(
    text
):

    text = str(
        text
    ).strip()

    match = re.search(
        r"LABEL\s*:\s*"
        r"(RELEVANT|CLOSER|INADEQUATE)",
        text,
        flags=re.IGNORECASE
    )

    if match:

        label = (
            match
            .group(1)
            .upper()
        )

    else:

        upper = text.upper()

        if "INADEQUATE" in upper:
            label = "INADEQUATE"

        elif "CLOSER" in upper:
            label = "CLOSER"

        elif "RELEVANT" in upper:
            label = "RELEVANT"

        else:
            label = "PARSE_ERROR"

    reason_match = re.search(
        r"REASON\s*:\s*(.+)",
        text,
        flags=
            re.IGNORECASE
            |
            re.DOTALL
    )

    reason = (
        reason_match
        .group(1)
        .strip()

        if reason_match
        else ""
    )

    return (
        label,
        reason
    )


def call_judge(
    client,
    fault,
    ground_truth,
    generated
):

    prompt = build_judge_prompt(
        fault,
        ground_truth,
        generated
    )

    last_error = ""

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            t0 = (
                time.perf_counter()
            )

            response = (
                client
                .responses
                .create(
                    model=
                        JUDGE_MODEL,

                    instructions=
                        JUDGE_INSTRUCTIONS,

                    input=
                        prompt,

                    max_output_tokens=
                        JUDGE_MAX_TOKENS,

                    temperature=0
                )
            )

            judge_ms = (
                time.perf_counter()
                -
                t0
            ) * 1000

            output_text = (
                response
                .output_text
                .strip()
            )

            if not output_text:
                raise RuntimeError(
                    "Empty judge output."
                )

            label, reason = (
                parse_judge_output(
                    output_text
                )
            )

            if label == "PARSE_ERROR":
                raise RuntimeError(
                    "Could not parse "
                    "judge output."
                )

            return {
                "label":
                    label,

                "reason":
                    reason,

                "judge_ms":
                    float(judge_ms),

                "error":
                    ""
            }

        except Exception as exc:

            last_error = str(
                exc
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    2 * attempt
                )

    return {
        "label":
            "API_ERROR",

        "reason":
            "",

        "judge_ms":
            np.nan,

        "error":
            last_error
    }


# ============================================================
# 24. LOAD TEST SET
# ============================================================

def load_test():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found:\n"
            f"{TEST_FILE}"
        )

    df = pd.read_csv(
        TEST_FILE
    ).reset_index(
        drop=True
    )

    for col in [
        TEST_FAULT_COL,
        TEST_REFERENCE_COL
    ]:

        if col not in df.columns:
            raise ValueError(
                f"Missing test column: "
                f"{col}"
            )

    if len(df) != EXPECTED_FULL_TEST:
        raise ValueError(
            f"Expected "
            f"{EXPECTED_FULL_TEST} "
            f"test queries; "
            f"found {len(df)}."
        )

    if TEST_QUERIES > len(df):
        raise ValueError(
            "TEST_QUERIES exceeds "
            "available test queries."
        )

    return (
        df
        .head(TEST_QUERIES)
        .copy()
    )


# ============================================================
# 25. SAVE GENERATION CHECKPOINT
# ============================================================

def save_generation_checkpoint(
    query_rows,
    retrieved_rows
):

    pd.DataFrame(
        query_rows
    ).to_csv(
        QUERY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        retrieved_rows
    ).to_csv(
        RETRIEVED_CSV,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 26. RUN GENERATION EXPERIMENT
# ============================================================

def run_generation(
    client,
    test,
    systems
):

    # --------------------------------------------------------
    # Resume generation if interrupted.
    # --------------------------------------------------------

    if QUERY_CSV.exists():

        old = pd.read_csv(
            QUERY_CSV
        )

        query_rows = (
            old.to_dict(
                "records"
            )
        )

        done = set(
            zip(
                old[
                    "query_id"
                ].astype(int),

                old[
                    "method"
                ].astype(str)
            )
        )

    else:

        query_rows = []
        done = set()

    if RETRIEVED_CSV.exists():

        retrieved_rows = (
            pd.read_csv(
                RETRIEVED_CSV
            )
            .to_dict(
                "records"
            )
        )

    else:

        retrieved_rows = []

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for query_id, row in (
        test.iterrows()
    ):

        query = str(
            row[
                TEST_FAULT_COL
            ]
        ).strip()

        truth = str(
            row[
                TEST_REFERENCE_COL
            ]
        ).strip()

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"QUERY "
            f"{query_id + 1}/"
            f"{len(test)}"
        )

        print(
            "=" * 80
        )

        for method, system in (
            systems.items()
        ):

            key = (
                int(query_id),
                method
            )

            if key in done:

                print(
                    f"{method}: "
                    f"already complete"
                )

                continue

            print(
                f"\n{method}"
            )

            # -----------------------------------------------
            # Retrieval
            # -----------------------------------------------

            ret = retrieve(
                query,
                system
            )

            records = (
                records_from_indices(
                    ret,
                    system
                )
            )

            # -----------------------------------------------
            # Generation
            # -----------------------------------------------

            generated, prompt_ms, llm_ms = (
                generate_resolution(
                    client,
                    query,
                    records
                )
            )

            e2e_ms = (
                ret["encoding_ms"]
                +
                ret["retrieval_ms"]
                +
                prompt_ms
                +
                llm_ms
            )

            qrow = {
                "query_id":
                    int(query_id),

                "method":
                    method,

                "kb_size":
                    system["kb_size"],

                "gmm_k":
                    system["gmm_k"],

                "top_l":
                    TOP_L,

                "retrieval_k":
                    RETRIEVAL_K,

                "query_fault":
                    query,

                "ground_truth_resolution":
                    truth,

                "generated_resolution":
                    generated,

                "selected_units":
                    json.dumps(
                        [
                            int(x)
                            for x in
                            ret[
                                "selected_units"
                            ]
                        ]
                    ),

                "selected_unit_cosine_scores":
                    json.dumps(
                        ret[
                            "selected_unit_scores"
                        ]
                    ),

                "candidate_count":
                    ret[
                        "candidate_count"
                    ],

                "search_space_reduction":
                    (
                        1.0
                        -
                        ret[
                            "candidate_count"
                        ]
                        /
                        system[
                            "kb_size"
                        ]
                    ),

                "routing_recall_at_1":
                    ret[
                        "routing_recall_at_1"
                    ],

                "routing_recall_at_5":
                    ret[
                        "routing_recall_at_5"
                    ],

                "retrieved_indices":
                    json.dumps(
                        [
                            int(x)
                            for x in
                            ret["indices"]
                        ]
                    ),

                "retrieved_scores":
                    json.dumps(
                        [
                            float(x)
                            for x in
                            ret["scores"]
                        ]
                    ),

                "rouge_l":
                    rouge_l(
                        truth,
                        generated
                    ),

                "encoding_ms":
                    ret[
                        "encoding_ms"
                    ],

                # "routing_ms":
                #     ret[
                #         "routing_ms"
                #     ],

                "retrieval_ms":
                    ret[
                        "retrieval_ms"
                    ],

                "prompt_ms":
                    prompt_ms,

                "llm_ms":
                    llm_ms,

                "e2e_ms":
                    e2e_ms
            }

            query_rows.append(
                qrow
            )

            # -----------------------------------------------
            # Retrieved examples
            # -----------------------------------------------

            for record in records:

                retrieved_rows.append(
                    {
                        "query_id":
                            int(query_id),

                        "method":
                            method,

                        "rank":
                            record[
                                "rank"
                            ],

                        "kb_index":
                            record[
                                "kb_index"
                            ],

                        "retrieved_fault":
                            record[
                                "fault"
                            ],

                        "retrieved_resolution":
                            record[
                                "resolution"
                            ],

                        "cosine_similarity":
                            record[
                                "cosine_similarity"
                            ]
                    }
                )

            done.add(
                key
            )

            save_generation_checkpoint(
                query_rows,
                retrieved_rows
            )

            print(
                f"  selected cluster(s) = "
                f"{ret['selected_units'].tolist()}"
            )

            print(
                f"  candidates = "
                f"{ret['candidate_count']}"
            )

            print(
                f"  routing R@1 = "
                f"{ret['routing_recall_at_1']:.4f}"
            )

            print(
                f"  ROUGE-L = "
                f"{qrow['rouge_l']:.4f}"
            )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    qdf = (
        pd.DataFrame(
            query_rows
        )
        .drop_duplicates(
            subset=[
                "query_id",
                "method"
            ],
            keep="last"
        )
        .sort_values(
            [
                "query_id",
                "method"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    rdf = (
        pd.DataFrame(
            retrieved_rows
        )
        .drop_duplicates(
            subset=[
                "query_id",
                "method",
                "rank"
            ],
            keep="last"
        )
        .sort_values(
            [
                "query_id",
                "method",
                "rank"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected_rows = (
        len(test)
        *
        len(systems)
    )

    if len(qdf) != expected_rows:

        raise RuntimeError(
            f"Expected "
            f"{expected_rows} "
            f"query-method rows; "
            f"found {len(qdf)}."
        )

    # --------------------------------------------------------
    # BERTScore
    # --------------------------------------------------------

    qdf = add_bertscore(
        qdf
    )

    qdf.to_csv(
        QUERY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    qdf.to_excel(
        QUERY_XLSX,
        index=False
    )

    rdf.to_csv(
        RETRIEVED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    rdf.to_excel(
        RETRIEVED_XLSX,
        index=False
    )

    return qdf

def print_running_judge_summary(judge_rows):

    df = pd.DataFrame(judge_rows)

    if df.empty:
        return

    # Keep only valid completed judgments
    df = df[
        df["judge_label"].isin(
            ["RELEVANT", "CLOSER", "INADEQUATE"]
        )
    ].copy()

    if df.empty:
        return

    # Only compare queries for which BOTH methods have been judged
    counts = (
        df.groupby("query_id")["method"]
        .nunique()
    )

    complete_query_ids = counts[
        counts == 2
    ].index

    paired = df[
        df["query_id"].isin(complete_query_ids)
    ].copy()

    if paired.empty:
        return

    n_queries = len(complete_query_ids)

    # Print running summary only every 10 paired queries
    if n_queries % 10 != 0:
        return

    def percentages(method):

        x = paired[
            paired["method"] == method
        ]["judge_label"]

        n = len(x)

        return {
            label: 100.0 * (x == label).sum() / n
            for label in [
                "RELEVANT",
                "CLOSER",
                "INADEQUATE"
            ]
        }

    s64 = percentages("scala_64")
    s10500 = percentages("scala_10500")

    delta = (
        s10500["RELEVANT"]
        - s64["RELEVANT"]
    )

    print("\n" + "-" * 70)
    print(
        f"RUNNING JUDGE SUMMARY — "
        f"{n_queries} paired queries completed"
    )
    print("-" * 70)

    print(
        f"{'':15}"
        f"{'SCALA-64':>15}"
        f"{'SCALA-10,500':>18}"
    )

    for label in [
        "RELEVANT",
        "CLOSER",
        "INADEQUATE"
    ]:

        print(
            f"{label:15}"
            f"{s64[label]:>14.2f}%"
            f"{s10500[label]:>17.2f}%"
        )

    print(
        f"\nΔ RELEVANT (10500 - 64): "
        f"{delta:+.2f} pp"
    )

    if delta > 0:
        print(
            ">>> SCALA-10,500 currently BETTER "
            "on RELEVANT"
        )

    elif delta < 0:
        print(
            ">>> SCALA-64 currently BETTER "
            "on RELEVANT"
        )

    else:
        print(
            ">>> Both currently TIED "
            "on RELEVANT"
        )

    print("-" * 70)
# ============================================================
# 27. RUN BLIND LLM JUDGE
# ============================================================

def run_judge(
    client,
    qdf
):

    # --------------------------------------------------------
    # Resume judge if interrupted.
    # --------------------------------------------------------

    if JUDGE_CSV.exists():

        old = pd.read_csv(
            JUDGE_CSV
        )

        judge_rows = (
            old.to_dict(
                "records"
            )
        )

        done = set()

        for _, row in old.iterrows():

            if str(
                row[
                    "judge_label"
                ]
            ) in {
                "RELEVANT",
                "CLOSER",
                "INADEQUATE"
            }:

                done.add(
                    (
                        int(
                            row[
                                "query_id"
                            ]
                        ),
                        str(
                            row[
                                "method"
                            ]
                        )
                    )
                )

    else:

        judge_rows = []
        done = set()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BLIND LLM JUDGE"
    )

    print(
        "=" * 80
    )

    for position, (_, row) in enumerate(
        qdf.iterrows(),
        start=1
    ):

        query_id = int(
            row["query_id"]
        )

        method = str(
            row["method"]
        )

        key = (
            query_id,
            method
        )

        if key in done:

            print(
                f"[{position}/"
                f"{len(qdf)}] "
                f"{method}: "
                f"already judged"
            )

            continue

        print(
            f"[{position}/"
            f"{len(qdf)}] "
            f"query={query_id} "
            f"method={method}"
        )

        # IMPORTANT:
        # Method identity is NOT passed to judge.
        judge = call_judge(
            client=client,

            fault=str(
                row[
                    "query_fault"
                ]
            ),

            ground_truth=str(
                row[
                    "ground_truth_resolution"
                ]
            ),

            generated=str(
                row[
                    "generated_resolution"
                ]
            )
        )

        judge_rows.append(
            {
                "query_id":
                    query_id,

                "method":
                    method,

                "query_fault":
                    row[
                        "query_fault"
                    ],

                "ground_truth_resolution":
                    row[
                        "ground_truth_resolution"
                    ],

                "generated_resolution":
                    row[
                        "generated_resolution"
                    ],

                "judge_label":
                    judge[
                        "label"
                    ],

                "judge_reason":
                    judge[
                        "reason"
                    ],

                "judge_ms":
                    judge[
                        "judge_ms"
                    ],

                "judge_error":
                    judge[
                        "error"
                    ]
            }
        )

        if judge[
            "label"
        ] in {
            "RELEVANT",
            "CLOSER",
            "INADEQUATE"
        }:

            done.add(
                key
            )

        pd.DataFrame(
            judge_rows
        ).to_csv(
            JUDGE_CSV,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "  ->",
            judge[
                "label"
            ]
        )

        # Show running comparison after both methods
        # for the current query have been judged.
        if method == "scala_10500":
            print_running_judge_summary(
                judge_rows
            )

    jdf = (
        pd.DataFrame(
            judge_rows
        )
        .drop_duplicates(
            subset=[
                "query_id",
                "method"
            ],
            keep="last"
        )
        .sort_values(
            [
                "query_id",
                "method"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected = len(
        qdf
    )

    if len(jdf) != expected:

        raise RuntimeError(
            f"Expected {expected} "
            f"judge rows; "
            f"found {len(jdf)}."
        )

    jdf.to_csv(
        JUDGE_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    jdf.to_excel(
        JUDGE_XLSX,
        index=False
    )

    return jdf


# ============================================================
# 28. BUILD FINAL COMPARISON
# ============================================================

def build_final_comparison(qdf):

    rows = []

    for method in [
        "scala_64",
        "scala_10500"
    ]:

        q = qdf[
            qdf["method"] == method
        ].copy()

        rows.append(
            {
                "method":
                    method,

                "num_queries":
                    len(q),

                "kb_size":
                    int(q["kb_size"].iloc[0]),

                "gmm_k":
                    int(q["gmm_k"].iloc[0]),

                "rouge_l":
                    q["rouge_l"].mean(),

                "bertscore_f1":
                    q["bertscore_f1"].mean(),

                "avg_encoding_ms":
                    q["encoding_ms"].mean(),

                "avg_retrieval_ms":
                    q["retrieval_ms"].mean(),

                "avg_prompt_ms":
                    q["prompt_ms"].mean(),

                "avg_llm_ms":
                    q["llm_ms"].mean(),

                "avg_e2e_ms":
                    q["e2e_ms"].mean()
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 29. BUILD 10,500 - 64 DELTAS
# ============================================================

def build_delta(
    summary
):

    s64 = (
        summary[
            summary["method"]
            ==
            "scala_64"
        ]
        .iloc[0]
    )

    s10500 = (
        summary[
            summary["method"]
            ==
            "scala_10500"
        ]
        .iloc[0]
    )

    delta = pd.DataFrame(
        [
            {
                "top_l":
                    TOP_L,

                "retrieval_k":
                    RETRIEVAL_K,

                "delta_rouge_l":
                    (
                        s10500[
                            "rouge_l"
                        ]
                        -
                        s64[
                            "rouge_l"
                        ]
                    ),

                "delta_bertscore_f1":
                    (
                        s10500[
                            "bertscore_f1"
                        ]
                        -
                        s64[
                            "bertscore_f1"
                        ]
                    ),

                "delta_relevant_percentage_points":
                    (
                        s10500[
                            "relevant_percentage"
                        ]
                        -
                        s64[
                            "relevant_percentage"
                        ]
                    ),

                "delta_closer_percentage_points":
                    (
                        s10500[
                            "closer_percentage"
                        ]
                        -
                        s64[
                            "closer_percentage"
                        ]
                    ),

                "delta_inadequate_percentage_points":
                    (
                        s10500[
                            "inadequate_percentage"
                        ]
                        -
                        s64[
                            "inadequate_percentage"
                        ]
                    )
            }
        ]
    )

    return delta


# ============================================================
# 30. MAIN
# ============================================================

def main():

    set_seed()

    print(
        "=" * 80
    )

    print(
        "SCALA COLD-START SCALE COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        "Device          :",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU             :",
            torch.cuda.get_device_name(
                0
            )
        )

    print(
        "Test queries    :",
        TEST_QUERIES
    )

    print(
        "Top-L           :",
        TOP_L
    )

    print(
        "Retrieval K     :",
        RETRIEVAL_K
    )

    print(
        "Generator       :",
        GENERATOR_MODEL
    )

    # print(
    #     "Judge           :",
    #     JUDGE_MODEL
    # )

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    if not os.getenv(
        "OPENAI_API_KEY"
    ):

        raise EnvironmentError(
            "OPENAI_API_KEY is not set."
        )

    client = OpenAI()

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test = load_test()

    # --------------------------------------------------------
    # Load both completely independent SCALA systems
    # --------------------------------------------------------

    scala64 = load_scala_system(
        name=
            "SCALA-64",

        kb_file=
            SCALA64_KB_FILE,

        ckpt_file=
            SCALA64_CKPT,

        raw_file=
            SCALA64_RAW,

        norm_file=
            SCALA64_NORM,

        gmm_file=
            SCALA64_GMM,

        gmm_means_norm_file=
            SCALA64_GMM_MEANS_NORM,

        lookup_file=
            SCALA64_LOOKUP,

        expected_kb=
            SCALA64_EXPECTED_KB,

        expected_gmm_k=
            SCALA64_EXPECTED_GMM_K
    )

    scala10500 = load_scala_system(
        name=
            "SCALA-10,500",

        kb_file=
            SCALA10500_KB_FILE,

        ckpt_file=
            SCALA10500_CKPT,

        raw_file=
            SCALA10500_RAW,

        norm_file=
            SCALA10500_NORM,

        gmm_file=
            SCALA10500_GMM,

        gmm_means_norm_file=
            SCALA10500_GMM_MEANS_NORM,

        lookup_file=
            SCALA10500_LOOKUP,

        expected_kb=
            SCALA10500_EXPECTED_KB,

        expected_gmm_k=
            SCALA10500_EXPECTED_GMM_K
    )

    systems = {
        "scala_64":
            scala64,

        "scala_10500":
            scala10500
    }

    # --------------------------------------------------------
    # Retrieval + generation OR reuse existing results
    # --------------------------------------------------------

    if RUN_GENERATION:

        qdf = run_generation(
            client,
            test,
            systems
        )

    else:

        print("\nSkipping retrieval/generation.")
        print("Loading existing generated results:")
        print(QUERY_CSV)

        if not QUERY_CSV.exists():
            raise FileNotFoundError(
                f"Existing query results not found:\n{QUERY_CSV}"
            )

        qdf = pd.read_csv(QUERY_CSV)

        expected_rows = TEST_QUERIES * len(systems)

        if len(qdf) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} existing rows "
                f"({TEST_QUERIES} queries x {len(systems)} methods), "
                f"found {len(qdf)}."
            )

        print(f"Loaded {len(qdf)} existing generated answers.")


    # --------------------------------------------------------
    # Blind LLM judge
    # --------------------------------------------------------

    # if RUN_JUDGE:

    #     jdf = run_judge(
    #         client,
    #         qdf
    #     )

    # else:

    #     raise RuntimeError(
    #         "RUN_JUDGE=False, but this script requires judge "
    #         "results to build the final comparison."
    #     )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary = (
        build_final_comparison(
            qdf,
            # jdf
        )
    )

    summary.to_csv(
        FINAL_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_excel(
        FINAL_XLSX,
        index=False
    )

    # Same summary also under generic names.
    summary.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_excel(
        SUMMARY_XLSX,
        index=False
    )

    # --------------------------------------------------------
    # Delta: 10,500 minus 64
    # --------------------------------------------------------

    # delta = build_delta(
    #     summary
    # )

    # delta.to_csv(
    #     DELTA_CSV,
    #     index=False,
    #     encoding="utf-8-sig"
    # )

    # delta.to_excel(
    #     DELTA_XLSX,
    #     index=False
    # )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"SCALA-64 VS SCALA-10,500 "
        f"| TOP-L={TOP_L} "
        f"| K={RETRIEVAL_K}"
    )

    print(
        "=" * 80
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # print(
    #     "\n"
    #     + "=" * 80
    # )

    # print(
    #     "SCALA-10,500 MINUS SCALA-64"
    # )

    # print(
    #     "=" * 80
    # )

    # print(
    #     delta.to_string(
    #         index=False
    #     )
    # )

    print(
        "\nResults saved under:"
    )

    print(
        RESULT_DIR
    )


# ============================================================
# 31. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()