import os
import time
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from rouge_score import rouge_scorer
from bert_score import score as bert_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

KB_FILE = "knowledge-base.xlsx"

INFERENCE_FILE = (
    "synthetic_inference_FINAL_1050_with_reference.csv"
)

# New output directory so old GPT-4.1-mini results
# are not accidentally resumed/reused.
OUTPUT_DIR = "synthetic_standard_rag_gpt5mini_argsort"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "standard_rag_results.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "standard_rag_summary.csv"
)

KB_EMBEDDING_FILE = os.path.join(
    OUTPUT_DIR,
    "standard_kb_embeddings.npy"
)

KB_METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "standard_kb_metadata.csv"
)

FAULT_COL = "Fault Description"
RESOLUTION_COL = "Resolution"

QUERY_COL = "Generated Fault Description"
REFERENCE_COL = "reference_resolution"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# GPT-5-mini for resolution generation
# ============================================================

GPT_MODEL = "gpt-5-mini"

TOP_K = 5

BATCH_SIZE = 128

CHECKPOINT_EVERY = 10

MAX_OUTPUT_TOKENS = 500




# ============================================================
# 2. OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 85)
print("STANDARD RAG — SYNTHETIC FINAL INFERENCE")
print("GPT-5-mini | FULL ARGSORT | TOP-5")
print("=" * 85)

kb_df = pd.read_excel(
    KB_FILE
)

test_df = pd.read_csv(
    INFERENCE_FILE
)


print("\nKnowledge base rows:", len(kb_df))
print("Inference rows     :", len(test_df))

print("\nKB columns:")
print(list(kb_df.columns))

print("\nInference columns:")
print(list(test_df.columns))


# ============================================================
# 4. VALIDATE COLUMNS
# ============================================================

for col in [
    FAULT_COL,
    RESOLUTION_COL
]:

    if col not in kb_df.columns:

        raise ValueError(
            f"Missing KB column: {col}"
        )


for col in [
    QUERY_COL,
    REFERENCE_COL
]:

    if col not in test_df.columns:

        raise ValueError(
            f"Missing inference column: {col}"
        )


if len(kb_df) != 10500:

    print(
        f"WARNING: expected 10500 KB records, "
        f"found {len(kb_df)}"
    )


if len(test_df) != 1050:

    print(
        f"WARNING: expected 1050 test queries, "
        f"found {len(test_df)}"
    )


# ============================================================
# 5. CLEAN TEXT
# ============================================================

kb_df[FAULT_COL] = (
    kb_df[FAULT_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

kb_df[RESOLUTION_COL] = (
    kb_df[RESOLUTION_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

test_df[QUERY_COL] = (
    test_df[QUERY_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

test_df[REFERENCE_COL] = (
    test_df[REFERENCE_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# 6. DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

print("\nDevice:", device)


# ============================================================
# 7. LOAD STANDARD PRETRAINED MINILM
#
# Standard RAG uses pretrained MiniLM.
# It does NOT use the contrastively trained encoder.
# ============================================================

print("\nLoading pretrained MiniLM...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device=device
)


# ============================================================
# 8. KB EMBEDDING — OFFLINE COST
#
# Not included in online inference latency.
# ============================================================

if os.path.exists(
    KB_EMBEDDING_FILE
):

    print(
        "\nExisting KB embeddings found."
    )

    kb_embeddings = np.load(
        KB_EMBEDDING_FILE
    )

    kb_embedding_time_sec = np.nan

    print(
        "Loaded:",
        kb_embeddings.shape
    )


else:

    print("\n" + "=" * 85)
    print("OFFLINE KB EMBEDDING")
    print("=" * 85)

    kb_start = time.perf_counter()

    kb_embeddings = embedding_model.encode(
        kb_df[FAULT_COL].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    kb_embedding_time_sec = (
        time.perf_counter()
        -
        kb_start
    )

    np.save(
        KB_EMBEDDING_FILE,
        kb_embeddings
    )

    print(
        "\nKB embedding shape:",
        kb_embeddings.shape
    )

    print(
        f"KB embedding time: "
        f"{kb_embedding_time_sec:.4f} sec"
    )


kb_embeddings = np.ascontiguousarray(
    kb_embeddings,
    dtype=np.float32
)


# ============================================================
# 9. SAVE KB METADATA
# ============================================================

kb_metadata = kb_df[
    [
        FAULT_COL,
        RESOLUTION_COL
    ]
].copy()

kb_metadata.insert(
    0,
    "kb_index",
    np.arange(
        len(kb_metadata)
    )
)

kb_metadata.to_csv(
    KB_METADATA_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. OPENAI CLIENT
# ============================================================

client = OpenAI()


# ============================================================
# 11. SYSTEM PROMPT
#
# Keep exactly the same later for CLIP+GMM.
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
# 12. BUILD RAG PROMPT
# ============================================================

def build_prompt(
    query,
    retrieved_rows
):

    context_parts = []

    for rank, (_, row) in enumerate(
        retrieved_rows.iterrows(),
        start=1
    ):

        context_parts.append(
            f"""
Retrieved Example {rank}

Fault:
{row[FAULT_COL]}

Resolution:
{row[RESOLUTION_COL]}
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
# 13. GPT GENERATION
# ============================================================

def generate_resolution(prompt):

    response = client.responses.create(
        model=GPT_MODEL,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        reasoning={
            "effort": "minimal"
        },
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    return response.output_text.strip()


# ============================================================
# 14. LOAD EXISTING RESULTS FOR RESUME
# ============================================================

if os.path.exists(
    OUTPUT_FILE
):

    existing_df = pd.read_csv(
        OUTPUT_FILE
    )

    completed_indices = set(
        existing_df.loc[
            existing_df["status"]
            ==
            "success",
            "query_index"
        ]
        .astype(int)
        .tolist()
    )

    results = existing_df.to_dict(
        "records"
    )

    print(
        f"\nResume enabled: "
        f"{len(completed_indices)} "
        f"successful queries already completed."
    )

else:

    completed_indices = set()

    results = []


# ============================================================
# 15. INFERENCE LOOP
# ============================================================

print("\n" + "=" * 85)
print("RUNNING STANDARD RAG")
print("=" * 85)


for i, row in test_df.iterrows():

    if i in completed_indices:
        continue


    query = row[QUERY_COL]
    reference = row[REFERENCE_COL]


    print("\n" + "=" * 85)

    print(
        f"QUERY {i + 1}/{len(test_df)}"
    )

    print("=" * 85)

    print("\nFault:")
    print(query)


    try:

        # ====================================================
        # A. DIRECT E2E TIMER START
        # ====================================================

        e2e_start = time.perf_counter()


        # ====================================================
        # B. QUERY ENCODING
        #
        # Encoding is measured separately.
        # It is NOT part of retrieval time.
        # ====================================================

        if torch.cuda.is_available():
            torch.cuda.synchronize()


        encode_start = time.perf_counter()


        query_embedding = embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]


        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )


        if torch.cuda.is_available():
            torch.cuda.synchronize()


        encode_end = time.perf_counter()


        query_encoding_sec = (
            encode_end
            -
            encode_start
        )


        # ====================================================
        # C. STANDARD RAG RETRIEVAL
        #
        # SAME LOGIC AS OLD EXAMPLE:
        #
        # cosine over full KB
        #       ↓
        # full argsort
        #       ↓
        # Top-5
        #       ↓
        # construct retrieved records
        #
        # Everything above is included in retrieval time.
        # ====================================================

        retrieval_start = time.perf_counter()


        # ----------------------------------------------------
        # Cosine similarity
        #
        # KB and query embeddings are L2-normalized.
        # Therefore dot product = cosine similarity.
        # ----------------------------------------------------

        similarities = (
            kb_embeddings
            @
            query_embedding
        )


        # ----------------------------------------------------
        # FULL ARGSORT
        #
        # This intentionally follows the old retrieve_top_k()
        # implementation.
        #
        # NO argpartition is used.
        # ----------------------------------------------------

        top_indices = np.argsort(
            similarities
        )[-TOP_K:][::-1]


        top_scores = similarities[
            top_indices
        ]


        # ----------------------------------------------------
        # Construct retrieved records INSIDE retrieval timer.
        # This follows the old example timing.
        # ----------------------------------------------------

        retrieved_rows = (
            kb_df
            .iloc[top_indices]
            .copy()
        )


        retrieval_end = time.perf_counter()


        retrieval_sec = (
            retrieval_end
            -
            retrieval_start
        )


        # ====================================================
        # D. BUILD PROMPT
        #
        # Outside retrieval timing.
        # ====================================================

        prompt_start = time.perf_counter()


        prompt = build_prompt(
            query,
            retrieved_rows
        )


        prompt_end = time.perf_counter()


        prompt_build_sec = (
            prompt_end
            -
            prompt_start
        )


        # ====================================================
        # E. GPT-5-mini GENERATION
        # ====================================================

        llm_start = time.perf_counter()


        predicted_resolution = (
            generate_resolution(
                prompt
            )
        )


        llm_end = time.perf_counter()


        llm_latency_sec = (
            llm_end
            -
            llm_start
        )


        # ====================================================
        # F. DIRECT E2E TIMER END
        # ====================================================

        e2e_end = time.perf_counter()


        e2e_latency_sec = (
            e2e_end
            -
            e2e_start
        )


        print("\nPredicted Resolution:")
        print(predicted_resolution)


        print("\nLatency")

        print(
            f"Query encoding : "
            f"{query_encoding_sec:.6f} sec"
        )

        print(
            f"Retrieval      : "
            f"{retrieval_sec:.6f} sec"
        )

        print(
            f"Prompt build   : "
            f"{prompt_build_sec:.6f} sec"
        )

        print(
            f"LLM            : "
            f"{llm_latency_sec:.3f} sec"
        )

        print(
            f"E2E            : "
            f"{e2e_latency_sec:.3f} sec"
        )


        # ====================================================
        # G. SAVE RECORD
        # ====================================================

        result = {

            "query_index":
                i,

            "fault_description":
                query,

            "reference_resolution":
                reference,

            "predicted_resolution":
                predicted_resolution,

            "status":
                "success",

            # Keep all timing directly in seconds
            "query_encoding_sec":
                query_encoding_sec,

            "retrieval_sec":
                retrieval_sec,

            "prompt_build_sec":
                prompt_build_sec,

            "llm_latency_sec":
                llm_latency_sec,

            "e2e_latency_sec":
                e2e_latency_sec
        }


        # ----------------------------------------------------
        # Store Top-5 retrieval information
        # ----------------------------------------------------

        for rank in range(
            TOP_K
        ):

            kb_index = int(
                top_indices[rank]
            )

            result[
                f"retrieved_{rank + 1}_index"
            ] = kb_index

            result[
                f"retrieved_{rank + 1}_similarity"
            ] = float(
                top_scores[rank]
            )

            result[
                f"retrieved_{rank + 1}_fault"
            ] = (
                kb_df.iloc[
                    kb_index
                ][FAULT_COL]
            )

            result[
                f"retrieved_{rank + 1}_resolution"
            ] = (
                kb_df.iloc[
                    kb_index
                ][RESOLUTION_COL]
            )


        results.append(
            result
        )


    except Exception as e:

        print("\nERROR:")
        print(str(e))


        results.append({

            "query_index":
                i,

            "fault_description":
                query,

            "reference_resolution":
                reference,

            "predicted_resolution":
                "",

            "status":
                "failed",

            "error":
                str(e)
        })


    # ========================================================
    # H. CHECKPOINT
    # ========================================================

    if len(results) % CHECKPOINT_EVERY == 0:

        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "\nCheckpoint saved."
        )


# ============================================================
# 16. FINAL SAVE BEFORE METRICS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        [
            "query_index",
            "status"
        ]
    )
    .drop_duplicates(
        subset=["query_index"],
        keep="last"
    )
    .sort_values(
        "query_index"
    )
    .reset_index(
        drop=True
    )
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. QUALITY EVALUATION
# ============================================================

successful = results_df[
    results_df["status"]
    ==
    "success"
].copy()


print("\n" + "=" * 85)
print("QUALITY EVALUATION")
print("=" * 85)

print(
    "Successful generations:",
    len(successful)
)


if len(successful) == 0:

    raise RuntimeError(
        "No successful generations. "
        "Stopping before ROUGE-L/BERTScore evaluation."
    )


predictions = (
    successful[
        "predicted_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


references = (
    successful[
        "reference_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


# ============================================================
# 18. ROUGE-L
# ============================================================

print("\nComputing ROUGE-L...")


rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)


rouge_l_scores = []


for prediction, reference in zip(
    predictions,
    references
):

    score = rouge.score(
        reference,
        prediction
    )

    rouge_l_scores.append(
        score["rougeL"].fmeasure
    )


successful[
    "rouge_l"
] = rouge_l_scores


mean_rouge_l = float(
    np.mean(
        rouge_l_scores
    )
)


print(
    f"Mean ROUGE-L F1: "
    f"{mean_rouge_l:.6f}"
)


# ============================================================
# 19. BERTSCORE
# ============================================================

print("\nComputing BERTScore...")


P, R, F1 = bert_score(
    predictions,
    references,
    lang="en",
    verbose=True,
    device=device
)


bert_precision = (
    P.cpu()
    .numpy()
)

bert_recall = (
    R.cpu()
    .numpy()
)

bert_f1 = (
    F1.cpu()
    .numpy()
)


successful[
    "bertscore_precision"
] = bert_precision

successful[
    "bertscore_recall"
] = bert_recall

successful[
    "bertscore_f1"
] = bert_f1


mean_bert_f1 = float(
    np.mean(
        bert_f1
    )
)


print(
    f"Mean BERTScore F1: "
    f"{mean_bert_f1:.6f}"
)


# ============================================================
# 20. MERGE METRICS BACK
# ============================================================

metric_cols = successful[
    [
        "query_index",
        "rouge_l",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1"
    ]
]


results_df = results_df.merge(
    metric_cols,
    on="query_index",
    how="left"
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 21. LATENCY SUMMARY
# ============================================================

def safe_mean(series):

    return float(
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
        .mean()
    )


def safe_median(series):

    return float(
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
        .median()
    )


def safe_p95(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    return float(
        np.percentile(
            values,
            95
        )
    )


summary = {

    "method":
        "Standard RAG",

    "generator_model":
        GPT_MODEL,

    "retrieval_method":
        "global_cosine_full_argsort_top5",

    "kb_records":
        len(kb_df),

    "test_queries":
        len(test_df),

    "successful_queries":
        len(successful),

    "top_k":
        TOP_K,

    "kb_embedding_time_sec":
        kb_embedding_time_sec,

    "mean_query_encoding_sec":
        safe_mean(
            successful[
                "query_encoding_sec"
            ]
        ),

    "median_query_encoding_sec":
        safe_median(
            successful[
                "query_encoding_sec"
            ]
        ),

    "p95_query_encoding_sec":
        safe_p95(
            successful[
                "query_encoding_sec"
            ]
        ),

    "mean_retrieval_sec":
        safe_mean(
            successful[
                "retrieval_sec"
            ]
        ),

    "median_retrieval_sec":
        safe_median(
            successful[
                "retrieval_sec"
            ]
        ),

    "p95_retrieval_sec":
        safe_p95(
            successful[
                "retrieval_sec"
            ]
        ),

    "mean_prompt_build_sec":
        safe_mean(
            successful[
                "prompt_build_sec"
            ]
        ),

    "mean_llm_latency_sec":
        safe_mean(
            successful[
                "llm_latency_sec"
            ]
        ),

    "median_llm_latency_sec":
        safe_median(
            successful[
                "llm_latency_sec"
            ]
        ),

    "p95_llm_latency_sec":
        safe_p95(
            successful[
                "llm_latency_sec"
            ]
        ),

    "mean_e2e_latency_sec":
        safe_mean(
            successful[
                "e2e_latency_sec"
            ]
        ),

    "median_e2e_latency_sec":
        safe_median(
            successful[
                "e2e_latency_sec"
            ]
        ),

    "p95_e2e_latency_sec":
        safe_p95(
            successful[
                "e2e_latency_sec"
            ]
        ),

    "mean_rouge_l":
        mean_rouge_l,

    "mean_bertscore_f1":
        mean_bert_f1
}


summary_df = pd.DataFrame(
    [summary]
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 85)
print("STANDARD RAG COMPLETE")
print("=" * 85)


print(
    f"Generator model     : "
    f"{GPT_MODEL}"
)

print(
    f"Successful queries  : "
    f"{len(successful)}/{len(test_df)}"
)

print(
    f"Top-K               : "
    f"{TOP_K}"
)

print(
    f"Mean ROUGE-L        : "
    f"{mean_rouge_l:.6f}"
)

print(
    f"Mean BERTScore F1   : "
    f"{mean_bert_f1:.6f}"
)


print("\nAVERAGE LATENCY")

print(
    f"Query encoding      : "
    f"{summary['mean_query_encoding_sec']:.6f} sec"
)

print(
    f"Retrieval           : "
    f"{summary['mean_retrieval_sec']:.6f} sec"
)

print(
    f"Prompt build        : "
    f"{summary['mean_prompt_build_sec']:.6f} sec"
)

print(
    f"LLM                 : "
    f"{summary['mean_llm_latency_sec']:.6f} sec"
)

print(
    f"E2E                 : "
    f"{summary['mean_e2e_latency_sec']:.6f} sec"
)


print("\nP95 LATENCY")

print(
    f"Query encoding      : "
    f"{summary['p95_query_encoding_sec']:.6f} sec"
)

print(
    f"Retrieval           : "
    f"{summary['p95_retrieval_sec']:.6f} sec"
)

print(
    f"LLM                 : "
    f"{summary['p95_llm_latency_sec']:.6f} sec"
)

print(
    f"E2E                 : "
    f"{summary['p95_e2e_latency_sec']:.6f} sec"
)


print("\nFiles saved:")

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)

print(
    KB_EMBEDDING_FILE
)