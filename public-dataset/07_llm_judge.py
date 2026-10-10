import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference",
    "rag_query_results.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "llm_judge"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DETAIL_FILE = os.path.join(
    OUTPUT_DIR,
    "llm_judge_results.xlsx"
)

DETAIL_CSV = os.path.join(
    OUTPUT_DIR,
    "llm_judge_results.csv"
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "llm_judge_checkpoint.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "llm_judge_summary.xlsx"
)

SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "llm_judge_summary.json"
)


# ------------------------------------------------------------
# JUDGE MODEL
# ------------------------------------------------------------

JUDGE_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 120

# First test only 5 records.
# After checking output, change this to None.
MAX_TEST_QUERIES = None

# Retry failed API requests
MAX_RETRIES = 3


# ============================================================
# API
# ============================================================

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        "In PowerShell run:\n"
        '$env:OPENAI_API_KEY="YOUR_API_KEY"'
    )

client = OpenAI()


# ============================================================
# JUDGE DEFINITIONS
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
Equivalent or appropriately detailed remediation should be considered
Relevant.

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
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_judge_prompt(
    fault,
    ground_truth,
    generated_resolution
):

    return f"""
FAULT:
{fault}

REFERENCE GROUND-TRUTH RESOLUTION:
{ground_truth}

GENERATED RESOLUTION:
{generated_resolution}

Classify the generated resolution according to the provided criteria.
""".strip()


def parse_judge_output(text):

    text = clean_text(text)

    match = re.search(
        r"LABEL\s*:\s*(RELEVANT|CLOSER|INADEQUATE)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        label = match.group(1).upper()
    else:
        # Defensive fallback
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
        flags=re.IGNORECASE | re.DOTALL
    )

    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        reason = ""

    return label, reason


def call_judge(
    fault,
    ground_truth,
    generated_resolution
):

    prompt = build_judge_prompt(
        fault,
        ground_truth,
        generated_resolution
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.responses.create(
                model=JUDGE_MODEL,
                instructions=JUDGE_INSTRUCTIONS,
                input=prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=0
            )

            output_text = response.output_text.strip()

            label, reason = parse_judge_output(
                output_text
            )

            return {
                "label": label,
                "reason": reason,
                "raw_output": output_text,
                "error": ""
            }

        except Exception as e:

            last_error = str(e)

            print(
                f"\nAPI error "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{last_error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    return {
        "label": "API_ERROR",
        "reason": "",
        "raw_output": "",
        "error": last_error
    }


# ============================================================
# LOAD RAG RESULTS
# ============================================================

print("=" * 90)
print("LOADING RAG RESULTS")
print("=" * 90)

df = pd.read_excel(INPUT_FILE)

print("Input file:")
print(INPUT_FILE)

print("\nRows:", len(df))

print("\nColumns:")
for column in df.columns:
    print(" ", column)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "query_fault",
    "ground_truth_resolution",
    "flat_generated_resolution",
    "gmm_generated_resolution"
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# LIMIT FOR INITIAL TEST
# ============================================================

if MAX_TEST_QUERIES is not None:

    df = (
        df.iloc[:MAX_TEST_QUERIES]
        .copy()
        .reset_index(drop=True)
    )


print("\nQueries being judged:", len(df))
print("Judge model:", JUDGE_MODEL)

print(
    "Expected judge calls:",
    len(df) * 2
)


# ============================================================
# RUN LLM JUDGE
# ============================================================

results = []

print("\n" + "=" * 90)
print("RUNNING LLM JUDGE")
print("=" * 90)


for idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Judging"
):

    query_id = (
        row["query_id"]
        if "query_id" in df.columns
        else idx
    )

    fault = clean_text(
        row["query_fault"]
    )

    ground_truth = clean_text(
        row["ground_truth_resolution"]
    )

    flat_answer = clean_text(
        row["flat_generated_resolution"]
    )

    gmm_answer = clean_text(
        row["gmm_generated_resolution"]
    )


    # --------------------------------------------------------
    # FLAT-RAG JUDGMENT
    # --------------------------------------------------------

    flat_result = call_judge(
        fault=fault,
        ground_truth=ground_truth,
        generated_resolution=flat_answer
    )


    # --------------------------------------------------------
    # GMM-RAG JUDGMENT
    # --------------------------------------------------------

    gmm_result = call_judge(
        fault=fault,
        ground_truth=ground_truth,
        generated_resolution=gmm_answer
    )


    result = {
        "query_id": query_id,

        "query_fault": fault,

        "ground_truth_resolution":
            ground_truth,

        "flat_generated_resolution":
            flat_answer,

        "flat_judge_label":
            flat_result["label"],

        "flat_judge_reason":
            flat_result["reason"],

        "flat_judge_raw":
            flat_result["raw_output"],

        "flat_judge_error":
            flat_result["error"],


        "gmm_generated_resolution":
            gmm_answer,

        "gmm_judge_label":
            gmm_result["label"],

        "gmm_judge_reason":
            gmm_result["reason"],

        "gmm_judge_raw":
            gmm_result["raw_output"],

        "gmm_judge_error":
            gmm_result["error"]
    }

    results.append(result)


    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    pd.DataFrame(
        results
    ).to_csv(
        CHECKPOINT_FILE,
        index=False
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

VALID_LABELS = [
    "RELEVANT",
    "CLOSER",
    "INADEQUATE"
]


def label_statistics(series):

    total_valid = series.isin(
        VALID_LABELS
    ).sum()

    stats = {}

    for label in VALID_LABELS:

        count = int(
            (series == label).sum()
        )

        if total_valid > 0:
            percentage = (
                count / total_valid
            ) * 100
        else:
            percentage = 0.0

        stats[label] = {
            "count": count,
            "percentage": percentage
        }

    stats["valid_total"] = int(
        total_valid
    )

    stats["parse_or_api_errors"] = int(
        len(series) - total_valid
    )

    return stats


flat_stats = label_statistics(
    results_df["flat_judge_label"]
)

gmm_stats = label_statistics(
    results_df["gmm_judge_label"]
)


# ============================================================
# BUILD SUMMARY TABLE
# ============================================================

summary_rows = []

for label in VALID_LABELS:

    summary_rows.append({

        "category": label,

        "flat_count":
            flat_stats[label]["count"],

        "flat_percentage":
            flat_stats[label]["percentage"],

        "gmm_count":
            gmm_stats[label]["count"],

        "gmm_percentage":
            gmm_stats[label]["percentage"]
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# OPTIONAL COMBINED ACCEPTABLE RATE
#
# Relevant + Closer
#
# Keep this as a supporting statistic only.
# Main paper result should still report all three categories.
# ============================================================

flat_acceptable = (
    flat_stats["RELEVANT"]["count"]
    +
    flat_stats["CLOSER"]["count"]
)

gmm_acceptable = (
    gmm_stats["RELEVANT"]["count"]
    +
    gmm_stats["CLOSER"]["count"]
)


if flat_stats["valid_total"] > 0:

    flat_acceptable_pct = (
        flat_acceptable
        /
        flat_stats["valid_total"]
        * 100
    )

else:
    flat_acceptable_pct = 0.0


if gmm_stats["valid_total"] > 0:

    gmm_acceptable_pct = (
        gmm_acceptable
        /
        gmm_stats["valid_total"]
        * 100
    )

else:
    gmm_acceptable_pct = 0.0


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_excel(
    DETAIL_FILE,
    index=False
)

results_df.to_csv(
    DETAIL_CSV,
    index=False
)

summary_df.to_excel(
    SUMMARY_FILE,
    index=False
)


summary_json = {

    "judge_model":
        JUDGE_MODEL,

    "number_of_queries":
        len(results_df),

    "flat_rag": {
        "relevant_count":
            flat_stats["RELEVANT"]["count"],

        "relevant_percentage":
            flat_stats["RELEVANT"]["percentage"],

        "closer_count":
            flat_stats["CLOSER"]["count"],

        "closer_percentage":
            flat_stats["CLOSER"]["percentage"],

        "inadequate_count":
            flat_stats["INADEQUATE"]["count"],

        "inadequate_percentage":
            flat_stats["INADEQUATE"]["percentage"],

        "relevant_plus_closer_percentage":
            flat_acceptable_pct,

        "errors":
            flat_stats["parse_or_api_errors"]
    },

    "gmm_rag": {
        "relevant_count":
            gmm_stats["RELEVANT"]["count"],

        "relevant_percentage":
            gmm_stats["RELEVANT"]["percentage"],

        "closer_count":
            gmm_stats["CLOSER"]["count"],

        "closer_percentage":
            gmm_stats["CLOSER"]["percentage"],

        "inadequate_count":
            gmm_stats["INADEQUATE"]["count"],

        "inadequate_percentage":
            gmm_stats["INADEQUATE"]["percentage"],

        "relevant_plus_closer_percentage":
            gmm_acceptable_pct,

        "errors":
            gmm_stats["parse_or_api_errors"]
    }
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_json,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("LLM-JUDGE RESULTS")
print("=" * 90)


print("\nFLAT-RAG")

print(
    f"Relevant  : "
    f"{flat_stats['RELEVANT']['count']} "
    f"({flat_stats['RELEVANT']['percentage']:.2f}%)"
)

print(
    f"Closer    : "
    f"{flat_stats['CLOSER']['count']} "
    f"({flat_stats['CLOSER']['percentage']:.2f}%)"
)

print(
    f"Inadequate: "
    f"{flat_stats['INADEQUATE']['count']} "
    f"({flat_stats['INADEQUATE']['percentage']:.2f}%)"
)

print(
    f"Relevant + Closer: "
    f"{flat_acceptable_pct:.2f}%"
)


print("\nGMM-RAG")

print(
    f"Relevant  : "
    f"{gmm_stats['RELEVANT']['count']} "
    f"({gmm_stats['RELEVANT']['percentage']:.2f}%)"
)

print(
    f"Closer    : "
    f"{gmm_stats['CLOSER']['count']} "
    f"({gmm_stats['CLOSER']['percentage']:.2f}%)"
)

print(
    f"Inadequate: "
    f"{gmm_stats['INADEQUATE']['count']} "
    f"({gmm_stats['INADEQUATE']['percentage']:.2f}%)"
)

print(
    f"Relevant + Closer: "
    f"{gmm_acceptable_pct:.2f}%"
)


print("\nFiles saved:")

print(DETAIL_FILE)
print(SUMMARY_FILE)
print(SUMMARY_JSON)