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

BASE_DIR = "/home/iiitb/Desktop/SCALA/synthetic-dataset"

STANDARD_FILE = os.path.join(
    BASE_DIR,
    "synthetic_standard_rag_gpt5mini_argsort",
    "standard_rag_results.csv"
)

CLIP_GMM_FILE = os.path.join(
    BASE_DIR,
    "synthetic_clip_gmm_K50_gpt5mini_argsort",
    "clip_gmm_K50_query_results.csv"
)

PCA_GMM_FILE = os.path.join(
    BASE_DIR,
    "synthetic_clip_pca50_gmm_K50_gpt5mini_argsort",
    "clip_pca50_gmm_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "synthetic_llm_judge_final"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DETAIL_CSV = os.path.join(
    OUTPUT_DIR,
    "llm_judge_results.csv"
)

DETAIL_XLSX = os.path.join(
    OUTPUT_DIR,
    "llm_judge_results.xlsx"
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "llm_judge_checkpoint.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "llm_judge_summary.csv"
)

SUMMARY_XLSX = os.path.join(
    OUTPUT_DIR,
    "llm_judge_summary.xlsx"
)

SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "llm_judge_summary.json"
)


# ============================================================
# JUDGE SETTINGS
# ============================================================

JUDGE_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 160

# ------------------------------------------------------------
# IMPORTANT:
# First set this to 5 and check the output.
# Then change it to None for all 1050 queries.
# ------------------------------------------------------------

MAX_TEST_QUERIES = None

MAX_RETRIES = 3


# ============================================================
# API
# ============================================================

if not os.getenv("OPENAI_API_KEY"):

    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        'Run:\nexport OPENAI_API_KEY="YOUR_API_KEY"'
    )


client = OpenAI()


# ============================================================
# JUDGE INSTRUCTIONS
# ============================================================

JUDGE_INSTRUCTIONS = """
You are an expert evaluator of fault-resolution recommendations.

You will receive:

1. A fault description.
2. A reference resolution.
3. A generated resolution.

Evaluate ONLY whether the generated resolution appropriately
addresses the given fault when compared with the reference resolution.

Assign exactly ONE of the following labels:

RELEVANT:
The generated resolution is technically appropriate for the fault
and is substantially consistent with the reference resolution.
It does not need to use the same wording as the reference.
Equivalent or appropriately detailed remediation should be considered
RELEVANT.

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
""".strip()


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_for_check(value):

    return " ".join(
        clean_text(value)
        .lower()
        .split()
    )


def build_judge_prompt(
    fault,
    reference,
    generated_resolution
):

    return f"""
FAULT:
{fault}

REFERENCE RESOLUTION:
{reference}

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
    reference,
    generated_resolution
):

    if not clean_text(generated_resolution):

        return {
            "label": "INADEQUATE",
            "reason": "The generated resolution is empty.",
            "raw_output": "",
            "error": ""
        }


    prompt = build_judge_prompt(
        fault,
        reference,
        generated_resolution
    )


    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = client.responses.create(
                model=JUDGE_MODEL,
                instructions=JUDGE_INSTRUCTIONS,
                input=prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )


            output_text = (
                response
                .output_text
                .strip()
            )


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

                time.sleep(
                    2 * attempt
                )


    return {
        "label": "API_ERROR",
        "reason": "",
        "raw_output": "",
        "error": last_error
    }


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    STANDARD_FILE,
    CLIP_GMM_FILE,
    PCA_GMM_FILE
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


# ============================================================
# LOAD STANDARD RAG
# ============================================================

print("=" * 90)
print("LOADING FINAL SYNTHETIC RESULTS")
print("=" * 90)


standard_df = pd.read_csv(
    STANDARD_FILE
)


required_standard_columns = [
    "query_index",
    "fault_description",
    "reference_resolution",
    "predicted_resolution",
    "status"
]


missing = [
    c
    for c in required_standard_columns
    if c not in standard_df.columns
]


if missing:

    raise ValueError(
        f"Missing Standard columns: {missing}"
    )


standard_df = (
    standard_df[
        standard_df["status"] == "success"
    ]
    .copy()
)


standard_df = (
    standard_df[
        [
            "query_index",
            "fault_description",
            "reference_resolution",
            "predicted_resolution"
        ]
    ]
    .rename(
        columns={
            "fault_description":
                "standard_fault",

            "reference_resolution":
                "standard_reference",

            "predicted_resolution":
                "standard_generated"
        }
    )
)


# ============================================================
# LOAD CLIP + GMM
# ============================================================

clip_df = pd.read_csv(
    CLIP_GMM_FILE
)


required_clip_columns = [
    "query_index",
    "query_fault",
    "reference_resolution",
    "generated_resolution",
    "status"
]


missing = [
    c
    for c in required_clip_columns
    if c not in clip_df.columns
]


if missing:

    raise ValueError(
        f"Missing CLIP+GMM columns: {missing}"
    )


clip_df = (
    clip_df[
        clip_df["status"] == "success"
    ]
    .copy()
)


clip_df = (
    clip_df[
        [
            "query_index",
            "query_fault",
            "reference_resolution",
            "generated_resolution"
        ]
    ]
    .rename(
        columns={
            "query_fault":
                "clip_fault",

            "reference_resolution":
                "clip_reference",

            "generated_resolution":
                "clip_generated"
        }
    )
)


# ============================================================
# LOAD CLIP + PCA50 + GMM
# ============================================================

pca_df = pd.read_csv(
    PCA_GMM_FILE
)


required_pca_columns = [
    "query_id",
    "fault",
    "reference_resolution",
    "predicted_resolution"
]


missing = [
    c
    for c in required_pca_columns
    if c not in pca_df.columns
]


if missing:

    raise ValueError(
        f"Missing PCA+GMM columns: {missing}"
    )


pca_df = (
    pca_df[
        [
            "query_id",
            "fault",
            "reference_resolution",
            "predicted_resolution"
        ]
    ]
    .rename(
        columns={
            "query_id":
                "query_index",

            "fault":
                "pca_fault",

            "reference_resolution":
                "pca_reference",

            "predicted_resolution":
                "pca_generated"
        }
    )
)


# ============================================================
# CHECK DUPLICATE QUERY IDs
# ============================================================

for name, frame in [
    ("Standard", standard_df),
    ("CLIP+GMM", clip_df),
    ("PCA+GMM", pca_df)
]:

    if frame["query_index"].duplicated().any():

        raise ValueError(
            f"Duplicate query IDs found in {name}"
        )


# ============================================================
# MERGE ALL THREE
# ============================================================

df = pd.merge(
    standard_df,
    clip_df,
    on="query_index",
    how="inner",
    validate="one_to_one"
)


df = pd.merge(
    df,
    pca_df,
    on="query_index",
    how="inner",
    validate="one_to_one"
)


df = (
    df
    .sort_values("query_index")
    .reset_index(drop=True)
)


print()
print(
    "Standard successful queries:",
    len(standard_df)
)

print(
    "CLIP+GMM successful queries:",
    len(clip_df)
)

print(
    "PCA+GMM queries:",
    len(pca_df)
)

print(
    "Matched queries:",
    len(df)
)


if len(df) != 1050:

    print(
        "\nWARNING: "
        f"expected 1050 matched queries, "
        f"found {len(df)}."
    )


# ============================================================
# VERIFY SAME TEST QUERIES
# ============================================================

standard_fault = (
    df["standard_fault"]
    .map(normalize_for_check)
)

clip_fault = (
    df["clip_fault"]
    .map(normalize_for_check)
)

pca_fault = (
    df["pca_fault"]
    .map(normalize_for_check)
)


if not (standard_fault == clip_fault).all():

    bad = df.loc[
        standard_fault != clip_fault,
        "query_index"
    ].tolist()

    raise ValueError(
        "Fault mismatch between Standard "
        "and CLIP+GMM:\n"
        + str(bad[:20])
    )


if not (standard_fault == pca_fault).all():

    bad = df.loc[
        standard_fault != pca_fault,
        "query_index"
    ].tolist()

    raise ValueError(
        "Fault mismatch between Standard "
        "and PCA+GMM:\n"
        + str(bad[:20])
    )


standard_reference = (
    df["standard_reference"]
    .map(normalize_for_check)
)

clip_reference = (
    df["clip_reference"]
    .map(normalize_for_check)
)

pca_reference = (
    df["pca_reference"]
    .map(normalize_for_check)
)


if not (
    standard_reference
    ==
    clip_reference
).all():

    bad = df.loc[
        standard_reference != clip_reference,
        "query_index"
    ].tolist()

    raise ValueError(
        "Reference mismatch between Standard "
        "and CLIP+GMM:\n"
        + str(bad[:20])
    )


if not (
    standard_reference
    ==
    pca_reference
).all():

    bad = df.loc[
        standard_reference != pca_reference,
        "query_index"
    ].tolist()

    raise ValueError(
        "Reference mismatch between Standard "
        "and PCA+GMM:\n"
        + str(bad[:20])
    )


# Use one canonical copy after verification.

df["query_fault"] = (
    df["standard_fault"]
)

df["reference_resolution"] = (
    df["standard_reference"]
)


# ============================================================
# OPTIONAL SMALL TEST
# ============================================================

if MAX_TEST_QUERIES is not None:

    df = (
        df
        .iloc[:MAX_TEST_QUERIES]
        .copy()
        .reset_index(drop=True)
    )


print()
print(
    "Queries being judged:",
    len(df)
)

print(
    "Judge model:",
    JUDGE_MODEL
)

print(
    "Expected judge calls:",
    len(df) * 3
)


# ============================================================
# RUN JUDGE
# ============================================================

results = []


print(
    "\n" + "=" * 90
)

print(
    "RUNNING FINAL THREE-METHOD LLM JUDGE"
)

print(
    "=" * 90
)


for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Judging"
):

    query_id = int(
        row["query_index"]
    )


    fault = clean_text(
        row["query_fault"]
    )


    reference = clean_text(
        row["reference_resolution"]
    )


    standard_answer = clean_text(
        row["standard_generated"]
    )


    clip_answer = clean_text(
        row["clip_generated"]
    )


    pca_answer = clean_text(
        row["pca_generated"]
    )


    # --------------------------------------------------------
    # STANDARD
    # --------------------------------------------------------

    standard_result = call_judge(
        fault,
        reference,
        standard_answer
    )


    # --------------------------------------------------------
    # CLIP + GMM
    # --------------------------------------------------------

    clip_result = call_judge(
        fault,
        reference,
        clip_answer
    )


    # --------------------------------------------------------
    # CLIP + PCA50 + GMM
    # --------------------------------------------------------

    pca_result = call_judge(
        fault,
        reference,
        pca_answer
    )


    result = {

        "query_index":
            query_id,

        "query_fault":
            fault,

        "reference_resolution":
            reference,


        # STANDARD
        "standard_generated_resolution":
            standard_answer,

        "standard_judge_label":
            standard_result["label"],

        "standard_judge_reason":
            standard_result["reason"],

        "standard_judge_raw":
            standard_result["raw_output"],

        "standard_judge_error":
            standard_result["error"],


        # CLIP + GMM
        "clip_gmm_generated_resolution":
            clip_answer,

        "clip_gmm_judge_label":
            clip_result["label"],

        "clip_gmm_judge_reason":
            clip_result["reason"],

        "clip_gmm_judge_raw":
            clip_result["raw_output"],

        "clip_gmm_judge_error":
            clip_result["error"],


        # PCA + GMM
        "pca_gmm_generated_resolution":
            pca_answer,

        "pca_gmm_judge_label":
            pca_result["label"],

        "pca_gmm_judge_reason":
            pca_result["reason"],

        "pca_gmm_judge_raw":
            pca_result["raw_output"],

        "pca_gmm_judge_error":
            pca_result["error"]
    }


    results.append(
        result
    )


    # --------------------------------------------------------
    # CHECKPOINT AFTER EVERY QUERY
    # --------------------------------------------------------

    pd.DataFrame(
        results
    ).to_csv(
        CHECKPOINT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# LABEL STATISTICS
# ============================================================

VALID_LABELS = [
    "RELEVANT",
    "CLOSER",
    "INADEQUATE"
]


def label_statistics(series):

    total_valid = (
        series
        .isin(VALID_LABELS)
        .sum()
    )


    stats = {}


    for label in VALID_LABELS:

        count = int(
            (series == label).sum()
        )


        if total_valid > 0:

            percentage = (
                count
                /
                total_valid
                *
                100
            )

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
        len(series)
        -
        total_valid
    )


    return stats


method_columns = {

    "standard_rag":
        "standard_judge_label",

    "clip_gmm":
        "clip_gmm_judge_label",

    "clip_pca50_gmm":
        "pca_gmm_judge_label"
}


all_stats = {

    method:
        label_statistics(
            results_df[column]
        )

    for method, column
    in method_columns.items()
}


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = []


for label in VALID_LABELS:

    row = {
        "category": label
    }


    for method in [
        "standard_rag",
        "clip_gmm",
        "clip_pca50_gmm"
    ]:

        row[
            f"{method}_count"
        ] = (
            all_stats[
                method
            ][
                label
            ][
                "count"
            ]
        )


        row[
            f"{method}_percentage"
        ] = (
            all_stats[
                method
            ][
                label
            ][
                "percentage"
            ]
        )


    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# RELEVANT + CLOSER
# ============================================================

acceptable_stats = {}


for method in [
    "standard_rag",
    "clip_gmm",
    "clip_pca50_gmm"
]:

    stats = all_stats[
        method
    ]


    acceptable_count = (
        stats[
            "RELEVANT"
        ][
            "count"
        ]
        +
        stats[
            "CLOSER"
        ][
            "count"
        ]
    )


    if stats[
        "valid_total"
    ] > 0:

        acceptable_percentage = (
            acceptable_count
            /
            stats[
                "valid_total"
            ]
            *
            100
        )

    else:

        acceptable_percentage = 0.0


    acceptable_stats[
        method
    ] = {
        "count":
            acceptable_count,

        "percentage":
            acceptable_percentage
    }


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    DETAIL_CSV,
    index=False,
    encoding="utf-8-sig"
)


results_df.to_excel(
    DETAIL_XLSX,
    index=False
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
    encoding="utf-8-sig"
)


summary_df.to_excel(
    SUMMARY_XLSX,
    index=False
)


summary_json = {

    "judge_model":
        JUDGE_MODEL,

    "number_of_queries":
        len(results_df)
}


for method in [
    "standard_rag",
    "clip_gmm",
    "clip_pca50_gmm"
]:

    stats = all_stats[
        method
    ]


    summary_json[
        method
    ] = {

        "relevant_count":
            stats["RELEVANT"]["count"],

        "relevant_percentage":
            stats["RELEVANT"]["percentage"],

        "closer_count":
            stats["CLOSER"]["count"],

        "closer_percentage":
            stats["CLOSER"]["percentage"],

        "inadequate_count":
            stats["INADEQUATE"]["count"],

        "inadequate_percentage":
            stats["INADEQUATE"]["percentage"],

        "relevant_plus_closer_count":
            acceptable_stats[
                method
            ][
                "count"
            ],

        "relevant_plus_closer_percentage":
            acceptable_stats[
                method
            ][
                "percentage"
            ],

        "errors":
            stats[
                "parse_or_api_errors"
            ]
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

print(
    "\n" + "=" * 90
)

print(
    "FINAL LLM-JUDGE RESULTS"
)

print(
    "=" * 90
)


display_names = {

    "standard_rag":
        "STANDARD RAG",

    "clip_gmm":
        "CLIP + GMM",

    "clip_pca50_gmm":
        "CLIP + PCA50 + GMM"
}


for method in [
    "standard_rag",
    "clip_gmm",
    "clip_pca50_gmm"
]:

    stats = all_stats[
        method
    ]


    print(
        "\n"
        +
        display_names[
            method
        ]
    )


    print(
        f"Relevant   : "
        f"{stats['RELEVANT']['count']} "
        f"("
        f"{stats['RELEVANT']['percentage']:.2f}%"
        f")"
    )


    print(
        f"Closer     : "
        f"{stats['CLOSER']['count']} "
        f"("
        f"{stats['CLOSER']['percentage']:.2f}%"
        f")"
    )


    print(
        f"Inadequate : "
        f"{stats['INADEQUATE']['count']} "
        f"("
        f"{stats['INADEQUATE']['percentage']:.2f}%"
        f")"
    )


    print(
        f"Relevant + Closer: "
        f"{acceptable_stats[method]['percentage']:.2f}%"
    )


    print(
        f"Errors     : "
        f"{stats['parse_or_api_errors']}"
    )


print(
    "\nFiles saved:"
)

print(
    DETAIL_CSV
)

print(
    DETAIL_XLSX
)

print(
    SUMMARY_CSV
)

print(
    SUMMARY_XLSX
)

print(
    SUMMARY_JSON
)

print(
    CHECKPOINT_FILE
)