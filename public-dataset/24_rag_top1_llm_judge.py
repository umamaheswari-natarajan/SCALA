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

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"


# ============================================================
# TOP-1 INFERENCE RESULTS
# ============================================================

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "rag_inference_top1",
    "rag_top1_query_results.xlsx"
)


# ============================================================
# TOP-1 JUDGE OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "rag_top1_llm_judge"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

DETAIL_FILE = os.path.join(
    OUTPUT_DIR,
    "rag_top1_llm_judge_results.xlsx"
)

DETAIL_CSV = os.path.join(
    OUTPUT_DIR,
    "rag_top1_llm_judge_results.csv"
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "rag_top1_llm_judge_checkpoint.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "rag_top1_llm_judge_summary.xlsx"
)

SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "rag_top1_llm_judge_summary.json"
)


# ============================================================
# JUDGE SETTINGS
# ============================================================

JUDGE_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 120

# None = all remaining test queries
MAX_TEST_QUERIES = None

MAX_RETRIES = 3


# ============================================================
# API
# ============================================================

if not os.getenv(
    "OPENAI_API_KEY"
):

    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        "In PowerShell run:\n"
        '$env:OPENAI_API_KEY="YOUR_API_KEY"'
    )


# ------------------------------------------------------------
# Disable OpenAI automatic retries.
# Our own retry loop controls retries.
# ------------------------------------------------------------

client = OpenAI(
    timeout=30.0,
    max_retries=0
)


# ============================================================
# JUDGE INSTRUCTIONS
#
# IMPORTANT:
# Keep EXACTLY the same evaluation criteria as Top-3 and
# the other methods.
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
""".strip()


# ============================================================
# HELPERS
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
        flags=(
            re.IGNORECASE
            |
            re.DOTALL
        )
    )

    if reason_match:

        reason = (
            reason_match
            .group(1)
            .strip()
        )

    else:

        reason = ""


    return (
        label,
        reason
    )


# ============================================================
# CALL JUDGE
# ============================================================

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

    last_error = ""

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = client.responses.create(

                model=JUDGE_MODEL,

                instructions=JUDGE_INSTRUCTIONS,

                input=prompt,

                max_output_tokens=MAX_OUTPUT_TOKENS,

                temperature=0
            )


            output_text = (
                response
                .output_text
                .strip()
            )


            (
                label,
                reason
            ) = parse_judge_output(
                output_text
            )


            return {

                "label":
                    label,

                "reason":
                    reason,

                "raw_output":
                    output_text,

                "error":
                    ""
            }


        except Exception as e:

            last_error = str(e)

            print(
                f"\nAPI error "
                f"(attempt "
                f"{attempt}/{MAX_RETRIES}): "
                f"{last_error}"
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

        "raw_output":
            "",

        "error":
            last_error
    }


# ============================================================
# LOAD TOP-1 RESULTS
# ============================================================

print(
    "=" * 90
)

print(
    "LOADING RETRIEVAL-AWARE GMM TOP-1 RESULTS"
)

print(
    "=" * 90
)


df = pd.read_excel(
    INPUT_FILE
)


print(
    "Input file:"
)

print(
    INPUT_FILE
)


print(
    "\nRows:",
    len(df)
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [

    "query_fault",

    "ground_truth_resolution",

    "gmm_top1_generated_resolution"
]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ============================================================
# RESUME FROM CHECKPOINT
# ============================================================

if os.path.exists(
    CHECKPOINT_FILE
):

    previous_df = pd.read_csv(
        CHECKPOINT_FILE
    )


    print(
        "\nExisting completed judgments:",
        len(previous_df)
    )


    completed_ids = set(

        previous_df[
            "query_id"
        ]
        .astype(str)
    )


    df = df[

        ~df[
            "query_id"
        ]
        .astype(str)
        .isin(
            completed_ids
        )

    ].copy()


    df = df.reset_index(
        drop=True
    )


    results = (
        previous_df
        .to_dict(
            "records"
        )
    )


else:

    print(
        "\nNo existing checkpoint found."
    )

    results = []


# ============================================================
# OPTIONAL TEST LIMIT
# ============================================================

if MAX_TEST_QUERIES is not None:

    df = (

        df
        .iloc[
            :MAX_TEST_QUERIES
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# RUN INFO
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "RETRIEVAL-AWARE GMM TOP-1 LLM JUDGE"
)

print(
    "=" * 90
)


print(
    "Remaining queries:",
    len(df)
)


print(
    "Remaining judge calls:",
    len(df)
)


print(
    "Judge model:",
    JUDGE_MODEL
)


# ============================================================
# RUN JUDGE
#
# One judge call per query.
# ============================================================

for idx, row in tqdm(

    df.iterrows(),

    total=len(df),

    desc="Judging Top-1 GMM"
):


    query_id = (

        row[
            "query_id"
        ]

        if "query_id"
        in row.index

        else idx
    )


    fault = clean_text(
        row[
            "query_fault"
        ]
    )


    ground_truth = clean_text(
        row[
            "ground_truth_resolution"
        ]
    )


    top1_answer = clean_text(
        row[
            "gmm_top1_generated_resolution"
        ]
    )


    # ========================================================
    # TOP-1 GMM JUDGMENT
    # ========================================================

    judge_result = call_judge(

        fault=fault,

        ground_truth=ground_truth,

        generated_resolution=top1_answer
    )


    result = {

        "query_id":
            query_id,

        "query_fault":
            fault,

        "ground_truth_resolution":
            ground_truth,

        "gmm_top1_generated_resolution":
            top1_answer,

        "judge_label":
            judge_result[
                "label"
            ],

        "judge_reason":
            judge_result[
                "reason"
            ],

        "judge_raw":
            judge_result[
                "raw_output"
            ],

        "judge_error":
            judge_result[
                "error"
            ]
    }


    results.append(
        result
    )


    # ========================================================
    # CHECKPOINT AFTER EVERY QUERY
    # ========================================================

    pd.DataFrame(
        results
    ).to_csv(

        CHECKPOINT_FILE,

        index=False
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# VALID LABELS
# ============================================================

VALID_LABELS = [

    "RELEVANT",

    "CLOSER",

    "INADEQUATE"
]


# ============================================================
# LABEL STATISTICS
# ============================================================

def label_statistics(series):

    total_valid = int(

        series
        .isin(
            VALID_LABELS
        )
        .sum()
    )


    stats = {}


    for label in VALID_LABELS:

        count = int(

            (
                series
                ==
                label
            )
            .sum()
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


        stats[
            label
        ] = {

            "count":
                count,

            "percentage":
                percentage
        }


    stats[
        "valid_total"
    ] = total_valid


    stats[
        "parse_or_api_errors"
    ] = int(

        len(series)
        -
        total_valid
    )


    return stats


top1_stats = label_statistics(

    results_df[
        "judge_label"
    ]
)


# ============================================================
# BUILD SUMMARY TABLE
# ============================================================

summary_rows = []


for label in VALID_LABELS:

    summary_rows.append({

        "category":
            label,

        "count":
            top1_stats[
                label
            ][
                "count"
            ],

        "percentage":
            top1_stats[
                label
            ][
                "percentage"
            ]
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# RELEVANT + CLOSER
# ============================================================

acceptable_count = (

    top1_stats[
        "RELEVANT"
    ][
        "count"
    ]

    +

    top1_stats[
        "CLOSER"
    ][
        "count"
    ]
)


if top1_stats[
    "valid_total"
] > 0:

    acceptable_percentage = (

        acceptable_count

        /

        top1_stats[
            "valid_total"
        ]

        *

        100
    )

else:

    acceptable_percentage = 0.0


# ============================================================
# ERROR COUNTS
# ============================================================

parse_error_count = int(

    (
        results_df[
            "judge_label"
        ]
        ==
        "PARSE_ERROR"
    )
    .sum()
)


api_error_count = int(

    (
        results_df[
            "judge_label"
        ]
        ==
        "API_ERROR"
    )
    .sum()
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary_json = {

    "method":
        "Retrieval-Aware GMM Top-1",

    "judge_model":
        JUDGE_MODEL,

    "temperature":
        0,

    "number_of_queries":
        int(
            len(
                results_df
            )
        ),

    "valid_judgements":
        int(
            top1_stats[
                "valid_total"
            ]
        ),

    "relevant_count":

        top1_stats[
            "RELEVANT"
        ][
            "count"
        ],

    "relevant_percentage":

        top1_stats[
            "RELEVANT"
        ][
            "percentage"
        ],

    "closer_count":

        top1_stats[
            "CLOSER"
        ][
            "count"
        ],

    "closer_percentage":

        top1_stats[
            "CLOSER"
        ][
            "percentage"
        ],

    "inadequate_count":

        top1_stats[
            "INADEQUATE"
        ][
            "count"
        ],

    "inadequate_percentage":

        top1_stats[
            "INADEQUATE"
        ][
            "percentage"
        ],

    "relevant_plus_closer_count":
        int(
            acceptable_count
        ),

    "relevant_plus_closer_percentage":
        float(
            acceptable_percentage
        ),

    "parse_error_count":
        parse_error_count,

    "api_error_count":
        api_error_count
}


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
    "RETRIEVAL-AWARE GMM TOP-1 LLM-JUDGE RESULTS"
)

print(
    "=" * 90
)


print(

    f"Relevant  : "
    f"{summary_json['relevant_count']} "
    f"("
    f"{summary_json['relevant_percentage']:.2f}%"
    f")"
)


print(

    f"Closer    : "
    f"{summary_json['closer_count']} "
    f"("
    f"{summary_json['closer_percentage']:.2f}%"
    f")"
)


print(

    f"Inadequate: "
    f"{summary_json['inadequate_count']} "
    f"("
    f"{summary_json['inadequate_percentage']:.2f}%"
    f")"
)


print(

    f"Relevant + Closer: "
    f"{summary_json['relevant_plus_closer_percentage']:.2f}%"
)


print(
    "\nParse errors:",
    summary_json[
        "parse_error_count"
    ]
)


print(
    "API errors:",
    summary_json[
        "api_error_count"
    ]
)


print(
    "\nFiles saved:"
)


print(
    DETAIL_FILE
)

print(
    SUMMARY_FILE
)

print(
    SUMMARY_JSON
)