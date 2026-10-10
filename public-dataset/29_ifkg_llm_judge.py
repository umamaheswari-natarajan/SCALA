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
    "ifkg_inference",
    "ifkg_query_results.xlsx"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ifkg_llm_judge"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_checkpoint.csv"
)


RESULT_XLSX = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_results.xlsx"
)


RESULT_CSV = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_results.csv"
)


SUMMARY_XLSX = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_summary.xlsx"
)


SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_summary.json"
)


# ============================================================
# JUDGE SETTINGS
# ============================================================

JUDGE_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 120

MAX_RETRIES = 3

# None = all queries
MAX_TEST_QUERIES = None


# ============================================================
# API
# ============================================================

if not os.getenv(
    "OPENAI_API_KEY"
):

    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )


client = OpenAI(
    timeout=30.0,
    max_retries=0
)


# ============================================================
# EXACT SAME JUDGE CRITERIA
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

def clean_text(
    value
):

    if pd.isna(
        value
    ):

        return ""

    return str(
        value
    ).strip()


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


def parse_judge_output(
    text
):

    text = clean_text(
        text
    )


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


            raw_output = (
                response.output_text
                .strip()
            )


            (
                label,
                reason
            ) = parse_judge_output(
                raw_output
            )


            return {
                "label":
                    label,

                "reason":
                    reason,

                "raw_output":
                    raw_output,

                "error":
                    ""
            }


        except Exception as e:

            last_error = str(
                e
            )


            print(
                f"\nAttempt "
                f"{attempt}/"
                f"{MAX_RETRIES} failed: "
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
# LOAD IFKG RESULTS
# ============================================================

df = pd.read_excel(
    INPUT_FILE
)


required_columns = [
    "query_id",
    "query_fault",
    "ground_truth_resolution",
    "ifkg_generated_resolution"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


# ============================================================
# RESUME
# ============================================================

if os.path.exists(
    CHECKPOINT_FILE
):

    previous_df = pd.read_csv(
        CHECKPOINT_FILE
    )


    print(
        "Existing completed judgments:",
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

    completed_ids = set()

    results = []


# ============================================================
# OPTIONAL LIMIT
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
# RUN
# ============================================================

print("=" * 90)
print("IFKG LLM JUDGE")
print("=" * 90)


print(
    "Remaining queries:",
    len(df)
)


print(
    "Remaining judge calls:",
    len(df)
)


for idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Judging IFKG"
):

    query_id = row[
        "query_id"
    ]


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


    generated = clean_text(
        row[
            "ifkg_generated_resolution"
        ]
    )


    judge = call_judge(
        fault,
        ground_truth,
        generated
    )


    results.append({

        "query_id":
            query_id,

        "query_fault":
            fault,

        "ground_truth_resolution":
            ground_truth,

        "ifkg_generated_resolution":
            generated,

        "judge_label":
            judge[
                "label"
            ],

        "judge_reason":
            judge[
                "reason"
            ],

        "judge_raw":
            judge[
                "raw_output"
            ],

        "judge_error":
            judge[
                "error"
            ]
    })


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
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


VALID_LABELS = [
    "RELEVANT",
    "CLOSER",
    "INADEQUATE"
]


valid_df = results_df[
    results_df[
        "judge_label"
    ].isin(
        VALID_LABELS
    )
]


total_valid = len(
    valid_df
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "baseline":
        "IFKG",

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
            total_valid
        )
}


for label in VALID_LABELS:

    count = int(
        (
            valid_df[
                "judge_label"
            ]
            ==
            label
        ).sum()
    )


    percentage = (
        count
        /
        total_valid
        *
        100
        if total_valid > 0
        else 0.0
    )


    summary[
        label.lower()
        +
        "_count"
    ] = count


    summary[
        label.lower()
        +
        "_percentage"
    ] = float(
        percentage
    )


# ============================================================
# RELEVANT + CLOSER
# ============================================================

acceptable_count = (

    summary[
        "relevant_count"
    ]

    +

    summary[
        "closer_count"
    ]
)


acceptable_percentage = (

    acceptable_count
    /
    total_valid
    *
    100

    if total_valid > 0

    else 0.0
)


summary[
    "relevant_plus_closer_count"
] = int(
    acceptable_count
)


summary[
    "relevant_plus_closer_percentage"
] = float(
    acceptable_percentage
)


# ============================================================
# ERRORS
# ============================================================

summary[
    "parse_error_count"
] = int(
    (
        results_df[
            "judge_label"
        ]
        ==
        "PARSE_ERROR"
    ).sum()
)


summary[
    "api_error_count"
] = int(
    (
        results_df[
            "judge_label"
        ]
        ==
        "API_ERROR"
    ).sum()
)


# ============================================================
# SAVE
# ============================================================

results_df.to_excel(
    RESULT_XLSX,
    index=False
)


results_df.to_csv(
    RESULT_CSV,
    index=False
)


pd.DataFrame(
    [
        summary
    ]
).to_excel(
    SUMMARY_XLSX,
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
# FINAL
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "IFKG LLM-JUDGE RESULTS"
)

print("=" * 90)


print(
    f"Relevant  : "
    f"{summary['relevant_count']} "
    f"({summary['relevant_percentage']:.2f}%)"
)


print(
    f"Closer    : "
    f"{summary['closer_count']} "
    f"({summary['closer_percentage']:.2f}%)"
)


print(
    f"Inadequate: "
    f"{summary['inadequate_count']} "
    f"({summary['inadequate_percentage']:.2f}%)"
)


print(
    "Relevant + Closer:",
    f"{summary['relevant_plus_closer_percentage']:.2f}%"
)


print(
    "\nParse errors:",
    summary[
        "parse_error_count"
    ]
)


print(
    "API errors:",
    summary[
        "api_error_count"
    ]
)


print("\nFiles saved:")

print(
    RESULT_XLSX
)

print(
    SUMMARY_XLSX
)

print(
    SUMMARY_JSON
)