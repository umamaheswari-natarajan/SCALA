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

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_rag",
    "plain_gmm_query_results.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "plain_gmm_llm_judge"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


JUDGE_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 120

# Use None for all 872 queries
MAX_TEST_QUERIES = None

MAX_RETRIES = 3


# ============================================================
# OPENAI API
# ============================================================

if not os.getenv("OPENAI_API_KEY"):

    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )


client = OpenAI(
    timeout=30.0,
    max_retries=0
)


# ============================================================
# JUDGE INSTRUCTIONS
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
Different wording is acceptable.

CLOSER:
The generated resolution is related to the correct remediation and
contains useful or technically appropriate actions, but it is incomplete,
too general, partially correct, or misses an important action.

INADEQUATE:
The generated resolution does not adequately address the fault,
is materially incorrect or unrelated, contradicts the reference,
or fails to provide a useful resolution.

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


def call_judge(
    fault,
    ground_truth,
    generated_resolution
):

    prompt = f"""
FAULT:
{fault}

REFERENCE GROUND-TRUTH RESOLUTION:
{ground_truth}

GENERATED RESOLUTION:
{generated_resolution}

Classify the generated resolution.
""".strip()


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


            text = response.output_text.strip()


            # ------------------------------------------------
            # Parse label
            # ------------------------------------------------

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

                label = "PARSE_ERROR"


            # ------------------------------------------------
            # Parse reason
            # ------------------------------------------------

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
                reason,
                ""
            )


        except Exception as e:

            last_error = str(e)

            print(
                f"Attempt "
                f"{attempt}/{MAX_RETRIES} "
                f"failed: {last_error}"
            )

            time.sleep(
                attempt * 2
            )


    return (
        "API_ERROR",
        "",
        last_error
    )


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_excel(
    INPUT_FILE
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "plain_gmm_llm_judge_checkpoint.csv"
)

if os.path.exists(CHECKPOINT_FILE):

    previous_df = pd.read_csv(
        CHECKPOINT_FILE
    )

    print(
        "Existing completed judgments:",
        len(previous_df)
    )

    completed_ids = set(
        previous_df["query_id"]
        .astype(str)
    )

    df = df[
        ~df["query_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()

    df = df.reset_index(
        drop=True
    )

    results = previous_df.to_dict(
        "records"
    )

else:

    print(
        "No existing checkpoint found."
    )

    results = []


if MAX_TEST_QUERIES is not None:

    df = (
        df.iloc[
            :MAX_TEST_QUERIES
        ]
        .copy()
        .reset_index(drop=True)
    )


print("=" * 90)
print("PLAIN GMM LLM JUDGE")
print("=" * 90)

print(
    "Remaining queries:",
    len(df)
)

print(
    "Remaining judge calls:",
    len(df)
)


# if MAX_TEST_QUERIES is not None:

#     df = (
#         df.iloc[
#             :MAX_TEST_QUERIES
#         ]
#         .copy()
#         .reset_index(drop=True)
#     )


# print("=" * 90)
# print("PLAIN GMM LLM JUDGE")
# print("=" * 90)

# print(
#     "Queries:",
#     len(df)
# )

# print(
#     "Expected judge calls:",
#     len(df)
# )


# ============================================================
# RUN JUDGE
# ============================================================




for idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Judging Plain GMM-RAG"
):


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


    generated_resolution = clean_text(
        row[
            "plain_gmm_generated_resolution"
        ]
    )


    (
        label,
        reason,
        error
    ) = call_judge(
        fault,
        ground_truth,
        generated_resolution
    )


    results.append({

        "query_index":
            row[
                "query_index"
            ]
            if "query_index"
            in row.index
            else idx,

        "query_id":
            row[
                "query_id"
            ],

        "query_fault":
            fault,

        "ground_truth_resolution":
            ground_truth,

        "plain_gmm_generated_resolution":
            generated_resolution,

        "judge_label":
            label,

        "judge_reason":
            reason,

        "judge_error":
            error
    })


    # ========================================================
    # CHECKPOINT
    # ========================================================

    pd.DataFrame(
        results
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "plain_gmm_llm_judge_checkpoint.csv"
        ),
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

valid_labels = [
    "RELEVANT",
    "CLOSER",
    "INADEQUATE"
]


valid_df = results_df[
    results_df[
        "judge_label"
    ].isin(
        valid_labels
    )
]


total_valid = len(
    valid_df
)


# ============================================================
# BUILD SUMMARY
# ============================================================

summary = {

    "baseline":
        "Plain GMM-RAG",

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


for label in valid_labels:

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
        else 0
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
    ] = percentage


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


summary[
    "relevant_plus_closer_count"
] = int(
    acceptable_count
)


summary[
    "relevant_plus_closer_percentage"
] = (

    acceptable_count
    /
    total_valid
    *
    100

    if total_valid > 0

    else 0
)


# ============================================================
# ERROR COUNTS
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
# SAVE RESULTS
# ============================================================

RESULT_XLSX = os.path.join(
    OUTPUT_DIR,
    "plain_gmm_llm_judge_results.xlsx"
)


SUMMARY_XLSX = os.path.join(
    OUTPUT_DIR,
    "plain_gmm_llm_judge_summary.xlsx"
)


SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "plain_gmm_llm_judge_summary.json"
)


results_df.to_excel(
    RESULT_XLSX,
    index=False
)


summary_df = pd.DataFrame(
    [
        summary
    ]
)


summary_df.to_excel(
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
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("PLAIN GMM LLM-JUDGE RESULTS")
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
    f"Relevant + Closer: "
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