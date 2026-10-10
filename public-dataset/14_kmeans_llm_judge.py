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
    "kmeans_rag",
    "kmeans_query_results.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "kmeans_llm_judge"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


JUDGE_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 120

MAX_TEST_QUERIES = None

MAX_RETRIES = 3


# ============================================================
# API
# ============================================================

if not os.getenv(
    "OPENAI_API_KEY"
):

    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )


client = OpenAI()


# ============================================================
# JUDGE PROMPT
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
"""


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
# LOAD
# ============================================================

df = pd.read_excel(
    INPUT_FILE
)


if MAX_TEST_QUERIES is not None:

    df = (
        df.iloc[
            :MAX_TEST_QUERIES
        ]
        .copy()
    )


print(
    "Queries:",
    len(df)
)


# ============================================================
# RUN
# ============================================================

results = []


for idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="K-Means Judge"
):


    label, reason, error = (
        call_judge(

            clean_text(
                row["query_fault"]
            ),

            clean_text(
                row[
                    "ground_truth_resolution"
                ]
            ),

            clean_text(
                row[
                    "kmeans_generated_resolution"
                ]
            )
        )
    )


    results.append({

        "query_id":
            row["query_id"],

        "query_fault":
            row["query_fault"],

        "ground_truth_resolution":
            row[
                "ground_truth_resolution"
            ],

        "kmeans_generated_resolution":
            row[
                "kmeans_generated_resolution"
            ],

        "judge_label":
            label,

        "judge_reason":
            reason,

        "judge_error":
            error
    })


    pd.DataFrame(
        results
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "kmeans_llm_judge_checkpoint.csv"
        ),
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


valid_labels = [
    "RELEVANT",
    "CLOSER",
    "INADEQUATE"
]


valid_df = results_df[
    results_df[
        "judge_label"
    ].isin(valid_labels)
]


total = len(valid_df)


summary = {

    "judge_model":
        JUDGE_MODEL,

    "number_of_queries":
        len(results_df),

    "valid_judgements":
        total
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
        count / total * 100
        if total > 0
        else 0
    )

    summary[
        label.lower()
        + "_count"
    ] = count

    summary[
        label.lower()
        + "_percentage"
    ] = percentage


acceptable = (
    summary[
        "relevant_count"
    ]
    +
    summary[
        "closer_count"
    ]
)


summary[
    "relevant_plus_closer_percentage"
] = (
    acceptable
    /
    total
    *
    100
    if total > 0
    else 0
)


# ============================================================
# SAVE
# ============================================================

results_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "kmeans_llm_judge_results.xlsx"
    ),
    index=False
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "kmeans_llm_judge_summary.json"
    ),
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

print("\n" + "=" * 90)
print("K-MEANS LLM-JUDGE RESULTS")
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