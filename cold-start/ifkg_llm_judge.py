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

IFKG_FILE = os.path.join(
    BASE_DIR,
    "synthetic_ifkg_gpt5mini",
    "ifkg_results_scored.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "synthetic_ifkg_llm_judge"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DETAIL_CSV = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_results.csv"
)

DETAIL_XLSX = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_results.xlsx"
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_checkpoint.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_summary.csv"
)

SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "ifkg_llm_judge_summary.json"
)


# ============================================================
# JUDGE SETTINGS
# SAME AS PREVIOUS THREE-METHOD EVALUATION
# ============================================================

JUDGE_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 160

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
# EXACT SAME RUBRIC AS PREVIOUS EVALUATION
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
# CHECK INPUT
# ============================================================

if not os.path.exists(IFKG_FILE):

    raise FileNotFoundError(
        f"Missing IFKG file:\n{IFKG_FILE}"
    )


# ============================================================
# LOAD IFKG RESULTS
# ============================================================

df = pd.read_csv(
    IFKG_FILE
)


required_columns = [
    "query_index",
    "fault_description",
    "reference_resolution",
    "predicted_resolution",
    "generation_status"
]


missing = [
    c
    for c in required_columns
    if c not in df.columns
]


if missing:

    raise ValueError(
        f"Missing IFKG columns: {missing}"
    )


if df["query_index"].duplicated().any():

    raise ValueError(
        "Duplicate query IDs found in IFKG results."
    )


df = (
    df
    .sort_values("query_index")
    .reset_index(drop=True)
)


print("=" * 90)
print("LOADING IFKG RESULTS")
print("=" * 90)

print(
    "Total queries:",
    len(df)
)

print(
    "Successful generations:",
    (
        df["generation_status"]
        ==
        "success"
    ).sum()
)

print(
    "No KG matches:",
    (
        df["generation_status"]
        ==
        "no_kg_match"
    ).sum()
)


if len(df) != 1050:

    raise ValueError(
        f"Expected 1050 queries, found {len(df)}."
    )


# ============================================================
# CHECK FOR UNEXPECTED STATUSES
# ============================================================

expected_statuses = {
    "success",
    "no_kg_match"
}

unexpected_statuses = set(
    df["generation_status"]
    .dropna()
    .astype(str)
    .unique()
) - expected_statuses


if unexpected_statuses:

    print(
        "WARNING: unexpected generation statuses:",
        unexpected_statuses
    )


# ============================================================
# RESUME FROM CHECKPOINT
# ============================================================

if os.path.exists(CHECKPOINT_FILE):

    checkpoint_df = pd.read_csv(
        CHECKPOINT_FILE
    )

    results = checkpoint_df.to_dict(
        "records"
    )

    completed_indices = set(
        checkpoint_df["query_index"]
        .astype(int)
        .tolist()
    )

    print(
        f"\nCheckpoint found: "
        f"{len(completed_indices)} queries completed."
    )

else:

    results = []
    completed_indices = set()

    print("\nStarting new IFKG judge run.")


# ============================================================
# RUN JUDGE
# ============================================================

print("\n" + "=" * 90)
print("RUNNING IFKG LLM JUDGE")
print("=" * 90)


for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Judging IFKG"
):

    query_id = int(
        row["query_index"]
    )


    if query_id in completed_indices:
        continue


    fault = clean_text(
        row["fault_description"]
    )

    reference = clean_text(
        row["reference_resolution"]
    )

    generated = clean_text(
        row["predicted_resolution"]
    )

    status = clean_text(
        row["generation_status"]
    )


    # ========================================================
    # NO KG MATCH
    #
    # No resolution exists, so this is an end-to-end failure.
    # Do NOT send [NO_KG_MATCH] to the LLM judge.
    # ========================================================

    if status == "no_kg_match":

        judge_result = {
            "label": "INADEQUATE",
            "reason":
                "No knowledge graph evidence was retrieved and "
                "no resolution was generated.",
            "raw_output": "",
            "error": ""
        }

        judge_source = "automatic_no_kg_match"


    # ========================================================
    # SUCCESSFUL GENERATION
    # ========================================================

    elif status == "success":

        judge_result = call_judge(
            fault,
            reference,
            generated
        )

        judge_source = "gpt-4.1-mini"


    # ========================================================
    # UNEXPECTED FAILURE STATUS
    # ========================================================

    else:

        judge_result = {
            "label": "INADEQUATE",
            "reason":
                f"Resolution generation did not complete "
                f"successfully (status={status}).",
            "raw_output": "",
            "error": ""
        }

        judge_source = "automatic_generation_failure"


    result = {

        "query_index":
            query_id,

        "fault_description":
            fault,

        "reference_resolution":
            reference,

        "predicted_resolution":
            generated,

        "generation_status":
            status,

        "judge_label":
            judge_result["label"],

        "judge_reason":
            judge_result["reason"],

        "judge_raw":
            judge_result["raw_output"],

        "judge_error":
            judge_result["error"],

        "judge_source":
            judge_source
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
# FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values("query_index")
    .reset_index(drop=True)
)


# ============================================================
# LABEL STATISTICS
# ============================================================

VALID_LABELS = [
    "RELEVANT",
    "CLOSER",
    "INADEQUATE"
]


valid_total = (
    results_df["judge_label"]
    .isin(VALID_LABELS)
    .sum()
)


stats = {}


for label in VALID_LABELS:

    count = int(
        (
            results_df["judge_label"]
            ==
            label
        ).sum()
    )

    percentage = (
        count
        /
        valid_total
        *
        100
        if valid_total > 0
        else 0.0
    )

    stats[label] = {
        "count": count,
        "percentage": percentage
    }


errors = int(
    len(results_df)
    -
    valid_total
)


acceptable_count = (
    stats["RELEVANT"]["count"]
    +
    stats["CLOSER"]["count"]
)


acceptable_percentage = (
    acceptable_count
    /
    valid_total
    *
    100
    if valid_total > 0
    else 0.0
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "category": label,
            "count": stats[label]["count"],
            "percentage": stats[label]["percentage"]
        }
        for label in VALID_LABELS
    ]
)


summary_json = {

    "judge_model":
        JUDGE_MODEL,

    "number_of_queries":
        len(results_df),

    "gpt_judged_queries":
        int(
            (
                results_df["judge_source"]
                ==
                "gpt-4.1-mini"
            ).sum()
        ),

    "automatic_no_kg_match":
        int(
            (
                results_df["judge_source"]
                ==
                "automatic_no_kg_match"
            ).sum()
        ),

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
        acceptable_count,

    "relevant_plus_closer_percentage":
        acceptable_percentage,

    "errors":
        errors
}


# ============================================================
# SAVE
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
print("FINAL IFKG LLM-JUDGE RESULTS")
print("=" * 90)

print(
    f"\nTotal queries       : "
    f"{len(results_df)}"
)

print(
    f"GPT-judged queries  : "
    f"{summary_json['gpt_judged_queries']}"
)

print(
    f"Automatic no-KG     : "
    f"{summary_json['automatic_no_kg_match']}"
)


print(
    f"\nRelevant   : "
    f"{stats['RELEVANT']['count']} "
    f"({stats['RELEVANT']['percentage']:.2f}%)"
)

print(
    f"Closer     : "
    f"{stats['CLOSER']['count']} "
    f"({stats['CLOSER']['percentage']:.2f}%)"
)

print(
    f"Inadequate : "
    f"{stats['INADEQUATE']['count']} "
    f"({stats['INADEQUATE']['percentage']:.2f}%)"
)

print(
    f"Relevant + Closer: "
    f"{acceptable_percentage:.2f}%"
)

print(
    f"Errors     : "
    f"{errors}"
)


print("\nFiles saved:")

print(DETAIL_CSV)
print(DETAIL_XLSX)
print(SUMMARY_CSV)
print(SUMMARY_JSON)
print(CHECKPOINT_FILE)