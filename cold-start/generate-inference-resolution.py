import os
import time
import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "synthetic_inference_FINAL_1050.csv"
OUTPUT_FILE = "synthetic_inference_FINAL_1050_with_reference.csv"

TEXT_COL = "Generated Fault Description"

MODEL = "gpt-4.1-mini"

# Save after every N successful records
SAVE_EVERY = 10

# Delay between calls
SLEEP_SECONDS = 0.2

# Retry settings
MAX_RETRIES = 5


# ============================================================
# OPENAI CLIENT
# ============================================================

# Recommended:
# Windows PowerShell:
# $env:OPENAI_API_KEY="your-key"
#
# Do NOT put the API key directly in this script.

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set."
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

if TEXT_COL not in df.columns:
    raise ValueError(
        f"'{TEXT_COL}' column not found.\n"
        f"Available columns: {list(df.columns)}"
    )

print("=" * 70)
print("GENERATING REFERENCE RESOLUTIONS")
print("=" * 70)

print("Input records :", len(df))
print("Model         :", MODEL)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

if "reference_resolution" not in df.columns:
    df["reference_resolution"] = ""

if "reference_status" not in df.columns:
    df["reference_status"] = ""

if "reference_latency_sec" not in df.columns:
    df["reference_latency_sec"] = None


# ============================================================
# RESUME FROM PREVIOUS RUN
# ============================================================

if os.path.exists(OUTPUT_FILE):

    print("\nExisting output found.")
    print("Resuming previous run...")

    old_df = pd.read_csv(OUTPUT_FILE)

    if len(old_df) == len(df):

        for col in [
            "reference_resolution",
            "reference_status",
            "reference_latency_sec"
        ]:

            if col in old_df.columns:
                df[col] = old_df[col]

        print(
            "Already completed:",
            (
                df["reference_status"]
                .fillna("")
                == "success"
            ).sum()
        )


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert in 5G, O-RAN, OpenAirInterface, mobile
networking, and telecom fault diagnosis.

You will receive one fault description.

Generate the most appropriate technical resolution for that
fault.

Requirements:
- Provide only the resolution.
- Do not repeat the fault description.
- Do not mention that you are an AI.
- Do not invent observations that are not necessary.
- Give a concise, technically actionable resolution.
- State the configuration change, corrective action, restart,
  validation, or troubleshooting step required to resolve the
  fault.
- Prefer approximately 1-3 sentences.
"""


def generate_resolution(fault):

    user_prompt = f"""
Fault Description:
{fault}

Provide the appropriate resolution.
"""

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            start = time.perf_counter()

            response = client.responses.create(
                model=MODEL,

                instructions=SYSTEM_PROMPT,

                input=user_prompt,

                temperature=0.2,

                max_output_tokens=180
            )

            latency = (
                time.perf_counter()
                - start
            )

            resolution = (
                response.output_text
                .strip()
            )

            if not resolution:
                raise ValueError(
                    "Empty resolution returned"
                )

            return (
                resolution,
                latency,
                "success"
            )

        except Exception as e:

            print(
                f"    Attempt {attempt}/"
                f"{MAX_RETRIES} failed: {e}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    min(2 ** attempt, 20)
                )

    return (
        "",
        None,
        "failed"
    )


# ============================================================
# GENERATE
# ============================================================

successful_since_save = 0

for idx, row in df.iterrows():

    # --------------------------------------------------------
    # Skip already completed rows
    # --------------------------------------------------------

    if (
        str(
            row.get(
                "reference_status",
                ""
            )
        ).strip()
        == "success"
    ):
        continue


    fault = str(
        row[TEXT_COL]
    ).strip()


    if not fault:

        df.at[
            idx,
            "reference_status"
        ] = "empty_fault"

        continue


    print(
        f"\n[{idx + 1}/{len(df)}]"
    )

    if "synthetic_id" in df.columns:
        print(
            "ID:",
            row["synthetic_id"]
        )

    print(
        "Fault:",
        fault[:160]
        + (
            "..."
            if len(fault) > 160
            else ""
        )
    )


    # --------------------------------------------------------
    # GPT CALL
    # --------------------------------------------------------

    resolution, latency, status = (
        generate_resolution(fault)
    )


    df.at[
        idx,
        "reference_resolution"
    ] = resolution

    df.at[
        idx,
        "reference_status"
    ] = status

    df.at[
        idx,
        "reference_latency_sec"
    ] = latency


    if status == "success":

        successful_since_save += 1

        print(
            "Resolution:",
            resolution
        )

        print(
            f"Latency: {latency:.3f} sec"
        )

    else:

        print(
            "FAILED after retries."
        )


    # --------------------------------------------------------
    # PERIODIC SAVE
    # --------------------------------------------------------

    if successful_since_save >= SAVE_EVERY:

        df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "\nCheckpoint saved."
        )

        successful_since_save = 0


    time.sleep(
        SLEEP_SECONDS
    )


# ============================================================
# FINAL SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

success_count = (
    df["reference_status"]
    .fillna("")
    .eq("success")
    .sum()
)

failed_count = (
    df["reference_status"]
    .fillna("")
    .eq("failed")
    .sum()
)

latencies = pd.to_numeric(
    df["reference_latency_sec"],
    errors="coerce"
)


print("\n" + "=" * 70)
print("REFERENCE GENERATION COMPLETE")
print("=" * 70)

print(
    "Total records :",
    len(df)
)

print(
    "Successful    :",
    success_count
)

print(
    "Failed        :",
    failed_count
)

print(
    "Mean latency  :",
    round(
        latencies.mean(),
        4
    ),
    "sec"
)

print(
    "P95 latency   :",
    round(
        latencies.quantile(0.95),
        4
    ),
    "sec"
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)