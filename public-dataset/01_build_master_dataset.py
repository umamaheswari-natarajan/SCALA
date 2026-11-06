from pathlib import Path
import os
import re
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Existing standardized 1245 records
COMBINED_1245_FILE = os.path.join(
    BASE_DIR,
    "combined_raw_1245.xlsx"
)

# 3163-row 5G dataset
FIVEG_FILE = os.path.join(
    BASE_DIR,
    "5G_Faults_Full_v2.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset_quality_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


RAW_MASTER_FILE = os.path.join(
    OUTPUT_DIR,
    "master_4408_raw.xlsx"
)

DUPLICATE_REPORT = os.path.join(
    OUTPUT_DIR,
    "master_4408_duplicate_report.xlsx"
)

FINAL_FILE = os.path.join(
    OUTPUT_DIR,
    "master_fault_resolution_final.xlsx"
)


# ============================================================
# HELPER: READ XLS / XLSX / CSV
# ============================================================

def read_file(path):

    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    elif ext == ".csv":
        return pd.read_csv(path)

    else:
        raise ValueError(
            f"Unsupported file type: {ext}"
        )


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# NORMALIZATION FOR DUPLICATE DETECTION
# ============================================================

def normalize_text(text):
    """
    Remove superficial differences only.

    Examples handled:
        commas vs semicolons
        different capitalization
        extra spaces
        punctuation differences
        'Cause:' formatting
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Cause: is treated as a separator
    text = re.sub(
        r"\bcause\s*:",
        " ",
        text
    )

    # replace separators with spaces
    text = re.sub(
        r"[;,|:/\\()\[\]{}]",
        " ",
        text
    )

    # remove remaining punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # collapse spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def token_fingerprint(text):
    """
    Allows the same content with slight ordering /
    punctuation differences to map to the same key.
    """

    normalized = normalize_text(text)

    tokens = normalized.split()

    return " ".join(
        sorted(tokens)
    )


# ============================================================
# 1. LOAD EXISTING 1245 DATASET
# ============================================================

print("\n" + "=" * 80)
print("LOADING EXISTING 1245 DATASET")
print("=" * 80)

df1 = read_file(
    COMBINED_1245_FILE
)

print("Shape:", df1.shape)
print("Columns:", df1.columns.tolist())


# Standardize
df1_std = pd.DataFrame()

df1_std["id"] = (
    df1["id"]
    .astype(str)
)

df1_std["fault_text"] = (
    df1["fault_text"]
    .apply(clean_text)
)

df1_std["root_cause"] = (
    df1["root_cause"]
    .apply(clean_text)
)

df1_std["resolution_text"] = (
    df1["resolution_text"]
    .apply(clean_text)
)

df1_std["source_dataset"] = (
    df1["source_dataset"]
    .astype(str)
)


print(
    "Loaded standardized records:",
    len(df1_std)
)


# ============================================================
# 2. LOAD 5G FULL DATASET
# ============================================================

print("\n" + "=" * 80)
print("LOADING 5G FAULTS FULL V2")
print("=" * 80)

df2 = read_file(
    FIVEG_FILE
)

print("Shape:", df2.shape)
print("Columns:", df2.columns.tolist())


df2_std = pd.DataFrame()

df2_std["id"] = [
    f"5GFULL_{i+1:05d}"
    for i in range(len(df2))
]

df2_std["fault_text"] = (
    df2["input"]
    .apply(clean_text)
)

# No explicit root cause field
df2_std["root_cause"] = ""

df2_std["resolution_text"] = (
    df2["output"]
    .apply(clean_text)
)

df2_std["source_dataset"] = (
    "5G_Faults_Full_v2"
)


print(
    "Loaded standardized records:",
    len(df2_std)
)


# ============================================================
# 3. COMBINE
# ============================================================

master = pd.concat(
    [
        df1_std,
        df2_std
    ],
    ignore_index=True
)


print("\n" + "=" * 80)
print("RAW MASTER DATASET")
print("=" * 80)

print(
    "Total raw records:",
    len(master)
)

print(
    "Source distribution:"
)

print(
    master[
        "source_dataset"
    ].value_counts()
)


# ============================================================
# 4. REMOVE EMPTY FAULT / RESOLUTION ROWS
# ============================================================

master = master[
    (
        master["fault_text"]
        .str.len() > 0
    )
    &
    (
        master["resolution_text"]
        .str.len() > 0
    )
].copy()


# Save original 4408-ish master
master.to_excel(
    RAW_MASTER_FILE,
    index=False
)


# ============================================================
# 5. NORMALIZED REPRESENTATIONS
# ============================================================

master["fault_norm"] = (
    master["fault_text"]
    .apply(normalize_text)
)

master["resolution_norm"] = (
    master["resolution_text"]
    .apply(normalize_text)
)

master["fault_fingerprint"] = (
    master["fault_text"]
    .apply(token_fingerprint)
)

master["resolution_fingerprint"] = (
    master["resolution_text"]
    .apply(token_fingerprint)
)


# ============================================================
# 6. DUPLICATES AFTER NORMALIZATION
# ============================================================

norm_duplicate_mask = (
    master.duplicated(
        subset=[
            "fault_norm",
            "resolution_norm"
        ],
        keep=False
    )
)


print(
    "\nRows involved in normalized duplicate groups:",
    norm_duplicate_mask.sum()
)


# ============================================================
# 7. DUPLICATES USING TOKEN FINGERPRINT
# ============================================================

finger_duplicate_mask = (
    master.duplicated(
        subset=[
            "fault_fingerprint",
            "resolution_fingerprint"
        ],
        keep=False
    )
)


print(
    "Rows involved in fingerprint duplicate groups:",
    finger_duplicate_mask.sum()
)


# ============================================================
# 8. CREATE DUPLICATE REPORT
# ============================================================

duplicate_mask = (
    norm_duplicate_mask
    |
    finger_duplicate_mask
)


duplicate_rows = (
    master[
        duplicate_mask
    ].copy()
)


if len(duplicate_rows) > 0:

    duplicate_rows[
        "duplicate_key"
    ] = (
        duplicate_rows[
            "fault_fingerprint"
        ]
        +
        "|||"
        +
        duplicate_rows[
            "resolution_fingerprint"
        ]
    )

    duplicate_rows[
        "duplicate_group"
    ] = (
        duplicate_rows
        .groupby(
            "duplicate_key"
        )
        .ngroup()
        + 1
    )

    duplicate_rows = (
        duplicate_rows
        .sort_values(
            [
                "duplicate_group",
                "source_dataset",
                "id"
            ]
        )
    )


report_columns = [
    "duplicate_group",
    "id",
    "fault_text",
    "root_cause",
    "resolution_text",
    "source_dataset"
]


if len(duplicate_rows) > 0:

    duplicate_rows[
        report_columns
    ].to_excel(
        DUPLICATE_REPORT,
        index=False
    )

else:

    pd.DataFrame(
        columns=report_columns
    ).to_excel(
        DUPLICATE_REPORT,
        index=False
    )


# ============================================================
# 9. REMOVE ONLY FULL PAIR DUPLICATES
# ============================================================

before = len(master)


final_df = (
    master
    .drop_duplicates(
        subset=[
            "fault_fingerprint",
            "resolution_fingerprint"
        ],
        keep="first"
    )
    .copy()
)


after = len(final_df)

removed = (
    before - after
)


# ============================================================
# 10. REMOVE HELPER COLUMNS
# ============================================================

helper_columns = [
    "fault_norm",
    "resolution_norm",
    "fault_fingerprint",
    "resolution_fingerprint"
]


final_df = (
    final_df
    .drop(
        columns=helper_columns
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 11. SAVE FINAL MASTER
# ============================================================

final_df.to_excel(
    FINAL_FILE,
    index=False
)


# ============================================================
# 12. FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET SUMMARY")
print("=" * 80)

print(
    "Raw combined rows              :",
    before
)

print(
    "Rows in duplicate groups       :",
    len(duplicate_rows)
)

print(
    "Duplicate copies removed       :",
    removed
)

print(
    "Final usable records           :",
    after
)

print(
    "Unique fault texts             :",
    final_df[
        "fault_text"
    ].nunique()
)

print(
    "Unique resolutions             :",
    final_df[
        "resolution_text"
    ].nunique()
)

print(
    "Unique fault-resolution pairs  :",
    final_df[
        [
            "fault_text",
            "resolution_text"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


print(
    "\nSource distribution:"
)

print(
    final_df[
        "source_dataset"
    ].value_counts()
)


print(
    "\nRaw master saved to:"
)

print(
    RAW_MASTER_FILE
)

print(
    "\nDuplicate report saved to:"
)

print(
    DUPLICATE_REPORT
)

print(
    "\nFINAL dataset saved to:"
)

print(
    FINAL_FILE
)