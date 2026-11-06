from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = os.path.join(
    BASE_DIR,
    "master_fault_resolution_final.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset_splits"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "train_fault_resolution.xlsx"
)

VAL_FILE = os.path.join(
    OUTPUT_DIR,
    "val_fault_resolution.xlsx"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "test_fault_resolution.xlsx"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "split_summary.xlsx"
)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_excel(INPUT_FILE)

print("=" * 80)
print("MASTER DATASET")
print("=" * 80)

print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

print("\nSource distribution:")
print(df["source_dataset"].value_counts())


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "id",
    "fault_text",
    "root_cause",
    "resolution_text",
    "source_dataset"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# REMOVE ACCIDENTAL EMPTY FAULT / RESOLUTION ROWS
# ============================================================

df["fault_text"] = df["fault_text"].fillna("").astype(str)
df["resolution_text"] = df["resolution_text"].fillna("").astype(str)
df["root_cause"] = df["root_cause"].fillna("").astype(str)

df = df[
    (df["fault_text"].str.strip() != "")
    &
    (df["resolution_text"].str.strip() != "")
].copy()

df = df.reset_index(drop=True)

print("\nRows after empty-row check:", len(df))


# ============================================================
# STEP 1:
# SPLIT 80% DEVELOPMENT / 20% TEST
# ============================================================

development_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["source_dataset"]
)


# ============================================================
# STEP 2:
# SPLIT DEVELOPMENT INTO:
#
# 87.5% TRAIN
# 12.5% VALIDATION
#
# because:
#
# 80% * 87.5% = 70%
# 80% * 12.5% = 10%
# ============================================================

train_df, val_df = train_test_split(
    development_df,
    test_size=0.125,
    random_state=RANDOM_STATE,
    stratify=development_df["source_dataset"]
)


# ============================================================
# RESET INDICES
# ============================================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# ADD SPLIT COLUMN
# ============================================================

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"


# ============================================================
# SAVE FILES
# ============================================================

train_df.to_excel(
    TRAIN_FILE,
    index=False
)

val_df.to_excel(
    VAL_FILE,
    index=False
)

test_df.to_excel(
    TEST_FILE,
    index=False
)


# ============================================================
# BUILD SUMMARY
# ============================================================

summary_rows = []

for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:

    source_counts = (
        split_df["source_dataset"]
        .value_counts()
        .to_dict()
    )

    row = {
        "split": split_name,
        "total_rows": len(split_df),
        "unique_fault_texts":
            split_df["fault_text"].nunique(),
        "unique_resolutions":
            split_df["resolution_text"].nunique()
    }

    for source, count in source_counts.items():
        row[source] = count

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_df.to_excel(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SPLIT SUMMARY")
print("=" * 80)

print("\nTRAIN")
print("Rows:", len(train_df))
print(train_df["source_dataset"].value_counts())

print("\nVALIDATION")
print("Rows:", len(val_df))
print(val_df["source_dataset"].value_counts())

print("\nTEST")
print("Rows:", len(test_df))
print(test_df["source_dataset"].value_counts())


# ============================================================
# CHECK TOTAL
# ============================================================

total_after_split = (
    len(train_df)
    +
    len(val_df)
    +
    len(test_df)
)

print("\nTotal after split:", total_after_split)

assert total_after_split == len(df), \
    "Split row count does not match original dataset!"


# ============================================================
# CHECK FOR ID OVERLAP
# ============================================================

train_ids = set(train_df["id"])
val_ids = set(val_df["id"])
test_ids = set(test_df["id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("ID overlap check: PASSED")


# ============================================================
# FINAL OUTPUT PATHS
# ============================================================

print("\nFiles saved:")

print("Train:")
print(TRAIN_FILE)

print("\nValidation:")
print(VAL_FILE)

print("\nTest:")
print(TEST_FILE)

print("\nSummary:")
print(SUMMARY_FILE)