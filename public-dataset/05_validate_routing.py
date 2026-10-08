import os
import time
import json
import joblib

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

GMM_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "gmm"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "routing_validation"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Saved embeddings
# ------------------------------------------------------------

TRAIN_RAW_FILE = os.path.join(
    EMBEDDING_DIR,
    "train_fault_embeddings_raw.npy"
)

TRAIN_NORM_FILE = os.path.join(
    EMBEDDING_DIR,
    "train_fault_embeddings_normalized.npy"
)

VAL_RAW_FILE = os.path.join(
    EMBEDDING_DIR,
    "val_fault_embeddings_raw.npy"
)

VAL_NORM_FILE = os.path.join(
    EMBEDDING_DIR,
    "val_fault_embeddings_normalized.npy"
)


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

TRAIN_METADATA_FILE = os.path.join(
    EMBEDDING_DIR,
    "train_embedding_metadata.xlsx"
)

VAL_METADATA_FILE = os.path.join(
    EMBEDDING_DIR,
    "val_embedding_metadata.xlsx"
)


# ------------------------------------------------------------
# Saved GMM
# ------------------------------------------------------------

GMM_FILE = os.path.join(
    GMM_MODEL_DIR,
    "best_gmm.joblib"
)


# ============================================================
# VALIDATION SETTINGS
# ============================================================

# Global nearest neighbours used as our reference
# "relevant" records.
REFERENCE_K_VALUES = [
    1,
    5,
    10
]


# Threshold values to test for soft GMM routing.
#
# IMPORTANT:
# If no cluster probability >= threshold,
# the query ABSTAINS.
THRESHOLDS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


# We also test fixed top-m GMM clusters.
TOP_M_VALUES = [
    1,
    2,
    3,
    5
]


# Main metric for automatic recommendation
PRIMARY_REFERENCE_K = 10

# Desired routing recall.
# Among configurations meeting this,
# choose the one searching the smallest KB fraction.
TARGET_ROUTING_RECALL = 0.95


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 80)
print("LOADING SAVED DATA")
print("=" * 80)


train_raw = np.load(
    TRAIN_RAW_FILE
)

train_norm = np.load(
    TRAIN_NORM_FILE
)

val_raw = np.load(
    VAL_RAW_FILE
)

val_norm = np.load(
    VAL_NORM_FILE
)


train_metadata = pd.read_excel(
    TRAIN_METADATA_FILE
)

val_metadata = pd.read_excel(
    VAL_METADATA_FILE
)


gmm = joblib.load(
    GMM_FILE
)


print(
    "Training raw embeddings       :",
    train_raw.shape
)

print(
    "Training normalized embeddings:",
    train_norm.shape
)

print(
    "Validation raw embeddings     :",
    val_raw.shape
)

print(
    "Validation normalized         :",
    val_norm.shape
)

print(
    "GMM components               :",
    gmm.n_components
)


# ============================================================
# SANITY CHECKS
# ============================================================

assert len(train_raw) == len(train_metadata), \
    "Training embedding/metadata mismatch"

assert len(val_raw) == len(val_metadata), \
    "Validation embedding/metadata mismatch"

assert train_raw.shape[1] == val_raw.shape[1], \
    "Train/validation embedding dimensions differ"

assert train_norm.shape == train_raw.shape
assert val_norm.shape == val_raw.shape


N_TRAIN = len(train_raw)
N_VAL = len(val_raw)
N_CLUSTERS = gmm.n_components


# ============================================================
# GET TRAINING CLUSTER ASSIGNMENTS
#
# These are the hard cluster labels for all 3049
# knowledge-base records.
# ============================================================

print("\n" + "=" * 80)
print("ASSIGNING TRAINING KB RECORDS TO GMM CLUSTERS")
print("=" * 80)


train_cluster_labels = gmm.predict(
    train_raw
)


# Build lookup:
#
# cluster_id -> array of training record indices

cluster_to_indices = {}

for cluster_id in range(
    N_CLUSTERS
):

    cluster_to_indices[
        cluster_id
    ] = np.where(
        train_cluster_labels
        ==
        cluster_id
    )[0]


cluster_sizes = {
    cluster_id:
        len(indices)

    for cluster_id, indices
    in cluster_to_indices.items()
}


print(
    "Smallest cluster:",
    min(cluster_sizes.values())
)

print(
    "Largest cluster :",
    max(cluster_sizes.values())
)

print(
    "Mean cluster size:",
    np.mean(
        list(
            cluster_sizes.values()
        )
    )
)


# ============================================================
# GMM POSTERIORS FOR VALIDATION FAULTS
#
# Shape:
#
# 436 x 55
#
# Each row:
#
# P(C1 | q), ..., P(C55 | q)
# ============================================================

print("\n" + "=" * 80)
print("COMPUTING VALIDATION GMM POSTERIORS")
print("=" * 80)


posterior_start = time.perf_counter()


val_posteriors = gmm.predict_proba(
    val_raw
)


posterior_time = (
    time.perf_counter()
    -
    posterior_start
)


print(
    "Posterior matrix shape:",
    val_posteriors.shape
)

print(
    "Total posterior computation time:",
    f"{posterior_time:.6f} sec"
)

print(
    "Average per-query GMM time:",
    f"{posterior_time / N_VAL * 1000:.6f} ms"
)


# ============================================================
# GLOBAL COSINE RETRIEVAL
#
# This is our reference retrieval.
#
# Validation query is compared against ALL 3049
# training faults.
#
# Since all vectors are normalized:
#
# cosine(q, x) = q dot x
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL RETRIEVAL REFERENCE")
print("=" * 80)


global_start = time.perf_counter()


global_similarity = np.matmul(
    val_norm,
    train_norm.T
)


global_time = (
    time.perf_counter()
    -
    global_start
)


MAX_REFERENCE_K = max(
    REFERENCE_K_VALUES
)


# argsort descending
global_rankings = np.argsort(
    -global_similarity,
    axis=1
)


global_top_indices = (
    global_rankings[
        :,
        :MAX_REFERENCE_K
    ]
)


print(
    "Similarity matrix shape:",
    global_similarity.shape
)

print(
    "Global retrieval total time:",
    f"{global_time:.6f} sec"
)

print(
    "Global retrieval avg/query:",
    f"{global_time / N_VAL * 1000:.6f} ms"
)


# ============================================================
# HELPER:
# GET ALL TRAINING RECORDS IN SELECTED CLUSTERS
# ============================================================

def candidate_indices_from_clusters(
    selected_clusters
):

    if len(selected_clusters) == 0:

        return np.array(
            [],
            dtype=int
        )


    arrays = [
        cluster_to_indices[
            int(cluster_id)
        ]

        for cluster_id
        in selected_clusters
    ]


    if len(arrays) == 0:

        return np.array(
            [],
            dtype=int
        )


    return np.unique(
        np.concatenate(
            arrays
        )
    )


# ============================================================
# HELPER:
# METRICS FOR ONE QUERY
# ============================================================

def evaluate_one_query(
    query_index,
    selected_clusters,
    reference_k
):

    # --------------------------------------------------------
    # Reference relevant records:
    #
    # globally retrieved top-K training faults
    # --------------------------------------------------------

    reference_indices = (
        global_rankings[
            query_index,
            :reference_k
        ]
    )


    reference_set = set(
        reference_indices.tolist()
    )


    # --------------------------------------------------------
    # Candidate set after GMM routing
    # --------------------------------------------------------

    candidates = (
        candidate_indices_from_clusters(
            selected_clusters
        )
    )


    candidate_set = set(
        candidates.tolist()
    )


    # --------------------------------------------------------
    # Abstention
    # --------------------------------------------------------

    abstained = (
        len(candidates) == 0
    )


    if abstained:

        return {
            "cluster_hit":
                0.0,

            "routing_recall":
                0.0,

            "retrieval_overlap":
                0.0,

            "candidate_count":
                0,

            "candidate_fraction":
                0.0,

            "candidate_reduction":
                1.0,

            "abstained":
                1
        }


    # --------------------------------------------------------
    # CLUSTER HIT RATE
    #
    # Does routed candidate set contain
    # AT LEAST ONE globally relevant item?
    # --------------------------------------------------------

    intersection = (
        reference_set
        &
        candidate_set
    )


    cluster_hit = float(
        len(intersection) > 0
    )


    # --------------------------------------------------------
    # ROUTING RECALL@K
    #
    # What fraction of global Top-K survived
    # GMM filtering?
    # --------------------------------------------------------

    routing_recall = (
        len(intersection)
        /
        reference_k
    )


    # --------------------------------------------------------
    # RETRIEVE TOP-K WITHIN ROUTED CANDIDATES
    #
    # Rank only the selected candidate records
    # by cosine similarity.
    # --------------------------------------------------------

    candidate_scores = (
        global_similarity[
            query_index,
            candidates
        ]
    )


    local_order = np.argsort(
        -candidate_scores
    )


    routed_top = (
        candidates[
            local_order[
                :min(
                    reference_k,
                    len(candidates)
                )
            ]
        ]
    )


    routed_top_set = set(
        routed_top.tolist()
    )


    # --------------------------------------------------------
    # RETRIEVAL OVERLAP@K
    #
    # How much of global Top-K is also in
    # routed Top-K?
    # --------------------------------------------------------

    retrieval_overlap = (
        len(
            reference_set
            &
            routed_top_set
        )
        /
        reference_k
    )


    candidate_count = len(
        candidates
    )


    candidate_fraction = (
        candidate_count
        /
        N_TRAIN
    )


    candidate_reduction = (
        1.0
        -
        candidate_fraction
    )


    return {
        "cluster_hit":
            cluster_hit,

        "routing_recall":
            routing_recall,

        "retrieval_overlap":
            retrieval_overlap,

        "candidate_count":
            candidate_count,

        "candidate_fraction":
            candidate_fraction,

        "candidate_reduction":
            candidate_reduction,

        "abstained":
            0
    }


# ============================================================
# ROUTING STRATEGIES
# ============================================================

def select_top_m_clusters(
    probabilities,
    m
):

    order = np.argsort(
        -probabilities
    )

    return order[:m]


def select_threshold_clusters(
    probabilities,
    threshold
):

    selected = np.where(
        probabilities
        >=
        threshold
    )[0]

    # IMPORTANT:
    #
    # No fallback to argmax.
    #
    # If none pass threshold,
    # we intentionally return empty
    # and record an abstention.

    return selected


# ============================================================
# RUN VALIDATION EXPERIMENTS
# ============================================================

summary_rows = []
detail_rows = []


print("\n" + "=" * 80)
print("VALIDATING GMM ROUTING")
print("=" * 80)


# ============================================================
# PART A:
# FIXED TOP-M CLUSTERS
# ============================================================

for m in TOP_M_VALUES:

    strategy_name = (
        f"top_{m}_clusters"
    )

    print(
        f"\nEvaluating {strategy_name}"
    )


    for reference_k in REFERENCE_K_VALUES:

        query_results = []


        start = time.perf_counter()


        for q in range(
            N_VAL
        ):

            probs = (
                val_posteriors[q]
            )


            selected_clusters = (
                select_top_m_clusters(
                    probs,
                    m
                )
            )


            metrics = (
                evaluate_one_query(
                    q,
                    selected_clusters,
                    reference_k
                )
            )


            query_results.append(
                metrics
            )


            # ------------------------------------------------
            # Store detailed result
            # ------------------------------------------------

            detail_rows.append({

                "strategy":
                    strategy_name,

                "reference_k":
                    reference_k,

                "query_index":
                    q,

                "query_id":
                    val_metadata
                    .iloc[q]["id"],

                "source_dataset":
                    val_metadata
                    .iloc[q][
                        "source_dataset"
                    ],

                "selected_clusters":
                    ",".join(
                        map(
                            str,
                            selected_clusters
                        )
                    ),

                "num_selected_clusters":
                    len(
                        selected_clusters
                    ),

                "max_gmm_probability":
                    float(
                        probs.max()
                    ),

                "posterior_entropy":
                    float(
                        -np.sum(
                            probs
                            *
                            np.log(
                                probs
                                +
                                1e-12
                            )
                        )
                    ),

                **metrics
            })


        elapsed = (
            time.perf_counter()
            -
            start
        )


        result_df = pd.DataFrame(
            query_results
        )


        summary = {

            "strategy":
                strategy_name,

            "reference_k":
                reference_k,

            "cluster_hit_rate":
                result_df[
                    "cluster_hit"
                ].mean(),

            "routing_recall":
                result_df[
                    "routing_recall"
                ].mean(),

            "retrieval_overlap":
                result_df[
                    "retrieval_overlap"
                ].mean(),

            "avg_candidate_count":
                result_df[
                    "candidate_count"
                ].mean(),

            "avg_candidate_fraction":
                result_df[
                    "candidate_fraction"
                ].mean(),

            "avg_candidate_reduction":
                result_df[
                    "candidate_reduction"
                ].mean(),

            "avg_selected_clusters":
                float(m),

            "abstention_rate":
                result_df[
                    "abstained"
                ].mean(),

            "routing_eval_time_seconds":
                elapsed,

            "avg_routing_eval_ms":
                (
                    elapsed
                    /
                    N_VAL
                    *
                    1000
                )
        }


        summary_rows.append(
            summary
        )


        print(
            f"  Reference@{reference_k}"
        )

        print(
            f"    Cluster Hit Rate : "
            f"{summary['cluster_hit_rate']:.4f}"
        )

        print(
            f"    Routing Recall   : "
            f"{summary['routing_recall']:.4f}"
        )

        print(
            f"    Retrieval overlap: "
            f"{summary['retrieval_overlap']:.4f}"
        )

        print(
            f"    Candidate reduction: "
            f"{summary['avg_candidate_reduction']:.4f}"
        )


# ============================================================
# PART B:
# THRESHOLD-BASED ROUTING
# ============================================================

for threshold in THRESHOLDS:

    strategy_name = (
        f"threshold_{threshold:.2f}"
    )


    print(
        f"\nEvaluating {strategy_name}"
    )


    for reference_k in REFERENCE_K_VALUES:

        query_results = []


        start = time.perf_counter()


        selected_cluster_counts = []


        for q in range(
            N_VAL
        ):

            probs = (
                val_posteriors[q]
            )


            selected_clusters = (
                select_threshold_clusters(
                    probs,
                    threshold
                )
            )


            selected_cluster_counts.append(
                len(
                    selected_clusters
                )
            )


            metrics = (
                evaluate_one_query(
                    q,
                    selected_clusters,
                    reference_k
                )
            )


            query_results.append(
                metrics
            )


            detail_rows.append({

                "strategy":
                    strategy_name,

                "reference_k":
                    reference_k,

                "query_index":
                    q,

                "query_id":
                    val_metadata
                    .iloc[q]["id"],

                "source_dataset":
                    val_metadata
                    .iloc[q][
                        "source_dataset"
                    ],

                "selected_clusters":
                    ",".join(
                        map(
                            str,
                            selected_clusters
                        )
                    ),

                "num_selected_clusters":
                    len(
                        selected_clusters
                    ),

                "max_gmm_probability":
                    float(
                        probs.max()
                    ),

                "posterior_entropy":
                    float(
                        -np.sum(
                            probs
                            *
                            np.log(
                                probs
                                +
                                1e-12
                            )
                        )
                    ),

                **metrics
            })


        elapsed = (
            time.perf_counter()
            -
            start
        )


        result_df = pd.DataFrame(
            query_results
        )


        summary = {

            "strategy":
                strategy_name,

            "threshold":
                threshold,

            "reference_k":
                reference_k,

            "cluster_hit_rate":
                result_df[
                    "cluster_hit"
                ].mean(),

            "routing_recall":
                result_df[
                    "routing_recall"
                ].mean(),

            "retrieval_overlap":
                result_df[
                    "retrieval_overlap"
                ].mean(),

            "avg_candidate_count":
                result_df[
                    "candidate_count"
                ].mean(),

            "avg_candidate_fraction":
                result_df[
                    "candidate_fraction"
                ].mean(),

            "avg_candidate_reduction":
                result_df[
                    "candidate_reduction"
                ].mean(),

            "avg_selected_clusters":
                float(
                    np.mean(
                        selected_cluster_counts
                    )
                ),

            "abstention_rate":
                result_df[
                    "abstained"
                ].mean(),

            "routing_eval_time_seconds":
                elapsed,

            "avg_routing_eval_ms":
                (
                    elapsed
                    /
                    N_VAL
                    *
                    1000
                )
        }


        summary_rows.append(
            summary
        )


        print(
            f"  Reference@{reference_k}"
        )

        print(
            f"    Cluster Hit Rate   : "
            f"{summary['cluster_hit_rate']:.4f}"
        )

        print(
            f"    Routing Recall     : "
            f"{summary['routing_recall']:.4f}"
        )

        print(
            f"    Retrieval overlap  : "
            f"{summary['retrieval_overlap']:.4f}"
        )

        print(
            f"    Candidate reduction: "
            f"{summary['avg_candidate_reduction']:.4f}"
        )

        print(
            f"    Avg clusters       : "
            f"{summary['avg_selected_clusters']:.2f}"
        )

        print(
            f"    Abstention rate    : "
            f"{summary['abstention_rate']:.4f}"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "routing_validation_summary.csv"
    ),
    index=False
)


summary_df.to_excel(
    os.path.join(
        RESULT_DIR,
        "routing_validation_summary.xlsx"
    ),
    index=False
)


# ============================================================
# SAVE QUERY-LEVEL DETAILS
# ============================================================

detail_df = pd.DataFrame(
    detail_rows
)


detail_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "routing_validation_details.csv"
    ),
    index=False
)


# ============================================================
# SELECT RECOMMENDED CONFIGURATION
#
# Primary:
# routing recall >= 0.95
#
# Among those:
# maximize candidate reduction.
#
# We use Reference@10.
# ============================================================

selection_df = summary_df[
    summary_df[
        "reference_k"
    ]
    ==
    PRIMARY_REFERENCE_K
].copy()


eligible = selection_df[
    selection_df[
        "routing_recall"
    ]
    >=
    TARGET_ROUTING_RECALL
].copy()


if len(eligible) > 0:

    recommended = (
        eligible
        .sort_values(
            [
                "avg_candidate_reduction",
                "routing_recall"
            ],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )

    selection_reason = (
        f"Met routing recall >= "
        f"{TARGET_ROUTING_RECALL:.2f}; "
        f"selected maximum candidate reduction."
    )


else:

    # --------------------------------------------------------
    # If nothing reaches target recall,
    # choose maximum routing recall;
    # candidate reduction breaks ties.
    # --------------------------------------------------------

    recommended = (
        selection_df
        .sort_values(
            [
                "routing_recall",
                "avg_candidate_reduction"
            ],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )

    selection_reason = (
        "No configuration met target routing recall; "
        "selected highest routing recall."
    )


# ============================================================
# SAVE RECOMMENDATION
# ============================================================

recommended_dict = {}


for key, value in (
    recommended.to_dict().items()
):

    if pd.isna(value):

        recommended_dict[key] = None

    elif isinstance(
        value,
        np.generic
    ):

        recommended_dict[key] = (
            value.item()
        )

    else:

        recommended_dict[key] = value


recommended_dict[
    "selection_reason"
] = selection_reason


with open(
    os.path.join(
        RESULT_DIR,
        "recommended_routing_config.json"
    ),
    "w"
) as f:

    json.dump(
        recommended_dict,
        f,
        indent=4
    )


# ============================================================
# PRINT BEST CONFIGURATIONS
# ============================================================

print("\n" + "=" * 80)
print("ROUTING VALIDATION COMPLETE")
print("=" * 80)


print(
    "\nReference Top-K:",
    PRIMARY_REFERENCE_K
)


print(
    "\nRecommended strategy:"
)

print(
    recommended[
        "strategy"
    ]
)


print(
    "Cluster Hit Rate:",
    f"{recommended['cluster_hit_rate']:.4f}"
)

print(
    "Routing Recall:",
    f"{recommended['routing_recall']:.4f}"
)

print(
    "Retrieval overlap:",
    f"{recommended['retrieval_overlap']:.4f}"
)

print(
    "Average candidate count:",
    f"{recommended['avg_candidate_count']:.2f}"
)

print(
    "Candidate reduction:",
    f"{recommended['avg_candidate_reduction']:.4f}"
)

print(
    "Average selected clusters:",
    f"{recommended['avg_selected_clusters']:.2f}"
)

print(
    "Abstention rate:",
    f"{recommended['abstention_rate']:.4f}"
)


print(
    "\nSelection reason:"
)

print(
    selection_reason
)


print(
    "\nResults saved to:"
)

print(
    RESULT_DIR
)