import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

GMM_DIR = "synthetic_gmm_selection_fine"

K = 120

GMM_FILE = os.path.join(
    GMM_DIR,
    f"gmm_K{K}.joblib"
)

TRAIN_RAW_FILE = os.path.join(
    GMM_DIR,
    "train_fault_embeddings_raw.npy"
)

TRAIN_NORM_FILE = os.path.join(
    GMM_DIR,
    "train_fault_embeddings_normalized.npy"
)

VAL_RAW_FILE = os.path.join(
    GMM_DIR,
    "validation_fault_embeddings_raw.npy"
)

VAL_NORM_FILE = os.path.join(
    GMM_DIR,
    "validation_fault_embeddings_normalized.npy"
)

OUTPUT_DIR = "synthetic_routing_validation_timed"

ROUTING_OPTIONS = [1, 3, 5]

GLOBAL_TOP_K = 5

# Number of warm-up queries before timing
WARMUP_QUERIES = 50

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 85)
print("STAGE 3: ROUTING VALIDATION + TIMING")
print("=" * 85)

train_raw = np.load(TRAIN_RAW_FILE)
train_norm = np.load(TRAIN_NORM_FILE)

val_raw = np.load(VAL_RAW_FILE)
val_norm = np.load(VAL_NORM_FILE)

N_TRAIN = len(train_raw)
N_VAL = len(val_raw)

print("\nShapes")
print("Train raw       :", train_raw.shape)
print("Train normalized:", train_norm.shape)
print("Validation raw  :", val_raw.shape)
print("Validation norm :", val_norm.shape)


# ============================================================
# 3. LOAD GMM
# ============================================================

print("\nLoading GMM...")

gmm = joblib.load(
    GMM_FILE
)

print(
    "GMM components:",
    gmm.n_components
)


# ============================================================
# 4. TRAIN RECORD -> CLUSTER ASSIGNMENT
# ============================================================

train_cluster_labels = gmm.predict(
    train_raw
)

cluster_to_indices = {}

for cluster_id in range(K):

    cluster_to_indices[cluster_id] = np.where(
        train_cluster_labels == cluster_id
    )[0]


cluster_sizes = np.array([
    len(cluster_to_indices[c])
    for c in range(K)
])

print("\nCluster size statistics")
print("Min   :", cluster_sizes.min())
print("Mean  :", cluster_sizes.mean())
print("Median:", np.median(cluster_sizes))
print("Max   :", cluster_sizes.max())


# ============================================================
# 5. GLOBAL REFERENCE RETRIEVAL
#
# This is NOT used for timing.
#
# We compute this once in batch to determine the true
# unrestricted Top-1 and Top-5 neighbors.
# ============================================================

print("\nComputing unrestricted global Top-5 reference...")

global_similarity = (
    val_norm
    @
    train_norm.T
)

global_top5_unsorted = np.argpartition(
    -global_similarity,
    kth=GLOBAL_TOP_K - 1,
    axis=1
)[:, :GLOBAL_TOP_K]

global_top5 = np.zeros_like(
    global_top5_unsorted
)

for i in range(N_VAL):

    idx = global_top5_unsorted[i]

    scores = global_similarity[
        i,
        idx
    ]

    order = np.argsort(
        -scores
    )

    global_top5[i] = idx[order]


global_top1 = (
    global_top5[:, 0]
)

print("Global reference ready.")


# ============================================================
# 6. UTILITY FUNCTIONS
# ============================================================

def percentile_95(values):

    return float(
        np.percentile(
            values,
            95
        )
    )


def timing_summary(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean_ms":
            float(values.mean() * 1000.0),

        "median_ms":
            float(np.median(values) * 1000.0),

        "p95_ms":
            float(
                np.percentile(
                    values,
                    95
                )
                *
                1000.0
            )
    }


# ============================================================
# 7. STANDARD RAG RETRIEVAL TIMING
#
# IMPORTANT:
# One query at a time.
#
# This measures cosine retrieval over ALL 8400 records.
#
# Query encoding is excluded deliberately because it is
# common to both systems.
# ============================================================

print("\n" + "=" * 85)
print("TIMING STANDARD GLOBAL COSINE RETRIEVAL")
print("=" * 85)


# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------

warmup_count = min(
    WARMUP_QUERIES,
    N_VAL
)

for i in range(warmup_count):

    q = val_norm[i]

    similarities = (
        train_norm
        @
        q
    )

    top_idx = np.argpartition(
        -similarities,
        GLOBAL_TOP_K - 1
    )[:GLOBAL_TOP_K]


# ------------------------------------------------------------
# Actual timing
# ------------------------------------------------------------

standard_times = []

standard_top5_results = []

for i in range(N_VAL):

    q = val_norm[i]

    start = time.perf_counter()

    similarities = (
        train_norm
        @
        q
    )

    top5_unsorted = np.argpartition(
        -similarities,
        GLOBAL_TOP_K - 1
    )[:GLOBAL_TOP_K]

    scores = similarities[
        top5_unsorted
    ]

    order = np.argsort(
        -scores
    )

    top5_sorted = (
        top5_unsorted[
            order
        ]
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    standard_times.append(
        elapsed
    )

    standard_top5_results.append(
        top5_sorted
    )


standard_times = np.array(
    standard_times
)

standard_stats = timing_summary(
    standard_times
)


print("\nStandard retrieval timing")
print(
    f"Mean   : "
    f"{standard_stats['mean_ms']:.6f} ms"
)

print(
    f"Median : "
    f"{standard_stats['median_ms']:.6f} ms"
)

print(
    f"P95    : "
    f"{standard_stats['p95_ms']:.6f} ms"
)


# ============================================================
# 8. PROPOSED ROUTING EVALUATION
# ============================================================

summary_rows = []
query_level_rows = []


for top_n_clusters in ROUTING_OPTIONS:

    print("\n" + "=" * 85)

    print(
        f"TOP-{top_n_clusters} CLUSTER ROUTING"
    )

    print("=" * 85)


    top1_preserved_count = 0
    at_least_one_top5_count = 0
    all_top5_preserved_count = 0

    total_top5_fraction = 0.0

    candidate_sizes = []

    gmm_times = []
    candidate_build_times = []
    routed_cosine_times = []
    proposed_total_times = []


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for i in range(warmup_count):

        q_raw = val_raw[i]
        q_norm = val_norm[i]


        probs = gmm.predict_proba(
            q_raw.reshape(
                1,
                -1
            )
        )[0]


        routed_clusters = np.argsort(
            -probs
        )[
            :top_n_clusters
        ]


        candidate_parts = [
            cluster_to_indices[c]
            for c in routed_clusters
        ]


        candidates = np.concatenate(
            candidate_parts
        )


        candidate_vectors = (
            train_norm[
                candidates
            ]
        )


        sims = (
            candidate_vectors
            @
            q_norm
        )


        k_local = min(
            GLOBAL_TOP_K,
            len(candidates)
        )


        _ = np.argpartition(
            -sims,
            k_local - 1
        )[
            :k_local
        ]


    # --------------------------------------------------------
    # Actual query-by-query evaluation
    # --------------------------------------------------------

    for i in range(N_VAL):

        q_raw = val_raw[i]
        q_norm = val_norm[i]


        # ====================================================
        # A. GMM ROUTING TIME
        # ====================================================

        gmm_start = time.perf_counter()


        probs = gmm.predict_proba(
            q_raw.reshape(
                1,
                -1
            )
        )[0]


        routed_clusters = np.argsort(
            -probs
        )[
            :top_n_clusters
        ]


        gmm_elapsed = (
            time.perf_counter()
            -
            gmm_start
        )


        # ====================================================
        # B. CANDIDATE BUILD TIME
        # ====================================================

        candidate_start = (
            time.perf_counter()
        )


        candidate_parts = [
            cluster_to_indices[c]
            for c in routed_clusters
        ]


        if len(candidate_parts) > 0:

            routed_candidates = np.concatenate(
                candidate_parts
            )

        else:

            routed_candidates = np.array(
                [],
                dtype=np.int64
            )


        candidate_elapsed = (
            time.perf_counter()
            -
            candidate_start
        )


        candidate_size = len(
            routed_candidates
        )

        candidate_sizes.append(
            candidate_size
        )


        # ====================================================
        # C. COSINE SEARCH WITHIN ROUTED CANDIDATES
        # ====================================================

        cosine_start = time.perf_counter()


        candidate_vectors = (
            train_norm[
                routed_candidates
            ]
        )


        similarities = (
            candidate_vectors
            @
            q_norm
        )


        local_k = min(
            GLOBAL_TOP_K,
            candidate_size
        )


        if local_k > 0:

            local_top_unsorted = np.argpartition(
                -similarities,
                local_k - 1
            )[:local_k]


            local_scores = similarities[
                local_top_unsorted
            ]


            local_order = np.argsort(
                -local_scores
            )


            local_top_sorted = (
                local_top_unsorted[
                    local_order
                ]
            )


            routed_top_global_indices = (
                routed_candidates[
                    local_top_sorted
                ]
            )

        else:

            routed_top_global_indices = np.array(
                [],
                dtype=np.int64
            )


        cosine_elapsed = (
            time.perf_counter()
            -
            cosine_start
        )


        # ====================================================
        # D. TOTAL PROPOSED RETRIEVAL TIME
        # ====================================================

        proposed_elapsed = (
            gmm_elapsed
            +
            candidate_elapsed
            +
            cosine_elapsed
        )


        gmm_times.append(
            gmm_elapsed
        )

        candidate_build_times.append(
            candidate_elapsed
        )

        routed_cosine_times.append(
            cosine_elapsed
        )

        proposed_total_times.append(
            proposed_elapsed
        )


        # ====================================================
        # E. GLOBAL REFERENCE PRESERVATION
        # ====================================================

        routed_candidate_set = set(
            routed_candidates.tolist()
        )


        ref_top1 = int(
            global_top1[i]
        )


        ref_top5 = (
            global_top5[i]
            .astype(int)
            .tolist()
        )


        top1_preserved = (
            ref_top1
            in
            routed_candidate_set
        )


        if top1_preserved:

            top1_preserved_count += 1


        top5_hits = sum(
            idx
            in
            routed_candidate_set
            for idx
            in
            ref_top5
        )


        top5_fraction = (
            top5_hits
            /
            GLOBAL_TOP_K
        )


        total_top5_fraction += (
            top5_fraction
        )


        if top5_hits >= 1:

            at_least_one_top5_count += 1


        if top5_hits == GLOBAL_TOP_K:

            all_top5_preserved_count += 1


        # ====================================================
        # F. CANDIDATE REDUCTION
        # ====================================================

        candidate_reduction = (
            1.0
            -
            (
                candidate_size
                /
                N_TRAIN
            )
        )


        # ====================================================
        # G. SPEEDUP FOR THIS QUERY
        # ====================================================

        standard_time = (
            standard_times[i]
        )


        if proposed_elapsed > 0:

            speedup = (
                standard_time
                /
                proposed_elapsed
            )

        else:

            speedup = np.nan


        query_level_rows.append({

            "query_index":
                i,

            "routing_top_clusters":
                top_n_clusters,

            "candidate_size":
                candidate_size,

            "candidate_reduction":
                candidate_reduction,

            "global_top1_preserved":
                int(
                    top1_preserved
                ),

            "global_top5_hits":
                top5_hits,

            "global_top5_fraction_preserved":
                top5_fraction,

            "standard_retrieval_ms":
                standard_time
                *
                1000.0,

            "gmm_routing_ms":
                gmm_elapsed
                *
                1000.0,

            "candidate_build_ms":
                candidate_elapsed
                *
                1000.0,

            "routed_cosine_ms":
                cosine_elapsed
                *
                1000.0,

            "proposed_total_retrieval_ms":
                proposed_elapsed
                *
                1000.0,

            "speedup_vs_standard":
                speedup
        })


    # ========================================================
    # 9. ROUTING SUMMARY
    # ========================================================

    candidate_sizes = np.asarray(
        candidate_sizes
    )

    gmm_times = np.asarray(
        gmm_times
    )

    candidate_build_times = np.asarray(
        candidate_build_times
    )

    routed_cosine_times = np.asarray(
        routed_cosine_times
    )

    proposed_total_times = np.asarray(
        proposed_total_times
    )


    top1_preservation = (
        top1_preserved_count
        /
        N_VAL
    )


    at_least_one_top5 = (
        at_least_one_top5_count
        /
        N_VAL
    )


    all_top5_preserved = (
        all_top5_preserved_count
        /
        N_VAL
    )


    mean_top5_fraction = (
        total_top5_fraction
        /
        N_VAL
    )


    avg_candidate_size = float(
        candidate_sizes.mean()
    )


    avg_candidate_reduction = (
        1.0
        -
        (
            avg_candidate_size
            /
            N_TRAIN
        )
    )


    gmm_stats = timing_summary(
        gmm_times
    )

    candidate_stats = timing_summary(
        candidate_build_times
    )

    cosine_stats = timing_summary(
        routed_cosine_times
    )

    proposed_stats = timing_summary(
        proposed_total_times
    )


    mean_speedup = (
        standard_times.mean()
        /
        proposed_total_times.mean()
    )


    median_speedup = (
        np.median(
            standard_times
        )
        /
        np.median(
            proposed_total_times
        )
    )


    summary_rows.append({

        "routing_top_clusters":
            top_n_clusters,

        "global_top1_preservation":
            top1_preservation,

        "at_least_one_global_top5_preserved":
            at_least_one_top5,

        "all_global_top5_preserved":
            all_top5_preserved,

        "mean_global_top5_fraction_preserved":
            mean_top5_fraction,

        "avg_candidate_size":
            avg_candidate_size,

        "median_candidate_size":
            float(
                np.median(
                    candidate_sizes
                )
            ),

        "avg_candidate_reduction":
            avg_candidate_reduction,

        "standard_mean_ms":
            standard_stats[
                "mean_ms"
            ],

        "standard_median_ms":
            standard_stats[
                "median_ms"
            ],

        "standard_p95_ms":
            standard_stats[
                "p95_ms"
            ],

        "gmm_mean_ms":
            gmm_stats[
                "mean_ms"
            ],

        "candidate_build_mean_ms":
            candidate_stats[
                "mean_ms"
            ],

        "routed_cosine_mean_ms":
            cosine_stats[
                "mean_ms"
            ],

        "proposed_mean_ms":
            proposed_stats[
                "mean_ms"
            ],

        "proposed_median_ms":
            proposed_stats[
                "median_ms"
            ],

        "proposed_p95_ms":
            proposed_stats[
                "p95_ms"
            ],

        "mean_speedup_vs_standard":
            mean_speedup,

        "median_speedup_vs_standard":
            median_speedup
    })


    print("\nRetrieval preservation")

    print(
        f"Global Top-1 preserved      : "
        f"{top1_preservation:.6f}"
    )

    print(
        f"At least one Top-5 preserved: "
        f"{at_least_one_top5:.6f}"
    )

    print(
        f"All Top-5 preserved         : "
        f"{all_top5_preserved:.6f}"
    )

    print(
        f"Mean Top-5 fraction kept    : "
        f"{mean_top5_fraction:.6f}"
    )


    print("\nCandidate reduction")

    print(
        f"Average candidate size      : "
        f"{avg_candidate_size:.2f}"
    )

    print(
        f"Candidate reduction         : "
        f"{avg_candidate_reduction:.6f}"
    )


    print("\nTiming")

    print(
        f"Standard mean retrieval     : "
        f"{standard_stats['mean_ms']:.6f} ms"
    )

    print(
        f"GMM routing mean            : "
        f"{gmm_stats['mean_ms']:.6f} ms"
    )

    print(
        f"Candidate build mean        : "
        f"{candidate_stats['mean_ms']:.6f} ms"
    )

    print(
        f"Routed cosine mean          : "
        f"{cosine_stats['mean_ms']:.6f} ms"
    )

    print(
        f"Proposed mean retrieval     : "
        f"{proposed_stats['mean_ms']:.6f} ms"
    )

    print(
        f"Standard P95 retrieval      : "
        f"{standard_stats['p95_ms']:.6f} ms"
    )

    print(
        f"Proposed P95 retrieval      : "
        f"{proposed_stats['p95_ms']:.6f} ms"
    )

    print(
        f"Mean speedup                : "
        f"{mean_speedup:.3f}x"
    )


# ============================================================
# 10. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "routing_timing_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


query_df = pd.DataFrame(
    query_level_rows
)

query_file = os.path.join(
    OUTPUT_DIR,
    "routing_timing_query_level.csv"
)

query_df.to_csv(
    query_file,
    index=False
)


# ============================================================
# 11. PLOT: PRESERVATION VS ROUTING
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    summary_df[
        "routing_top_clusters"
    ],
    summary_df[
        "global_top1_preservation"
    ],
    marker="o",
    label="Global Top-1 preserved"
)

plt.plot(
    summary_df[
        "routing_top_clusters"
    ],
    summary_df[
        "mean_global_top5_fraction_preserved"
    ],
    marker="o",
    label="Mean Global Top-5 fraction"
)

plt.xlabel(
    "Number of routed clusters"
)

plt.ylabel(
    "Preservation score"
)

plt.title(
    "Retrieval Preservation with GMM Routing"
)

plt.xticks(
    ROUTING_OPTIONS
)

plt.ylim(
    0,
    1
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "routing_preservation.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 12. PLOT: CANDIDATE REDUCTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    summary_df[
        "routing_top_clusters"
    ],
    summary_df[
        "avg_candidate_reduction"
    ],
    marker="o"
)

plt.xlabel(
    "Number of routed clusters"
)

plt.ylabel(
    "Candidate reduction ratio"
)

plt.title(
    "Candidate Reduction with GMM Routing"
)

plt.xticks(
    ROUTING_OPTIONS
)

plt.ylim(
    0,
    1
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "routing_candidate_reduction.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 13. PLOT: STANDARD VS PROPOSED LATENCY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    summary_df[
        "routing_top_clusters"
    ],
    summary_df[
        "standard_mean_ms"
    ],
    marker="o",
    label="Standard global retrieval"
)

plt.plot(
    summary_df[
        "routing_top_clusters"
    ],
    summary_df[
        "proposed_mean_ms"
    ],
    marker="o",
    label="GMM-routed retrieval"
)

plt.xlabel(
    "Number of routed clusters"
)

plt.ylabel(
    "Mean retrieval latency (ms)"
)

plt.title(
    "Standard vs GMM-Routed Retrieval Latency"
)

plt.xticks(
    ROUTING_OPTIONS
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "standard_vs_routed_latency.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 14. PLOT: SPEEDUP
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    summary_df[
        "routing_top_clusters"
    ],
    summary_df[
        "mean_speedup_vs_standard"
    ],
    marker="o"
)

plt.axhline(
    1.0,
    linestyle="--"
)

plt.xlabel(
    "Number of routed clusters"
)

plt.ylabel(
    "Speedup vs standard retrieval"
)

plt.title(
    "Retrieval Speedup from GMM Routing"
)

plt.xticks(
    ROUTING_OPTIONS
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "routing_speedup.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 15. FINAL TABLE
# ============================================================

print("\n" + "=" * 85)
print("FINAL ROUTING + TIMING SUMMARY")
print("=" * 85)

display_cols = [

    "routing_top_clusters",

    "global_top1_preservation",

    "mean_global_top5_fraction_preserved",

    "avg_candidate_size",

    "avg_candidate_reduction",

    "standard_mean_ms",

    "proposed_mean_ms",

    "proposed_p95_ms",

    "mean_speedup_vs_standard"
]

print(
    summary_df[
        display_cols
    ].to_string(
        index=False
    )
)


print("\nSaved:")
print(summary_file)
print(query_file)