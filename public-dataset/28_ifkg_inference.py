import os
import re
import json
import time

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from openai import OpenAI
from neo4j import GraphDatabase

from rouge_score import rouge_scorer
from bert_score import score as bert_score


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Uma\IIIT-B\IIITB-IBN-ORAN-WCNC\SCALA"

TEST_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "test_fault_resolution.xlsx"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ifkg_inference"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

CHECKPOINT_FILE = os.path.join(
    RESULT_DIR,
    "ifkg_query_results_checkpoint.csv"
)

RESULT_CSV = os.path.join(
    RESULT_DIR,
    "ifkg_query_results.csv"
)

RESULT_XLSX = os.path.join(
    RESULT_DIR,
    "ifkg_query_results.xlsx"
)

SUMMARY_JSON = os.path.join(
    RESULT_DIR,
    "ifkg_overall_summary.json"
)


# ============================================================
# MODELS
#
# Same model everywhere, as decided.
# ============================================================

LLM_MODEL = "gpt-4.1-mini"

CYPHER_MAX_OUTPUT_TOKENS = 500

ANSWER_MAX_OUTPUT_TOKENS = 500


# ============================================================
# EXPERIMENT
#
# First debugging run:
# MAX_TEST_QUERIES = 5
#
# Final:
# MAX_TEST_QUERIES = None
# ============================================================

MAX_TEST_QUERIES = None


# ============================================================
# CYPHER SETTINGS
# ============================================================

MAX_CYPHER_RETRIES = 2

MAX_KG_RESULTS = 10


# ============================================================
# BERTSCORE
# ============================================================

BERTSCORE_LANG = "en"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

if not OPENAI_API_KEY:

    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )


client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60.0,
    max_retries=0
)


# ============================================================
# LOCAL NEO4J
#
# ONLY CHANGE PASSWORD.
# ============================================================

NEO4J_URI = "bolt://localhost:7687"

NEO4J_USERNAME = "neo4j"

NEO4J_PASSWORD = "iiitbresearch"


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def percentile(
    values,
    p
):

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            p
        )
    )


def latency_summary(
    series
):

    values = (
        series
        .dropna()
        .astype(float)
        .tolist()
    )

    if len(values) == 0:

        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "std_ms": 0.0,
            "p95_ms": 0.0
        }

    return {
        "mean_ms":
            float(
                np.mean(values)
            ),

        "median_ms":
            float(
                np.median(values)
            ),

        "std_ms":
            float(
                np.std(values)
            ),

        "p95_ms":
            percentile(
                values,
                95
            )
    }


# ============================================================
# VERIFY NEO4J
# ============================================================

print("=" * 90)
print("TESTING NEO4J")
print("=" * 90)


with driver.session() as session:

    result = session.run(
        """
        MATCH (n)
        RETURN count(n) AS nodes
        """
    )

    graph_nodes = int(
        result.single()[
            "nodes"
        ]
    )


print(
    "Graph nodes:",
    graph_nodes
)


if graph_nodes == 0:

    driver.close()

    raise RuntimeError(
        "Neo4j graph is empty. "
        "Finish IFKG construction first."
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_excel(
    TEST_FILE
)


required_columns = [
    "id",
    "fault_text",
    "resolution_text"
]


for column in required_columns:

    if column not in test_df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


if MAX_TEST_QUERIES is not None:

    test_df = (
        test_df
        .iloc[
            :MAX_TEST_QUERIES
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


print(
    "Test queries:",
    len(test_df)
)


# ============================================================
# GRAPH SCHEMA FOR TEXT-TO-CYPHER
# ============================================================

GRAPH_SCHEMA = """
Neo4j graph schema:

NODE: TrainingRecord
Properties:
- record_id
- fault_text
- resolution_text
- source_dataset

NODE: Chunk
Properties:
- chunk_id
- chunk_index
- text
- token_count

NODE: Entity
Properties:
- name
- type

STRUCTURAL RELATIONSHIPS:

(:TrainingRecord)-[:HAS_CHUNK]->(:Chunk)

(:Chunk)-[:MENTIONS]->(:Entity)

Entities are also connected to other Entity nodes through
semantic relationships extracted from the training data.
Examples include:

CAUSES
AFFECTS
OCCURS_IN
DEPENDS_ON
RESOLVED_BY
RESOLVES
REQUIRES
CONFIGURED_WITH
INDICATES
TRIGGERS
USES
CONNECTED_TO
ASSOCIATED_WITH

Other valid semantic relation types may also exist.

The graph contains TRAINING data only.
""".strip()


# ============================================================
# TEXT-TO-CYPHER INSTRUCTIONS
#
# IFKG:
#
# natural-language fault
#        |
#        v
#       LLM
#        |
#        v
#     Cypher
# ============================================================

CYPHER_INSTRUCTIONS = f"""
You translate a network or IT fault description into a
READ-ONLY Neo4j Cypher query for retrieving useful historical
fault-resolution evidence.

{GRAPH_SCHEMA}

The purpose of the query is to obtain graph evidence useful
for resolving the new fault.

Important requirements:

1. Return ONLY the Cypher query.
2. Do not return markdown fences.
3. NEVER modify the database.
4. Do not use CREATE, MERGE, DELETE, SET, REMOVE, DROP,
   LOAD CSV, or write procedures.
5. Prefer traversing:

   TrainingRecord -> HAS_CHUNK -> Chunk -> MENTIONS -> Entity

6. You may additionally traverse semantic Entity-to-Entity
   relationships when useful.
7. Search case-insensitively.
8. Prefer CONTAINS rather than exact equality for textual
   matching unless exact matching is clearly appropriate.
9. Retrieve historical TrainingRecord information whenever
   possible so the downstream model can see fault-resolution
   evidence.
10. Return useful scalar fields such as:

    record_id,
    fault_text,
    resolution_text,
    entity names,
    relation types

11. Return at most {MAX_KG_RESULTS} rows.

A useful general query pattern is:

MATCH (tr:TrainingRecord)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
WHERE ...
RETURN DISTINCT
    tr.record_id AS record_id,
    tr.fault_text AS fault_text,
    tr.resolution_text AS resolution_text,
    collect(DISTINCT e.name)[0..10] AS entities
LIMIT {MAX_KG_RESULTS}

Adapt the actual WHERE conditions to the supplied fault.
""".strip()


# ============================================================
# CLEAN CYPHER
# ============================================================

def clean_cypher(
    text
):

    text = safe_text(
        text
    )

    text = re.sub(
        r"^```(?:cypher)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    if text.endswith(";"):
        text = text[:-1].strip()

    return text


# ============================================================
# READ-ONLY VALIDATION
# ============================================================

FORBIDDEN_CYPHER = [
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD CSV",
    "FOREACH"
]


def validate_read_only_cypher(
    query
):

    upper = query.upper()

    for token in FORBIDDEN_CYPHER:

        if token in upper:

            return False

    if "RETURN" not in upper:

        return False

    return True


# ============================================================
# GENERATE CYPHER
# ============================================================

def generate_cypher(
    query_fault,
    correction_context=""
):

    if correction_context:

        user_prompt = f"""
NEW FAULT
---------
{query_fault}

THE PREVIOUS CYPHER QUERY FAILED.

ERROR / CORRECTION INFORMATION
------------------------------
{correction_context}

Generate a corrected READ-ONLY Cypher query.
""".strip()

    else:

        user_prompt = f"""
NEW FAULT
---------
{query_fault}

Generate the most appropriate read-only Cypher query to
retrieve useful fault-resolution evidence from the graph.
""".strip()


    start = time.perf_counter()


    response = client.responses.create(
        model=LLM_MODEL,
        instructions=CYPHER_INSTRUCTIONS,
        input=user_prompt,
        max_output_tokens=CYPHER_MAX_OUTPUT_TOKENS,
        temperature=0
    )


    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    query = clean_cypher(
        response.output_text
    )


    return (
        query,
        elapsed_ms
    )


# ============================================================
# SERIALIZE NEO4J VALUES
# ============================================================

def serializable_value(
    value
):

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool
        )
    ):

        return value

    if isinstance(
        value,
        list
    ):

        return [
            serializable_value(v)
            for v in value
        ]

    if isinstance(
        value,
        dict
    ):

        return {
            str(k):
                serializable_value(v)
            for k, v in value.items()
        }

    try:
        return dict(value)

    except Exception:
        return str(value)


# ============================================================
# EXECUTE CYPHER
# ============================================================

def execute_cypher(
    query
):

    start = time.perf_counter()


    with driver.session() as session:

        result = session.run(
            query
        )

        rows = []

        for record in result:

            data = record.data()

            clean_data = {
                key:
                    serializable_value(value)
                for key, value
                in data.items()
            }

            rows.append(
                clean_data
            )


    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    return (
        rows,
        elapsed_ms
    )


# ============================================================
# TEXT-TO-CYPHER + GRAPH RETRIEVAL
#
# Allows one correction retry for syntactically invalid Cypher.
# All retry time is included in E2E.
# ============================================================

def retrieve_from_kg(
    query_fault
):

    total_cypher_ms = 0.0

    total_kg_ms = 0.0

    cypher_calls = 0

    last_query = ""

    last_error = ""


    correction = ""


    for attempt in range(
        MAX_CYPHER_RETRIES + 1
    ):

        try:

            query, cypher_ms = (
                generate_cypher(
                    query_fault,
                    correction
                )
            )


            cypher_calls += 1

            total_cypher_ms += (
                cypher_ms
            )


            last_query = query


            if not validate_read_only_cypher(
                query
            ):

                raise ValueError(
                    "Generated query was not valid "
                    "read-only Cypher."
                )


            rows, kg_ms = execute_cypher(
                query
            )


            total_kg_ms += kg_ms


            return {
                "cypher":
                    query,

                "rows":
                    rows,

                "cypher_ms":
                    total_cypher_ms,

                "kg_ms":
                    total_kg_ms,

                "cypher_calls":
                    cypher_calls,

                "retry_count":
                    max(
                        0,
                        cypher_calls - 1
                    ),

                "success":
                    True,

                "error":
                    ""
            }


        except Exception as e:

            last_error = str(e)

            correction = (
                f"Previous query:\n"
                f"{last_query}\n\n"
                f"Neo4j/error message:\n"
                f"{last_error}"
            )


    return {
        "cypher":
            last_query,

        "rows":
            [],

        "cypher_ms":
            total_cypher_ms,

        "kg_ms":
            total_kg_ms,

        "cypher_calls":
            cypher_calls,

        "retry_count":
            max(
                0,
                cypher_calls - 1
            ),

        "success":
            False,

        "error":
            last_error
    }


# ============================================================
# BUILD KG CONTEXT
# ============================================================

def build_kg_context(
    rows
):

    if len(rows) == 0:

        return (
            "No explicit graph records were returned "
            "for this fault."
        )


    blocks = []


    for rank, row in enumerate(
        rows,
        start=1
    ):

        block = (
            f"KG Result {rank}\n"
            f"{json.dumps(row, ensure_ascii=False)}"
        )

        blocks.append(
            block
        )


    return "\n\n".join(
        blocks
    )


# ============================================================
# FINAL GENERATION PROMPT
#
# IFKG paper:
#
# query results
#     |
#     v
#    LLM
#     |
#     v
# natural-language response
# ============================================================

ANSWER_INSTRUCTIONS = """
You are a network and IT fault-resolution assistant.

You receive:

1. a new fault description
2. evidence retrieved from a fault-resolution knowledge graph

Generate a concise and technically appropriate resolution for
the new fault.

Use the graph evidence as supporting context.
Do not mention Cypher, Neo4j, graph nodes, database rows,
or that a knowledge graph was used.

Do not copy irrelevant details.

Return only the recommended resolution.
""".strip()


def generate_resolution(
    query_fault,
    kg_context
):

    prompt_start = time.perf_counter()


    user_prompt = f"""
NEW FAULT
---------
{query_fault}

KNOWLEDGE-GRAPH EVIDENCE
------------------------
{kg_context}

TASK
----
Provide the most appropriate resolution for the new fault.
""".strip()


    prompt_ms = (
        time.perf_counter()
        -
        prompt_start
    ) * 1000


    llm_start = time.perf_counter()


    response = client.responses.create(
        model=LLM_MODEL,
        instructions=ANSWER_INSTRUCTIONS,
        input=user_prompt,
        max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS,
        temperature=0
    )


    llm_ms = (
        time.perf_counter()
        -
        llm_start
    ) * 1000


    generated = (
        response.output_text
        .strip()
    )


    return (
        generated,
        prompt_ms,
        llm_ms
    )


# ============================================================
# RESUME CHECKPOINT
# ============================================================

query_results = []


if os.path.exists(
    CHECKPOINT_FILE
):

    old_df = pd.read_csv(
        CHECKPOINT_FILE
    )


    query_results = (
        old_df
        .to_dict(
            "records"
        )
    )


    completed_ids = set(
        old_df[
            "query_id"
        ]
        .astype(str)
    )


    print(
        "Existing completed queries:",
        len(completed_ids)
    )


else:

    completed_ids = set()


# ============================================================
# INFERENCE LOOP
# ============================================================

print("\n" + "=" * 90)
print("IFKG INFERENCE")
print("=" * 90)

print(
    "Queries:",
    len(test_df)
)

print(
    "Model:",
    LLM_MODEL
)


for q, row in tqdm(
    test_df.iterrows(),
    total=len(test_df),
    desc="IFKG inference"
):

    query_id = safe_text(
        row[
            "id"
        ]
    )


    if str(query_id) in completed_ids:
        continue


    query_fault = safe_text(
        row[
            "fault_text"
        ]
    )


    ground_truth = safe_text(
        row[
            "resolution_text"
        ]
    )


    source_dataset = safe_text(
        row.get(
            "source_dataset",
            ""
        )
    )


    # ========================================================
    # COMPLETE QUERY TIMER
    # ========================================================

    e2e_start = time.perf_counter()


    # ========================================================
    # 1. TEXT -> CYPHER
    # 2. EXECUTE CYPHER
    # ========================================================

    kg_result = retrieve_from_kg(
        query_fault
    )


    rows = kg_result[
        "rows"
    ]


    # ========================================================
    # IFKG RETRIEVAL DIAGNOSTICS
    # ========================================================

    kg_result_count = len(
        rows
    )


    unique_record_ids = set()


    for item in rows:

        if (
            isinstance(
                item,
                dict
            )
            and
            item.get(
                "record_id"
            ) is not None
        ):

            unique_record_ids.add(
                str(
                    item[
                        "record_id"
                    ]
                )
            )


    unique_training_records = len(
        unique_record_ids
    )


    # ========================================================
    # 3. BUILD CONTEXT
    # ========================================================

    kg_context = build_kg_context(
        rows
    )


    # ========================================================
    # 4. GRAPH RESULTS -> NATURAL-LANGUAGE RESOLUTION
    # ========================================================

    try:

        (
            generated_resolution,
            prompt_ms,
            answer_llm_ms
        ) = generate_resolution(
            query_fault,
            kg_context
        )


        answer_api_error = ""


    except Exception as e:

        generated_resolution = ""

        prompt_ms = 0.0

        answer_llm_ms = 0.0

        answer_api_error = str(e)


    # ========================================================
    # TRUE E2E WALL-CLOCK LATENCY
    #
    # Includes:
    #
    # text-to-Cypher GPT
    # Neo4j query
    # correction retry if needed
    # context/prompt preparation
    # final GPT answer generation
    # ========================================================

    e2e_ms = (
        time.perf_counter()
        -
        e2e_start
    ) * 1000


    result = {

        "query_index":
            int(q),

        "query_id":
            query_id,

        "query_fault":
            query_fault,

        "ground_truth_resolution":
            ground_truth,

        "source_dataset":
            source_dataset,


        # ----------------------------------------------------
        # IFKG RETRIEVAL
        # ----------------------------------------------------

        "generated_cypher":
            kg_result[
                "cypher"
            ],

        "cypher_success":
            bool(
                kg_result[
                    "success"
                ]
            ),

        "cypher_calls":
            int(
                kg_result[
                    "cypher_calls"
                ]
            ),

        "cypher_retry_count":
            int(
                kg_result[
                    "retry_count"
                ]
            ),

        "cypher_error":
            kg_result[
                "error"
            ],

        "kg_result_count":
            int(
                kg_result_count
            ),

        "unique_training_records":
            int(
                unique_training_records
            ),

        "kg_hit":
            bool(
                kg_result_count > 0
            ),

        "kg_results_json":
            json.dumps(
                rows,
                ensure_ascii=False
            ),


        # ----------------------------------------------------
        # FINAL GENERATED RESOLUTION
        # ----------------------------------------------------

        "ifkg_generated_resolution":
            generated_resolution,

        "answer_api_error":
            answer_api_error,


        # ----------------------------------------------------
        # LATENCIES
        # ----------------------------------------------------

        "ifkg_text_to_cypher_ms":
            float(
                kg_result[
                    "cypher_ms"
                ]
            ),

        "ifkg_kg_query_ms":
            float(
                kg_result[
                    "kg_ms"
                ]
            ),

        "ifkg_prompt_ms":
            float(
                prompt_ms
            ),

        "ifkg_llm_ms":
            float(
                answer_llm_ms
            ),

        "ifkg_e2e_ms":
            float(
                e2e_ms
            )
    }


    query_results.append(
        result
    )


    pd.DataFrame(
        query_results
    ).to_csv(
        CHECKPOINT_FILE,
        index=False
    )


# ============================================================
# FINAL RESULT DF
# ============================================================

results_df = pd.DataFrame(
    query_results
)


# ============================================================
# QUALITY INPUTS
# ============================================================

references = (
    results_df[
        "ground_truth_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


predictions = (
    results_df[
        "ifkg_generated_resolution"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


# ============================================================
# ROUGE-L
#
# Same setup as other baselines.
# ============================================================

print(
    "\nCalculating ROUGE-L..."
)


rouge = rouge_scorer.RougeScorer(
    [
        "rougeL"
    ],
    use_stemmer=True
)


rouge_scores = []


for reference, prediction in zip(
    references,
    predictions
):

    result = rouge.score(
        reference,
        prediction
    )


    rouge_scores.append(
        result[
            "rougeL"
        ].fmeasure
    )


results_df[
    "ifkg_rougeL"
] = rouge_scores


# ============================================================
# BERTSCORE
#
# Same setup as existing experiments:
#
# lang="en"
# device same CPU/GPU selection
# ============================================================

print(
    "\nCalculating BERTScore..."
)


P, R, F1 = bert_score(
    predictions,
    references,
    lang=BERTSCORE_LANG,
    verbose=True,
    device=DEVICE
)


results_df[
    "ifkg_bertscore_precision"
] = P.cpu().numpy()


results_df[
    "ifkg_bertscore_recall"
] = R.cpu().numpy()


results_df[
    "ifkg_bertscore_f1"
] = F1.cpu().numpy()


# ============================================================
# SUMMARY
# ============================================================

n_queries = len(
    results_df
)


cypher_success_count = int(
    results_df[
        "cypher_success"
    ].astype(bool).sum()
)


kg_hit_count = int(
    results_df[
        "kg_hit"
    ].astype(bool).sum()
)


answer_error_count = int(
    (
        results_df[
            "answer_api_error"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        >
        0
    ).sum()
)


summary = {

    "baseline":
        "IFKG",

    "num_test_queries":
        int(
            n_queries
        ),

    "llm_model":
        LLM_MODEL,

    "retrieval_method":
        "LLM text-to-Cypher + Neo4j knowledge graph",

    "candidate_reduction":
        "N/A",

    "top5_retrieval_overlap":
        "N/A",


    # ========================================================
    # QUALITY
    # ========================================================

    "avg_rougeL":
        float(
            results_df[
                "ifkg_rougeL"
            ].mean()
        ),

    "avg_bertscore_f1":
        float(
            results_df[
                "ifkg_bertscore_f1"
            ].mean()
        ),


    # ========================================================
    # IFKG RETRIEVAL METRICS
    # ========================================================

    "avg_kg_result_count":
        float(
            results_df[
                "kg_result_count"
            ].mean()
        ),

    "avg_unique_training_records":
        float(
            results_df[
                "unique_training_records"
            ].mean()
        ),

    "kg_hit_count":
        int(
            kg_hit_count
        ),

    "kg_hit_rate":
        float(
            kg_hit_count
            /
            n_queries
            if n_queries > 0
            else 0
        ),

    "cypher_success_count":
        int(
            cypher_success_count
        ),

    "cypher_success_rate":
        float(
            cypher_success_count
            /
            n_queries
            if n_queries > 0
            else 0
        ),

    "avg_cypher_calls":
        float(
            results_df[
                "cypher_calls"
            ].mean()
        ),

    "avg_cypher_retry_count":
        float(
            results_df[
                "cypher_retry_count"
            ].mean()
        ),

    "answer_api_error_count":
        int(
            answer_error_count
        ),


    # ========================================================
    # LATENCY
    # ========================================================

    "text_to_cypher_latency":
        latency_summary(
            results_df[
                "ifkg_text_to_cypher_ms"
            ]
        ),

    "kg_query_latency":
        latency_summary(
            results_df[
                "ifkg_kg_query_ms"
            ]
        ),

    "prompt_latency":
        latency_summary(
            results_df[
                "ifkg_prompt_ms"
            ]
        ),

    "llm_latency":
        latency_summary(
            results_df[
                "ifkg_llm_ms"
            ]
        ),

    "e2e_latency":
        latency_summary(
            results_df[
                "ifkg_e2e_ms"
            ]
        )
}


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    RESULT_CSV,
    index=False
)


results_df.to_excel(
    RESULT_XLSX,
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


driver.close()


# ============================================================
# FINAL PRINT
# ============================================================

print("\n" + "=" * 90)
print("IFKG INFERENCE COMPLETE")
print("=" * 90)


print("\nQUALITY")


print(
    "ROUGE-L:",
    f"{summary['avg_rougeL']:.4f}"
)


print(
    "BERTScore F1:",
    f"{summary['avg_bertscore_f1']:.4f}"
)


print("\nIFKG RETRIEVAL")


print(
    "Average KG rows:",
    f"{summary['avg_kg_result_count']:.2f}"
)


print(
    "Average unique historical records:",
    f"{summary['avg_unique_training_records']:.2f}"
)


print(
    "KG hit rate:",
    f"{summary['kg_hit_rate'] * 100:.2f}%"
)


print(
    "Cypher success rate:",
    f"{summary['cypher_success_rate'] * 100:.2f}%"
)


print(
    "Average Cypher calls:",
    f"{summary['avg_cypher_calls']:.3f}"
)


print("\nMEAN LATENCY")


print(
    "Text-to-Cypher:",
    f"{summary['text_to_cypher_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "KG query:",
    f"{summary['kg_query_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "Prompt:",
    f"{summary['prompt_latency']['mean_ms']:.3f}",
    "ms"
)


print(
    "Final LLM:",
    f"{summary['llm_latency']['mean_ms']:.3f}",
    "ms"
)


print("\nEND-TO-END")


print(
    "Mean E2E:",
    f"{summary['e2e_latency']['mean_ms'] / 1000:.3f}",
    "sec"
)


print(
    "P95 E2E:",
    f"{summary['e2e_latency']['p95_ms'] / 1000:.3f}",
    "sec"
)


print("\nSaved:")

print(
    RESULT_XLSX
)

print(
    SUMMARY_JSON
)