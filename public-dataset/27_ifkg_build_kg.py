import os
import re
import json
import time
import hashlib

import pandas as pd
from tqdm import tqdm

import tiktoken
from openai import OpenAI
from neo4j import GraphDatabase


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset_splits",
    "train_fault_resolution.xlsx"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ifkg"
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
    "ifkg_build_checkpoint.csv"
)

CHUNK_FILE = os.path.join(
    RESULT_DIR,
    "ifkg_chunks.xlsx"
)

EXTRACTION_FILE = os.path.join(
    RESULT_DIR,
    "ifkg_extractions.xlsx"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "ifkg_build_summary.json"
)


# ============================================================
# OPENAI SETTINGS
# ============================================================

LLM_MODEL = "gpt-4.1-mini"

MAX_OUTPUT_TOKENS = 1600


# ============================================================
# IFKG CHUNKING SETTINGS
# ============================================================

CHUNK_SIZE = 256
CHUNK_OVERLAP = 20


# ============================================================
# TEST MODE
#
# FIRST RUN:
# keep this as 5
#
# FULL RUN:
# change to None
# ============================================================

MAX_TRAIN_RECORDS = None


# ============================================================
# RETRY SETTINGS
# ============================================================

MAX_RETRIES = 5

RETRY_SLEEP_SECONDS = 5


# ============================================================
# OPENAI API KEY
#
# Keep API key in PowerShell environment variable.
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

if not OPENAI_API_KEY:

    raise EnvironmentError(
        "OPENAI_API_KEY is not set.\n\n"
        "In PowerShell run:\n"
        '$env:OPENAI_API_KEY="your-openai-key"'
    )


# ============================================================
# LOCAL NEO4J SETTINGS
#
# YOU ONLY NEED TO CHANGE THE PASSWORD BELOW.
# ============================================================

NEO4J_URI = "bolt://localhost:7687"

NEO4J_USERNAME = "neo4j"

NEO4J_PASSWORD = "iiitbresearch"


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60.0,
    max_retries=0
)


# ============================================================
# NEO4J DRIVER
# ============================================================

print("=" * 90)
print("NEO4J CONFIGURATION")
print("=" * 90)

print(
    "URI:",
    NEO4J_URI
)

print(
    "Username:",
    NEO4J_USERNAME
)


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)


# ============================================================
# TOKENIZER
# ============================================================

try:

    tokenizer = (
        tiktoken.encoding_for_model(
            LLM_MODEL
        )
    )

except KeyError:

    tokenizer = (
        tiktoken.get_encoding(
            "o200k_base"
        )
    )


# ============================================================
# TEST NEO4J CONNECTION
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "TESTING NEO4J CONNECTION"
)

print("=" * 90)


try:

    with driver.session() as session:

        result = session.run(
            "RETURN 1 AS test"
        )

        value = result.single()[
            "test"
        ]


    print(
        "Neo4j connection successful:",
        value
    )


except Exception as e:

    driver.close()

    raise RuntimeError(
        "\nCould not connect to local Neo4j.\n"
        "Please make sure:\n"
        "1. Neo4j is running.\n"
        "2. Bolt is available at localhost:7687.\n"
        "3. Username is neo4j.\n"
        "4. NEO4J_PASSWORD in this script is correct.\n\n"
        f"Original error:\n{e}"
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "LOADING TRAINING DATA"
)

print("=" * 90)


train_df = pd.read_excel(
    TRAIN_FILE
)


required_columns = [
    "id",
    "fault_text",
    "resolution_text"
]


for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Missing required column: "
            f"{column}"
        )


if MAX_TRAIN_RECORDS is not None:

    train_df = (
        train_df
        .iloc[
            :MAX_TRAIN_RECORDS
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


print(
    "Training records being processed:",
    len(train_df)
)


# ============================================================
# HELPERS
# ============================================================

def safe_text(
    value
):

    if pd.isna(
        value
    ):

        return ""

    return str(
        value
    ).strip()


def clean_text(
    text
):

    text = str(
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# BUILD TRAINING DOCUMENT
#
# Every training item contributes:
#
# fault description
# +
# resolution
# ============================================================

def build_document(
    row
):

    fault = safe_text(
        row[
            "fault_text"
        ]
    )

    resolution = safe_text(
        row[
            "resolution_text"
        ]
    )


    return (
        "Fault Description:\n"
        f"{fault}\n\n"
        "Resolution:\n"
        f"{resolution}"
    )


# ============================================================
# TOKEN-BASED CHUNKING
#
# 256 tokens
# 20-token overlap
# ============================================================

def chunk_text_by_tokens(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    token_ids = tokenizer.encode(
        text
    )


    if len(
        token_ids
    ) <= chunk_size:

        return [
            {
                "text":
                    text,

                "token_count":
                    len(
                        token_ids
                    )
            }
        ]


    chunks = []

    start = 0


    while start < len(
        token_ids
    ):

        end = min(
            start + chunk_size,
            len(
                token_ids
            )
        )


        current_tokens = token_ids[
            start:end
        ]


        chunk_text = tokenizer.decode(
            current_tokens
        )


        chunks.append({

            "text":
                chunk_text,

            "token_count":
                len(
                    current_tokens
                )
        })


        if end >= len(
            token_ids
        ):

            break


        start = (
            end
            -
            overlap
        )


    return chunks


# ============================================================
# UNIQUE CHUNK ID
# ============================================================

def make_chunk_id(
    record_id,
    chunk_index,
    chunk_text
):

    digest = hashlib.md5(
        chunk_text.encode(
            "utf-8"
        )
    ).hexdigest()[
        :12
    ]


    return (
        f"{record_id}_"
        f"{chunk_index}_"
        f"{digest}"
    )


# ============================================================
# JOINT ENTITY + RELATIONSHIP EXTRACTION
#
# One GPT call per chunk.
# ============================================================

IFKG_EXTRACTION_PROMPT = """
You are an information extraction system for building a
knowledge graph from telecommunications, networking,
IT-support, and fault-resolution text.

From the supplied text, jointly extract:

1. technically meaningful entities
2. technically meaningful relationships between those entities

Return ONLY valid JSON with exactly this structure:

{
  "entities": [
    {
      "name": "entity name",
      "type": "entity type"
    }
  ],
  "relationships": [
    {
      "source": "source entity name",
      "relation": "relationship type",
      "target": "target entity name"
    }
  ]
}

Useful entity types may include:

FAULT
COMPONENT
SERVICE
PROTOCOL
INTERFACE
DEVICE
SOFTWARE
CONFIGURATION
ERROR
ALARM
CAUSE
ACTION
PARAMETER
NETWORK_FUNCTION
TECHNOLOGY

Useful relationship types may include:

CAUSES
AFFECTS
OCCURS_IN
CONNECTED_TO
DEPENDS_ON
RESOLVED_BY
CONFIGURED_WITH
INDICATES
REQUIRES
USES
TRIGGERS
ASSOCIATED_WITH

Rules:

1. Extract only information supported by the supplied text.
2. Do not invent entities or relationships.
3. Keep entity names concise.
4. Avoid duplicate entities.
5. Keep relationship names concise and uppercase.
6. Relationship source and target should correspond to
   meaningful entities in the extracted graph.
7. Return an empty entities list if no entities exist.
8. Return an empty relationships list if no relationships exist.
""".strip()


def extract_graph_information(
    chunk
):

    user_prompt = f"""
TEXT
----
{chunk}

Extract the entities and relationships required to construct
the knowledge graph.
""".strip()


    last_error = ""


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            start = time.perf_counter()


            response = client.responses.create(
                model=LLM_MODEL,
                instructions=IFKG_EXTRACTION_PROMPT,
                input=user_prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=0
            )


            elapsed_ms = (
                time.perf_counter()
                -
                start
            ) * 1000


            return (
                response.output_text.strip(),
                elapsed_ms
            )


        except Exception as e:

            last_error = str(
                e
            )


            print(
                f"\nExtraction attempt "
                f"{attempt}/"
                f"{MAX_RETRIES} "
                f"failed: "
                f"{last_error}"
            )


            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_SLEEP_SECONDS
                    *
                    attempt
                )


    raise RuntimeError(
        "Graph extraction failed after "
        f"{MAX_RETRIES} attempts.\n"
        f"{last_error}"
    )


# ============================================================
# CLEAN GPT JSON
# ============================================================

def clean_json_output(
    text
):

    text = safe_text(
        text
    )


    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(
        r"^```\s*",
        "",
        text
    )


    text = re.sub(
        r"\s*```$",
        "",
        text
    )


    return text.strip()


# ============================================================
# PARSE GPT OUTPUT
# ============================================================

def parse_graph_output(
    text
):

    try:

        data = json.loads(
            clean_json_output(
                text
            )
        )

    except Exception:

        return (
            [],
            [],
            True
        )


    raw_entities = data.get(
        "entities",
        []
    )


    raw_relationships = data.get(
        "relationships",
        []
    )


    entities = []


    if isinstance(
        raw_entities,
        list
    ):

        for item in raw_entities:

            if not isinstance(
                item,
                dict
            ):

                continue


            name = safe_text(
                item.get(
                    "name",
                    ""
                )
            )


            entity_type = safe_text(
                item.get(
                    "type",
                    "ENTITY"
                )
            )


            if not name:
                continue


            entities.append({

                "name":
                    name,

                "type":
                    entity_type
            })


    relationships = []


    if isinstance(
        raw_relationships,
        list
    ):

        for item in raw_relationships:

            if not isinstance(
                item,
                dict
            ):

                continue


            source = safe_text(
                item.get(
                    "source",
                    ""
                )
            )


            relation = safe_text(
                item.get(
                    "relation",
                    ""
                )
            )


            target = safe_text(
                item.get(
                    "target",
                    ""
                )
            )


            if (
                not source
                or
                not relation
                or
                not target
            ):

                continue


            relationships.append({

                "source":
                    source,

                "relation":
                    relation,

                "target":
                    target
            })


    return (
        entities,
        relationships,
        False
    )


# ============================================================
# NORMALIZE NEO4J VALUES
# ============================================================

def normalize_entity_name(
    value
):

    return safe_text(
        value
    )


def normalize_entity_type(
    value
):

    value = safe_text(
        value
    ).upper()


    value = re.sub(
        r"[^A-Z0-9_]",
        "_",
        value
    )


    if not value:

        value = "ENTITY"


    return value


def normalize_relation_type(
    value
):

    value = safe_text(
        value
    ).upper()


    value = re.sub(
        r"[^A-Z0-9_]",
        "_",
        value
    )


    if not value:

        value = "RELATED_TO"


    return value


# ============================================================
# CREATE CONSTRAINTS
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "CREATING NEO4J CONSTRAINTS"
)

print("=" * 90)


with driver.session() as session:

    session.run(
        """
        CREATE CONSTRAINT entity_name_unique
        IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE e.name IS UNIQUE
        """
    )


    session.run(
        """
        CREATE CONSTRAINT training_record_unique
        IF NOT EXISTS
        FOR (r:TrainingRecord)
        REQUIRE r.record_id IS UNIQUE
        """
    )


    session.run(
        """
        CREATE CONSTRAINT chunk_unique
        IF NOT EXISTS
        FOR (c:Chunk)
        REQUIRE c.chunk_id IS UNIQUE
        """
    )


print(
    "Neo4j constraints ready."
)


# ============================================================
# INSERT TRAINING RECORD
# ============================================================

def insert_training_record(
    record_id,
    fault_text,
    resolution_text,
    source_dataset
):

    start = time.perf_counter()


    with driver.session() as session:

        session.run(
            """
            MERGE (r:TrainingRecord {
                record_id: $record_id
            })

            SET
                r.fault_text =
                    $fault_text,

                r.resolution_text =
                    $resolution_text,

                r.source_dataset =
                    $source_dataset
            """,

            record_id=
                str(
                    record_id
                ),

            fault_text=
                fault_text,

            resolution_text=
                resolution_text,

            source_dataset=
                source_dataset
        )


    return (
        time.perf_counter()
        -
        start
    ) * 1000


# ============================================================
# INSERT CHUNK
# ============================================================

def insert_chunk(
    record_id,
    chunk_id,
    chunk_index,
    chunk_text,
    token_count
):

    start = time.perf_counter()


    with driver.session() as session:

        session.run(
            """
            MATCH (r:TrainingRecord {
                record_id: $record_id
            })

            MERGE (c:Chunk {
                chunk_id: $chunk_id
            })

            SET
                c.chunk_index =
                    $chunk_index,

                c.text =
                    $chunk_text,

                c.token_count =
                    $token_count

            MERGE
                (r)-[:HAS_CHUNK]->(c)
            """,

            record_id=
                str(
                    record_id
                ),

            chunk_id=
                chunk_id,

            chunk_index=
                int(
                    chunk_index
                ),

            chunk_text=
                chunk_text,

            token_count=
                int(
                    token_count
                )
        )


    return (
        time.perf_counter()
        -
        start
    ) * 1000


# ============================================================
# INSERT ENTITIES
# ============================================================

def insert_entities(
    entities,
    chunk_id
):

    start = time.perf_counter()

    inserted = 0


    with driver.session() as session:

        for entity in entities:

            name = normalize_entity_name(
                entity.get(
                    "name",
                    ""
                )
            )


            entity_type = normalize_entity_type(
                entity.get(
                    "type",
                    "ENTITY"
                )
            )


            if not name:
                continue


            session.run(
                """
                MATCH (c:Chunk {
                    chunk_id: $chunk_id
                })

                MERGE (e:Entity {
                    name: $name
                })

                ON CREATE SET
                    e.type =
                        $entity_type

                MERGE
                    (c)-[:MENTIONS]->(e)
                """,

                chunk_id=
                    chunk_id,

                name=
                    name,

                entity_type=
                    entity_type
            )


            inserted += 1


    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    return (
        inserted,
        elapsed_ms
    )


# ============================================================
# INSERT RELATIONSHIPS
# ============================================================

def insert_relationships(
    relationships,
    chunk_id
):

    start = time.perf_counter()

    inserted = 0


    with driver.session() as session:

        for relationship in relationships:

            source = normalize_entity_name(
                relationship.get(
                    "source",
                    ""
                )
            )


            target = normalize_entity_name(
                relationship.get(
                    "target",
                    ""
                )
            )


            relation_type = normalize_relation_type(
                relationship.get(
                    "relation",
                    "RELATED_TO"
                )
            )


            if (
                not source
                or
                not target
            ):

                continue


            query = f"""
            MATCH (c:Chunk {{
                chunk_id: $chunk_id
            }})

            MERGE (s:Entity {{
                name: $source
            }})

            MERGE (t:Entity {{
                name: $target
            }})

            MERGE
                (s)-[r:{relation_type}]->(t)

            SET
                r.last_chunk_id =
                    $chunk_id

            MERGE
                (c)-[:MENTIONS]->(s)

            MERGE
                (c)-[:MENTIONS]->(t)
            """


            session.run(
                query,

                chunk_id=
                    chunk_id,

                source=
                    source,

                target=
                    target
            )


            inserted += 1


    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000


    return (
        inserted,
        elapsed_ms
    )


# ============================================================
# PREPARE CHUNKS
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "PREPARING 256-TOKEN CHUNKS"
)

print("=" * 90)


chunking_start = time.perf_counter()


all_chunks = []


for row_index, row in train_df.iterrows():

    record_id = safe_text(
        row[
            "id"
        ]
    )


    source_dataset = safe_text(
        row.get(
            "source_dataset",
            ""
        )
    )


    fault_text = safe_text(
        row[
            "fault_text"
        ]
    )


    resolution_text = safe_text(
        row[
            "resolution_text"
        ]
    )


    document = clean_text(
        build_document(
            row
        )
    )


    chunks = chunk_text_by_tokens(
        document
    )


    for chunk_index, chunk_info in enumerate(
        chunks
    ):

        chunk_text = safe_text(
            chunk_info[
                "text"
            ]
        )


        token_count = int(
            chunk_info[
                "token_count"
            ]
        )


        chunk_id = make_chunk_id(
            record_id,
            chunk_index,
            chunk_text
        )


        all_chunks.append({

            "row_index":
                int(
                    row_index
                ),

            "record_id":
                record_id,

            "source_dataset":
                source_dataset,

            "fault_text":
                fault_text,

            "resolution_text":
                resolution_text,

            "chunk_index":
                int(
                    chunk_index
                ),

            "chunk_id":
                chunk_id,

            "chunk_text":
                chunk_text,

            "token_count":
                token_count
        })


chunking_time_seconds = (
    time.perf_counter()
    -
    chunking_start
)


chunks_df = pd.DataFrame(
    all_chunks
)


chunks_df.to_excel(
    CHUNK_FILE,
    index=False
)


print(
    "Training records:",
    len(train_df)
)


print(
    "Total chunks:",
    len(chunks_df)
)


print(
    "Mean chunks per record:",
    len(chunks_df)
    /
    len(train_df)
)


print(
    "Maximum chunk tokens:",
    int(
        chunks_df[
            "token_count"
        ].max()
    )
)


print(
    "Chunking time:",
    f"{chunking_time_seconds:.3f} sec"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if os.path.exists(
    CHECKPOINT_FILE
):

    previous_df = pd.read_csv(
        CHECKPOINT_FILE
    )


    build_results = (
        previous_df
        .to_dict(
            "records"
        )
    )


    completed_chunks = set(
        previous_df[
            "chunk_id"
        ]
        .astype(str)
    )


    print(
        "\nExisting completed chunks:",
        len(
            completed_chunks
        )
    )


else:

    build_results = []

    completed_chunks = set()


# ============================================================
# BUILD KNOWLEDGE GRAPH
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "BUILDING IFKG KNOWLEDGE GRAPH"
)

print("=" * 90)


print(
    "Model:",
    LLM_MODEL
)


print(
    "GPT calls per chunk: 1"
)


print(
    "Extraction: entities + relationships"
)


print(
    "Chunks:",
    len(chunks_df)
)


run_start = time.perf_counter()


for _, chunk_row in tqdm(

    chunks_df.iterrows(),

    total=
        len(
            chunks_df
        ),

    desc=
        "Building IFKG"
):


    chunk_id = safe_text(
        chunk_row[
            "chunk_id"
        ]
    )


    if chunk_id in completed_chunks:

        continue


    record_id = safe_text(
        chunk_row[
            "record_id"
        ]
    )


    source_dataset = safe_text(
        chunk_row[
            "source_dataset"
        ]
    )


    fault_text = safe_text(
        chunk_row[
            "fault_text"
        ]
    )


    resolution_text = safe_text(
        chunk_row[
            "resolution_text"
        ]
    )


    chunk_index = int(
        chunk_row[
            "chunk_index"
        ]
    )


    chunk_text = safe_text(
        chunk_row[
            "chunk_text"
        ]
    )


    token_count = int(
        chunk_row[
            "token_count"
        ]
    )


    chunk_start = time.perf_counter()


    # ========================================================
    # 1. INSERT TRAINING RECORD
    # ========================================================

    record_insert_ms = (
        insert_training_record(
            record_id,
            fault_text,
            resolution_text,
            source_dataset
        )
    )


    # ========================================================
    # 2. INSERT CHUNK
    # ========================================================

    chunk_insert_ms = insert_chunk(
        record_id,
        chunk_id,
        chunk_index,
        chunk_text,
        token_count
    )


    # ========================================================
    # 3. ONE GPT CALL
    #
    # Extract both entities and relationships
    # ========================================================

    (
        raw_output,
        extraction_llm_ms
    ) = extract_graph_information(
        chunk_text
    )


    (
        entities,
        relationships,
        parse_error
    ) = parse_graph_output(
        raw_output
    )


    # ========================================================
    # 4. INSERT ENTITIES
    # ========================================================

    (
        entities_inserted,
        entity_insert_ms
    ) = insert_entities(
        entities,
        chunk_id
    )


    # ========================================================
    # 5. INSERT RELATIONSHIPS
    # ========================================================

    (
        relationships_inserted,
        relationship_insert_ms
    ) = insert_relationships(
        relationships,
        chunk_id
    )


    # ========================================================
    # TOTAL TIMES
    # ========================================================

    neo4j_total_ms = (

        record_insert_ms
        +
        chunk_insert_ms
        +
        entity_insert_ms
        +
        relationship_insert_ms
    )


    chunk_total_ms = (
        time.perf_counter()
        -
        chunk_start
    ) * 1000


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "record_id":
            record_id,

        "chunk_id":
            chunk_id,

        "chunk_index":
            chunk_index,

        "source_dataset":
            source_dataset,

        "token_count":
            token_count,


        "entity_count":
            int(
                len(
                    entities
                )
            ),

        "relationship_count":
            int(
                len(
                    relationships
                )
            ),


        "entities_inserted":
            int(
                entities_inserted
            ),

        "relationships_inserted":
            int(
                relationships_inserted
            ),


        "parse_error":
            bool(
                parse_error
            ),


        "extraction_llm_ms":
            float(
                extraction_llm_ms
            ),


        "record_insert_ms":
            float(
                record_insert_ms
            ),

        "chunk_insert_ms":
            float(
                chunk_insert_ms
            ),

        "entity_insert_ms":
            float(
                entity_insert_ms
            ),

        "relationship_insert_ms":
            float(
                relationship_insert_ms
            ),

        "neo4j_total_ms":
            float(
                neo4j_total_ms
            ),


        "chunk_total_ms":
            float(
                chunk_total_ms
            ),


        "entities_json":
            json.dumps(
                entities,
                ensure_ascii=False
            ),


        "relationships_json":
            json.dumps(
                relationships,
                ensure_ascii=False
            )
    }


    build_results.append(
        result
    )


    # ========================================================
    # CHECKPOINT AFTER EVERY CHUNK
    # ========================================================

    pd.DataFrame(
        build_results
    ).to_csv(
        CHECKPOINT_FILE,
        index=False
    )


# ============================================================
# CURRENT RUN WALL TIME
# ============================================================

current_run_wall_time_seconds = (
    time.perf_counter()
    -
    run_start
)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    build_results
)


results_df.to_excel(
    EXTRACTION_FILE,
    index=False
)


# ============================================================
# GRAPH STATISTICS
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "GRAPH STATISTICS"
)

print("=" * 90)


with driver.session() as session:

    total_nodes = session.run(
        """
        MATCH (n)
        RETURN count(n) AS count
        """
    ).single()[
        "count"
    ]


    entity_nodes = session.run(
        """
        MATCH (e:Entity)
        RETURN count(e) AS count
        """
    ).single()[
        "count"
    ]


    record_nodes = session.run(
        """
        MATCH (r:TrainingRecord)
        RETURN count(r) AS count
        """
    ).single()[
        "count"
    ]


    chunk_nodes = session.run(
        """
        MATCH (c:Chunk)
        RETURN count(c) AS count
        """
    ).single()[
        "count"
    ]


    total_relationships = session.run(
        """
        MATCH ()-[r]->()
        RETURN count(r) AS count
        """
    ).single()[
        "count"
    ]


# ============================================================
# OFFLINE CONSTRUCTION TIMES
# ============================================================

total_extraction_llm_seconds = (
    results_df[
        "extraction_llm_ms"
    ].sum()
    /
    1000
)


total_neo4j_seconds = (
    results_df[
        "neo4j_total_ms"
    ].sum()
    /
    1000
)


total_chunk_processing_seconds = (
    results_df[
        "chunk_total_ms"
    ].sum()
    /
    1000
)


total_offline_construction_seconds = (
    chunking_time_seconds
    +
    total_chunk_processing_seconds
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "method":
        "IFKG",

    "llm_model":
        LLM_MODEL,


    "training_records":
        int(
            len(
                train_df
            )
        ),


    "chunk_unit":
        "tokens",

    "chunk_size_tokens":
        int(
            CHUNK_SIZE
        ),

    "chunk_overlap_tokens":
        int(
            CHUNK_OVERLAP
        ),


    "total_chunks":
        int(
            len(
                chunks_df
            )
        ),


    "mean_chunks_per_record":
        float(
            len(
                chunks_df
            )
            /
            len(
                train_df
            )
        ),


    "maximum_chunk_tokens":
        int(
            chunks_df[
                "token_count"
            ].max()
        ),


    "extraction_calls_per_chunk":
        1,

    "extraction_type":
        "joint entities and relationships",


    "chunking_time_seconds":
        float(
            chunking_time_seconds
        ),


    "llm_extraction_time_seconds":
        float(
            total_extraction_llm_seconds
        ),


    "neo4j_insertion_time_seconds":
        float(
            total_neo4j_seconds
        ),


    "chunk_processing_time_seconds":
        float(
            total_chunk_processing_seconds
        ),


    "total_offline_construction_time_seconds":
        float(
            total_offline_construction_seconds
        ),


    "current_run_wall_time_seconds":
        float(
            current_run_wall_time_seconds
        ),


    "mean_llm_extraction_ms":
        float(
            results_df[
                "extraction_llm_ms"
            ].mean()
        ),


    "mean_neo4j_insertion_ms":
        float(
            results_df[
                "neo4j_total_ms"
            ].mean()
        ),


    "mean_chunk_processing_ms":
        float(
            results_df[
                "chunk_total_ms"
            ].mean()
        ),


    "total_extracted_entities":
        int(
            results_df[
                "entity_count"
            ].sum()
        ),


    "total_extracted_relationships":
        int(
            results_df[
                "relationship_count"
            ].sum()
        ),


    "parse_errors":
        int(
            results_df[
                "parse_error"
            ].sum()
        ),


    "graph_total_nodes":
        int(
            total_nodes
        ),


    "graph_entity_nodes":
        int(
            entity_nodes
        ),


    "graph_training_record_nodes":
        int(
            record_nodes
        ),


    "graph_chunk_nodes":
        int(
            chunk_nodes
        ),


    "graph_total_relationships":
        int(
            total_relationships
        )
}


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# CLOSE NEO4J
# ============================================================

driver.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "IFKG CONSTRUCTION COMPLETE"
)

print("=" * 90)


print("\nDATA")


print(
    "Training records:",
    summary[
        "training_records"
    ]
)


print(
    "Total chunks:",
    summary[
        "total_chunks"
    ]
)


print(
    "Chunk size:",
    summary[
        "chunk_size_tokens"
    ],
    "tokens"
)


print(
    "Overlap:",
    summary[
        "chunk_overlap_tokens"
    ],
    "tokens"
)


print(
    "GPT extraction calls/chunk:",
    summary[
        "extraction_calls_per_chunk"
    ]
)


print("\nEXTRACTION")


print(
    "Entities:",
    summary[
        "total_extracted_entities"
    ]
)


print(
    "Relationships:",
    summary[
        "total_extracted_relationships"
    ]
)


print(
    "Parse errors:",
    summary[
        "parse_errors"
    ]
)


print("\nGRAPH")


print(
    "Total nodes:",
    summary[
        "graph_total_nodes"
    ]
)


print(
    "Entity nodes:",
    summary[
        "graph_entity_nodes"
    ]
)


print(
    "TrainingRecord nodes:",
    summary[
        "graph_training_record_nodes"
    ]
)


print(
    "Chunk nodes:",
    summary[
        "graph_chunk_nodes"
    ]
)


print(
    "Total graph relationships:",
    summary[
        "graph_total_relationships"
    ]
)


print("\nOFFLINE CONSTRUCTION TIME")


print(
    "Chunking:",
    f"{summary['chunking_time_seconds']:.3f}",
    "sec"
)


print(
    "GPT extraction:",
    f"{summary['llm_extraction_time_seconds']:.3f}",
    "sec"
)


print(
    "Neo4j insertion:",
    f"{summary['neo4j_insertion_time_seconds']:.3f}",
    "sec"
)


print(
    "Total chunk processing:",
    f"{summary['chunk_processing_time_seconds']:.3f}",
    "sec"
)


print(
    "Total offline construction:",
    f"{summary['total_offline_construction_time_seconds']:.3f}",
    "sec"
)


print("\nFILES")


print(
    CHUNK_FILE
)


print(
    CHECKPOINT_FILE
)


print(
    EXTRACTION_FILE
)


print(
    SUMMARY_FILE
)