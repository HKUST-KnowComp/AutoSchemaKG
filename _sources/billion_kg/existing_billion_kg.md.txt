# Hosting Existing Billion-Scale Knowledge Graphs

This guide explains how to host and use our pre-constructed billion-scale knowledge graphs (Wiki, Pes2o, Common Crawl) using Neo4j Community Edition.

## Option 1: Quick Start (Pre-built Server)

The easiest way to get started is to download the pre-configured Neo4j server with the data already imported.

1.  **Download**: Get the zipped server from our [Huggingface Dataset](https://huggingface.co/datasets/gzone0111/AutoSchemaKG).

    You can download the specific dataset you need:

    | Dataset | File Name | Size | Description |
    |---------|-----------|------|-------------|
    | **Common Crawl** | `neo4j-server-cc.zip` | ~213 GB | Large-scale web crawl data. |
    | **Wiki** | `neo4j-server-wiki.zip` | ~74.1 GB | Wikipedia-based knowledge graph. |
    | **Pes2o** | `neo4j-server-pes2o.zip` | ~53.2 GB | Academic papers dataset. |

    **Download via CLI:**
    ```bash
    # Install huggingface-cli
    pip install huggingface_hub

    # Download specific file (e.g., Wiki)
    hf download gzone0111/AutoSchemaKG 'ATLAS Neo4j Server Zip/neo4j-server-wiki.zip' --local-dir . --repo-type dataset
    ```

2.  **Unzip**: Extract the downloaded file.
3.  **Run**: Start the server using the bash file in the unzipped folder.

```bash
# Replace {dataset-name} with wiki, pes2o, or cc
./neo4j-server-{dataset-name}/bin/neo4j start
```

## Option 2: Build from Source

If you prefer to build the database yourself from the raw CSV files, follow these steps.

### 1. Setup Neo4j

We provide scripts to download Neo4j Community Edition, install required plugins (APOC, GDS), and configure the environment.

```bash
cd neo4j_scripts
sh get_neo4j_cc.sh    # For Common Crawl
sh get_neo4j_pes2o.sh # For Pes2o
sh get_neo4j_wiki.sh  # For Wiki
```

**Configuration**:
- Copy `AutoschemaKG/neo4j_scripts/neo4j.conf` to `neo4j-server-{dataset}/conf/neo4j.conf`.
- Update `dbms.default_database` to your desired dataset name (e.g., `wiki-csv-json-text`).
- Configure ports (Bolt, HTTP, HTTPS) to avoid conflicts if running multiple servers.

### 2. Prepare Data

1.  **Download Data**: Download the CSV dumps from our [Huggingface Dataset](https://huggingface.co/datasets/gzone0111/AutoSchemaKG/tree/main). You need to download all the zip files from the `ATLAS Neo4j Dump` folder.

    **Download via CLI:**
    ```bash
    hf download gzone0111/AutoSchemaKG --include "ATLAS Neo4j Dump/*" --local-dir . --repo-type dataset
    ```

2.  **Decompress**: Run the `decompress_csv_files.sh` script to decompress all zip files in parallel to a `decompressed` directory. Then move the files to the `./import` directory of your Neo4j server.

    **Storage Requirements:**
    Ensure you have sufficient disk space. Approximate sizes after import and decompression:
    
    | Directory | Size |
    |-----------|------|
    | `./neo4j-server-wiki` | 342 GB |
    | `./neo4j-server-cc` | 907 GB |
    | `./neo4j-server-pes2o` | 249 GB |
    | `./import` (Raw CSVs) | 2.3 TB |

3.  **Add Numeric IDs**: (Optional if using provided processed CSVs) If building from raw extraction output, you may need to add numeric IDs for vector indexing.

### 3. Import Data

Use `neo4j-admin import` to load the CSVs. This is much faster than Cypher `LOAD CSV` for large datasets.

**Example: Importing Wiki Graph**

```bash
./neo4j-server-wiki/bin/neo4j-admin database import full wiki-csv-json-text \
    --nodes=./import/text_nodes_en_simple_wiki_v0_from_json_with_numeric_id.csv \
    ./import/triple_nodes_en_simple_wiki_v0_from_json_without_emb_with_numeric_id.csv \
    ./import/concept_nodes_en_simple_wiki_v0_from_json_without_emb.csv \
    --relationships=./import/text_edges_en_simple_wiki_v0_from_json.csv \
    ./import/triple_edges_en_simple_wiki_v0_from_json_without_emb_full_concept_with_numeric_id.csv \
    ./import/concept_edges_en_simple_wiki_v0_from_json_without_emb.csv \
    --overwrite-destination \
    --multiline-fields=true \
    --id-type=string \
    --verbose --skip-bad-relationships=true
```

*(Refer to the notebook or repository scripts for Pes2o and CC import commands)*

## Hosting the RAG API

Once the Neo4j server is running, you can host the ATLAS RAG API to perform retrieval.

```bash
python example/example_scripts/neo4j_kg/atlas_api_server_demo.py
```

Ensure you configure the `LargeKGConfig` in the script to point to your Neo4j instance (URI, username, password) and the correct FAISS indices.

## Usage Example

You can query the hosted RAG API using an OpenAI-compatible client. (ref: `example/example_scripts/neo4j_kg/atlas_api_client_demo.py`)

```python
from openai import OpenAI

# Point to your hosted API
base_url = "http://0.0.0.0:10089/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)

message = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions based on the knowledge graph.",
    },
    {
        "role": "user",
        "content": "Question: Who is Alex Mercer?",
    }
]

response = client.chat.completions.create(
    model="llama",
    messages=message,
    max_tokens=2048,
    temperature=0.5,
    extra_body = {
        "retriever_config": {
            "topN": 5,
            "number_of_source_nodes_per_ner": 1,
            "sampling_area": 10 
        }
    }
)

print(response.choices[0].message.content)
```
