# File Formats & Data Loading

AutoSchemaKG supports various input formats for knowledge graph construction and benchmarking.

## Input Data Formats

### 1. Raw Text (Corpus)

For building a Knowledge Graph from scratch, the system typically expects a directory of text files or a structured JSON/JSONL corpus.

**JSONL Format (Recommended):**
Each line is a JSON object representing a document.
```json
{"id": "title_1", "text": "Full text content of the document...", "metadata": {"lang": "en"}}
{"id": "title_2", "text": "Another document content...", "metadata": {"lang": "en"}}
```

**Directory of Text Files:**
You can also point `data_directory` to a folder containing `.txt` or `.md` files. The filename is often used as the document ID.

### 2. Benchmark Datasets

For evaluation, the system supports standard QA dataset formats like HotpotQA, 2WikiMultihopQA, and MuSiQue.

**Standard QA Format (JSON):**
```json
[
  {
    "_id": "5a7a06935542990198eaf050",
    "question": "Which magazine was published first, Arthur's Magazine or First for Women?",
    "answer": "Arthur's Magazine",
    "supporting_facts": [["Arthur's Magazine", 0], ["First for Women", 0]],
    "context": [
      ["Arthur's Magazine", ["Arthur's Magazine (1844–1846) was an American literary periodical..."]],
      ["First for Women", ["First for Women is a woman's magazine published by Bauer Media Group..."]]
    ]
  }
]
```

## PDF & Document Processing

AutoSchemaKG includes utilities to convert unstructured documents (PDFs) into Markdown/Text for processing.

> **Note:** Complete example scripts and configuration details are available in the [GitHub repository](https://github.com/HKUST-KnowComp/AutoSchemaKG/tree/main/example/pdf_md_conversion).

### Workflow

The general pipeline for PDFs is:

1.  **Convert to Markdown**: Use the provided tools (based on `marker-pdf`) to extract text and structure.
2.  **Convert to JSON**: Transform the Markdown output into the JSONL format required by AutoSchemaKG.
3.  **KG Extraction**: Run the `KnowledgeGraphExtractor` on the processed data.

### Quick Start

1.  **Install Dependencies**:
    ```bash
    # Create a separate environment recommended
    conda create --name pdf-marker pip python=3.10
    conda activate pdf-marker
    pip install 'marker-pdf[full]' google-genai
    ```

2.  **Configure**: Edit `config.yaml` to set your LLM service (Azure OpenAI or Gemini) and input/output paths.

3.  **Run Conversion**:
    ```bash
    # Convert PDF to Markdown
    bash run.sh
    
    # Convert Markdown to JSON (from AutoSchemaKG root)
    python -m atlas_rag.kg_construction.utils.md_processing.markdown_to_json \
        --input example_data/md_data \
        --output example_data
    ```

## Output Formats

### 1. GraphML

The final Knowledge Graph is often exported as `.graphml`, which can be opened in networkx.

### 2. CSV (Triples & Concepts)

Intermediate results are stored as CSVs:
-   **Triples CSV**: `triple_nodes, triple_edges, text_nodes, text_edges`
-   **Concepts CSV**: `triple_edges, concept_edges, concept_nodes`

### 3. NetworkX / GraphDatabase

For programmatic access, graphs are manipulated as NetworkX objects or stored as Graph Database instances.
