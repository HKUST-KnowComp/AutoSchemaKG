# Parallel Knowledge Graph Generation

This guide demonstrates how to parallelize knowledge graph extraction and concept generation using multiple LLM instances running on different ports. By distributing the workload across multiple LLM servers, you can significantly speed up the extraction process for large datasets.

> **Note:** Complete example scripts are available in the [GitHub repository](https://github.com/HKUST-KnowComp/AutoSchemaKG/tree/main/example/example_scripts/parallel_generation).

## Overview

The parallel generation approach divides your dataset into **shards** and processes each shard with a separate LLM instance. Each instance runs on a different port, allowing multiple extractions to happen simultaneously.

The parallel workflow includes two stages:
1. **Triple Extraction**: Extract knowledge graph triples from raw text in parallel
2. **Concept Generation**: Generate concepts and mappings for extracted entities in parallel

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Original Dataset                         │
└─────────────────────────────────────────────────────────────┘
                         │
                         ├─────────────────────────┐
                         │                         │
                         ▼                         ▼
              ┌─────────────────┐      ┌─────────────────┐
              │   Shard 0       │      │   Shard 1       │
              │   Chunks        │      │   Chunks        │
              └─────────────────┘      └─────────────────┘
                         │                         │
                         ▼                         ▼
              ┌─────────────────┐      ┌─────────────────┐
              │ LLM Instance 0  │      │ LLM Instance 1  │
              │ (localhost:8135)│      │ (localhost:8136)│
              └─────────────────┘      └─────────────────┘
                         │                         │
                         ▼                         ▼
              ┌─────────────────┐      ┌─────────────────┐
              │   Shard 0       │      │   Shard 1       │
              │    Triple       │      │    Triple       │
              │  Extraction     │      │   Extraction    │
              └─────────────────┘      └─────────────────┘
                         │                         │
                         ▼                         ▼
              ┌─────────────────┐      ┌─────────────────┐
              │   Shard 0       │      │   Shard 1       │
              │    Concept      │      │    Concept      │
              │   Generation    │      │   Generation    │
              └─────────────────┘      └─────────────────┘
                         │                         │
                         ▼                         ▼
                         │                         │
                         └─────────┬───────────────┘
                                   ▼
                        ┌─────────────────────┐
                        │   Merged Results    │
                        └─────────────────────┘
```

## Prerequisites

### Starting Multiple vLLM Instances

Start multiple LLM server instances on different ports. For 3 parallel instances:

```bash
# Terminal 1: Port 8135
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8135 --max-model-len 8192 --gpu-memory-utilization 0.9

# Terminal 2: Port 8136
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8136 --max-model-len 8192 --gpu-memory-utilization 0.9

# Terminal 3: Port 8137
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8137 --max-model-len 8192 --gpu-memory-utilization 0.9
```

**Tips:**
- Use `tmux` or `screen` to manage multiple terminals
- Adjust `--gpu-memory-utilization` based on available GPU memory
- Use `CUDA_VISIBLE_DEVICES` to assign different GPUs to each instance

## Example Scripts

The example implementation includes the following scripts (available on [GitHub](https://github.com/HKUST-KnowComp/AutoSchemaKG/tree/main/example/example_scripts/parallel_generation)):

- **`1_slice_kg_extraction.py`**: Triple extraction script for a single shard
- **`1_slice_kg_extraction.sh`**: Launches parallel triple extraction across all shards
- **`2_concept_generation.py`**: Concept generation script for a single shard
- **`2_concept_generation.sh`**: Launches parallel concept generation across all shards
- **`3_final_to_graphml.py`**: Final GraphML conversion script
- **`run_full_pipeline.sh`**: Master script that runs both stages sequentially

## Configuration

### Shell Script Configuration

Edit the variables at the top of the shell scripts (`1_slice_kg_extraction.sh` and `2_concept_generation.sh`):

```bash
TOTAL_SHARDS=3                    # Number of parallel processes
BASE_PORT=8135                    # Starting port number
LOG_DIR="/path/to/log"           # Directory for log files
SCRIPT_DIR="/path/to/scripts"    # Directory containing Python scripts
KEYWORD="musique"                 # Dataset keyword for filtering
```

### Port Assignment

Ports are automatically assigned based on shard number: `BASE_PORT + shard_number`

Example with `BASE_PORT=8135` and `TOTAL_SHARDS=3`:
- Shard 0 → Port 8135
- Shard 1 → Port 8136
- Shard 2 → Port 8137

### Python Script Configuration

The Python scripts (`1_slice_kg_extraction.py` and `2_concept_generation.py`) support the following command-line arguments:

```python
parser.add_argument("--shard", type=int, default=0, 
                    help="Shard number to process")
parser.add_argument("--total_shards", type=int, default=1,
                    help="Total number of shards")
parser.add_argument("--port", type=int, default=8135,
                    help="Port number for the LLM API")
parser.add_argument("--keyword", type=str, default="musique",
                    help="Keyword for filtering data files")
```

## Usage

### Quick Start (Recommended)

Clone the repository and navigate to the parallel generation examples:

```bash
cd example/example_scripts/parallel_generation

# 1. Start LLM instances on ports 8135, 8136, 8137 (see Prerequisites)
# 2. Configure TOTAL_SHARDS and BASE_PORT in the scripts
# 3. Run the complete pipeline
./run_full_pipeline.sh
```

### Manual Workflow

If you prefer to run each stage separately:

#### Stage 1: Triple Extraction

```bash
chmod +x 1_slice_kg_extraction.sh
./1_slice_kg_extraction.sh

# Monitor progress in real-time
tail -f log/shard_*.log
```

#### Stage 2: Concept Generation

After triple extraction completes:

```bash
chmod +x 2_concept_generation.sh
./2_concept_generation.sh

# Monitor progress in real-time
tail -f log/concept_shard_*.log
```

## Implementation Details

### Triple Extraction Script

Here's a simplified example of the parallel triple extraction implementation:

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator, GenerationConfig
from openai import OpenAI
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--total_shards", type=int, default=1)
parser.add_argument("--port", type=int, default=8135)
parser.add_argument("--keyword", type=str, default="musique")
args = parser.parse_args()

# Initialize LLM client pointing to specific port
client = OpenAI(
    base_url=f"http://localhost:{args.port}/v1",
    api_key="EMPTY"
)

# Configure generation
gen_config = GenerationConfig(temperature=0.5, max_tokens=16384)
triple_generator = LLMGenerator(
    client, 
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_workers=64, 
    default_config=gen_config
)

# Configure KG extraction for this specific shard
kg_extraction_config = ProcessingConfig(
    model_path="Qwen2.5-7B-Instruct",
    data_directory="/path/to/data",
    filename_pattern=args.keyword,
    batch_size_triple=64,
    output_directory=f'/path/to/output',
    current_shard_triple=args.shard,      # Current shard number
    total_shards_triple=args.total_shards, # Total number of shards
    max_new_tokens=16384
)

# Run extraction for this shard
kg_extractor = KnowledgeGraphExtractor(
    model=triple_generator, 
    config=kg_extraction_config
)
kg_extractor.run_extraction()
```

### Shell Script Launcher

The shell script launches multiple Python processes in parallel:

```bash
#!/bin/bash

TOTAL_SHARDS=3
BASE_PORT=8135
LOG_DIR="./log"

run_shard() {
    local shard_num=$1
    local port=$((BASE_PORT + shard_num))
    
    python 1_slice_kg_extraction.py \
        --shard "${shard_num}" \
        --total_shards "${TOTAL_SHARDS}" \
        --port "${port}" \
        --keyword "musique" \
        > "${LOG_DIR}/shard_${shard_num}.log" 2>&1 &
}

# Launch all shards in parallel
for ((i=0; i<TOTAL_SHARDS; i++)); do
    run_shard $i
    sleep 1  # Stagger startup
done

# Wait for all processes to complete
wait
```

## Performance Considerations

### Speedup Calculation

With N shards running in parallel, the theoretical speedup is close to N×, though actual performance depends on:
- GPU memory availability
- Network bandwidth (if using remote LLM APIs)
- Disk I/O for reading input and writing output
- Load balancing across shards

## See Also

- [Full pipeline example notebook](https://github.com/HKUST-KnowComp/AutoSchemaKG/blob/main/example/atlas_full_pipeline.ipynb)
- [Triple extraction documentation](https://github.com/HKUST-KnowComp/AutoSchemaKG/tree/main/atlas_rag/kg_construction)
- [vLLM deployment guide](https://docs.vllm.ai/en/latest/)
