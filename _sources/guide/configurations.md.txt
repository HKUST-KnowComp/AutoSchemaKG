# Configuration Guide

AutoSchemaKG uses several configuration classes to manage settings for different parts of the pipeline: knowledge graph construction, LLM generation, retrieval, and benchmarking.

## Knowledge Graph Construction

The `ProcessingConfig` class controls the pipeline for extracting triples and generating concepts from text.

```python
from atlas_rag.kg_construction.triple_config import ProcessingConfig
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to the model or model identifier. |
| `data_directory` | str | Required | Directory containing input text files. |
| `filename_pattern` | str | Required | Pattern to filter input files (e.g., "musique"). |
| `output_directory` | str | `"./generation_result_debug"` | Directory to save output files. |
| `batch_size_triple` | int | `16` | Batch size for triple extraction. |
| `batch_size_concept` | int | `64` | Batch size for concept generation. |
| `total_shards_triple` | int | `1` | Total number of shards for parallel triple extraction. |
| `current_shard_triple` | int | `0` | Current shard index for triple extraction. |
| `total_shards_concept` | int | `1` | Total number of shards for parallel concept generation. |
| `current_shard_concept` | int | `0` | Current shard index for concept generation. |
| `debug_mode` | bool | `False` | Enable debug logging. |
| `resume_from` | int | `0` | Index to resume processing from. |
| `record` | bool | `False` | Whether to record processing metrics. |
| `max_workers` | int | `8` | Number of parallel workers. (Useful in speeding up when using OpenAI API) |
| `remove_doc_spaces` | bool | `False` | Whether to remove spaces from documents during preprocessing. |
| `allow_empty` | bool | `True` | Allow empty results without raising errors. |
| `include_concept` | bool | `True` | Whether to perform concept generation after triple extraction. |
| `deduplicate_text` | bool | `False` | Whether to deduplicate input text. |
| `triple_extraction_prompt_path` | str | `None` | Path to custom triple extraction prompt. |
| `triple_extraction_schema_path` | str | `None` | Path to custom triple extraction schema. |
| `benchmark` | bool | `False` | Enable benchmarking mode (e.g., for GPU hours). |

## LLM Generation

The `GenerationConfig` class provides a unified interface for configuring LLM generation parameters across different backends (OpenAI, vLLM, HuggingFace, etc.).

```python
from atlas_rag.llm_generator import GenerationConfig
```

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | int | `8192` | Maximum number of tokens to generate. |
| `temperature` | float | `0.7` | Sampling temperature. |
| `top_p` | float | `None` | Nucleus sampling probability. |
| `top_k` | int | `None` | Top-k sampling. |
| `do_sample` | bool | `True` | Whether to use sampling or greedy decoding. |
| `seed` | int | `None` | Random seed for reproducibility. |
| `stop` | str/List[str] | `None` | Stop sequences. |

### Repetition Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frequency_penalty` | float | `None` | Penalize new tokens based on their existing frequency (-2.0 to 2.0). |
| `presence_penalty` | float | `None` | Penalize new tokens based on whether they appear in the text (-2.0 to 2.0). |
| `repetition_penalty` | float | `None` | Penalty for repeating tokens (typically 1.0-2.0). |

### Advanced Sampling (vLLM/SGLang)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_p` | float | `None` | Minimum probability threshold. |
| `use_beam_search` | bool | `False` | Whether to use beam search. |
| `ignore_eos` | bool | `False` | Whether to ignore the EOS token. |
| `skip_special_tokens` | bool | `True` | Whether to skip special tokens in output. |

### Guided Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `guided_json` | Union[str, Dict] | `None` | JSON schema for guided generation. |
| `guided_regex` | str | `None` | Regex pattern for guided generation. |
| `guided_choice` | List[str] | `None` | List of allowed choices. |
| `guided_grammar` | str | `None` | Context-free grammar for guided generation. |

### OpenAI Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_format` | Dict | `None` | Format of the response (e.g., `{"type": "json_object"}`). |
| `tools` | List[Dict] | `None` | List of tools/functions. |
| `tool_choice` | Union[str, Dict] | `None` | Tool choice strategy. |

## Retrieval & Inference

The `InferenceConfig` class controls the settings for the retrieval and reasoning pipeline.

```python
from atlas_rag.retriever.inference_config import InferenceConfig
```

### General Retrieval

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keyword` | str | `"musique"` | Dataset keyword. |
| `topk` | int | `5` | Number of top passages/results to retrieve. |

### Think on Graph (ToG)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic_prune` | bool | `True` | Whether to prune topics. |
| `temperature_exploration` | float | `0.0` | Temperature for exploration phase. |
| `temperature_reasoning` | float | `0.0` | Temperature for reasoning phase. |
| `num_sents_for_reasoning` | int | `10` | Number of sentences to use for reasoning. |
| `remove_unnecessary_rel` | bool | `True` | Whether to remove unnecessary relationships. |
| `Dmax` | int | `3` | Maximum depth for search. |
| `Wmax` | int | `3` | Maximum width for search. |

### HippoRAG 1&2

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_filter_edges` | bool | `True` | Whether to filter edges. |
| `hipporag_mode` | str | `"query2edge"` | Mode: `"query2edge"` or `"query2node"`. |
| `weight_adjust` | float | `1.0` | Weight adjustment factor. |
| `topk_edges` | int | `50` | Number of top edges to retrieve before filtering. |
| `topk_nodes` | int | `10` | Number of top nodes to retrieve for graph exploration. |  
| `ppr_alpha` | float | `0.99` | Damping factor for Presonalized PageRank (PPR). |
| `ppr_max_iter` | int | `2000` | Maximum iterations for PPR. |
| `ppr_tol` | float | `1e-7` | Tolerance for PPR convergence. |

## Benchmarking

The `BenchMarkConfig` class configures the evaluation and benchmarking process.

```python
from atlas_rag.evaluation.benchmark import BenchMarkConfig
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | `"hotpotqa"` | Name of the dataset. |
| `question_file` | str | `"hotpotqa"` | Path to the question file. |
| `graph_file` | str | `"hotpotqa_concept.graphml"` | Path to the graph file (unused in some contexts). |
| `include_events` | bool | `False` | Whether to include events in evaluation. |
| `include_concept` | bool | `False` | Whether to include concepts in evaluation. |
| `reader_model_name` | str | `"meta-llama/Llama-2-7b-chat-hf"` | Model used for reading/answering. |
| `encoder_model_name` | str | `"nvidia/NV-Embed-v2"` | Model used for encoding/embedding. |
| `number_of_samples` | int | `-1` | Number of samples to evaluate (-1 for all). |
| `react_max_iterations` | int | `5` | Maximum iterations for ReAct agent. |
| `result_dir` | str | `"./result"` | Directory to store results. |
| `upper_bound_mode` | bool | `False` | Enable upper bound evaluation mode. |
| `topN` | int | `5` | Number of top passages to retrieve for evaluation. |
