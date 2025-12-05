# Retrieval Augmentation

AutoSchemaKG provides a suite of retrievers designed to work with the constructed Knowledge Graph. These retrievers implement different strategies for finding relevant information, ranging from simple vector similarity to complex multi-hop reasoning.

## Base Classes

All retrievers inherit from `BaseRetriever` or its specialized subclasses:

-   **`BaseRetriever`**: The abstract base class for all retrieval operations.
-   **`BaseEdgeRetriever`**: Specialized for retrieving edges (triples) from the graph.
-   **`BasePassageRetriever`**: Specialized for retrieving text passages or documents.

## Available Retrievers

### 1. Simple Retrievers

These provide baseline retrieval capabilities using vector similarity.

#### `SimpleGraphRetriever`
Retrieves the most relevant edges (triples) from the Knowledge Graph based on the semantic similarity between the query and the edge embeddings.

```python
from atlas_rag.retriever import SimpleGraphRetriever

retriever = SimpleGraphRetriever(
    llm_generator=llm_generator,
    sentence_encoder=sentence_encoder,
    data=data  # Dictionary containing KG, embeddings, and indices
)

edges, _ = retriever.retrieve("query", topN=5)
```

#### `SimpleTextRetriever`
Performs standard dense passage retrieval. It finds the most relevant text chunks/documents by comparing the query embedding with passage embeddings.

```python
from atlas_rag.retriever import SimpleTextRetriever

retriever = SimpleTextRetriever(
    passage_dict=passage_dict,
    sentence_encoder=sentence_encoder,
    data=data
)

passages, ids = retriever.retrieve("query", topN=5)
```

### 2. HippoRAG

HippoRAG is an advanced retrieval method that leverages the Knowledge Graph structure to improve retrieval accuracy, especially for multi-hop questions. It uses Personalized PageRank (PPR) to propagate relevance scores across the graph.

#### `HippoRAGRetriever` & `HippoRAG2Retriever`
These retrievers implement the HippoRAG algorithm. They support different modes for mapping queries to graph components:

-   **`query2edge`**: Maps the query directly to relevant edges in the graph.
-   **`query2node`**: Maps the query to relevant nodes.
-   **`ner2node`**: Extracts named entities from the query and maps them to graph nodes.

```python
from atlas_rag.retriever import HippoRAG2Retriever
from atlas_rag.retriever.inference_config import InferenceConfig

config = InferenceConfig(
    hipporag_mode="query2edge",
    ppr_alpha=0.99
)

retriever = HippoRAG2Retriever(
    llm_generator=llm_generator,
    sentence_encoder=sentence_encoder,
    data=data,
    inference_config=config
)

# Returns relevant passages identified via graph propagation
passages, ids = retriever.retrieve("query", topN=5)
```

### 3. Think on Graph (ToG)

The ToG (Think-on-Graph) retriever implement a reasoning-based approach where an LLM explores the graph step-by-step to find answer paths.

#### `TogRetriever`, `TogV3Retriever (Turbo Version)` 
These classes implement different versions of the ToG algorithm. The general process involves:
1.  **Entity Extraction**: Identifying topic entities in the query.
2.  **Search**: Exploring paths from these entities in the Knowledge Graph.
3.  **Prune**: Using the LLM or scoring functions to prune irrelevant paths.
4.  **Reasoning**: Checking if the current paths are sufficient to answer the question.

```python
from atlas_rag.retriever import TogRetriever

retriever = TogRetriever(
    llm_generator=llm_generator,
    sentence_encoder=sentence_encoder,
    data=data,
    inference_config=config
)

# Returns a generated answer or reasoning path
result = retriever.retrieve("query", topN=5)
```

### 4. Subgraph Retriever

#### `SubgraphRetriever`
Retrieves a multi-hop subgraph surrounding the entities identified in the query. This is useful for providing a broader context to the LLM.

```python
from atlas_rag.retriever import SubgraphRetriever

retriever = SubgraphRetriever(
    llm_generator=llm_generator,
    sentence_encoder=sentence_encoder,
    data=data,
    config=config
)

subgraph_str = retriever.retrieve("query")
```

### 5. Upper Bound Retriever

#### `UpperBoundRetriever`
A utility retriever used for benchmarking. It assumes access to the "gold" (correct) passages and retrieves them directly, filling the remaining slots with random passages if necessary. This helps in establishing an upper bound for performance during evaluation.

## Configuration

Retrievers are configured using the `InferenceConfig` class. Key parameters include:

-   `topk`: Number of results to retrieve.
-   `Dmax`: Maximum depth for graph search (ToG).
-   `Wmax`: Maximum width/branching factor (ToG).
-   `hipporag_mode`: Strategy for HippoRAG (`query2edge`, `query2node`, etc.).
-   `ppr_alpha`: Damping factor for PageRank.

See the [Configuration Guide](configurations.md) for full details.
