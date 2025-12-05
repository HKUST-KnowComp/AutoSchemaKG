# Quick Start

This guide walks you through the full pipeline of creating a Knowledge Graph (KG) and performing Retrieval-Augmented Generation (RAG) using `AutoSchemaKG`.

## Prerequisites

Ensure you have the necessary packages installed and your environment configured. You will need an LLM endpoint (e.g., OpenAI compatible) and an embedding model.

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI

# Initialize LLM Client (assuming you are hosting LLM instance with vllm)
client = OpenAI(base_url="http://0.0.0.0:8129/v1", api_key="EMPTY")
llm_generator = LLMGenerator(client=client, model_name="Qwen/Qwen2.5-7B-Instruct")
```

## 1. Knowledge Graph Construction

The construction process involves extracting triples from your data, converting them to CSV, and finally to GraphML format.

### Configuration

Set up the processing configuration. Point `data_directory` to your source documents.

```python
kg_extraction_config = ProcessingConfig(
      model_path="Qwen/Qwen2.5-7B-Instruct",
      data_directory='example/example_data',
      filename_pattern='Dulce', # Only process files containing this substring in their filename
      output_directory='example/generated/test_data',
)

kg_extractor = KnowledgeGraphExtractor(model=llm_generator, config=kg_extraction_config)
```

### Execution

Run the extraction pipeline and convert the results to GraphML.

```python
# 1. Extract Triples
kg_extractor.run_extraction()

# 2. Convert to CSV
kg_extractor.convert_json_to_csv()

# 3. Schema Induction
kg_extractor.generate_concept_csv_temp()

# 4. Concert Concept to CSV
kg_extractor.create_concept_csv()

# 5. Convert to GraphML for NetworkX
kg_extractor.convert_to_graphml()
```

## 2. Retrieval Augmented Generation (RAG)

Once you have your Knowledge Graph in GraphML format, you can index it and perform retrieval.

### Setup Embeddings

Initialize the embedding model.

```python
from sentence_transformers import SentenceTransformer
from atlas_rag.vectorstore.embedding_model import SentenceEmbedding

encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(encoder_model_name, trust_remote_code=True)
sentence_encoder = SentenceEmbedding(sentence_model)
```

### Create Index

Create embeddings and the FAISS index for your graph.

```python
from atlas_rag.vectorstore import create_embeddings_and_index

# Use the output directory from the KG construction step
working_directory = 'example/example_data/test_data' 

data = create_embeddings_and_index(
    sentence_encoder=sentence_encoder,
    model_name=encoder_model_name,
    working_directory=working_directory,
    keyword='test_data',
    include_concept=False, # Set to True if you generated concepts
    include_events=False,
    normalize_embeddings=True
)
```

### Perform Retrieval

Initialize the retriever (e.g., `HippoRAG2Retriever`) and ask a question.

```python
from atlas_rag.retriever import HippoRAG2Retriever

# Initialize Retriever
hipporag2_retriever = HippoRAG2Retriever(
    llm_generator=llm_generator,
    sentence_encoder=sentence_encoder,
    data=data,
)

# Retrieve Context
query = "Your question here?"
content, sorted_context_ids = hipporag2_retriever.retrieve(query, topN=3)

print(f"Retrieved content: {content}")
```

### Generate Answer

Use the retrieved context to generate an answer with the LLM.

```python
sorted_context = "\n".join(content)
response = llm_generator.generate_with_context(
    query, 
    sorted_context, 
    max_new_tokens=2048, 
    temperature=0.5
)

print(response)
```
