# Custom Knowledge Graph Extraction

This guide demonstrates how to customize the knowledge graph extraction process using your own prompts and validation schemas. This allows you to define specific extraction rules, output formats, and structural constraints for your knowledge graph.

## Overview

The custom extraction process involves three main components:
1. **Custom Prompt**: Defines the instructions given to the LLM for extracting triples.
2. **Custom Schema**: Defines the JSON schema used to validate the extracted triples.
3. **Extraction Script**: Configures the `KnowledgeGraphExtractor` to use your custom files.

## 1. Creating a Custom Prompt

Create a JSON file (e.g., `custom_prompt.json`) to define your extraction prompts. The file should support language keys (like `"en"`) and contain specific instructions for the system and the extraction tasks.

You can define multiple keys (e.g., `"triple_extraction"`, `"time_extraction"`) to represent different stages of extraction. The system will iterate through each key and perform the corresponding extraction task.

**Example `custom_prompt.json`:**

```json
{
  "en": {
    "system": "You are a helpful assistant",
    "triple_extraction": "You are an expert knowledge graph constructor.\nYour task is to extract factual information from the provided text and represent it strictly as a ***JSON array*** of knowledge graph triples.\n\n### Output Format\n- The output must be a **JSON array**.\n- Each element in the array must be a **JSON object** with exactly three non-empty keys:\n  - \"subject\": the main entity, concept, event, or attribute.\n  - \"relation\": a concise, descriptive phrase or verb that describes the relationship.\n  - \"object\": the entity, concept, value, event, or attribute that the subject has a relationship with.\n\n### Constraints\n- **Do not include any text other than the JSON output.**\n- Extract **all possible and relevant triples**.\n- If no triples can be extracted, return exactly: `[]`.",
    "time_extraction": "You are an expert in temporal extraction..."
  }
}
```

## 2. Creating a Custom Schema

Create a JSON schema file (e.g., `custom_schema.json`) to enforce the structure of the extracted data. This ensures that the LLM output conforms to your expected format.

**Important**: If you defined multiple extraction keys in your `custom_prompt.json` (e.g., `"triple_extraction"`, `"time_extraction"`), you must provide a corresponding schema for **each** key in this file.

**Example `custom_schema.json`:**

```json
{
    "triple_extraction": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "subject": { "type": "string" },
                "relation": { "type": "string" },
                "object": { "type": "string" }
            },
            "required": ["subject", "relation", "object"]
        }
    },
    "time_extraction": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "event": { "type": "string" },
                "timestamp": { "type": "string" }
            },
            "required": ["event", "timestamp"]
        }
    }
}
```

## 3. Running Custom Extraction

Use the `KnowledgeGraphExtractor` class with a `ProcessingConfig` that points to your custom prompt and schema files.

**Example Script (`custom_kg_extraction.py`):**

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI

# Initialize LLM
client = OpenAI(base_url="http://0.0.0.0:8129/v1", api_key="EMPTY")
triple_generator = LLMGenerator(client=client, model_name="Qwen/Qwen2.5-7B-Instruct")

# Configure Extraction with Custom Files
kg_extraction_config = ProcessingConfig(
      model_path="Qwen/Qwen2.5-7B-Instruct",
      data_directory="benchmark_data",
      filename_pattern="musique",
      output_directory="example",
      triple_extraction_prompt_path='example/example_scripts/custom_extraction/custom_benchmark/custom_prompt.json', # Path to your custom prompt
      triple_extraction_schema_path='example/example_scripts/custom_extraction/custom_benchmark/custom_schema.json', # Path to your custom schema
)

# Run Extraction
kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)
kg_extractor.run_extraction()
kg_extractor.convert_json_to_csv()
kg_extractor.convert_to_graphml()
```
