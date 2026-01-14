# Multilingual Support

AutoSchemaKG provides built-in support for processing documents in multiple languages, with the ability to extend to additional languages through custom prompts.

## Built-in Language Support

AutoSchemaKG comes with pre-configured prompts for the following languages:

- **English** (`en`)
- **Simplified Chinese** (`zh-CN`) - Mainland China
- **Traditional Chinese** (`zh-HK`) - Hong Kong

These languages have complete support for:
- Triple extraction (entity relations, event entities, event relations)
- Concept generation (events, entities, relations)

## Using Built-in Languages

### 1. Prepare Your Data

Each document in your corpus must include language metadata:

```json
[
    {
        "id": "1",
        "text": "The quick brown fox jumps over the lazy dog.",
        "metadata": {
            "lang": "en"
        }
    },
    {
        "id": "2",
        "text": "话说天下大势，分久必合，合久必分。",
        "metadata": {
            "lang": "zh-CN"
        }
    },
    {
        "id": "3",
        "text": "話說天下大勢，分久必合，合久必分。",
        "metadata": {
            "lang": "zh-HK"
        }
    }
]
```

**Required Fields:**
- `id`: Unique document identifier
- `text`: Document content in the specified language
- `metadata.lang`: Language code (`en`, `zh-CN`, or `zh-HK`)

> **Note:** If `metadata.lang` is not specified, the system defaults to English (`en`).

### 2. Run Knowledge Graph Extraction

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI

# Initialize LLM client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

model_name = "Qwen/Qwen2.5-7B-Instruct"
triple_generator = LLMGenerator(client, model_name=model_name)

# Configure extraction
kg_extraction_config = ProcessingConfig(
    model_path=model_name,
    data_directory="example_data/multilingual_data",
    filename_pattern="RomanceOfTheThreeKingdom",
    batch_size_triple=16,
    batch_size_concept=64,
    output_directory="generated/multilingual_output",
    max_new_tokens=8192
)

# Create extractor
kg_extractor = KnowledgeGraphExtractor(
    model=triple_generator, 
    config=kg_extraction_config
)

# Extract triples (automatically uses correct language from metadata)
kg_extractor.run_extraction()

# Convert to CSV
kg_extractor.convert_json_to_csv()

# Generate concepts - specify language explicitly
kg_extractor.generate_concept_csv_temp(language='zh-CN')  # For Simplified Chinese
# kg_extractor.generate_concept_csv_temp(language='zh-HK')  # For Traditional Chinese
# kg_extractor.generate_concept_csv_temp(language='en')     # For English
```

### 3. How Language Detection Works

**Automatic Triple Extraction:**
- The system reads `metadata.lang` from each document
- Automatically selects the corresponding prompt from `TRIPLE_INSTRUCTIONS`
- Applies language-specific extraction rules

**Manual Concept Generation:**
- You must explicitly specify the language when generating concepts
- Use the same language code as in your document metadata

## Adding Custom Language Support

To add support for additional languages (e.g., Japanese, Korean, French), you need to create custom prompt files.

### Step 1: Create Custom Triple Extraction Prompts

Create a JSON file with prompts for your target language(s):

**`custom_prompts/multilingual_triple_prompt.json`:**
```json
{
    "ja": {
        "system": "あなたは常に有効なJSON配列形式で応答する役立つアシスタントです",
        "entity_relation": "与えられたテキストから重要なエンティティとそれらの関係を抽出し、簡潔に要約してください。関係は、先頭エンティティと末尾エンティティの情報を繰り返すことなく、エンティティ間の接続を簡潔に捉える必要があります。\n\n次のJSON形式で厳密に出力してください：\n[\n    {\n        \"Head\": \"{名詞}\",\n        \"Relation\": \"{動詞}\",\n        \"Tail\": \"{名詞}\"\n    }...\n]\n\n以下はテキストです：",
        "event_entity": "...",
        "event_relation": "..."
    },
    "ko": {
        "system": "당신은 항상 유효한 JSON 배열 형식으로 응답하는 유용한 도우미입니다",
        "entity_relation": "주어진 텍스트에서 중요한 개체와 그들 간의 관계를 추출하고 간결하게 요약하세요...",
        "event_entity": "...",
        "event_relation": "..."
    }
}
```

### Step 2: Create Custom Concept Generation Prompts

Create prompts for concept abstraction in your target language:

**`custom_prompts/multilingual_concept_prompt.json`:**
```json
{
    "ja": {
        "event": "イベントを提供します。このイベントの抽象的なイベントとして、1〜2語を含むいくつかのフレーズを提供する必要があります...",
        "entity": "エンティティを提供します。このエンティティの抽象的なエンティティとして、1〜2語を含むいくつかのフレーズを提供する必要があります...",
        "relation": "関係を提供します。この関係の抽象的な関係として、1〜2語を含むいくつかのフレーズを提供する必要があります..."
    },
    "ko": {
        "event": "이벤트를 제공합니다. 이 이벤트의 추상적 이벤트에 대해 1-2단어를 포함하는 여러 구문을 제공해야 합니다...",
        "entity": "엔티티를 제공합니다. 이 엔티티의 추상적 엔티티에 대해 1-2단어를 포함하는 여러 구문을 제공해야 합니다...",
        "relation": "관계를 제공합니다. 이 관계의 추상적 관계에 대해 1-2단어를 포함하는 여러 구문을 제공해야 합니다..."
    }
}
```

### Step 3: Configure with Custom Prompts

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator

# Configure with custom prompts
kg_extraction_config = ProcessingConfig(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_directory="example_data/japanese_data",
    filename_pattern="japanese_corpus",
    batch_size_triple=16,
    batch_size_concept=64,
    output_directory="generated/japanese_output",
    
    # Specify custom prompt files
    triple_extraction_prompt_path="custom_prompts/multilingual_triple_prompt.json",
    triple_extraction_schema_path="custom_prompts/custom_schema.json",  # Optional
    
    max_new_tokens=8192
)

kg_extractor = KnowledgeGraphExtractor(
    model=triple_generator, 
    config=kg_extraction_config
)

# Run extraction with custom prompts
kg_extractor.run_extraction()
kg_extractor.convert_json_to_csv()

# Generate concepts for Japanese
kg_extractor.generate_concept_csv_temp(language='ja')
```

### Step 4: Update Your Data

Make sure your corpus includes the correct language codes:

```json
[
    {
        "id": "1",
        "text": "吾輩は猫である。名前はまだ無い。",
        "metadata": {
            "lang": "ja"
        }
    },
    {
        "id": "2",
        "text": "나는 고양이로소이다. 이름은 아직 없다.",
        "metadata": {
            "lang": "ko"
        }
    }
]
```

## Prompt Structure Reference

When creating custom language prompts, follow this structure:

### Triple Extraction Prompts

```json
{
    "<lang_code>": {
        "system": "System message for the LLM in target language",
        "entity_relation": "Instructions for extracting entity-relation-entity triples",
        "event_entity": "Instructions for extracting event-entity relationships",
        "event_relation": "Instructions for extracting event-event relationships"
    }
}
```

### Concept Generation Prompts

```json
{
    "<lang_code>": {
        "event": "Instructions for generating abstract concepts from events",
        "entity": "Instructions for generating abstract concepts from entities",
        "relation": "Instructions for generating abstract concepts from relations"
    }
}
```

## Example Files

- **Sample multilingual data**: `example/example_data/multilingual_data/`
- **Built-in prompts**: `atlas_rag/llm_generator/prompt/triple_extraction_prompt.py`
- **Complete tutorial**: `example/multilingual_processing.md`

## Tips for Custom Languages

1. **Prompt Quality**: High-quality, native-language prompts yield better extraction results
2. **Test Incrementally**: Start with a small dataset to verify your prompts work correctly
3. **JSON Format**: Ensure your prompts explicitly request JSON output format
4. **Examples**: Include few-shot examples in your target language when possible
5. **Language Models**: Use multilingual or language-specific models for best results (e.g., Qwen for Chinese, GPT-4 for general multilingual support)
