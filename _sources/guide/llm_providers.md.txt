# LLM Providers

AutoSchemaKG's `LLMGenerator` is designed to be backend-agnostic, supporting various LLM providers through a unified interface. This allows you to switch between proprietary APIs (like OpenAI) and open-source models (via vLLM, HuggingFace, or other serving frameworks) with minimal code changes.

## OpenAI-Compatible APIs

The primary way to interface with LLMs is through the OpenAI-compatible API standard. This supports:
- **OpenAI** (GPT-4, GPT-3.5)
- **DeepInfra** (Llama 3, Mixtral, Qwen)
- **Together AI**
- **vLLM** (running as a server)
- **LocalAI** / **Ollama** / **LiteLLM**

### Configuration

To use an OpenAI-compatible provider, initialize the `OpenAI` client with the appropriate `base_url` and `api_key`, then pass it to `LLMGenerator`.

```python
from openai import OpenAI
from atlas_rag.llm_generator import LLMGenerator, GenerationConfig

# 1. Configure Generation Parameters
gen_config = GenerationConfig(
    temperature=0.5,
    max_tokens=4096,
    top_p=0.9
)

# 2. Initialize Client (Example: DeepInfra)
client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key="YOUR_API_KEY"
)

# 3. Initialize Generator
generator = LLMGenerator(
    client=client,
    model_name="meta-llama/Llama-3-70b-chat-hf",
    max_workers=10,  # Number of concurrent requests
    default_config=gen_config
)
```

### Using Local vLLM Server

You can run a local LLM using vLLM and connect to it as if it were an OpenAI API.

1. **Start vLLM Server:**
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-7B-Instruct \
       --port 8000
   ```

2. **Connect via Python:**
   ```python
   client = OpenAI(
       base_url="http://localhost:8000/v1",
       api_key="EMPTY"  # vLLM usually doesn't require a key locally
   )
   
   generator = LLMGenerator(
       client=client,
       model_name="Qwen/Qwen2.5-7B-Instruct",
       default_config=gen_config
   )
   ```

## Azure OpenAI

For Azure OpenAI, use the `AzureOpenAI` client.

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="YOUR_AZURE_API_KEY",
    api_version="2023-05-15",
    azure_endpoint="https://your-resource.openai.azure.com"
)

generator = LLMGenerator(
    client=client,
    model_name="gpt-4", # Deployment name
    default_config=gen_config
)
```

## Native Local Models (HuggingFace / vLLM Offline)

*Note: Direct support for offline `vLLM` or `HuggingFace` pipelines (without the API server) is supported via specific `LLMGenerator` subclasses or configurations. Ensure you have the necessary packages installed (`vllm`, `transformers`, `torch`).*

The `GenerationConfig` class includes specific parameters for these backends:

- **vLLM**: `min_p`, `use_beam_search`, `guided_json`, `guided_regex`
- **HuggingFace**: `repetition_penalty`, `truncation`, `padding`

Example of configuring backend-specific parameters:

```python
gen_config = GenerationConfig(
    temperature=0.7,
    # vLLM specific
    min_p=0.05,
    guided_json=my_json_schema,
    # HuggingFace specific
    repetition_penalty=1.1
)
```
