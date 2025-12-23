from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI
from configparser import ConfigParser
import argparse
parser = argparse.ArgumentParser(description="Custom KG Extraction")
parser.add_argument("--keyword", type=str, default="musique", help="Keyword for extraction")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name for LLM")
args = parser.parse_args()
# Load OpenRouter API key from config file
config = ConfigParser()
config.read('config.ini')
client = OpenAI(
  base_url="https://api.deepinfra.com/v1/openai",
  api_key=config['settings']['DEEPINFRA_API_KEY'],
)
triple_generator = LLMGenerator(client=client, model_name=args.model)
filename_pattern = args.keyword
# get model name for after slash
dir_name = args.model.split("/")[-1]
output_directory = f'example/example_scripts/custom_extraction/{dir_name}/{filename_pattern}'
data_directory = f'example/example_data/'
# triple_generator = LLMGenerator(client, model_name=model_name)
model_name = args.model
kg_extraction_config = ProcessingConfig(
      model_path=model_name,
      data_directory=data_directory,
      filename_pattern=filename_pattern,
      batch_size_triple=16,
      batch_size_concept=16,
      output_directory=f"{output_directory}",
      max_new_tokens=8192,
      max_workers=5,
      remove_doc_spaces=True, # For removing duplicated spaces in the document text
      include_concept=False, # Whether to include concept nodes and edges in the knowledge graph
      triple_extraction_prompt_path='example/example_scripts/custom_extraction/custom_benchmark/custom_prompt.json',
      triple_extraction_schema_path='example/example_scripts/custom_extraction/custom_benchmark/custom_schema.json',
)
kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)
# construct entity&event graph
kg_extractor.run_extraction()
# Convert Triples Json to CSV
kg_extractor.convert_json_to_csv()
# convert csv to graphml for networkx
kg_extractor.convert_to_graphml()