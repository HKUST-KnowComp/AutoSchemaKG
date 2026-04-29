from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator, GenerationConfig
from configparser import ConfigParser
from argparse import ArgumentParser
from openai import OpenAI
import time
import torch

parser = ArgumentParser(description="Generate knowledge graph slices from text data.")
parser.add_argument("--shard", type=int, help="Shard number to process.", default=0)
parser.add_argument("--total_shards", type=int, help="Total number of slices to process.", default=1)
parser.add_argument("--port", type=int, help="Port number for the OpenAI API client.", default=8135)
parser.add_argument("--keyword", type=str, help="Keyword for filtering data files.", default="musique")
args = parser.parse_args()
if __name__ == "__main__":
    config = ConfigParser()
    config.read('config.ini')
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    keyword = args.keyword
    client = OpenAI(
        base_url=f"https://api.deepinfra.com/v1/openai",
        api_key=config['settings']['BAI_DEEPINFRA_API_KEY'],
    )
    
    gen_config = GenerationConfig(temperature=0.5, max_tokens=32768)
    triple_generator = LLMGenerator(client, model_name=model_name,max_workers=64, default_config=gen_config)
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    start_time = time.time()
    kg_extraction_config = ProcessingConfig(
        model_path=model_name,
        data_directory="/home/httsangaj/projects/AutoSchemaKG/benchmark_data/atlas_graphml",
        filename_pattern=keyword,
        batch_size_triple=64,
        batch_size_concept=64,
        output_directory=f'/data/httsangaj/atlas/rebuttal/{keyword}_reconstruct/{model_name}',
        current_shard_triple=args.shard,
        total_shards_triple=args.total_shards,
        current_shard_concept=args.shard,
        total_shards_concept=args.total_shards,
        record=False,
        max_new_tokens=16384,
        benchmark=False,
        allow_empty=False
    )
    kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)
    kg_extractor.run_extraction()
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    kg_extractor.convert_json_to_csv()
    kg_extractor.generate_concept_csv_temp(language='en')
    kg_extractor.create_concept_csv()
    kg_extractor.convert_to_graphml()