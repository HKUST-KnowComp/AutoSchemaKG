#!/usr/bin/env python3
"""
Knowledge Graph Extraction Pipeline
Extracts entities, relations, and events from text data using transformer models.
"""
import re
import json
import os
import argparse
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import torch
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import json_repair
import copy

from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.kg_construction.utils.json_processing.json_to_csv import json2csv
from atlas_rag.kg_construction.concept_generation import generate_concept
from atlas_rag.kg_construction.utils.csv_processing.merge_csv import merge_csv_files
from atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml import csvs_to_graphml, csvs_to_temp_graphml
from atlas_rag.kg_construction.concept_to_csv import all_concept_triples_csv_to_csv
from atlas_rag.kg_construction.utils.csv_processing.csv_add_numeric_id import add_csv_columns
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.vectorstore.create_neo4j_index import create_faiss_index
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator.prompt.triple_extraction_prompt import TRIPLE_INSTRUCTIONS
from atlas_rag.llm_generator.format.validate_json_schema import ATLAS_SCHEMA
 
# Prompt instructions and schema, will be modified during the initialization of KnowledgeGraphExtractor according to the config if provided

class TextChunker:
    """Handles text chunking based on character limits."""
    
    def __init__(self, config: ProcessingConfig):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
    
    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        # 1. Split text into words first to avoid cutting inside a word
        words = text.split(' ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # +1 for the space we removed during split
            word_len = len(word) + 1 
            
            # If adding this word exceeds chunk size, save the current chunk
            if current_length + word_len > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    
                    # LOGIC FOR OVERLAP:
                    # Keep the last N words that fit within the overlap limit
                    # This is harder to do perfectly with simple loops.
                    # Usually, we start a new chunk with the current word.
                    
                    # Simplified overlap approach (Keep last X words roughly):
                    overlap_len = 0
                    new_start_index = len(current_chunk)
                    
                    # Walk backwards to find how many words fit in the overlap
                    while new_start_index > 0 and overlap_len < self.chunk_overlap:
                        new_start_index -= 1
                        overlap_len += len(current_chunk[new_start_index]) + 1
                    
                    current_chunk = current_chunk[new_start_index:]
                    current_length = overlap_len
            
            current_chunk.append(word)
            current_length += word_len
            
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

class DatasetProcessor:
    """Processes and prepares dataset for knowledge graph extraction."""
    
    def __init__(self, config: ProcessingConfig, prompt_instructions: Dict, result_schema: Dict):
        self.config = config
        self.chunker = TextChunker(config)
        self.benchmark = config.benchmark
        self.deduplicate_text = config.deduplicate_text
        self.prompt_instructions = prompt_instructions
        self.result_schema = result_schema

    def filter_language_content(self, sample: Dict[str, Any]) -> bool:
        """Check if content is in English."""
        metadata = sample.get("metadata", {})
        language = metadata.get("lang", "en")  # Default to English if not specified
        supported_languages = list(self.prompt_instructions.keys())
        return language in supported_languages
    
    
    def create_sample_chunks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from a single sample."""
        original_text = sample.get("text", "")
        if self.config.remove_doc_spaces:
            original_text = re.sub(r'\s+', ' ',original_text).strip()
        text_chunks = self.chunker.split_text(original_text)
        chunks = []
        self.text_set = set()  # For deduplication
        for chunk_idx, chunk_text in enumerate(text_chunks):
            if self.deduplicate_text:
                if chunk_text in self.text_set:
                    continue
                self.text_set.add(chunk_text)
            chunk_data = {
                "id": sample["id"],
                "text": chunk_text,
                "chunk_id": chunk_idx,
                "metadata": sample["metadata"]
            }
            chunks.append(chunk_data)
            
        return chunks
    
    def prepare_dataset(self, raw_dataset) -> List[Dict[str, Any]]:
        """Process raw dataset into chunks suitable for processing with generalized slicing."""
        
        processed_samples = []
        if self.benchmark:
            raw_dataset = raw_dataset.select(range(min(100, len(raw_dataset))))
        total_texts = len(raw_dataset)
        # Handle edge cases
        if total_texts == 0:
            print(f"No texts found for shard {self.config.current_shard_triple+1}/{self.config.total_shards_triple}")
            return processed_samples
        
        # Calculate base and remainder for fair distribution
        base_texts_per_shard = total_texts // self.config.total_shards_triple
        remainder = total_texts % self.config.total_shards_triple
        
        # Calculate start index
        if self.config.current_shard_triple < remainder:
            start_idx = self.config.current_shard_triple * (base_texts_per_shard + 1)
        else:
            start_idx = remainder * (base_texts_per_shard + 1) + (self.config.current_shard_triple - remainder) * base_texts_per_shard
        
        # Calculate end index
        if self.config.current_shard_triple < remainder:
            end_idx = start_idx + (base_texts_per_shard + 1)
        else:
            end_idx = start_idx + base_texts_per_shard
        
        # Ensure indices are within bounds
        start_idx = min(start_idx, total_texts)
        end_idx = min(end_idx, total_texts)
        
        print(f"Processing shard {self.config.current_shard_triple+1}/{self.config.total_shards_triple} "
            f"(texts {start_idx}-{end_idx-1} of {total_texts}, {end_idx - start_idx} documents)")
        
        # Process documents in assigned shard
        for idx in range(start_idx, end_idx):
            sample = raw_dataset[idx]
            
            # Filter by language
            if not self.filter_language_content(sample):
                print(f"Unsupported language in sample {idx}, skipping.")
                continue
                
            # Create chunks
            chunks = self.create_sample_chunks(sample)
            processed_samples.extend(chunks)
            
            # Debug mode early termination
            if self.config.debug_mode and len(processed_samples) >= 20:
                print("Debug mode: Stopping at 20 chunks")
                break
        
        print(f"Generated {len(processed_samples)} chunks for shard {self.config.current_shard_triple+1}/{self.config.total_shards_triple}")
        return processed_samples

class CustomDataLoader:
    """Custom data loader for knowledge graph extraction."""
    
    def __init__(self, dataset, processor: DatasetProcessor, prompt_instructions: Dict, result_schema: Dict):
        self.raw_dataset = dataset
        self.processor = processor
        self.processed_data = processor.prepare_dataset(dataset)
        self.prompt_instructions = prompt_instructions
        self.result_schema = result_schema
        self.stage_to_prompt_dict = {
            "stage_1": "entity_relation",
            "stage_2": "event_entity",
            "stage_3": "event_relation"
        }

    def __len__(self) -> int:
        return len(self.processed_data)
    
    def create_batch_instructions(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        messages_dict = {key: [] for key in self.result_schema.keys()}
        for item in batch_data:
            # get language
            language = item.get("metadata",{}).get("lang", "en")
            system_msg = self.prompt_instructions.get(language, self.prompt_instructions["en"])['system'] 

            for key in self.result_schema.keys():
                stage_msg = self.prompt_instructions.get(language, self.prompt_instructions["en"])[key] + '\n' + item["text"]
                messages_dict[key].append([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": stage_msg}
                ])
        
        return messages_dict
    
    def __iter__(self):
        """Iterate through batches."""
        batch_size = self.processor.config.batch_size_triple
        start_idx = self.processor.config.resume_from * batch_size
        
        for i in tqdm(range(start_idx, len(self.processed_data), batch_size)):
            batch_data = self.processed_data[i:i + batch_size]
            
            # Prepare instructions
            instructions = self.create_batch_instructions(batch_data)
            
            # Extract batch information
            batch_ids = [item["id"] for item in batch_data]
            batch_metadata = [item["metadata"] for item in batch_data]
            batch_texts = [item["text"] for item in batch_data]
            
            yield instructions, batch_ids, batch_texts, batch_metadata

class OutputParser:
    """Parses model outputs and extracts structured data."""
    def __init__(self):
        pass

    def extract_structured_data(self, outputs: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract structured data from model outputs."""
        results = []
        for output in outputs:
            parsed_data = json_repair.loads(output)
            results.append(parsed_data)
            
        return results

class KnowledgeGraphExtractor:
    """Main class for knowledge graph extraction pipeline."""
    
    def __init__(self, model:LLMGenerator, config: ProcessingConfig):
        self.config = config
        self.model = model
        self.model_name = model.model_name
        self.parser = OutputParser()        

        # Create instance-specific copies of defaults
        self.prompt_instructions = copy.deepcopy(TRIPLE_INSTRUCTIONS)
        self.result_schema = copy.deepcopy(ATLAS_SCHEMA)
        
        # check both prompt and schema is provided if one of them is provided
        self.custom_prompt = None
        self.custom_schema = None
        if self.config.triple_extraction_prompt_path is not None:
            assert self.config.triple_extraction_schema_path is not None, "Schema path must be provided if prompt path is provided"
            with open(self.config.triple_extraction_prompt_path, 'r') as f:
                self.custom_prompt = json_repair.loads(f.read())
            self.prompt_instructions = self.custom_prompt
            print(f"Using custom kg extraction prompt:\n{self.custom_prompt}")
        if self.config.triple_extraction_schema_path is not None:
            assert self.config.triple_extraction_prompt_path is not None, "Prompt path must be provided if schema path is provided"
            with open(self.config.triple_extraction_schema_path, 'r') as f:
                self.custom_schema = json_repair.loads(f.read())
            self.result_schema = self.custom_schema 
            print(f"Using custom kg extraction schema:\n{self.custom_schema}")

            # validate one-to-one and one-to-many triple schema
            for triple_schema in self.custom_schema.values():
                print(triple_schema)
                triples = triple_schema['items']['properties']
                # check for one-to-one or one-to-many
                one_to_many = any(prop.get('type') == "array" for prop in triples.values())
                one_to_one = not one_to_many
                if one_to_many:
                    continue
                # check relation exist
                assert "Relation" in list(triples.keys()) or "relation" in list(triples.keys()), "Relation field must be present in triple schema for one-to-one triple"
                if one_to_one:
                    assert len(triples) == 3, "One-to-one triple schema must have exactly three fields: subject, relation, object"

    def set_model(self, model:LLMGenerator):
        """Set or update the LLM model."""
        self.model = model

    def load_dataset(self) -> Any:
        """Load and prepare dataset."""
        data_path = Path(self.config.data_directory)
        all_files = os.listdir(data_path)
        
        valid_files = [
            filename for filename in all_files
            if filename.startswith(self.config.filename_pattern) and
            (filename.endswith(".json.gz") or filename.endswith(".json") or filename.endswith(".jsonl") or filename.endswith(".jsonl.gz"))
        ]
        
        print(f"Found data files: {valid_files}")
        data_files = valid_files
        dataset_config = {"train": data_files}
        return load_dataset(self.config.data_directory, data_files=dataset_config["train"])
    
    def process_stage(self, instructions: Dict[str, str], result_schema: Dict) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
        """Process first stage: entity-relation extraction."""
        outputs = self.model.triple_extraction(messages=instructions,
                                                max_tokens=self.config.max_new_tokens if self.config.max_new_tokens else self.model.config.max_tokens, 
                                                repetition_penalty=self.config.repetition_penalty if self.config.repetition_penalty else self.model.config.repetition_penalty,
                                                record=self.config.record,
                                                result_schema=result_schema, 
                                                allow_empty=self.config.allow_empty)
        if self.config.record:
            text_outputs = [output[0] for output in outputs]
        else:
            text_outputs = outputs
        structured_data = self.parser.extract_structured_data(text_outputs)
        return outputs, structured_data
    
    def create_output_filename(self) -> str:
        """Create output filename with timestamp and shard info."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name_safe = self.config.model_path.replace("/", "_")
        
        filename = (f"{model_name_safe}_{self.config.filename_pattern}_output_"
                   f"{timestamp}_{self.config.current_shard_triple + 1}_in_{self.config.total_shards_triple}.json")
        
        extraction_dir = os.path.join(self.config.output_directory, "kg_extraction")
        os.makedirs(extraction_dir, exist_ok=True)
        
        return os.path.join(extraction_dir, filename)

    def prepare_result_dict(self, id: str, text: str, metadata: Dict[str, Any], stage_outputs: List[Any], result_keys: List[str]) -> Dict[str, Any]:
        """Prepare result dictionary for a single sample."""
        sample_dict = {}
        sample_dict['id'] = id
        sample_dict['original_text'] = text
        sample_dict['metadata'] = metadata
        llm_outputs, result_dicts = stage_outputs
        if self.config.record:
            llm_usage = [output[1] for output in llm_outputs]
            llm_outputs = [output[0] for output in llm_outputs]

        for key, llm_output, result_dict in zip(result_keys, llm_outputs, result_dicts):
            sample_dict[f'{key}_dict'] = result_dict
            sample_dict[f'{key}_output'] = llm_output
        if self.config.record:
            for key, llm_usage in zip(result_keys, llm_usage):
                sample_dict[f'{key}_usage'] = llm_usage

        if 'date_download' in sample_dict['metadata']:
            sample_dict['metadata']['date_download'] = str(sample_dict['metadata']['date_download'])
        return sample_dict

    def debug_print_result(self, result: Dict[str, Any]):
        """Print result for debugging."""
        for key, value in result.items():
            print(f"{key}: {value}")
            print("-" * 100)
    
    def run_extraction(self):
        """Run the complete knowledge graph extraction pipeline."""
        # Setup
        os.makedirs(self.config.output_directory+'/kg_extraction', exist_ok=True)
        dataset = self.load_dataset()
        
        if self.config.debug_mode:
            print("Debug mode: Processing only 20 samples")
        
        # Create data processor and loader
        processor = DatasetProcessor(self.config, self.prompt_instructions, self.result_schema)
        data_loader = CustomDataLoader(dataset["train"], processor, self.prompt_instructions, self.result_schema)
        
        output_file = self.create_output_filename()
        print(f"Model: {self.config.model_path}")
        
        batch_counter = 0
        if self.config.record:
            extraction_start_time = time.time()
        with torch.no_grad():
            with open(output_file, "w") as output_stream:
                for batch in data_loader:
                    batch_counter += 1
                    messages_dict, batch_ids, batch_texts, batch_metadata = batch
                    
                    stage_outputs = {}
                    result_keys = list(messages_dict.keys())
                    for key in result_keys:
                        batch_result = self.process_stage(messages_dict[key], result_schema = self.result_schema[key])
                        # each batch_result is (llm_rawoutput, triple-dict), and at the same time llm_rawoutput can be (llm_rawoutput, llm_usage)
                        
                        stage_outputs[key] = batch_result


                    # Write results
                    print(f"Processed {batch_counter} batches ({batch_counter * self.config.batch_size_triple} chunks)")
                    for i in range(len(batch_ids)):
                        id = batch_ids[i]
                        text = batch_texts[i]
                        metadata = batch_metadata[i]

                        triple_dicts = []
                        llm_outputs = []
                        for key in result_keys:
                            temp_batch_result = stage_outputs[key]
                            llm_output, triple_dict = temp_batch_result
    
                            triple_dicts.append(triple_dict[i])
                            llm_outputs.append(llm_output[i])

                        result = self.prepare_result_dict(id, text, metadata, (llm_outputs, triple_dicts), result_keys)

                        if self.config.debug_mode:
                            self.debug_print_result(result)
   
                        output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_stream.flush()
            if self.config.record and self.config.benchmark:
                # add the total time taken for extraction at the last json object
                total_extraction_time = time.time() - extraction_start_time
                # read the json and modify the last json object
                with open(output_file, "r") as f:
                    lines = f.readlines()
                if len(lines) > 0:
                    last_line = lines[-1]
                    try:
                        last_json = json.loads(last_line)
                        last_json['total_extraction_time_seconds'] = total_extraction_time
                        lines[-1] = json.dumps(last_json, ensure_ascii=False) + "\n"
                        with open(output_file, "w") as f:
                            f.writelines(lines)
                    except:
                        print("Failed to add extraction time to the last json object")

    def convert_json_to_csv(self):
        json2csv(
            dataset = self.config.filename_pattern, 
            output_dir=f"{self.config.output_directory}/triples_csv",
            data_dir=f"{self.config.output_directory}/kg_extraction",
            schema = self.result_schema,
            custom = self.custom_schema != None
        )
        csvs_to_temp_graphml(
            triple_node_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            triple_edge_file=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv",
            config = self.config
        )
    
    def generate_concept_csv_temp(self, batch_size: int = None, **kwargs):
        if self.config.benchmark and self.config.record:
            concept_start_time = time.time()
        generate_concept(
            model=self.model,
            input_file=f"{self.config.output_directory}/triples_csv/missing_concepts_{self.config.filename_pattern}_from_json.csv",
            output_folder=f"{self.config.output_directory}/concepts",
            output_file="concept.json",
            logging_file=f"{self.config.output_directory}/concepts/logging.txt",
            config=self.config,
            batch_size=batch_size if batch_size else self.config.batch_size_concept,
            shard=self.config.current_shard_concept,
            num_shards=self.config.total_shards_concept,
            record = self.config.record,
            **kwargs
        )
        if self.config.benchmark and self.config.record:  
            # add line to the logging file
            total_concept_time = time.time() - concept_start_time
            with open(f"{self.config.output_directory}/concepts/logging.txt", "a") as f:
                f.write(f"Total concept generation time: {total_concept_time:.2f} seconds\n")
    
    def create_concept_csv(self):
        merge_csv_files(
            output_file=f"{self.config.output_directory}/triples_csv/{self.config.filename_pattern}_from_json_with_concept.csv",
            input_dir=f"{self.config.output_directory}/concepts",
        )
        all_concept_triples_csv_to_csv(
            node_file=f'{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv',
            edge_file=f'{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv',
            concepts_file=f'{self.config.output_directory}/triples_csv/{self.config.filename_pattern}_from_json_with_concept.csv',
            output_node_file=f'{self.config.output_directory}/concept_csv/concept_nodes_{self.config.filename_pattern}_from_json_with_concept.csv',
            output_edge_file=f'{self.config.output_directory}/concept_csv/concept_edges_{self.config.filename_pattern}_from_json_with_concept.csv',
            output_full_concept_triple_edges=f'{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv',
        )
        
    def convert_to_graphml(self):
        csvs_to_graphml(
            triple_node_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            text_node_file=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            triple_edge_file=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv" if self.config.include_concept else \
                f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv",
            text_edge_file=f"{self.config.output_directory}/triples_csv/text_edges_{self.config.filename_pattern}_from_json.csv",
            concept_node_file=f"{self.config.output_directory}/concept_csv/concept_nodes_{self.config.filename_pattern}_from_json_with_concept.csv" if self.config.include_concept else None,
            concept_edge_file=f"{self.config.output_directory}/concept_csv/concept_edges_{self.config.filename_pattern}_from_json_with_concept.csv" if self.config.include_concept else None,
            output_file=f"{self.config.output_directory}/kg_graphml/{self.config.filename_pattern}_graph.graphml",
            include_concept = self.config.include_concept
        )
    
    def add_numeric_id(self):
        add_csv_columns(
            node_csv=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            edge_csv=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            text_csv=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            node_with_numeric_id=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb_with_numeric_id.csv",
            edge_with_numeric_id=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb_with_numeric_id.csv",
            text_with_numeric_id=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json_with_numeric_id.csv",
        )

    def compute_kg_embedding(self, encoder_model:BaseEmbeddingModel, batch_size: int = 2048):
        encoder_model.compute_kg_embedding(
            node_csv_without_emb=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            node_csv_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
            edge_csv_without_emb=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            edge_csv_file=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept_with_emb.csv",
            text_node_csv_without_emb=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            text_node_csv=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
            batch_size = 2048
        )

    def create_faiss_index(self, index_type="HNSW,Flat", faiss_gpu=True):
        create_faiss_index(self.config.output_directory, self.config.filename_pattern, index_type, faiss_gpu)

def parse_command_line_arguments() -> ProcessingConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Extraction Pipeline")
    
    parser.add_argument("-m", "--model", type=str, required=True,
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Model path for knowledge extraction")
    parser.add_argument("--data_dir", type=str, default="your_data_dir",
                       help="Directory containing input data")
    parser.add_argument("--file_name", type=str, default="en_simple_wiki_v0",
                       help="Filename pattern to match")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--output_dir", type=str, default="./generation_result_debug",
                       help="Output directory for results")
    parser.add_argument("--total_shards_triple", type=int, default=1,
                       help="Total number of data shards")
    parser.add_argument("--shard", type=int, default=0,
                       help="Current shard index")
    parser.add_argument("--bit8", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--resume", type=int, default=0,
                       help="Resume from specific batch")
    
    args = parser.parse_args()
    
    return ProcessingConfig(
        model_path=args.model,
        data_directory=args.data_dir,
        filename_pattern=args.file_name,
        batch_size=args.batch_size,
        output_directory=args.output_dir,
        total_shards_triple=args.total_shards_triple,
        current_shard_triple=args.shard,
        use_8bit=args.bit8,
        debug_mode=args.debug,
        resume_from=args.resume
    )

def main():
    """Main entry point for the knowledge graph extraction pipeline."""
    config = parse_command_line_arguments()
    extractor = KnowledgeGraphExtractor(config)
    extractor.run_extraction()

if __name__ == "__main__":
    main() 