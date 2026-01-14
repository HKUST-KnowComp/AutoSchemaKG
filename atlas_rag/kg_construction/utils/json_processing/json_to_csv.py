from tqdm import tqdm
import argparse
import os
import csv 
import json
import re
import hashlib

# Increase the field size limit
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB limit


# Function to compute a hash ID from text
def compute_hash_id(text):
    # Use SHA-256 to generate a hash
    text = text + '_text'
    hash_object = hashlib.sha256(text.encode('utf-8'))
    return hash_object.hexdigest()  # Return hash as a hex string

def clean_text(text):
    # remove NUL as well
    
    new_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\v", " ").replace("\f", " ").replace("\b", " ").replace("\a", " ").replace("\e", " ").replace(";", ",")
    new_text = new_text.replace("\x00", "")
    new_text = re.sub(r'\s+', ' ', new_text).strip()

    return new_text

def remove_NUL(text):
    return text.replace("\x00", "")

def custom_schema_json_2_csv(dataset, data_dir, output_dir, schema):
    visited_nodes = set()
    visited_hashes = set()

    all_entities = set()
    all_events = set()
    all_relations = set()

    file_dir_list = [f for f in os.listdir(data_dir) if dataset in f]
    file_dir_list = sorted(file_dir_list)
    print("Loading data from the json files")
    print("Number of files: ", len(file_dir_list))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Define output file paths
    node_csv_without_emb = os.path.join(output_dir, f"triple_nodes_{dataset}_from_json_without_emb.csv")
    edge_csv_without_emb = os.path.join(output_dir, f"triple_edges_{dataset}_from_json_without_emb.csv")
    node_text_file = os.path.join(output_dir, f"text_nodes_{dataset}_from_json.csv")
    edge_text_file = os.path.join(output_dir, f"text_edges_{dataset}_from_json.csv")
    missing_concepts_file = os.path.join(output_dir, f"missing_concepts_{dataset}_from_json.csv")

    # Open CSV files for writing
    with open(node_text_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_node_text, \
         open(edge_text_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_edge_text, \
         open(node_csv_without_emb, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_node, \
         open(edge_csv_without_emb, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_edge:

        csv_writer_node_text = csv.writer(csvfile_node_text)
        csv_writer_edge_text = csv.writer(csvfile_edge_text)
        writer_node = csv.writer(csvfile_node)
        writer_edge = csv.writer(csvfile_edge)

        # Write headers
        csv_writer_node_text.writerow(["text_id:ID", "original_text", ":LABEL"])
        csv_writer_edge_text.writerow([":START_ID", ":END_ID", ":TYPE"])
        writer_node.writerow(["name:ID", "type", "concepts", "synsets", ":LABEL"])
        writer_edge.writerow([":START_ID", ":END_ID", "relation", "concepts", "synsets", ":TYPE"])
        
        # Process each file
        for file_dir in tqdm(file_dir_list):
            print("Processing file for file ids: ", file_dir)
            with open(os.path.join(data_dir, file_dir), "r") as jsonfile:
                for line in jsonfile:
                    data = json.loads(line.strip())
                    original_text = data["original_text"]
                    original_text = remove_NUL(original_text)
                    if "Here is the passage." in original_text:
                        original_text = original_text.split("Here is the passage.")[-1]
                    eot_token = "<|eot_id|>"
                    original_text = original_text.split(eot_token)[0]

                    text_hash_id = compute_hash_id(original_text)

                    # Write the original text as nodes
                    if text_hash_id not in visited_hashes:
                        visited_hashes.add(text_hash_id)
                        csv_writer_node_text.writerow([text_hash_id, original_text, "Text"])

                    file_id = str(data["id"])
                   
                    triple_name_keys = list(schema.keys())
                    for key in triple_name_keys:
                        triple_dict = data[f'{key}_dict']
                        schema_items = schema[key]['items']['properties']
                        # check if one to many or one to one
                        one_to_many = False
                        one_to_one = False
                        for item in schema_items.values():
                            if item['type'] == "array":
                                one_to_many = True
                                break
                        if not one_to_many:
                            one_to_one = True
                        
                        # process for case one to one
                        if one_to_one:
                            triple_keys = list(schema_items.keys())
                            entity_keys = []
                            node_types = []
                            for key in triple_keys:
                                if key not in ["Relation", "relation"]:
                                    entity_keys.append(key)
                                    node_types.append(schema_items[key]['node_type'] if 'node_type' in schema_items[key] else 'entity')
                            assert len(entity_keys) == 2, f"One-to-one triple schema must have exactly two entities: {entity_keys}"    
                            for triple in triple_dict:
                               # get the three keys
                               # We can only ensure either "Relation" or "relation" is always exist for edge name for one-to-one triple
                                assert not ("Relation" not in triple and "relation" not in triple), f"Missing relation key for {triple}"
                                
                                # check if it is relation or Relation for relation key
                                head_entity, relation, tail_entity = (triple[key] for key in triple_keys)
                                if head_entity is None or tail_entity is None or relation is None:
                                    continue
                                if head_entity.isspace() or tail_entity.isspace() or relation.isspace():
                                    continue
                                if len(head_entity) == 0 or len(tail_entity) == 0 or len(relation) == 0:
                                    continue

                                # Add nodes to files
                                if head_entity not in visited_nodes:
                                    visited_nodes.add(head_entity)
                                    all_entities.add(head_entity)
                                    writer_node.writerow([head_entity, node_types[0], [], [], "Node"])
                                    csv_writer_edge_text.writerow([head_entity, text_hash_id, "Source"])

                                if tail_entity not in visited_nodes:
                                    visited_nodes.add(tail_entity)
                                    all_entities.add(tail_entity)
                                    writer_node.writerow([tail_entity, node_types[1], [], [], "Node"])
                                    csv_writer_edge_text.writerow([tail_entity, text_hash_id, "Source"])

                                all_relations.add(relation)
                                writer_edge.writerow([head_entity, tail_entity, relation, [], [], "Relation"])
                        
                        if one_to_many:
                            triple_keys = list(schema_items.keys())
                            entity_keys = []
                            relation = None
                            # get the three keys
                            # For one-to-many triple, We cannot ensure either "Relation" or "relation" is always exist for edge name, but we can check if it exist
                            
                            for key in triple_keys:
                                # check if it is relation or Relation for relation key
                                if key not in ["Relation", "relation"]:
                                    entity_keys.append(key)
                                else:
                                    relation = triple[key]
                            if relation is None:
                                relation = "is participated by"
                            assert len(entity_keys) == 2, f"One-to-many triple schema must have exactly two entities: {entity_keys}"  
                            for triple in triple_dict:
                                many_entity = None
                                many_entity_key = None
                                many_entity_node_type = None
                                for key in entity_keys:
                                    if type(triple[key]) == list:
                                        many_entity = triple[key]
                                        many_entity_key = key
                                        many_entity_node_type = schema_items[key]['items']['node_type'] if 'node_type' in schema_items[key]['items'] else 'entity'
                                        break
                                one_entity = None
                                one_entity_node_type = None
                                for key in entity_keys:
                                    if key != many_entity_key:
                                        one_entity = triple[key]
                                        one_entity_node_type = schema_items[key]['node_type'] if 'node_type' in schema_items[key] else 'event'
                                if many_entity is None or one_entity is None or relation is None:
                                    continue
                                if one_entity.isspace() or relation.isspace():
                                    continue
                                if len(many_entity) == 0 or len(one_entity) == 0 or len(relation) == 0:
                                    continue
                                # Add nodes to files
                                if one_entity not in visited_nodes:
                                    visited_nodes.add(one_entity)
                                    all_events.add(one_entity)
                                    writer_node.writerow([one_entity, one_entity_node_type, [], [], "Node"])
                                    csv_writer_edge_text.writerow([head_entity, text_hash_id, "Source"])

                                for entity in many_entity:
                                    if entity is None or entity.isspace():
                                        continue
                                    if entity not in visited_nodes:
                                        visited_nodes.add(entity)
                                        all_entities.add(entity)
                                        writer_node.writerow([entity, many_entity_node_type, [], [], "Node"])
                                        csv_writer_edge_text.writerow([entity, text_hash_id, "Source"])

                                    all_relations.add(relation)
                                    writer_edge.writerow([head_entity, entity, relation, [], [], "Relation"])
                                    
    with open(missing_concepts_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Type"])
        for entity in all_entities:
            writer.writerow([entity, "Entity"])
        for event in all_events:
            writer.writerow([event, "Event"])
        for relation in all_relations:
            writer.writerow([relation, "Relation"])

def json2csv(dataset, data_dir, output_dir, schema, custom, test=False):
    """
    Convert JSON files to CSV files for nodes, edges, and missing concepts.

    Args:
        dataset (str): Name of the dataset.
        data_dir (str): Directory containing the JSON files.
        output_dir (str): Directory to save the output CSV files.
        test (bool): If True, run in test mode (process only 3 files).
    """
    if custom:
        return custom_schema_json_2_csv(dataset, data_dir, output_dir, schema)
    visited_nodes = set()
    visited_hashes = set()

    all_entities = set()
    all_events = set()
    all_relations = set()

    file_dir_list = [f for f in os.listdir(data_dir) if dataset in f]
    file_dir_list = sorted(file_dir_list)
    if test:
        file_dir_list = file_dir_list[:3]
    print("Loading data from the json files")
    print("Number of files: ", len(file_dir_list))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Define output file paths
    node_csv_without_emb = os.path.join(output_dir, f"triple_nodes_{dataset}_from_json_without_emb.csv")
    edge_csv_without_emb = os.path.join(output_dir, f"triple_edges_{dataset}_from_json_without_emb.csv")
    node_text_file = os.path.join(output_dir, f"text_nodes_{dataset}_from_json.csv")
    edge_text_file = os.path.join(output_dir, f"text_edges_{dataset}_from_json.csv")
    missing_concepts_file = os.path.join(output_dir, f"missing_concepts_{dataset}_from_json.csv")

    if test:
        node_text_file = os.path.join(output_dir, f"text_nodes_{dataset}_from_json_test.csv")
        edge_text_file = os.path.join(output_dir, f"text_edges_{dataset}_from_json_test.csv")
        node_csv_without_emb = os.path.join(output_dir, f"triple_nodes_{dataset}_from_json_without_emb_test.csv")
        edge_csv_without_emb = os.path.join(output_dir, f"triple_edges_{dataset}_from_json_without_emb_test.csv")
        missing_concepts_file = os.path.join(output_dir, f"missing_concepts_{dataset}_from_json_test.csv")

    # Open CSV files for writing
    with open(node_text_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_node_text, \
         open(edge_text_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_edge_text, \
         open(node_csv_without_emb, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_node, \
         open(edge_csv_without_emb, "w", newline='', encoding='utf-8', errors='ignore') as csvfile_edge:

        csv_writer_node_text = csv.writer(csvfile_node_text)
        csv_writer_edge_text = csv.writer(csvfile_edge_text)
        writer_node = csv.writer(csvfile_node)
        writer_edge = csv.writer(csvfile_edge)

        # Write headers
        csv_writer_node_text.writerow(["text_id:ID", "original_text", ":LABEL"])
        csv_writer_edge_text.writerow([":START_ID", ":END_ID", ":TYPE"])
        writer_node.writerow(["name:ID", "type", "concepts", "synsets", ":LABEL"])
        writer_edge.writerow([":START_ID", ":END_ID", "relation", "concepts", "synsets", ":TYPE"])
        
        # Process each file
        for file_dir in tqdm(file_dir_list):
            print("Processing file for file ids: ", file_dir)
            with open(os.path.join(data_dir, file_dir), "r") as jsonfile:
                for line in jsonfile:
                    data = json.loads(line.strip())
                    original_text = data["original_text"]
                    original_text = remove_NUL(original_text)
                    if "Here is the passage." in original_text:
                        original_text = original_text.split("Here is the passage.")[-1]
                    eot_token = "<|eot_id|>"
                    original_text = original_text.split(eot_token)[0]

                    text_hash_id = compute_hash_id(original_text)

                    # Write the original text as nodes
                    if text_hash_id not in visited_hashes:
                        visited_hashes.add(text_hash_id)
                        csv_writer_node_text.writerow([text_hash_id, original_text, "Text"])

                    file_id = str(data["id"])
                    entity_relation_dict = data["entity_relation_dict"]
                    event_entity_relation_dict = data["event_entity_dict"]
                    event_relation_dict = data["event_relation_dict"]

                    # Process entity triples
                    entity_triples = []
                    for entity_triple in entity_relation_dict:
                        try:
                            assert isinstance(entity_triple["Head"], str)
                            assert isinstance(entity_triple["Relation"], str)
                            assert isinstance(entity_triple["Tail"], str)

                            head_entity = entity_triple["Head"]
                            relation = entity_triple["Relation"]
                            tail_entity = entity_triple["Tail"]

                            # Clean the text
                            head_entity = clean_text(head_entity)
                            relation = clean_text(relation)
                            tail_entity = clean_text(tail_entity)

                            if head_entity.isspace() or len(head_entity) == 0 or tail_entity.isspace() or len(tail_entity) == 0:
                                continue

                            entity_triples.append((head_entity, relation, tail_entity))
                        except:
                            print(f"Error processing entity triple: {entity_triple}")
                            continue

                    # Process event triples
                    event_triples = []
                    for event_triple in event_relation_dict:
                        try:
                            assert isinstance(event_triple["Head"], str)
                            assert isinstance(event_triple["Relation"], str)
                            assert isinstance(event_triple["Tail"], str)
                            head_event = event_triple["Head"]
                            relation = event_triple["Relation"]
                            tail_event = event_triple["Tail"]
                            
                            # Clean the text
                            head_event = clean_text(head_event)
                            relation = clean_text(relation)
                            tail_event = clean_text(tail_event)

                            if head_event.isspace() or len(head_event) == 0 or tail_event.isspace() or len(tail_event) == 0:
                                continue

                            event_triples.append((head_event, relation, tail_event))
                        except:
                            print(f"Error processing event triple: {event_triple}")

                    # Process event-entity triples
                    event_entity_triples = []
                    for event_entity_participations in event_entity_relation_dict:
                        if "Event" not in event_entity_participations or "Entity" not in event_entity_participations:
                            continue
                        if not isinstance(event_entity_participations["Event"], str) or not isinstance(event_entity_participations["Entity"], list):
                            continue

                        for entity in event_entity_participations["Entity"]:
                            if not isinstance(entity, str):
                                continue

                            entity = clean_text(entity)
                            event = clean_text(event_entity_participations["Event"])

                            if event.isspace() or len(event) == 0 or entity.isspace() or len(entity) == 0:
                                continue

                            event_entity_triples.append((event, "is participated by", entity))

                    # Write nodes and edges to CSV files
                    for entity_triple in entity_triples:
                        head_entity, relation, tail_entity = entity_triple
                        if head_entity is None or tail_entity is None or relation is None:
                            continue
                        if head_entity.isspace() or tail_entity.isspace() or relation.isspace():
                            continue
                        if len(head_entity) == 0 or len(tail_entity) == 0 or len(relation) == 0:
                            continue

                        # Add nodes to files
                        if head_entity not in visited_nodes:
                            visited_nodes.add(head_entity)
                            all_entities.add(head_entity)
                            writer_node.writerow([head_entity, "entity", [], [], "Node"])
                            csv_writer_edge_text.writerow([head_entity, text_hash_id, "Source"])

                        if tail_entity not in visited_nodes:
                            visited_nodes.add(tail_entity)
                            all_entities.add(tail_entity)
                            writer_node.writerow([tail_entity, "entity", [], [], "Node"])
                            csv_writer_edge_text.writerow([tail_entity, text_hash_id, "Source"])

                        all_relations.add(relation)
                        writer_edge.writerow([head_entity, tail_entity, relation, [], [], "Relation"])

                    for event_triple in event_triples:
                        head_event, relation, tail_event = event_triple
                        if head_event is None or tail_event is None or relation is None:
                            continue
                        if head_event.isspace() or tail_event.isspace() or relation.isspace():
                            continue
                        if len(head_event) == 0 or len(tail_event) == 0 or len(relation) == 0:
                            continue

                        # Add nodes to files
                        if head_event not in visited_nodes:
                            visited_nodes.add(head_event)
                            all_events.add(head_event)
                            writer_node.writerow([head_event, "event", [], [], "Node"])
                            csv_writer_edge_text.writerow([head_event, text_hash_id, "Source"])

                        if tail_event not in visited_nodes:
                            visited_nodes.add(tail_event)
                            all_events.add(tail_event)
                            writer_node.writerow([tail_event, "event", [], [], "Node"])
                            csv_writer_edge_text.writerow([tail_event, text_hash_id, "Source"])

                        all_relations.add(relation)
                        writer_edge.writerow([head_event, tail_event, relation, [], [], "Relation"])

                    for event_entity_triple in event_entity_triples:
                        head_event, relation, tail_entity = event_entity_triple
                        if head_event is None or tail_entity is None or relation is None:
                            continue
                        if head_event.isspace() or tail_entity.isspace() or relation.isspace():
                            continue
                        if len(head_event) == 0 or len(tail_entity) == 0 or len(relation) == 0:
                            continue

                        # Add nodes to files
                        if head_event not in visited_nodes:
                            visited_nodes.add(head_event)
                            all_events.add(head_event)
                            writer_node.writerow([head_event, "event", [], [], "Node"])
                            csv_writer_edge_text.writerow([head_event, text_hash_id, "Source"])

                        if tail_entity not in visited_nodes:
                            visited_nodes.add(tail_entity)
                            all_entities.add(tail_entity)
                            writer_node.writerow([tail_entity, "entity", [], [], "Node"])
                            csv_writer_edge_text.writerow([tail_entity, text_hash_id, "Source"])

                        all_relations.add(relation)
                        writer_edge.writerow([head_event, tail_entity, relation, [], [], "Relation"])

    # Write missing concepts to CSV
    with open(missing_concepts_file, "w", newline='', encoding='utf-8', errors='ignore') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Type"])
        for entity in all_entities:
            writer.writerow([entity, "Entity"])
        for event in all_events:
            writer.writerow([event, "Event"])
        for relation in all_relations:
            writer.writerow([relation, "Relation"])

    print("Data to CSV completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="[pes2o_abstract, en_simple_wiki_v0, cc_en]")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the graph raw JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV files")
    parser.add_argument("--test", action="store_true", help="Test the script")
    args = parser.parse_args()
    json2csv(dataset=args.dataset, data_dir=args.data_dir, output_dir=args.output_dir, test=args.test)