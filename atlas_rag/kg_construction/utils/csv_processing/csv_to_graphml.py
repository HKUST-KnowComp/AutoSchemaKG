import json
import networkx as nx
import csv
import ast
import hashlib
import os
from atlas_rag.kg_construction.triple_config import ProcessingConfig
import pickle
import html
import re

# Regex to match *illegal* XML characters (XML 1.0 spec)
_ILLEGAL_XML_RE = re.compile(
    "[" +
    "\x00-\x08" +
    "\x0B" +
    "\x0C" +
    "\x0E-\x1F" +
    "\uD800-\uDFFF" +   # Surrogates
    "\uFFFE\uFFFF" +    # Noncharacters
    "]"
)

def sanitize_xml_string(s: str) -> str:
    """Remove illegal XML characters from a string."""
    return _ILLEGAL_XML_RE.sub("", s)


def get_node_id(entity_name, entity_to_id=None):
    """Returns existing or creates new nX ID for an entity using a hash-based approach."""
    if entity_to_id is None:
        entity_to_id = {}
    if entity_name not in entity_to_id:
        # Use a hash function to generate a unique ID
        entity_name = entity_name+'_entity'
        hash_object = hashlib.sha256(entity_name.encode('utf-8'))
        hash_hex = hash_object.hexdigest()  # Get the hexadecimal representation of the hash
        # Use the first 8 characters of the hash as the ID (you can adjust the length as needed)
        entity_to_id[entity_name] = hash_hex
    return entity_to_id[entity_name]

def csvs_to_temp_graphml(triple_node_file, triple_edge_file, config:ProcessingConfig=None):
    g = nx.DiGraph()
    entity_to_id = {}

    # Add triple nodes
    with open(triple_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["name:ID"]
            mapped_id = get_node_id(node_id, entity_to_id)
            if mapped_id not in g.nodes:
                g.add_node(mapped_id, id=node_id, type=row["type"]) 
            

    # Add triple edges
    with open(triple_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = get_node_id(row[":END_ID"], entity_to_id)
            # Ensure referenced nodes exist (prevent silent orphan creation by add_edge)
            for node_id, name in [(start_id, row[":START_ID"]), (end_id, row[":END_ID"])]:
                if node_id not in g.nodes:
                    print(f"Warning: Node '{name}' referenced in edge but missing from node CSV. Adding as type='Unknown'.")
                    g.add_node(node_id, id=name, type="Unknown")
            # Check if edge already exists to prevent duplicates
            if not g.has_edge(start_id, end_id):
                g.add_edge(start_id, end_id, relation=row["relation"], type=row[":TYPE"])

    # save graph to 
    output_name = f"{config.output_directory}/kg_graphml/{config.filename_pattern}_without_concept.pkl"
    # check if output file directory exists
    output_dir = os.path.dirname(output_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # store the graph to a pickle file
    with open(output_name, 'wb') as output_file:
        pickle.dump(g, output_file)
    
def validate_graphml(output_file):
    """Validate that a GraphML file can be read back correctly."""
    try:
        # Try to read the file back
        test_graph = nx.read_graphml(output_file)
        node_count = test_graph.number_of_nodes()
        edge_count = test_graph.number_of_edges()
        print(f"GraphML validation successful: {node_count} nodes, {edge_count} edges")
        return True
    except Exception as e:
        print(f"ERROR: GraphML validation failed: {str(e)}")
        # Optionally print the line number where the error occurred
        if hasattr(e, 'position'):
            line_no = e.position[0]
            print(f"Error at line {line_no}")
            
            # Read the problematic line
            with open(output_file, 'r') as f:
                lines = f.readlines()
                if line_no - 1 < len(lines):
                    print(f"Problematic line: {lines[line_no-1].strip()}")
        return False

def csvs_to_graphml(triple_node_file, text_node_file, triple_edge_file, text_edge_file, 
                    concept_node_file = None, concept_edge_file = None,
                    output_file = "kg.graphml",
                    include_concept = True):
    '''
    Convert multiple CSV files into a single GraphML file.
    
    Types of nodes to be added to the graph:
    - Triple nodes: Nodes representing triples, with properties like subject, predicate, object.
    - Text nodes: Nodes representing text, with properties like text content.
    - Concept nodes: Nodes representing concepts, with properties like concept name and type.

    Types of edges to be added to the graph:
    - Triple edges: Edges representing relationships between triples, with properties like relation type.
    - Text edges: Edges representing relationships between text and nodes, with properties like text type.
    - Concept edges: Edges representing relationships between concepts and nodes, with properties like concept type.
    
    DiGraph networkx attributes:
    Node:
    - type: Type of the node (e.g., entity, event, text, concept).
    - file_id: List of text IDs the node is associated with.
    - id: Node Name 
    Edge:
    - relation: relation name
    - file_id: List of text IDs the edge is associated with.
    - type: Type of the edge (e.g., Source, Relation, Concept).
    - synsets: List of synsets associated with the edge.
    
    '''
    def safe_sanitize(value):
        """Safely sanitize any value for XML output."""
        if value is None:
            return ""
        return sanitize_xml_string(str(value))
    
    g = nx.DiGraph()
    entity_to_id = {}

    # Add triple nodes
    with open(triple_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["name:ID"]
            mapped_id = get_node_id(node_id, entity_to_id)
            # Check if node already exists to prevent duplicates
            if mapped_id not in g.nodes:
                g.add_node(mapped_id, id=safe_sanitize(node_id), type=safe_sanitize(row["type"]))

    # Add text nodes
    with open(text_node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["text_id:ID"]
            # Check if node already exists to prevent duplicates
            if node_id not in g.nodes:
                g.add_node(safe_sanitize(node_id), 
                          file_id=safe_sanitize(node_id), 
                          id=safe_sanitize(row["original_text"]), 
                          type="passage")

    # Add concept nodes
    if concept_node_file is not None:
        with open(concept_node_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = row["concept_id:ID"]
                # Check if node already exists to prevent duplicates
                if node_id not in g.nodes:
                    g.add_node(safe_sanitize(node_id), 
                              file_id="concept_file", 
                              id=safe_sanitize(row["name"]), 
                              type="concept")

    # Add file id for triple nodes and concept nodes when add the edges
    
    # Add triple edges
    with open(triple_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = get_node_id(row[":END_ID"], entity_to_id)
            # Ensure referenced nodes exist (prevent silent orphan creation by add_edge)
            for node_id, name in [(start_id, row[":START_ID"]), (end_id, row[":END_ID"])]:
                if node_id not in g.nodes:
                    print(f"Warning: Node '{name}' referenced in edge but missing from node CSV. Adding as type='Unknown'.")
                    g.add_node(node_id, id=safe_sanitize(name), type="Unknown")
            # Check if edge already exists to prevent duplicates
            if not g.has_edge(start_id, end_id):
                g.add_edge(start_id, end_id,
                          relation=safe_sanitize(row["relation"]),
                          type=safe_sanitize(row[":TYPE"]))
                # Add file_id to start and end nodes if they are triple or concept nodes
                for node_id in [start_id, end_id]:
                    if g.nodes[node_id]['type'] in ['triple', 'concept'] and 'file_id' not in g.nodes[node_id]:
                        g.nodes[node_id]['file_id'] = safe_sanitize(row.get("file_id", "triple_file"))
            
            if include_concept and "concepts" in row:
                try:
                    # Add concepts to the edge
                    concepts = ast.literal_eval(row["concepts"])
                    for concept in concepts:
                        concept_str = safe_sanitize(concept)
                        if "concepts" not in g.edges[start_id, end_id]:
                            g.edges[start_id, end_id]['concepts'] = concept_str
                        else:
                            # Avoid duplicate concepts by checking if concept is already in the list
                            current_concepts = g.edges[start_id, end_id]['concepts'].split(",")
                            if concept_str not in current_concepts:
                                g.edges[start_id, end_id]['concepts'] += "," + concept_str
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse concepts for edge {start_id}->{end_id}: {e}")
                    # Skip malformed concepts
                    pass
            

    # Add text edges
    with open(text_edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_id = get_node_id(row[":START_ID"], entity_to_id)
            end_id = safe_sanitize(row[":END_ID"])
            # Check if edge already exists to prevent duplicates
            if not g.has_edge(start_id, end_id):
                g.add_edge(start_id, end_id, 
                          relation="mention in", 
                          type=safe_sanitize(row[":TYPE"]))
                # Add file_id to start node if it is a triple or concept node
                if 'file_id' in g.nodes[start_id]:
                    current_file_id = g.nodes[start_id]['file_id']
                    g.nodes[start_id]['file_id'] = safe_sanitize(current_file_id + "," + str(end_id))
                else:
                    g.nodes[start_id]['file_id'] = safe_sanitize(str(end_id))

    # Add concept edges between triple nodes and concept nodes
    if concept_edge_file is not None:
        with open(concept_edge_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                start_id = get_node_id(row[":START_ID"], entity_to_id)
                end_id = safe_sanitize(row[":END_ID"])  # end id is concept node id
                if not g.has_edge(start_id, end_id):
                    g.add_edge(start_id, end_id, 
                              relation=safe_sanitize(row["relation"]), 
                              type=safe_sanitize(row[":TYPE"]))

    # Write to GraphML
    # check if output file directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        nx.write_graphml(g, output_file, infer_numeric_types=True)
        if validate_graphml(output_file):
            print(f"Successfully created GraphML file: {output_file}")
        else:
            print(f"Failed to create valid GraphML file: {output_file}")
    except Exception as e:
        print(f"Error writing GraphML file with numeric inference: {e}")
        # Try writing without numeric type inference
        try:
            nx.write_graphml(g, output_file, infer_numeric_types=False)
            print(f"Successfully created GraphML file (without numeric inference): {output_file}")
        except Exception as e2:
            print(f"Failed to write GraphML file even without numeric inference: {e2}")
            # Save as pickle as fallback
            pickle_file = output_file.replace('.graphml', '.pkl')
            with open(pickle_file, 'wb') as f:
                pickle.dump(g, f)
            print(f"Saved graph as pickle file instead: {pickle_file}")
            raise e2
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert CSV files to GraphML format.')
    parser.add_argument('--triple_node_file', type=str, required=True, help='Path to the triple node CSV file.')
    parser.add_argument('--text_node_file', type=str, required=True, help='Path to the text node CSV file.')
    parser.add_argument('--concept_node_file', type=str, required=True, help='Path to the concept node CSV file.')
    parser.add_argument('--triple_edge_file', type=str, required=True, help='Path to the triple edge CSV file.')
    parser.add_argument('--text_edge_file', type=str, required=True, help='Path to the text edge CSV file.')
    parser.add_argument('--concept_edge_file', type=str, required=True, help='Path to the concept edge CSV file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output GraphML file.')

    args = parser.parse_args()
    
    csvs_to_graphml(args.triple_node_file, args.text_node_file, args.concept_node_file,
                    args.triple_edge_file, args.text_edge_file, args.concept_edge_file,
                    args.output_file)