import os
import pickle
import networkx as nx
from tqdm import tqdm
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
import faiss
import numpy as np
import torch

def compute_graph_embeddings(node_list, edge_list_string, sentence_encoder: BaseEmbeddingModel, batch_size=40, normalize_embeddings: bool = False):
    # Encode in batches
    node_embeddings = []
    for i in tqdm(range(0, len(node_list), batch_size), desc="Encoding nodes"):
        batch = node_list[i:i + batch_size]
        node_embeddings.extend(sentence_encoder.encode(batch, normalize_embeddings = normalize_embeddings))

    edge_embeddings = []
    for i in tqdm(range(0, len(edge_list_string), batch_size), desc="Encoding edges"):
        batch = edge_list_string[i:i + batch_size]
        edge_embeddings.extend(sentence_encoder.encode(batch, normalize_embeddings = normalize_embeddings))

    return node_embeddings, edge_embeddings

def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    
    faiss_index = faiss.IndexHNSWFlat(dimension, 64, faiss.METRIC_INNER_PRODUCT)
    X = np.array(embeddings).astype('float32')

    # normalize the vectors
    faiss.normalize_L2(X)

    # batched add
    for i in tqdm(range(0, X.shape[0], 32)):
        faiss_index.add(X[i:i+32])
    return faiss_index

def compute_text_embeddings(text_list, sentence_encoder: BaseEmbeddingModel, batch_size = 40, normalize_embeddings: bool = False):
    """Separated text embedding computation"""
    text_embeddings = []
    
    for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding texts"):
        batch = text_list[i:i + batch_size]
        embeddings = sentence_encoder.encode(batch, normalize_embeddings=normalize_embeddings)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        text_embeddings.extend(embeddings)
    return text_embeddings

def create_embeddings_and_index(sentence_encoder, model_name: str, working_directory: str, keyword: str, include_events: bool, include_concept: bool,
                                 normalize_embeddings: bool = True, 
                                 text_batch_size = 40,
                                 node_and_edge_batch_size = 256,
                                 **kwargs):
    # Extract the last part of the encoder_model_name for simplified reference
    encoder_model_name = model_name.split('/')[-1]
    
    print(f"Using encoder model: {encoder_model_name}")
    graph_dir = f"{working_directory}/kg_graphml/{keyword}_graph.graphml"
    if not os.path.exists(graph_dir):
        raise FileNotFoundError(f"Graph file {graph_dir} does not exist. Please check the path or generate the graph first.")

    node_index_path = f"{working_directory}/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_node_faiss.index"
    node_list_path = f"{working_directory}/precompute/{keyword}_event{include_events}_concept{include_concept}_node_list.pkl"
    edge_index_path = f"{working_directory}/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_edge_faiss.index"
    edge_list_path = f"{working_directory}/precompute/{keyword}_event{include_events}_concept{include_concept}_edge_list.pkl"
    node_embeddings_path = f"{working_directory}/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_node_embeddings.pkl"
    edge_embeddings_path = f"{working_directory}/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_edge_embeddings.pkl"
    text_embeddings_path = f"{working_directory}/precompute/{keyword}_{encoder_model_name}_text_embeddings.pkl"
    text_index_path = f"{working_directory}/precompute/{keyword}_text_faiss.index"
    original_text_list_path = f"{working_directory}/precompute/{keyword}_text_list.pkl"
    original_text_dict_with_node_id_path = f"{working_directory}/precompute/{keyword}_original_text_dict_with_node_id.pkl"

    if not os.path.exists(f"{working_directory}/precompute"):
        os.makedirs(f"{working_directory}/precompute", exist_ok=True)

    print(f"Loading graph from {graph_dir}")
    with open(graph_dir, "rb") as f:
        KG: nx.DiGraph = nx.read_graphml(f)

    node_list = list(KG.nodes)
    text_list = [node for node in tqdm(node_list) if "passage" in KG.nodes[node]["type"]]
    
    if not include_events and not include_concept:
        node_list = [node for node in tqdm(node_list) if "entity" in KG.nodes[node]["type"]]
    elif include_events and not include_concept:
        node_list = [node for node in tqdm(node_list) if "event" in KG.nodes[node]["type"] or "entity" in KG.nodes[node]["type"]]
    elif include_events and include_concept:
        node_list = [node for node in tqdm(node_list) if "event" in KG.nodes[node]["type"] or "concept" in KG.nodes[node]["type"] or "entity" in KG.nodes[node]["type"]]
    else:
        raise ValueError("Invalid combination of include_events and include_concept")

    edge_list = list(KG.edges)
    node_set = set(node_list)
    node_list_string = [KG.nodes[node]["id"] for node in node_list]

    # Filter edges based on node list
    edge_list_index = [i for i, edge in tqdm(enumerate(edge_list)) if edge[0] in node_set and edge[1] in node_set]
    edge_list = [edge_list[i] for i in edge_list_index]
    edge_list_string = [f"{KG.nodes[edge[0]]['id']} {KG.edges[edge]['relation']} {KG.nodes[edge[1]]['id']}" for edge in edge_list]

    original_text_list = []
    original_text_dict_with_node_id = {}
    for text_node in text_list:
        text = KG.nodes[text_node]["id"].strip()
        original_text_list.append(text)
        original_text_dict_with_node_id[text_node] = text

    assert len(original_text_list) == len(original_text_dict_with_node_id)

    with open(original_text_list_path, "wb") as f:
        pickle.dump(original_text_list, f)
    with open(original_text_dict_with_node_id_path, "wb") as f:
        pickle.dump(original_text_dict_with_node_id, f)

    if not os.path.exists(text_index_path) or not os.path.exists(text_embeddings_path):
        print("Computing text embeddings...")
        text_embeddings = compute_text_embeddings(original_text_list, sentence_encoder, text_batch_size, normalize_embeddings)  
        text_faiss_index = build_faiss_index(text_embeddings)  
        faiss.write_index(text_faiss_index, text_index_path)
        with open(text_embeddings_path, "wb") as f:
            pickle.dump(text_embeddings, f)
    else:
        print("Text embeddings already computed.")
        with open(text_embeddings_path, "rb") as f:
            text_embeddings = pickle.load(f)
        text_faiss_index = faiss.read_index(text_index_path)

    if not os.path.exists(node_embeddings_path) or not os.path.exists(edge_embeddings_path):
        print("Node and edge embeddings not found, computing...")
        node_embeddings, edge_embeddings = compute_graph_embeddings(node_list_string, edge_list_string, sentence_encoder, node_and_edge_batch_size, normalize_embeddings=normalize_embeddings)  # Assumes this function is defined
    else:
        with open(node_embeddings_path, "rb") as f:
            node_embeddings = pickle.load(f)
        with open(edge_embeddings_path, "rb") as f:
            edge_embeddings = pickle.load(f)
        print("Graph embeddings already computed")
    
    if not os.path.exists(node_index_path):
        node_faiss_index = build_faiss_index(node_embeddings)
        faiss.write_index(node_faiss_index, node_index_path)
    else:
        node_faiss_index = faiss.read_index(node_index_path)
    
    if not os.path.exists(edge_index_path):
        edge_faiss_index = build_faiss_index(edge_embeddings)
        faiss.write_index(edge_faiss_index, edge_index_path)
    else:
        edge_faiss_index = faiss.read_index(edge_index_path)

    if not os.path.exists(node_embeddings_path):
        with open(node_embeddings_path, "wb") as f:
            pickle.dump(node_embeddings, f)

    if not os.path.exists(edge_embeddings_path):
        with open(edge_embeddings_path, "wb") as f:
            pickle.dump(edge_embeddings, f)

    with open(node_list_path, "wb") as f:
        pickle.dump(node_list, f)

    with open(edge_list_path, "wb") as f:
        pickle.dump(edge_list, f)

    print("Node and edge embeddings already computed.")
    # Return all required indices, embeddings, and lists
    return {
        "KG": KG,
        "node_faiss_index": node_faiss_index,
        "edge_faiss_index": edge_faiss_index,
        "text_faiss_index": text_faiss_index,
        "node_embeddings": node_embeddings,
        "edge_embeddings": edge_embeddings,
        "text_embeddings": text_embeddings,
        "node_list": node_list,
        "edge_list": edge_list,
        "text_dict": original_text_dict_with_node_id,
    }
