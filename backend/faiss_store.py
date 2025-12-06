# backend/faiss_store.py
import faiss
import numpy as np
import os
import pickle
import pandas as pd

INDEX_DIR = "vectors"

def ensure_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)

def create_faiss_index(embedding_dim):
    index = faiss.IndexFlatIP(embedding_dim)
    return index

def add_embeddings_to_index(index, embeddings, metadata_list, doc_id):
    vecs = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vecs)
    index.add(vecs)
    meta_path = os.path.join(INDEX_DIR, f"{doc_id}_meta.pkl")
    if os.path.exists(meta_path):
        existing = pd.read_pickle(meta_path)
        new = pd.DataFrame(metadata_list)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = pd.DataFrame(metadata_list)
    combined.to_pickle(meta_path)
    index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
    faiss.write_index(index, index_path)
    return index_path, meta_path

def load_index(doc_id):
    index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
    meta_path = os.path.join(INDEX_DIR, f"{doc_id}_meta.pkl")
    if not os.path.exists(index_path):
        return None, None
    index = faiss.read_index(index_path)
    meta = pd.read_pickle(meta_path)
    return index, meta

def search_index(index, query_vec, top_k=5):
    q = np.array([query_vec]).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, top_k)
    return D[0], I[0]
