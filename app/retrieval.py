# Retrieval logic will go here 

import numpy as np
from typing import List, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index for cosine similarity search (inner product on normalized vectors).
    """
    # Normalize embeddings for cosine similarity (in-place)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_faiss_index(index: faiss.IndexFlatIP, query_embedding: np.ndarray, top_k: int) -> List[int]:
    """
    Search the FAISS index for Top-K most similar vectors.
    Returns indices of the top_k results.
    """
    # Normalize query for cosine similarity (in-place)
    query = query_embedding.copy()
    faiss.normalize_L2(query)
    D, I = index.search(query, top_k)
    return I[0].tolist()


def embed_query(query: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> np.ndarray:
    """
    Embed a query string using the same embedding model as the chunks.
    Returns a 2D numpy array (1, dim).
    """
    model = SentenceTransformer(model_name)
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding


def hybrid_retrieve(
    chunks: List[Dict],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
    metadata_filter: Optional[Dict] = None
) -> List[Dict]:
    """
    Retrieve Top-K most relevant chunks using vector search and optional metadata filtering.
    - chunks: list of dicts with 'text' and 'metadata'
    - embeddings: np.ndarray of chunk embeddings
    - query_embedding: np.ndarray of shape (1, dim)
    - metadata_filter: dict of metadata key-value pairs to filter on (optional)
    Returns: List of chunk dicts (with text and metadata)
    """
    if metadata_filter:
        filtered_indices = [i for i, c in enumerate(chunks)
                            if all(c['metadata'].get(k) == v for k, v in metadata_filter.items())]
        if not filtered_indices:
            return []
        filtered_embeddings = embeddings[filtered_indices]
        faiss.normalize_L2(filtered_embeddings)
        query = query_embedding.copy()
        faiss.normalize_L2(query)
        index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
        index.add(filtered_embeddings)
        D, I = index.search(query, min(top_k, len(filtered_indices)))
        result_indices = [filtered_indices[i] for i in I[0]]
    else:
        faiss.normalize_L2(embeddings)
        query = query_embedding.copy()
        faiss.normalize_L2(query)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        D, I = index.search(query, min(top_k, len(chunks)))
        result_indices = I[0].tolist()
    return [chunks[i] for i in result_indices]


if __name__ == '__main__':
    # Example usage for testing the retrieval pipeline
    import sys
    sys.path.append('.')  # Ensure app/data_utils.py is importable if running from project root
    from app.data_utils import load_documents, clean_text, chunk_text, embed_chunks

    DATA_DIR = '/Users/amin/Documents/University/Term 6/Data Mining/Final Project/End-to-End-RAG-Pipeline-with-FastAPI/data'  # Adjust path as needed
    print('Loading and processing documents...')
    docs = load_documents(DATA_DIR)
    if not docs:
        print('No documents found in data directory.')
        exit()
    # Clean and chunk all documents
    all_chunks = []
    all_metadata = []
    for doc in docs:
        cleaned = clean_text(doc['text'])
        chunks = chunk_text(cleaned, chunk_size=300, overlap=50, method='sentence')
        for chunk in chunks:
            all_chunks.append({'text': chunk, 'metadata': doc['metadata'].copy()})
    print(f'Total chunks: {len(all_chunks)}')
    # Embed all chunks
    chunk_texts = [c['text'] for c in all_chunks]
    print('Embedding all chunks...')
    embeddings = embed_chunks(chunk_texts)
    # Build FAISS index
    print('Building FAISS index...')
    index = build_faiss_index(embeddings)
    # Example query
    query = 'What are the main rivers in France?'
    print(f'Embedding query: "{query}"')
    query_emb = embed_query(query)
    # Vector search
    print('Searching for top 3 relevant chunks (vector search)...')
    top_k = 3
    top_indices = search_faiss_index(index, query_emb, top_k)
    for i, idx in enumerate(top_indices):
        print(f'[{i+1}] {all_chunks[idx]["text"][:200]}...')
    # Hybrid search (simulate metadata filter)
    print('\nTesting hybrid retrieval with metadata filter (filename of first doc)...')
    first_filename = docs[0]['metadata']['filename']
    results = hybrid_retrieve(
        all_chunks, embeddings, query_emb, top_k=2, metadata_filter={'filename': first_filename}
    )
    for i, chunk in enumerate(results):
        print(f'[Hybrid {i+1}] {chunk["text"][:200]}...') 