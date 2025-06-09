# Data processing utilities will go here 

import os
import re
import unicodedata
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional: for sentence-based chunking
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize


def load_documents(data_dir: str) -> List[Dict]:
    """
    Load raw text documents from a directory.
    Returns a list of dicts with 'text' and 'metadata' (filename as metadata).
    """
    documents = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.txt'):
            path = os.path.join(data_dir, fname)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            documents.append({
                'text': text,
                'metadata': {'filename': fname}
            })
    return documents


def clean_text(text: str) -> str:
    """
    Normalize and clean text (remove noise, fix encoding, etc).
    """
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable/control characters
    text = ''.join(c for c in text if c.isprintable())
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    method: str = 'sentence'  # 'sentence' or 'fixed'
) -> List[str]:
    """
    Split text into chunks using either sentence-based or fixed-length method.
    - 'sentence': group sentences into chunks of ~chunk_size words, with overlap.
    - 'fixed': split text into chunks of chunk_size words, with overlap.
    """
    if method == 'sentence':
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_len = 0
        i = 0
        while i < len(sentences):
            sent_words = sentences[i].split()
            if current_len + len(sent_words) <= chunk_size:
                current_chunk.append(sentences[i])
                current_len += len(sent_words)
                i += 1
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                # Overlap: keep last N words
                if overlap > 0 and current_chunk:
                    overlap_words = ' '.join(current_chunk).split()[-overlap:]
                    current_chunk = [' '.join(overlap_words)]
                    current_len = len(overlap_words)
                else:
                    current_chunk = []
                    current_len = 0
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks
    elif method == 'fixed':
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            chunks.append(' '.join(chunk))
            i += chunk_size - overlap if chunk_size > overlap else chunk_size
        return chunks
    else:
        raise ValueError("Unknown chunking method: choose 'sentence' or 'fixed'")


def attach_metadata(docs: List[Dict], metadata: Dict) -> List[Dict]:
    """
    Attach metadata to each document chunk.
    """
    for doc in docs:
        doc['metadata'].update(metadata)
    return docs


def embed_chunks(chunks: List[str], model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> np.ndarray:
    """
    Embed text chunks using a sentence-transformers model.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings 

if __name__ == '__main__':
    # Example usage for testing the data processing pipeline
    DATA_DIR = '/Users/amin/Documents/University/Term 6/Data Mining/Final Project/End-to-End-RAG-Pipeline-with-FastAPI/data'  # Adjust path as needed
    print('Loading documents...')
    docs = load_documents(DATA_DIR)
    print(f'Loaded {len(docs)} documents.')
    if docs:
        # Clean the first document
        raw_text = docs[0]['text']
        cleaned = clean_text(raw_text)
        print(f'First 300 chars of cleaned text:\n{cleaned[:300]}\n')
        # Chunk the cleaned text (sentence-based)
        chunks = chunk_text(cleaned, chunk_size=300, overlap=50, method='sentence')
        print(f'Number of chunks (sentence-based): {len(chunks)}')
        print(f'First chunk:\n{chunks[0][:300]}\n')
        # Chunk the cleaned text (fixed-length)
        fixed_chunks = chunk_text(cleaned, chunk_size=300, overlap=50, method='fixed')
        print(f'Number of chunks (fixed-length): {len(fixed_chunks)}')
        print(f'First fixed chunk:\n{fixed_chunks[0][:300]}\n')
        # Embed the first 3 chunks
        to_embed = chunks[:3]
        print('Embedding first 3 chunks...')
        embeddings = embed_chunks(to_embed)
        print(f'Embeddings shape: {embeddings.shape}')
    else:
        print('No documents found in data directory.') 