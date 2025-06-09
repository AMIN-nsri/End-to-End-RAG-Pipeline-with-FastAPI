import logging
from fastapi import FastAPI, HTTPException
from app.schemas import RetrieveRequest, RetrieveResponse, GenerateRequest, GenerateResponse, ChunkWithMetadata
from app.data_utils import load_documents, clean_text, chunk_text, embed_chunks
from app.retrieval import build_faiss_index, search_faiss_index, embed_query, hybrid_retrieve
from app.generation import call_llm_with_context
from app.config import DATA_DIR, CHUNK_SIZE, OVERLAP, EMBED_MODEL, DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P
import os
import numpy as np
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global state for data, embeddings, and index ---
all_chunks = []
embeddings = None
faiss_index = None
executor = ThreadPoolExecutor(max_workers=4)

@app.on_event('startup')
def startup_event():
    global all_chunks, embeddings, faiss_index
    logger.info('Loading and processing documents...')
    docs = load_documents(DATA_DIR)
    all_chunks = []
    for doc in docs:
        cleaned = clean_text(doc['text'])
        chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=OVERLAP, method='sentence')
        for chunk in chunks:
            all_chunks.append({'text': chunk, 'metadata': doc['metadata'].copy()})
    logger.info(f'Total chunks: {len(all_chunks)}')
    chunk_texts = [c['text'] for c in all_chunks]
    logger.info('Embedding all chunks...')
    embeddings = embed_chunks(chunk_texts, model_name=EMBED_MODEL)
    logger.info('Building FAISS index...')
    faiss_index = build_faiss_index(embeddings)
    logger.info('Startup complete.')

# --- Caching for query embeddings ---
@lru_cache(maxsize=128)
def cached_embed_query(query: str, model_name: str = EMBED_MODEL):
    return embed_query(query, model_name=model_name)

@app.post('/retrieve', response_model=RetrieveResponse)
async def retrieve_chunks(request: RetrieveRequest):
    global all_chunks, embeddings
    if not all_chunks or embeddings is None:
        raise HTTPException(status_code=500, detail='Data not loaded.')
    assert embeddings is not None, 'Embeddings must not be None'  # type assertion for linter
    embeddings_np: np.ndarray = embeddings
    loop = asyncio.get_event_loop()
    # Use cached embedding for the query
    query_emb = await loop.run_in_executor(executor, cached_embed_query, request.query, EMBED_MODEL)
    try:
        results = await loop.run_in_executor(
            executor,
            lambda: hybrid_retrieve(
                all_chunks,
                embeddings_np,
                query_emb,
                request.top_k,
                request.metadata_filter
            )
        )
    except Exception as e:
        logger.error(f'Retrieval error: {e}')
        raise HTTPException(status_code=500, detail=f'Retrieval error: {e}')
    return RetrieveResponse(
        chunks=[ChunkWithMetadata(text=c['text'], metadata=c['metadata']) for c in results]
    )

@app.post('/generate', response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    global all_chunks, embeddings
    if not all_chunks or embeddings is None:
        raise HTTPException(status_code=500, detail='Data not loaded.')
    assert embeddings is not None, 'Embeddings must not be None'  # type assertion for linter
    embeddings_np: np.ndarray = embeddings
    loop = asyncio.get_event_loop()
    # Ensure defaults for optional parameters
    model_name = request.model_name or DEFAULT_LLM_MODEL
    temperature = request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE
    max_tokens = request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_TOKENS
    top_p = request.top_p if request.top_p is not None else DEFAULT_TOP_P
    query_emb = await loop.run_in_executor(executor, cached_embed_query, request.query, EMBED_MODEL)
    try:
        results = await loop.run_in_executor(
            executor,
            lambda: hybrid_retrieve(
                all_chunks,
                embeddings_np,
                query_emb,
                request.top_k,
                request.metadata_filter
            )
        )
        context_chunks = [c['text'] for c in results]
        answer = await loop.run_in_executor(
            executor,
            lambda: call_llm_with_context(
                context_chunks,
                request.query,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
        )
    except Exception as e:
        logger.error(f'Generation error: {e}')
        raise HTTPException(status_code=500, detail=f'Generation error: {e}')
    return GenerateResponse(
        answer=answer,
        chunks=[ChunkWithMetadata(text=c['text'], metadata=c['metadata']) for c in results]
    ) 