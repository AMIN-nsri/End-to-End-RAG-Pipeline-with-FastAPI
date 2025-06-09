# Pydantic models for request/response will go here 
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ChunkWithMetadata(BaseModel):
    text: str
    metadata: Dict[str, str]

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(3, description="Number of top chunks to retrieve")
    metadata_filter: Optional[Dict[str, str]] = Field(None, description="Optional metadata filter")

class RetrieveResponse(BaseModel):
    chunks: List[ChunkWithMetadata]

class GenerateRequest(BaseModel):
    query: str
    top_k: int = Field(3, description="Number of top chunks to use as context")
    metadata_filter: Optional[Dict[str, str]] = Field(None, description="Optional metadata filter")
    model_name: Optional[str] = Field('meta-llama/Meta-Llama-3-70B-Instruct-Turbo', description="LLM model name")
    temperature: Optional[float] = Field(0.2, description="LLM temperature")
    max_tokens: Optional[int] = Field(512, description="Max tokens for LLM output")
    top_p: Optional[float] = Field(0.95, description="LLM top_p parameter")

class GenerateResponse(BaseModel):
    answer: str
    chunks: List[ChunkWithMetadata] 