# Configuration variables will go here 

import os

# Data and model configuration
DATA_DIR = os.getenv('DATA_DIR', './data')
CHUNK_SIZE = 300
OVERLAP = 50
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_LLM_MODEL = 'meta-llama/Meta-Llama-3-70B-Instruct-Turbo'
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOP_P = 0.95

# API keys
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY') 