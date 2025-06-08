# End-to-End RAG Pipeline with FastAPI

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using FastAPI. It provides endpoints for retrieving relevant text chunks and generating answers using an LLM, with robust data processing and retrieval strategies.

## Features
- Data cleaning, chunking, and embedding
- Vector and hybrid retrieval (FAISS)
- LLM-based answer generation (TogetherAI API)
- Modular FastAPI endpoints: `/retrieve` and `/generate`
- (Bonus) Simple Streamlit UI

## Setup
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set TogetherAI API key:**
   - Create a `.env` file with:  
     `TOGETHER_API_KEY=your_api_key_here`

## Running the API
```bash
uvicorn app.main:app --reload
```

## Endpoints
- `POST /retrieve`: Retrieve Top-K relevant chunks
- `POST /generate`: Generate answer using retrieved context

## Bonus: Run the UI
```bash
streamlit run app/ui.py
```

---

## Project Structure
```
app/
  main.py
  data_utils.py
  retrieval.py
  generation.py
  schemas.py
  config.py
```

data/  # For raw and processed data
``` 