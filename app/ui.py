import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Demo", page_icon="🔎", layout="centered")
st.title("🔎 Retrieval-Augmented Generation Demo")

st.markdown(
    """
    Enter your question about France's geography, and get an answer grounded in the provided context.
    """
)

query = st.text_input("Your question", placeholder="e.g., What are the main rivers in France?")
top_k = st.slider("Number of context chunks", min_value=1, max_value=10, value=3)

if st.button("Generate Answer", type="primary") and query:
    with st.spinner("Retrieving and generating answer..."):
        payload = {
            "query": query,
            "top_k": top_k,
            "temperature": 0.2,
            "max_tokens": 256
        }
        try:
            resp = requests.post(f"{API_URL}/generate", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            st.success("Answer generated!")
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Retrieved Context Chunks")
            for i, chunk in enumerate(data["chunks"], 1):
                with st.expander(f"Chunk {i} (metadata: {chunk['metadata']})"):
                    st.write(chunk["text"])
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with FastAPI, Streamlit, and TogetherAI") 