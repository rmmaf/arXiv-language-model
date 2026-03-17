"""Streamlit front-end for the arXiv Hybrid RAG pipeline."""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("API_REQUEST_TIMEOUT", "600"))
ARXIV_ABS_BASE = "https://arxiv.org/abs"

st.set_page_config(page_title="arXiv RAG", page_icon="📄", layout="centered")
st.title("arXiv Research Assistant")
st.caption("Hybrid semantic + lexical search over arXiv papers with a local LLM")

# --- Sidebar: tenant authentication ---
api_key = st.sidebar.text_input("API Key", type="password", help="Tenant API key for authentication")
if not api_key:
    st.sidebar.warning("Enter your API Key to use the assistant.")

# --- Main form ---
question = st.text_input("Question", placeholder="e.g. What are the latest advances in vision transformers?")
top_k = st.number_input("top_k", min_value=1, max_value=10, value=3, step=1)

ask_clicked = st.button("Ask", type="primary", disabled=(not question or not api_key))

if ask_clicked and question and api_key:
    with st.spinner("Searching papers and generating answer..."):
        try:
            response = httpx.post(
                f"{API_URL}/api/v1/ask",
                json={"question": question, "top_k": top_k},
                headers={"X-API-Key": api_key},
                timeout=API_TIMEOUT,
            )
            if response.status_code == 401:
                st.error("Invalid or inactive API key. Check your key and try again.")
                st.stop()
            if response.status_code == 429:
                st.warning("Rate limit exceeded. Please wait a moment and try again.")
                st.stop()
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except httpx.RequestError as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
            st.stop()

    st.subheader("Answer")
    st.markdown(data["answer"])

    st.subheader("Sources")
    for src in data["sources"]:
        link = f"{ARXIV_ABS_BASE}/{src['paper_id']}"
        st.markdown(f"- [{src['title']}]({link})  \n  score: `{src['score']:.4f}`")

    st.caption(f"Processing time: {data['processing_time_seconds']:.2f}s")
