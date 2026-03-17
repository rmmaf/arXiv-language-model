"""Streamlit front-end for the arXiv Hybrid RAG pipeline."""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("API_REQUEST_TIMEOUT", "600"))
ARXIV_ABS_BASE = "https://arxiv.org/abs"
TENANT_NAME = os.getenv("TENANT_NAME", "")

page_title = f"arXiv RAG — {TENANT_NAME}" if TENANT_NAME else "arXiv RAG"
st.set_page_config(page_title=page_title, page_icon="📄", layout="centered")
st.title("arXiv Research Assistant")
if TENANT_NAME:
    st.caption(f"Tenant: **{TENANT_NAME}**")
st.caption("Hybrid semantic + lexical search over arXiv papers with a local LLM")

# --- Session state initialisation ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "sources" not in st.session_state:
    st.session_state.sources = []

# --- Sidebar ---
default_key = os.getenv("API_KEY_DEFAULT", "")
api_key = st.sidebar.text_input(
    "API Key", value=default_key, type="password",
    help="Tenant API key for authentication",
)
if not api_key:
    st.sidebar.warning("Enter your API Key to use the assistant.")

st.sidebar.divider()
top_k = st.sidebar.number_input("top_k", min_value=1, max_value=10, value=3, step=1)
fetch_new_papers = st.sidebar.checkbox(
    "Search new papers",
    value=False,
    help="Check this to force a new arXiv search + PDF download for your next question.",
)

st.sidebar.divider()
if st.sidebar.button("Restart conversation", type="secondary", use_container_width=True):
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.sources = []
    st.rerun()

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Display current sources (collapsed) ---
if st.session_state.sources:
    with st.expander("Current paper sources"):
        for src in st.session_state.sources:
            link = f"{ARXIV_ABS_BASE}/{src['paper_id']}"
            st.markdown(f"- [{src['title']}]({link})  \n  score: `{src['score']:.4f}`")

# --- Chat input ---
if question := st.chat_input("Ask about arXiv papers...", disabled=not api_key):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    is_first_message = st.session_state.conversation_id is None
    should_fetch = fetch_new_papers or is_first_message

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating answer..." if should_fetch else "Generating answer..."):
            try:
                payload = {
                    "question": question,
                    "top_k": top_k,
                    "fetch_new_papers": should_fetch,
                }
                if st.session_state.conversation_id:
                    payload["conversation_id"] = st.session_state.conversation_id

                response = httpx.post(
                    f"{API_URL}/api/v1/ask",
                    json=payload,
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

        st.session_state.conversation_id = data["conversation_id"]
        if data["sources"]:
            st.session_state.sources = data["sources"]

        answer = data["answer"]
        st.markdown(answer)
        st.caption(f"Processing time: {data['processing_time_seconds']:.2f}s")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
