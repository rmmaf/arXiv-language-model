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
if "custom_docs" not in st.session_state:
    st.session_state.custom_docs = []

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

# --- Custom Document Management ---
st.sidebar.divider()
st.sidebar.subheader("Custom Documents")

if api_key:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF", type=["pdf"], key="pdf_uploader",
    )
    if uploaded_file is not None:
        if st.sidebar.button("Upload", use_container_width=True):
            with st.sidebar:
                with st.spinner("Processing PDF..."):
                    try:
                        resp = httpx.post(
                            f"{API_URL}/api/v1/documents/",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                            headers={"X-API-Key": api_key},
                            timeout=API_TIMEOUT,
                        )
                        if resp.status_code == 201:
                            doc = resp.json()
                            st.success(f"Uploaded **{doc['filename']}** ({doc['total_chunks']} chunks)")
                            st.session_state.custom_docs = []
                        else:
                            st.error(f"Upload failed: {resp.text}")
                    except httpx.RequestError as exc:
                        st.error(f"Could not reach API: {exc}")

    if not st.session_state.custom_docs:
        try:
            resp = httpx.get(
                f"{API_URL}/api/v1/documents/",
                headers={"X-API-Key": api_key},
                timeout=30,
            )
            if resp.status_code == 200:
                st.session_state.custom_docs = resp.json()
        except httpx.RequestError:
            pass

    selected_doc_ids: list[str] = []
    if st.session_state.custom_docs:
        st.sidebar.caption("Select documents to include as context:")
        for doc in st.session_state.custom_docs:
            col1, col2 = st.sidebar.columns([4, 1])
            checked = col1.checkbox(
                doc["filename"],
                key=f"doc_{doc['id']}",
                help=f"{doc['total_chunks']} chunks — uploaded {doc['uploaded_at']}",
            )
            if checked:
                selected_doc_ids.append(doc["id"])
            if col2.button("🗑", key=f"del_{doc['id']}"):
                try:
                    del_resp = httpx.delete(
                        f"{API_URL}/api/v1/documents/{doc['id']}",
                        headers={"X-API-Key": api_key},
                        timeout=30,
                    )
                    if del_resp.status_code == 204:
                        st.session_state.custom_docs = []
                        st.rerun()
                except httpx.RequestError:
                    pass
    else:
        st.sidebar.caption("No custom documents uploaded yet.")

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Display current sources (collapsed) ---
if st.session_state.sources:
    with st.expander("Current sources"):
        for src in st.session_state.sources:
            source_type = src.get("source_type", "arxiv")
            if source_type == "arxiv":
                link = f"{ARXIV_ABS_BASE}/{src['paper_id']}"
                st.markdown(f"- [{src['title']}]({link})  \n  score: `{src['score']:.4f}`")
            else:
                st.markdown(f"- 📎 **{src['title']}** (custom upload)  \n  score: `{src['score']:.4f}`")

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
                if selected_doc_ids:
                    payload["custom_document_ids"] = selected_doc_ids

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
