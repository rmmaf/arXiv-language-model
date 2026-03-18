"""Streamlit front-end for the arXiv Hybrid RAG pipeline.

Supports persistent multi-conversation sessions, background task polling
with a stop button, and automatic session restoration on reload.
"""

import os
import time

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("API_REQUEST_TIMEOUT", "600"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2"))
ARXIV_ABS_BASE = "https://arxiv.org/abs"
TENANT_NAME = os.getenv("TENANT_NAME", "")

page_title = f"arXiv RAG — {TENANT_NAME}" if TENANT_NAME else "arXiv RAG"
st.set_page_config(page_title=page_title, page_icon="📄", layout="centered")
st.title("arXiv Research Assistant")
if TENANT_NAME:
    st.caption(f"Tenant: **{TENANT_NAME}**")
st.caption("Hybrid semantic + lexical search over arXiv papers with a local LLM")

# ------------------------------------------------------------------ #
#  Session state defaults
# ------------------------------------------------------------------ #
_DEFAULTS = {
    "messages": [],
    "conversation_id": None,
    "sources": [],
    "custom_docs": [],
    "conversation_list": [],
    "conversations_loaded": False,
    "pending_task_id": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------------------------------------------------------ #
#  Helper: load a conversation from the backend
# ------------------------------------------------------------------ #
def _load_conversation(conv_id: str, api_key: str) -> bool:
    """Fetch conversation detail and populate session state. Returns True on success."""
    try:
        resp = httpx.get(
            f"{API_URL}/api/v1/conversations/{conv_id}",
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]}
            for m in data.get("messages", [])
        ]
        st.session_state.conversation_id = data["conversation_id"]
        st.session_state.sources = [
            {
                "paper_id": s.get("paper_id", ""),
                "title": s.get("title", ""),
                "score": s.get("score", 0),
                "source_type": s.get("source_type", "arxiv"),
            }
            for s in data.get("sources", [])
        ]
        st.session_state.pending_task_id = data.get("pending_task_id")
        return True
    except httpx.RequestError:
        return False


def _refresh_conversation_list(api_key: str) -> None:
    """Fetch the conversation list for the current tenant."""
    try:
        resp = httpx.get(
            f"{API_URL}/api/v1/conversations",
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        if resp.status_code == 200:
            st.session_state.conversation_list = resp.json()
    except httpx.RequestError:
        pass


# ------------------------------------------------------------------ #
#  Sidebar: API key
# ------------------------------------------------------------------ #
default_key = os.getenv("API_KEY_DEFAULT", "")
api_key = st.sidebar.text_input(
    "API Key", value=default_key, type="password",
    help="Tenant API key for authentication",
)
if not api_key:
    st.sidebar.warning("Enter your API Key to use the assistant.")

# ------------------------------------------------------------------ #
#  Sidebar: Conversation list (only when authenticated)
# ------------------------------------------------------------------ #
if api_key:
    if not st.session_state.conversations_loaded:
        _refresh_conversation_list(api_key)
        st.session_state.conversations_loaded = True
        if st.session_state.conversation_list and not st.session_state.conversation_id:
            most_recent = st.session_state.conversation_list[0]
            _load_conversation(most_recent["id"], api_key)

    st.sidebar.divider()

    if st.sidebar.button("+ New conversation", use_container_width=True):
        try:
            resp = httpx.post(
                f"{API_URL}/api/v1/conversations",
                headers={"X-API-Key": api_key},
                timeout=30,
            )
            if resp.status_code == 201:
                new_conv = resp.json()
                st.session_state.conversation_id = new_conv["id"]
                st.session_state.messages = []
                st.session_state.sources = []
                st.session_state.pending_task_id = None
                _refresh_conversation_list(api_key)
                st.rerun()
        except httpx.RequestError as exc:
            st.sidebar.error(f"Could not create conversation: {exc}")

    for conv in st.session_state.conversation_list:
        is_active = conv["id"] == st.session_state.conversation_id
        has_pending = conv.get("pending_task_id") is not None
        label = conv["title"] or "New conversation"
        if has_pending:
            label = "⏳ " + label

        col_btn, col_del = st.sidebar.columns([5, 1])
        btn_type = "primary" if is_active else "secondary"
        if col_btn.button(
            label, key=f"conv_{conv['id']}", type=btn_type, use_container_width=True,
        ):
            if not is_active:
                _load_conversation(conv["id"], api_key)
                st.rerun()

        if col_del.button("🗑", key=f"del_conv_{conv['id']}"):
            try:
                httpx.delete(
                    f"{API_URL}/api/v1/conversations/{conv['id']}",
                    headers={"X-API-Key": api_key},
                    timeout=30,
                )
                if conv["id"] == st.session_state.conversation_id:
                    st.session_state.conversation_id = None
                    st.session_state.messages = []
                    st.session_state.sources = []
                    st.session_state.pending_task_id = None
                _refresh_conversation_list(api_key)
                st.rerun()
            except httpx.RequestError:
                pass

# ------------------------------------------------------------------ #
#  Sidebar: Settings
# ------------------------------------------------------------------ #
st.sidebar.divider()
top_k = st.sidebar.number_input("top_k", min_value=1, max_value=10, value=3, step=1)
fetch_new_papers = st.sidebar.checkbox(
    "Search new papers",
    value=False,
    help="Check this to force a new arXiv search + PDF download for your next question.",
)

# ------------------------------------------------------------------ #
#  Sidebar: Custom Document Management
# ------------------------------------------------------------------ #
st.sidebar.divider()
st.sidebar.subheader("Custom Documents")

selected_doc_ids: list[str] = []
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


# ------------------------------------------------------------------ #
#  Display chat history
# ------------------------------------------------------------------ #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------ #
#  Display current sources (collapsed)
# ------------------------------------------------------------------ #
if st.session_state.sources:
    with st.expander("Current sources"):
        for src in st.session_state.sources:
            source_type = src.get("source_type", "arxiv")
            if source_type == "arxiv":
                link = f"{ARXIV_ABS_BASE}/{src['paper_id']}"
                st.markdown(f"- [{src['title']}]({link})  \n  score: `{src['score']:.4f}`")
            else:
                st.markdown(f"- 📎 **{src['title']}** (custom upload)  \n  score: `{src['score']:.4f}`")


# ------------------------------------------------------------------ #
#  Poll a pending task (runs on page load if pending, or after submit)
# ------------------------------------------------------------------ #
def _poll_task(task_id: str, api_key: str) -> None:
    """Show a polling UI with a stop button. Blocks the Streamlit script until
    the task completes, is cancelled, or errors out."""
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        stop_clicked = st.button("⏹ Stop", key="stop_btn")

        if stop_clicked:
            try:
                httpx.post(
                    f"{API_URL}/api/v1/tasks/{task_id}/cancel",
                    headers={"X-API-Key": api_key},
                    timeout=30,
                )
            except httpx.RequestError:
                pass
            st.session_state.pending_task_id = None
            status_placeholder.warning("Response generation was stopped.")
            _refresh_conversation_list(api_key)
            if st.session_state.conversation_id:
                _load_conversation(st.session_state.conversation_id, api_key)
            st.rerun()
            return

        status_placeholder.markdown("⏳ Searching papers and generating answer...")

        try:
            resp = httpx.get(
                f"{API_URL}/api/v1/tasks/{task_id}",
                headers={"X-API-Key": api_key},
                timeout=30,
            )
            if resp.status_code != 200:
                status_placeholder.error("Could not check task status.")
                st.session_state.pending_task_id = None
                return

            data = resp.json()
            status = data.get("status", "processing")

            if status == "completed":
                st.session_state.pending_task_id = None
                result = data.get("result", {})
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                processing_time = result.get("processing_time_seconds", 0)

                status_placeholder.empty()
                st.markdown(answer)
                st.caption(f"Processing time: {processing_time:.2f}s")

                if sources:
                    st.session_state.sources = [
                        {
                            "paper_id": s.get("paper_id", ""),
                            "title": s.get("title", ""),
                            "score": s.get("score", 0),
                            "source_type": s.get("source_type", "arxiv"),
                        }
                        for s in sources
                    ]

                if st.session_state.conversation_id:
                    _load_conversation(st.session_state.conversation_id, api_key)
                _refresh_conversation_list(api_key)
                st.rerun()
                return

            if status in ("cancelled", "error"):
                st.session_state.pending_task_id = None
                error_msg = data.get("error_message", "")
                if status == "cancelled":
                    status_placeholder.warning("Response generation was stopped.")
                else:
                    status_placeholder.error(f"An error occurred: {error_msg}")
                if st.session_state.conversation_id:
                    _load_conversation(st.session_state.conversation_id, api_key)
                _refresh_conversation_list(api_key)
                st.rerun()
                return

            # Still processing -- wait and rerun to poll again
            time.sleep(POLL_INTERVAL)
            st.rerun()

        except httpx.RequestError as exc:
            status_placeholder.error(f"Could not reach the API: {exc}")
            st.session_state.pending_task_id = None


# Resume polling if there's a pending task from a previous page load
if st.session_state.pending_task_id and api_key:
    _poll_task(st.session_state.pending_task_id, api_key)

# ------------------------------------------------------------------ #
#  Chat input
# ------------------------------------------------------------------ #
if question := st.chat_input("Ask about arXiv papers...", disabled=not api_key):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    is_first_message = st.session_state.conversation_id is None
    should_fetch = fetch_new_papers or is_first_message

    try:
        payload: dict = {
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
            timeout=30,
        )
        if response.status_code == 401:
            st.error("Invalid or inactive API key. Check your key and try again.")
            st.stop()
        if response.status_code == 429:
            st.warning("Rate limit exceeded. Please wait a moment and try again.")
            st.stop()
        response.raise_for_status()
        data = response.json()

        st.session_state.conversation_id = data["conversation_id"]
        st.session_state.pending_task_id = data["task_id"]

        _refresh_conversation_list(api_key)
        st.rerun()

    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        st.stop()
    except httpx.RequestError as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        st.stop()
