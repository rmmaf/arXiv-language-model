"""Admin dashboard — one tab per tenant with full RAG interface + tenant CRUD."""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("API_REQUEST_TIMEOUT", "600"))
ARXIV_ABS_BASE = "https://arxiv.org/abs"

st.set_page_config(page_title="Admin — arXiv RAG", page_icon="⚙️", layout="wide")
st.title("Tenant Administration")

# --------------- Sidebar: admin authentication ---------------

default_admin_key = os.getenv("ADMIN_API_KEY", "")
admin_key = st.sidebar.text_input(
    "Admin Key", value=default_admin_key, type="password",
    help="ADMIN_API_KEY required for tenant management",
)
if not admin_key:
    st.warning("Enter your Admin Key in the sidebar to continue.")
    st.stop()

# --------------- Fetch tenants ---------------


def _fetch_tenants() -> list[dict] | None:
    try:
        resp = httpx.get(
            f"{API_URL}/api/v1/admin/tenants",
            headers={"X-Admin-Key": admin_key},
            timeout=API_TIMEOUT,
        )
    except httpx.RequestError as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        return None
    if resp.status_code == 403:
        st.error("Invalid admin key.")
        return None
    resp.raise_for_status()
    return resp.json()


all_tenants = _fetch_tenants()
if all_tenants is None:
    st.stop()

active_tenants = [t for t in all_tenants if t["is_active"]]

# --------------- Build tabs ---------------

if not active_tenants:
    tab_names = ["+ New Tenant"]
else:
    tab_names = [t["name"] for t in active_tenants] + ["+ New Tenant"]

tabs = st.tabs(tab_names)

# --------------- Tenant RAG tabs ---------------

for tab, tenant in zip(tabs[: len(active_tenants)], active_tenants):
    tid = tenant["id"]
    with tab:
        col_info, col_actions = st.columns([3, 1])
        with col_info:
            st.caption(
                f"Rate limit: **{tenant['rate_limit']} req/min** · "
                f"Created: {tenant['created_at']}"
            )
        with col_actions:
            with st.popover("Deactivate tenant"):
                st.warning(f"This will deactivate **{tenant['name']}**.")
                if st.button("Confirm deactivation", key=f"deactivate_{tid}"):
                    try:
                        dr = httpx.delete(
                            f"{API_URL}/api/v1/admin/tenants/{tid}",
                            headers={"X-Admin-Key": admin_key},
                            timeout=API_TIMEOUT,
                        )
                        dr.raise_for_status()
                        st.success("Tenant deactivated.")
                        st.rerun()
                    except httpx.HTTPStatusError as exc:
                        st.error(f"Error: {exc.response.status_code}")

        st.divider()

        question = st.text_input(
            "Question",
            placeholder="e.g. What are the latest advances in vision transformers?",
            key=f"question_{tid}",
        )
        top_k = st.number_input(
            "top_k", min_value=1, max_value=10, value=3, step=1,
            key=f"topk_{tid}",
        )

        ask_clicked = st.button(
            "Ask", type="primary",
            disabled=not question,
            key=f"ask_{tid}",
        )

        if ask_clicked and question:
            with st.spinner("Searching papers and generating answer..."):
                try:
                    response = httpx.post(
                        f"{API_URL}/api/v1/ask",
                        json={"question": question, "top_k": top_k},
                        headers={"X-API-Key": tenant["api_key"]},
                        timeout=API_TIMEOUT,
                    )
                    if response.status_code == 401:
                        st.error("Tenant API key rejected. The tenant may have been deactivated.")
                        st.stop()
                    if response.status_code == 429:
                        st.warning("Rate limit exceeded for this tenant.")
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
                st.markdown(
                    f"- [{src['title']}]({link})  \n  score: `{src['score']:.4f}`"
                )

            st.caption(f"Processing time: {data['processing_time_seconds']:.2f}s")

# --------------- New Tenant tab ---------------

with tabs[-1]:
    st.subheader("Create a new tenant")

    new_name = st.text_input("Tenant name", key="new_tenant_name")
    new_rate_limit = st.number_input(
        "Rate limit (req/min)", min_value=1, max_value=1000, value=30, step=1,
        key="new_tenant_rate_limit",
    )

    if st.button("Create tenant", type="primary", disabled=not new_name):
        try:
            cr = httpx.post(
                f"{API_URL}/api/v1/admin/tenants",
                json={"name": new_name, "rate_limit": new_rate_limit},
                headers={
                    "X-Admin-Key": admin_key,
                    "Content-Type": "application/json",
                },
                timeout=API_TIMEOUT,
            )
            if cr.status_code == 201:
                created = cr.json()
                st.success(f"Tenant **{created['name']}** created!")
                st.markdown("**API Key** (copy it now — it won't be shown again):")
                st.code(created["api_key"])
            else:
                cr.raise_for_status()
        except httpx.HTTPStatusError as exc:
            st.error(f"Error: {exc.response.status_code} — {exc.response.text}")
        except httpx.RequestError as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
