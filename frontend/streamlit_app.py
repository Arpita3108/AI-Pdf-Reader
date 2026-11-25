"""Streamlit UI.  Run:  streamlit run frontend/streamlit_app.py"""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="AI PDF Research Assistant", page_icon="📚", layout="wide")
st.title("📚 AI PDF Research Assistant")
st.caption("Ask questions about your PDFs. Every answer is grounded in the documents and cited.")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 600), **kw)
    except requests.ConnectionError:
        st.error(f"Cannot reach the backend at {API}. Start it with: uvicorn app.main:app")
        st.stop()
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, detail
    return r.json(), None


if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar: upload + document management ----------------
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Process uploads", type="primary", use_container_width=True):
        with st.spinner("Extracting, chunking and embedding..."):
            payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
            data, err = api("POST", "/upload", files=payload)
        if err:
            st.error(err)
        else:
            for r in data["results"]:
                if r["status"] == "ok":
                    st.success(f"{r['document']}: {r['pages']} pages, {r['chunks']} chunks")
                else:
                    st.error(f"{r['document']}: {r['error']}")

    docs_resp, _ = api("GET", "/documents")
    docs = (docs_resp or {}).get("documents", [])
    names = [d["name"] for d in docs]
    st.divider()
    if not docs:
        st.info("No documents yet.")
    for d in docs:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{d['name']}**  \n{d['pages']} pages · {d['chunks']} chunks")
        if c2.button("🗑", key=f"del-{d['name']}", help="Remove this document"):
            api("DELETE", f"/documents/{d['name']}")
            st.rerun()

    scope = st.multiselect("Search only in", names, help="Leave empty to search all documents.")
    top_k = st.slider("Passages to retrieve", 1, 10, 5)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            mark = "✅ cited" if s["cited"] else "retrieved"
            st.markdown(f"**[{s['id']}] {s['document']} — page {s['page']}** · similarity {s['score']:.2f} · {mark}")
            st.caption(s["snippet"])


# ---------------- Chat ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources", []))

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            data, err = api("POST", "/ask", json={"question": question, "top_k": top_k,
                                                   "documents": scope or None})
        if err:
            st.error(err)
        else:
            st.markdown(data["answer"])
            render_sources(data["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
            )
