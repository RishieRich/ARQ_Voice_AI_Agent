import shutil
import time
from pathlib import Path
from typing import List

import streamlit as st

from voice_rag.config.settings import PDF_DIR
from voice_rag.rag.ingest import build_vector_store_from_pdfs, load_existing_vector_store
from voice_rag.rag.qa import answer_question_marathi


# -----------------------------
# Page config + small UI polish
# -----------------------------
st.set_page_config(
    page_title="ARQ Voice RAG (Marathi) – Text v0",
    page_icon="🗣️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      .stTextArea textarea { font-size: 0.95rem; }
      .small-note { color: #8a8a8a; font-size: 0.85rem; }
      .badge { display:inline-block; padding: 0.15rem 0.5rem; border-radius: 999px;
               background: #f1f5f9; border: 1px solid #e2e8f0; margin-right: 0.25rem;
               font-size: 0.80rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helper utilities
# -----------------------------
def save_uploaded_pdfs(uploaded_files) -> List[Path]:
    """Persist uploaded PDF files to the local data directory."""
    saved = []
    for uf in uploaded_files:
        dest = PDF_DIR / uf.name
        with dest.open("wb") as f:
            shutil.copyfileobj(uf, f)
        saved.append(dest)
    return saved


def kb_exists() -> bool:
    """Lightweight check: if Chroma dir has any files, assume KB exists."""
    from voice_rag.config.settings import settings

    return settings.chroma_dir.exists() and any(settings.chroma_dir.iterdir())


# -----------------------------
# Header
# -----------------------------
st.title("🗣️ ARQ Voice RAG (Marathi) – Text Prototype (v0)")
st.write(
    "PDF knowledge base वर आधारित Q&A. आत्ता **text-only**, पुढे STT/TTS आणि LangGraph orchestration add करणार."
)

st.markdown(
    """
    <span class="badge">Local Ollama</span>
    <span class="badge">Chroma Vector DB</span>
    <span class="badge">LangChain RAG</span>
    <span class="badge">Marathi Answer</span>
    """,
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------
# Sidebar: KB setup + status
# -----------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    st.subheader("1) PDF Upload")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs are stored locally under voice_rag/data/pdfs",
    )

    build_clicked = st.button("📚 Build / Refresh Knowledge Base", use_container_width=True)

    st.subheader("2) Knowledge Base Status")
    if kb_exists():
        st.success("KB ready ✅ (Chroma store exists)")
    else:
        st.warning("KB not built yet ⚠️ Upload PDFs & build KB")

    st.markdown('<p class="small-note">Tip: First build can take time (embeddings download).</p>', unsafe_allow_html=True)

# -----------------------------
# Build KB action
# -----------------------------
if build_clicked:
    if not uploaded_files:
        st.error("Please upload at least one PDF first.")
    else:
        saved_paths = save_uploaded_pdfs(uploaded_files)
        with st.spinner("Building vector store from PDFs..."):
            t0 = time.time()
            build_vector_store_from_pdfs(saved_paths)
            t1 = time.time()

        st.success(f"Knowledge base built ✅  ({len(saved_paths)} PDF(s), {t1 - t0:.1f}s)")
        st.toast("KB is ready", icon="✅")


# -----------------------------
# Main area: Chat
# -----------------------------
col_left, col_right = st.columns([2.2, 1.0], gap="large")

with col_left:
    st.subheader("💬 Chat (Ask in Marathi / English → Answer in Marathi)")

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {role, content}

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    user_prompt = st.chat_input("Type your question here (Marathi/English) ...")

    if user_prompt:
        if not kb_exists():
            st.error("KB is not ready. Upload PDFs and build the KB first.")
        else:
            # Add user msg
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.write(user_prompt)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking... (Ollama + RAG)"):
                    answer = answer_question_marathi(user_prompt.strip())
                st.write(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})


with col_right:
    st.subheader("🧪 Debug / Inspection")

    st.markdown("**Current Mode:** Text-only (Voice nodes will be added next)")

    st.markdown("**Planned nodes:**")
    st.write(
        "- STT (Whisper) → Language router → Retrieval → Answer (RAG) → TTS (Marathi)\n"
        "- Later: LangGraph state + LangSmith traces"
    )

    st.divider()

    st.markdown("**Local Files**")
    st.write(f"PDF folder: `{PDF_DIR}`")

    # Optional: show the list of PDFs currently stored
    try:
        pdfs = sorted([p.name for p in PDF_DIR.glob("*.pdf")])
        if pdfs:
            st.write("PDFs stored:")
            st.code("\n".join(pdfs))
        else:
            st.write("No PDFs saved yet.")
    except Exception as e:
        st.write("Could not list PDFs:", e)

    st.divider()

    st.markdown("**(Optional) Retriever check**")
    st.caption("This does not show chunks yet. We'll add retrieved snippets in next iteration.")

    if st.button("🔎 Quick KB Load Test", use_container_width=True):
        try:
            _ = load_existing_vector_store()
            st.success("Chroma store loaded ✅")
        except Exception as e:
            st.error(f"Failed to load Chroma store: {e}")
