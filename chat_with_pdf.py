"""
INFO-5940-A1 — Single-file Streamlit RAG (LangChain + Chroma)
Meets assignment requirements:
  (1) Provided Codespace setup: respects devcontainer env; no secrets in code.
  (2) TXT upload support (frontend + backend).
  (3) RAG pipeline + conversational UI:
      - Chunking (configurable), retrieval via Chroma, grounded answers, citations.
  (4) PDF support via pypdf (per-page parsing).
  (5) Multiple document uploads + unified index (with optional namespaces).
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple
import time

import streamlit as st
from langchain_chroma import Chroma  # NOTE: use langchain-chroma (no deprecation)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pypdf import PdfReader
from openai import OpenAI


# ----------------------------- Logging -----------------------------------------
logger = logging.getLogger("rag_app")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ----------------------------- Config ------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """App configuration (defaults work with Cornell proxy devcontainer)."""

    # Auth/endpoint (read from devcontainer runtime env)
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    base_url: Optional[str] = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")

    # Models
    chat_model: str = os.getenv("OPENAI_MODEL", "openai.gpt-5")
    embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "openai.text-embedding-ada.002")

    # Storage (persisted in container workspace)
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("data/chroma")
    uploads_dir: Path = Path("data/uploads")

    # Chunking defaults
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval defaults
    top_k: int = int(os.getenv("TOP_K", "4"))
    use_mmr: bool = os.getenv("USE_MMR", "True").lower() == "true"


# ----------------------- Embeddings -------------------------
# Simplified: use fixed embedding model


# --------------------------- Vector Store (Chroma) -----------------------------
class VectorStoreManager:
    """Manages the lifecycle of the Chroma vector store."""

    def __init__(self, cfg: AppConfig, embeddings: OpenAIEmbeddings) -> None:
        self.cfg = cfg
        self.embeddings = embeddings
        self.store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.cfg.chroma_dir),
        )

    def clear(self) -> None:
        if self.cfg.chroma_dir.exists():
            shutil.rmtree(self.cfg.chroma_dir, ignore_errors=True)
        self.cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.cfg.chroma_dir),
        )

    def add_documents(self, docs: Sequence[Document]) -> None:
        if not docs:
            return
        self.store.add_documents(list(docs))

    def as_retriever(self, k: int, use_mmr: bool):
        search_type = "mmr" if use_mmr else "similarity"
        kwargs = {"k": k}
        if use_mmr:
            kwargs["lambda_mult"] = 0.5
        return self.store.as_retriever(search_type=search_type, search_kwargs=kwargs)


# ----------------------- Document Loading & Chunking ---------------------------
class DocumentLoader:
    """Parses .txt and .pdf files into Documents; splits into overlapping chunks."""

    SUPPORTED = {".txt", ".pdf"}

    @staticmethod
    def load_paths(paths: Iterable[Path]) -> List[Document]:
        out: List[Document] = []
        for path in paths:
            suffix = path.suffix.lower()
            if suffix == ".txt":
                out.append(
                    Document(
                        page_content=path.read_text(encoding="utf-8", errors="ignore"),
                        metadata={"source": path.name},
                    )
                )
            elif suffix == ".pdf":
                out.extend(DocumentLoader._read_pdf(path))
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        return out

    @staticmethod
    def _read_pdf(path: Path) -> List[Document]:
        reader = PdfReader(str(path))
        pages: List[Document] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(Document(page_content=text, metadata={"source": path.name, "page": i}))
        return pages

    @staticmethod
    def chunk(docs: Sequence[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        return splitter.split_documents(list(docs))


# ------------------------------- RAG Core --------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user using only the provided context. "
    "If the answer is not in the context, say you don't know. Provide concise citations."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}\n\n"
            "Answer succinctly in 2–6 sentences.",
        ),
    ]
)


def format_docs_for_prompt(docs: Sequence[Document]) -> str:
    sections: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "document")
        page = d.metadata.get("page")
        tag = f"[{src}{f' p.{page+1}' if page is not None else ''}]"
        sections.append(f"{tag}\n{d.page_content}")
    return "\n\n---\n\n".join(sections)


class RAGService:
    """Coordinates retrieval and generation with LangChain LCEL."""

    def __init__(self, cfg: AppConfig, vsm: VectorStoreManager) -> None:
        self.cfg = cfg
        self.vsm = vsm
        self.llm = ChatOpenAI(
            model=self.cfg.chat_model,
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
            temperature=0,
        )
        self._chain = RunnableLambda(self._with_context) | PROMPT | self.llm | StrOutputParser()

    def _with_context(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        retriever = self.vsm.as_retriever(self.cfg.top_k, self.cfg.use_mmr)
        docs = retriever.invoke(question)
        return {"question": question, "context": format_docs_for_prompt(docs), "_docs": docs}

    def answer(self, question: str) -> dict[str, Any]:
        text = self._chain.invoke({"question": question})
        docs = self._with_context({"question": question})["_docs"]
        return {"answer": text, "sources": docs}


# ----------------------------- Streamlit UI -----------------------------------
st.set_page_config(page_title="📚 RAG Chat (TXT/PDF)", page_icon="📚", layout="wide")

SESSION_VSM = "vsm"
SESSION_RAG = "rag"
SESSION_HIST = "hist"
SESSION_CFG = "cfg"


def ensure_dirs(cfg: AppConfig) -> None:
    for d in (cfg.data_dir, cfg.chroma_dir, cfg.uploads_dir):
        d.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def bootstrap(cfg: AppConfig) -> Tuple[VectorStoreManager, RAGService]:
    ensure_dirs(cfg)

    if not cfg.api_key:
        st.error("Missing API key. Set OPENAI_API_KEY (or API_KEY) in your devcontainer or environment.")
        st.stop()

    embed = OpenAIEmbeddings(
        model=cfg.embed_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )
    vsm = VectorStoreManager(cfg, embed)
    rag = RAGService(cfg, vsm)
    return vsm, rag


def sidebar(cfg: AppConfig) -> AppConfig:
    st.sidebar.header("Settings")
    chat_model = st.sidebar.text_input("Chat model", value=cfg.chat_model)
    st.sidebar.caption(f"Base URL: {cfg.base_url or 'OpenAI default'}")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        top_k = st.sidebar.slider("Top-K", 2, 10, cfg.top_k)
    with c2:
        use_mmr = st.sidebar.toggle("MMR", cfg.use_mmr)

    csize = st.sidebar.slider("Chunk size", 256, 2000, cfg.chunk_size, step=64)
    cover = st.sidebar.slider("Chunk overlap", 0, 400, cfg.chunk_overlap, step=16)

    if st.sidebar.button("🔁 Reset chat", use_container_width=True):
        st.session_state[SESSION_HIST] = []

    if st.sidebar.button("🧹 Clear vectors", use_container_width=True):
        if SESSION_VSM in st.session_state:
            st.session_state[SESSION_VSM].clear()
        st.success("Cleared vector store.")

    # Return a new config instance with tuned values (immutability principle)
    return AppConfig(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        chat_model=chat_model,
        embed_model=cfg.embed_model,
        data_dir=cfg.data_dir,
        chroma_dir=cfg.chroma_dir,
        uploads_dir=cfg.uploads_dir,
        chunk_size=csize,
        chunk_overlap=cover,
        top_k=top_k,
        use_mmr=use_mmr,
    )


def save_uploads(files: Sequence["UploadedFile"], dst_dir: Path) -> List[Path]:  # type: ignore[name-defined]
    """
    Save uploaded files into `dst_dir` safely.

    - Ensures `dst_dir` exists.
    - Sanitizes filename via os.path.basename to avoid path traversal.
    - Adds a timestamp suffix if a file with the same name already exists.
    """
    paths: List[Path] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        # sanitize filename to avoid path traversal
        name = os.path.basename(getattr(f, "name", "uploaded_file"))
        dest = dst_dir / name
        # avoid clobbering existing files: append timestamp
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            dest = dst_dir / f"{stem}_{int(time.time())}{suffix}"
        # write bytes (Streamlit UploadedFile provides .read())
        dest.write_bytes(f.read())
        paths.append(dest)
    return paths


def ingest_files(
    paths: Sequence[Path],
    cfg: AppConfig,
    vsm: VectorStoreManager,
) -> int:
    docs = DocumentLoader.load_paths(paths)
    chunks = DocumentLoader.chunk(docs, cfg.chunk_size, cfg.chunk_overlap)
    vsm.add_documents(chunks)
    return len(chunks)


def render_sources(sources: Sequence[Document]) -> None:
    if not sources:
        return
    st.markdown("**Sources:**")
    for s in sources:
        meta = s.metadata
        name = meta.get("source", "document")
        page = meta.get("page")
        st.write(f"• {name}{f' (p.{page+1})' if page is not None else ''}")


def main() -> None:
    base_cfg = AppConfig()
    if SESSION_CFG not in st.session_state:
        st.session_state[SESSION_CFG] = base_cfg
    cfg = sidebar(st.session_state[SESSION_CFG])
    st.session_state[SESSION_CFG] = cfg

    if SESSION_VSM not in st.session_state or SESSION_RAG not in st.session_state:
        vsm, rag = bootstrap(cfg)
        st.session_state[SESSION_VSM] = vsm
        st.session_state[SESSION_RAG] = rag

    if SESSION_HIST not in st.session_state:
        st.session_state[SESSION_HIST] = []

    st.title("📚 Retrieval-Augmented Chat (TXT/PDF)")
    st.write("Upload documents, click **Ingest**, then ask questions. Answers are grounded with citations.")

    # Upload + ingest
    with st.expander("➕ Add documents (.txt, .pdf) — multiple allowed", expanded=True):
        uploads = st.file_uploader("Choose file(s)", type=["txt", "pdf"], accept_multiple_files=True)
        if st.button("📥 Ingest files", type="primary"):
            if not uploads:
                st.warning("Please select at least one file.")
            else:
                with st.spinner("Indexing…"):
                    paths = save_uploads(uploads, st.session_state[SESSION_CFG].uploads_dir)
                    n_chunks = ingest_files(paths, cfg, st.session_state[SESSION_VSM])
                st.success(f"Indexed {len(paths)} file(s) into {n_chunks} chunk(s). Ready to chat.")

    # History
    for role, msg in st.session_state[SESSION_HIST]:
        with st.chat_message(role):
            st.markdown(msg)

    # Chat
    user_msg = st.chat_input("Ask about your documents…")
    if user_msg:
        st.session_state[SESSION_HIST].append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    res = st.session_state[SESSION_RAG].answer(user_msg)
                    st.markdown("**Answer:**\n\n" + res["answer"])
                    render_sources(res.get("sources", []))
                    st.session_state[SESSION_HIST].append(("assistant", res["answer"]))
                except Exception as exc:
                    st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
