from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from voice_rag.config.settings import settings


# Text splitter for PDF pages
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""],
)


# lazy global embeddings so we don’t reload model each call
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def load_and_split_pdfs(pdf_paths: Iterable[Path]):
    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()
        split_docs = TEXT_SPLITTER.split_documents(pdf_docs)
        docs.extend(split_docs)
    return docs


def build_vector_store_from_pdfs(pdf_paths: Iterable[Path]) -> Chroma:
    docs = load_and_split_pdfs(pdf_paths)
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(settings.chroma_dir),
    )

    vectorstore.persist()
    return vectorstore


def load_existing_vector_store() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
