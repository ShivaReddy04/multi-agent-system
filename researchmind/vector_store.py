from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.abspath(os.path.join(DB_DIR, "chroma_db"))

os.makedirs(DB_DIR, exist_ok=True)

# Lazy-loaded embedding model: creating the embedding instance can be
# expensive (downloads / weight loading). Defer instantiation until first use.
_embedding_model = None


def get_embedding_model():
    """Return a cached HuggingFaceEmbeddings instance, creating it on first use."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def save_report(report_text, topic="report"):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(report_text)

    # Tag every chunk with its source topic so retrieval can be scoped
    # to a single report instead of the whole collection.
    metadatas = [{"topic": topic} for _ in chunks]

    embedding_model = get_embedding_model()

    Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=DB_PATH
    )


def get_retriever(topic=None):

    embedding_model = get_embedding_model()

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    search_kwargs = {"k": 4}

    # When a topic is given, restrict retrieval to that report's chunks.
    if topic:
        search_kwargs["filter"] = {"topic": topic}

    return db.as_retriever(
        search_kwargs=search_kwargs
    )
