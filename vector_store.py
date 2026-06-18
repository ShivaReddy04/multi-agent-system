from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def save_report(report_text, topic="report"):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(report_text)

    # Tag every chunk with its source topic so retrieval can be scoped
    # to a single report instead of the whole collection.
    metadatas = [{"topic": topic} for _ in chunks]

    Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=DB_PATH
    )


def get_retriever(topic=None):

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
