from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def save_report(report_text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(report_text)

    Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )


def get_retriever():

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    return db.as_retriever(
        search_kwargs={"k": 4}
    )
