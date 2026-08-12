from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


DB_PATH = "chroma_db"
COLLECTION_NAME = "pdf_documents"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"


embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_URL,
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)