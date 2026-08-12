from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import DB_PATH, OLLAMA_URL, EMBEDDING_MODEL


COLLECTION_NAME = "pdf_documents"

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