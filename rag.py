from pathlib import Path
import hashlib
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import OLLAMA_URL, EMBEDDING_MODEL, DB_PATH


PDF_DIR = Path("documents")
COLLECTION_NAME = "pdf_documents"

BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def make_chunk_id(
    source: str,
    page: int,
    chunk_number: int,
    content: str,
) -> str:
    value = f"{source}|{page}|{chunk_number}|{content}"

    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def index_document():
    try:
        pdf_files = sorted(PDF_DIR.rglob("*.pdf"))

        logger.info("Найдено PDF-файлов: %s",len(pdf_files))

        if not pdf_files:
            raise ValueError(f"В папке '{PDF_DIR}' не найдено PDF-файлов")

        documents = []

        for pdf_path in pdf_files:
            logger.info("Загрузка: %s",pdf_path)
            loader = PyPDFLoader(str(pdf_path))

            pdf_pages = loader.load()
            relative_path = str(pdf_path.relative_to(PDF_DIR))

            for page_document in pdf_pages:
                page_document.metadata["source"] = relative_path

            documents.extend(pdf_pages)

            logger.info("  Страниц загружено: %s",len(pdf_pages))

        logger.info("Всего страниц: %s",len(documents))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = []
        chunk_ids = []

        for page_document in documents:
            source = page_document.metadata.get("source", "unknown")

            page = page_document.metadata.get("page", 0,)

            page_chunks = text_splitter.split_documents([page_document])

            for chunk_number, chunk in enumerate(page_chunks):
                chunk.metadata["source"] = source
                chunk.metadata["page"] = page
                chunk.metadata["chunk_number"] = chunk_number

                chunk_id = make_chunk_id(
                    source=source,
                    page=page,
                    chunk_number=chunk_number,
                    content=chunk.page_content,
                )

                chunks.append(chunk)
                chunk_ids.append(chunk_id)

        logger.info("Всего создано чанков: %s", len(chunks))

        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_PATH,
            embedding_function=embeddings,
        )

        logger.info("Получение существующих ID из Chroma...")

        existing_data = vectorstore.get(include=[])
        existing_ids = set(existing_data.get("ids", []))

        logger.info("В Chroma уже существует чанков: %s",len(existing_ids))

        new_chunks = []
        new_chunk_ids = []

        for chunk, chunk_id in zip(chunks, chunk_ids,):

            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_chunk_ids.append(chunk_id)

        logger.info("Новых чанков для добавления: %s",len(new_chunks))

        if not new_chunks:
            logger.info("Новых чанков нет. База уже актуальна.")
        else:

            total = len(new_chunks)
            for start in range(0, total, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total)

                batch_documents = new_chunks[start:end]

                batch_ids = new_chunk_ids[start:end]

                logger.info( "Добавление чанков %s-%s из %s...",start + 1, end, total)

                vectorstore.add_documents(documents=batch_documents, ids=batch_ids)

            logger.info("Добавлено новых чанков: %s",total)


        logger.info("Индексация завершена успешно.")

    except Exception:
        logger.exception("Ошибка при индексации документов")

if __name__ == "__main__":
    index_document()
