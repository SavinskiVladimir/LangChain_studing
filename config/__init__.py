from dotenv import load_dotenv
import os

load_dotenv()

try:
    OLLAMA_URL = os.getenv("OLLAMA_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    DB_PATH = os.getenv("DB_PATH")
    LLM_MODEL = os.getenv("LLM_MODEL")
except Exception as e:
    print(f'Ошибка при получении констант: {e}')

