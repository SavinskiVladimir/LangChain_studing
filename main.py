from langchain_ollama import ChatOllama
from retreiver import retriever

model = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

docs = retriever.invoke(
    "Как изменить IP-адрес?"
)

context = "\n\n".join(
    doc.page_content
    for doc in docs
)

prompt = f"""
Ты — помощник по документации.

Отвечай только на основе предоставленного контекста.

Контекст:

{context}

Вопрос:

Что необходимо делать?
"""

response = model.invoke(prompt)

print(response.content)