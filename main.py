from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

query = "Какие есть дни недели?"
response = model.invoke(query)

print(f'Вопрос: {query}, ответ: {response.content}')