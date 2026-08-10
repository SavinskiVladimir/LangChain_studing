from langchain_ollama import ChatOllama
from retreiver import retriever


model = ChatOllama(
    model="llama3.1:8b",
    base_url="http://127.0.0.1:11434",
    temperature=0,
)

def ask_question(question: str):
    docs = retriever.invoke(question)

    if not docs:
        return "В документах не найдено подходящей информации."

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # Формируем prompt
    prompt = f"""
Ты — помощник по предоставленной документации.

Отвечай только на основе информации из контекста ниже.

Если в контексте нет информации, необходимой для ответа,
честно скажи:

"В предоставленных документах нет информации по этому вопросу."

Не придумывай факты и не используй знания, которых нет
в предоставленном контексте.

Контекст:
--------------------
{context}
--------------------

Вопрос:
{question}

Ответ:
"""

    response = model.invoke(prompt)
    return response.content


print("Введите вопрос по вашим документам.")
print("Для выхода введите: exit")
print()

while True:
    question = input("Вы: ").strip()

    if not question:
        continue

    if question.lower() in {
        "exit",
        "quit",
        "выход",
    }:
        print("До свидания!")
        break

    try:
        answer = ask_question(question)

        print()
        print("AI:")
        print(answer)

        print()

    except Exception as e:

        print()
        print("Ошибка:", e)
        print()
