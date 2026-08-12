from langchain_ollama import ChatOllama

from retreiver import retriever
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import LLM_MODEL, OLLAMA_URL

model = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
)

chat_history = InMemoryChatMessageHistory()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Ты — помощник по предоставленной документации.
    
        Отвечай только на основе информации из контекста ниже.
    
        Если в контексте нет информации, необходимой для ответа,
        честно скажи:
    
        "В предоставленных документах нет информации по этому вопросу."
    
        Не придумывай факты и не используй знания, которых нет
        в предоставленном контексте.
        """
    ),
    MessagesPlaceholder(
        variable_name='chat_history'
    ),
    (
        "human",
        """
        Контекст:
        {context}
        Вопрос:
        {question}
        """
    ),
])


def build_context(docs):
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "неизвестный файл")
        page = doc.metadata.get("page", None)

        if page is not None:
            page += 1

        context_parts.append(
            f"""
            Источник: {source}
            Страница: {page}
    
            {doc.page_content}
            """
        )

        return "\n\n".join(context_parts)

def ask_question(question):
    docs = retriever.invoke(question)
    if not docs:
        answer = ("Не найдено подходящей информации")

        chat_history.add_user_message(question)
        chat_history.add_ai_message(answer)

        return answer, []

    context = build_context(docs)

    messages = prompt.invoke({
        'chat_history': chat_history.messages,
        'context': context,
        'question': question,
    })

    responses = model.invoke(messages)

    chat_history.add_user_message(question)
    chat_history.add_ai_message(responses.content)

    return responses.content, docs


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

    if question.lower() == "clear":
        chat_history.clear()
        print("История очищена\n")
        continue

    try:
        answer, docs = ask_question(question)

        print()
        print("AI:")
        print(answer)

        print()
        print("Источники:")
        sources = set()
        for doc in docs:
            source = doc.metadata.get("source", "неизвестный файл")
            page = doc.metadata.get("page", None)

            if page is not None:
                page += 1

            sources.add((source, page))

        for source, page in sorted(sources):
            if page is not None:
                print(f"- {source}, стр. {page}")
            else:
                print(f"- {source}")
        print()

    except Exception as e:
        print(f"Ошибка: {e}")
