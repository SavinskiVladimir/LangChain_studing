import os
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

model = ChatOpenAI(model="gpt-4o-mini")

response = model.invoke("Какой сегодня день?")

print(response.content)
