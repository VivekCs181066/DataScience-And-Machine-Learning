from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# chat_prompt = ChatPromptTemplate.from_messages([
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Translate to Frence"),
#     HumanMessage(content="{sentence}")
# ])
# messages=chat_prompt.format_messages(sentence="I love programming")
# print(llm.invoke(messages).content)
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="{sentence}")
])
messages = chat_prompt.format_messages(sentence="I love programming")
# print(llm.invoke(messages).content)
print(messages)