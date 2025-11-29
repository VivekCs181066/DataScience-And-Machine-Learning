from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# chat_prompt=ChatPromptTemplate.from_messages([
#     ("system : Your are a helpful assistent. Translate the users sentence to Frence "),
#     ("Human : {sentence}")
# ])
# messages=chat_prompt.format_messages(sentence="I love Programming")
# response=llm.invoke(messages)
# print(response.content)

# 2. ChatPromptTemplate with Multiple Variables
# chat_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a travel assistant."),
#     ("human", "What is the capital of {country}? And what is its population?")
# ])

# messages = chat_prompt.format_messages(country="Japan")
# response = llm.invoke(messages)
# print(response.content)

# 3. ChatPromptTemplate with Example Conversation
# chat_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant."),
#     ("human", "Hello!"),
#     ("ai", "Hi there! How can I help you today?"),
#     ("human", "{question}")
# ])

# messages = chat_prompt.format_messages(question="Tell me a joke about AI.")
# response = llm.invoke(messages)
# print(response.content)

# Creates a chat prompt from a single template string (not often used for multi-turn, but possible):
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following: {question}"
)
messages = chat_prompt.format_messages(question="What is LangChain?")
print(messages)
print(llm.invoke(messages))