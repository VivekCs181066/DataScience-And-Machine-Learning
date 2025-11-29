from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
messages = [
        ("system", "Translate the user sentence to French."),
        ("human", "I love programming."),
    ]
response=llm.invoke(messages)
print(response)
