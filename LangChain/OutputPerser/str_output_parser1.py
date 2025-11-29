from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation")
model = ChatHuggingFace(llm=llm)

# 1--> Detail report
tempate1 = PromptTemplate.from_template(
    template="You are a helpful assistant. Generate a detailed report on the following topic: {topic}"
)
# 2--> Summary report
template2 = PromptTemplate.from_template(
    template="Write a 5 line summary on the following text: {text}"
)

prompt1 = tempate1.invoke({"topic": "Black holes"})
response1 = model.invoke(prompt1)
print("Detailed Report:\n", response1.content)
prompt2 = template2.invoke({"text": response1.content})
print("Summary Report:\n", prompt2)
response2 = model.invoke(prompt2)
print("Final Summary:\n", response2.content)