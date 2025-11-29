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
parser=StrOutputParser()
# Data flow for the pipeline:
# 1. tempate1: Formats the initial prompt using the input variable "topic" (e.g., "block hole").
# 2. model: Sends the formatted prompt to the HuggingFace chat model and gets a detailed report as a response.
# 3. parser: Extracts the text content from the model's response.
# 4. template2: Formats a new prompt for summarization, inserting the detailed report as the "text" variable.
# 5. model: Sends the summary prompt to the HuggingFace chat model and gets a summary as a response.
# 6. parser: Extracts the final summary text from the model's response.
# The final output is a concise 5-line summary of the detailed report on the given topic.
chain = tempate1 | model | parser | template2 | model | parser
result=chain.invoke({"topic": "block hole"})
print(result)