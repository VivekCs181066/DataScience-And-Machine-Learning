from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation")
model = ChatHuggingFace(llm=llm)

jsonParsor=JsonOutputParser()
template1=PromptTemplate(
    template="Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction":jsonParsor.get_format_instructions()}
)
# prompt=template1.format()
# print(prompt)
# result=model.invoke(prompt)
# # print(result)
# res=jsonParsor.parse(result.content)
# print("res:",res)

# Now lets do the same with the chain that we did from 16 to 21 line
chain=template1 | model |jsonParsor
res=chain.invoke({'topic':'Artificial Inteligence in Agriculture'})
print("res:",res)