from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation")
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age:  int =Field(gt=18,description="Age of the person")
    city: str =Field(description="Name of the city the person belongs to")

pydantic_perser=PydanticOutputParser(pydantic_object=Person)
template=PromptTemplate(
    template=(
        "Generate ONLY a valid JSON object for a fictional person from the country: {country}.\n"
        "The JSON must match this schema:\n"
        "{format_instruction}\n"
        "Do not include any explanation, code, or text—just the JSON object."
    ),
    input_variables=["country"],
    partial_variables={'format_instruction':pydantic_perser.get_format_instructions()}
)
prompt=template.invoke({'country':'India'})
print("prompt:",prompt)
res=model.invoke(prompt)
import re

raw = res.content.strip()

# Remove code block markers and any text before the JSON object
# Remove triple backticks and language hints
raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
raw = re.sub(r"\n?```$", "", raw)

# Remove any text before the first curly brace
json_start = raw.find("{")
if json_start != -1:
    raw = raw[json_start:]

final_res = pydantic_perser.parse(raw)
print("final_res:", final_res)