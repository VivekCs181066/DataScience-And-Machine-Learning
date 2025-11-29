from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create a prompt template with variables
# prompt = PromptTemplate.from_template("Translate '{sentence}' to French.")

# # Format the prompt with your input
# formatted_prompt = prompt.format(sentence="I love programming.")

# # Call the model
# response = llm.invoke(formatted_prompt)
# print(response.content)

# 1. from_template method to create a prompt template from a string with placeholders.
prompt= PromptTemplate.from_template("What is the capital of {country} ,and the {financial} caliptal?")
formatted_prompt=prompt.format(country="France",financial="financial")
response=llm.invoke(formatted_prompt)
print(response.content)

# 2.. format method to fill in the placeholders with actual values.
prompt= PromptTemplate.from_template("What is the capital of {country} ,and the {financial} caliptal?")
formatted_prompt = prompt.format(country="America", financial="financial")
response = llm.invoke(formatted_prompt)
print(response.content)

# 3. format_prompt method to create a prompt template from a PromptValue object.
prompt_value=prompt.format_prompt(country="india",financial="financial")
response=llm.invoke(prompt_value)
print(response.content)

# 4. from_examples method to create a prompt template using example pairs.    
examples = [
    "What is the capital of France?\The capital of Farance is Paris.",
    "What is the capital of Germany?\n The capital of Germany is Berlin"
]

prompt = PromptTemplate.from_examples(
    examples=examples,
    input_variables=["country"],
    suffix="What is the capital of {country}?"
)

formatted_prompt = prompt.format(country="India")
print(formatted_prompt)
response=llm.invoke(formatted_prompt)
print(response.content)