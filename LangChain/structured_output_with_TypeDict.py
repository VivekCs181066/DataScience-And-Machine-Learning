# from typing import TypedDict
# class Person(TypedDict):
#     name:str
#     age:int
# newPerson:Person={'name':'Alice','age':30}
# print(newPerson)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Schema definition using TypedDict
class Review(TypedDict):
    summary:str
    sentiment:str
structured_model=llm.with_structured_output(Review)
response=structured_model.invoke("These shoes fit true to size and were comfortable right out of the box with no break-in period needed. I would definitely buy them again.")
print(response)

# Sometimes the llm could not understand the structure so we can provide a prompt to help it using the annotation from Typing
from typing import Annotated
class AnnotatedReview(TypedDict):
    summary:Annotated[str,'A brief summary of the review in one sentence.']
    sentiment:Annotated[str,"The overall sentiment of the review, either 'positive', 'negative', or 'neutral'."]
annotated_structured_model=llm.with_structured_output(AnnotatedReview)
response=annotated_structured_model.invoke("These shoes fit true to size and were comfortable right out of the box with no break-in period needed. I would definitely buy them again.")
print(response)

#  TypeDict with Optional fields
from typing import Optional
class OptionalReview(TypedDict):
    key_themes:Annotated[list[str],'A list of key themes mentioned in the review.']
    summary:str
    sentiment:Optional[str]
    pros:Annotated[Optional[list[str]],'A list of pros mentioned in the review, if any.']
    cons:Annotated[Optional[list[str]],'A list of cons mentioned in the review, if any.']
optional_structured_model=llm.with_structured_output(OptionalReview)
response=optional_structured_model.invoke("""These shoes fit true to size and were comfortable right out of the box with no break-in period needed.
The cushioning is fantastic; I wore them for a 12-hour shift and experienced zero foot fatigue.
Great value for money—the material feels premium and sturdy, rivaling much more expensive brands.
They look exactly like the pictures, and I’ve already received several compliments on the design.
I recommend ordering a half-size up if you have wide feet, as the toe box runs a little narrow.
The soles have excellent grip and traction, making them perfect for running on slippery or wet surfaces.
Lightweight and breathable, these are my new go-to shoes for the gym and summer walks.
Bought these as a gift, and the recipient loved them; the packaging was secure and arrived on time.
While the style is great, the arch support is a bit lacking, so you might need insoles if you have high arches.
Overall, a solid purchase that combines style and durability—I will definitely be buying another pair in a different color.""")
print(response)