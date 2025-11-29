from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
from typing import Literal, Optional, Annotated
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Define Pydantic model for structured output
class ReviewModel(BaseModel):
    key_themes: list[str] =Field(description="A list of key themes mentioned in the review.")
    summary: str = Field(description="A brief summary of the review in one sentence.")
    sentiment: Literal['positive', 'negative'] = Field(description="The overall sentiment of the review, either 'positive', 'negative'.")
    pros:Optional[list[str]] = Field(default=None, description="A list of pros mentioned in the review, if any.")
    cons:Optional[list[str]] =Field(default=None, description="A list of cons mentioned in the review, if any.")
    name:Optional[str] =Field(default=None,description="Name of the reviewer")
structured_model = llm.with_structured_output(ReviewModel)
response = structured_model.invoke("""These shoes fit true to size and were comfortable right out of the box with no break-in period needed.
The cushioning is fantastic; I wore them for a 12-hour shift and experienced zero foot fatigue.
Great value for money—the material feels premium and sturdy, rivaling much more expensive brands.
They look exactly like the pictures, and I’ve already received several compliments on the design.
I recommend ordering a half-size up if you have wide feet, as the toe box runs a little narrow.
The soles have excellent grip and traction, making them perfect for running on slippery or wet surfaces.
Lightweight and breathable, these are my new go-to shoes for the gym and summer walks.
Bought these as a gift, and the recipient loved them; the packaging was secure and arrived on time.
While the style is great, the arch support is a bit lacking, so you might need insoles if you have high arches.
Overall, a solid purchase that combines style and durability—I will definitely be buying another pair in a different color. My name is Alex.""")
print(response.name) 