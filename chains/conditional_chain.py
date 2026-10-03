from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment of the feedback"
    )


model = ChatOllama(
    model="llama3.2:1b",
    format="json",
    temperature=0
)


parser = PydanticOutputParser(
    pydantic_object=Feedback
)


prompt1 = PromptTemplate(
    template="""
Classify the sentiment of this feedback.

Return ONLY this JSON format:
{{"sentiment": "negative"}}

The sentiment MUST be exactly one of:
positive
negative
neutral

Feedback:
{feedback}
""",
    input_variables=["feedback"]
)


classification_chain = prompt1 | model | parser


result = classification_chain.invoke({
    "feedback": "This is a poor product. It broke after a week of use and the customer service was unhelpful."
})


print(result)
print(result.sentiment)