from typing import TypedDict

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


class Review(TypedDict):
    summary: str
    sentiment: str


llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
    max_new_tokens=200,
    temperature=0.7,
)

chat_model = ChatHuggingFace(llm=llm)
structured_model = chat_model.with_structured_output(Review)

result = structured_model.invoke(
    "The hardware is great, but the software feels bloated. There are some bugs that need to be fixed, but overall it's a decent product."
)

print(result)