from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv("chains/.env")

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Lionel Messi"})

print(result)

chain.get_graph().print_ascii();