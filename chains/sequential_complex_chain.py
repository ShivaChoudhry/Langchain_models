from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Generate the 5 pointer summary from the following report: {report}",
    input_variables=["report"]
)

model = ChatOllama(
    model="llama3.2:1b"
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Unemployment in the Indian IT sector"})

print(result)