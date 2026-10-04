from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


prompt1 = PromptTemplate(
    template="Write a joke about {topic}.",
    input_variables=["topic"]
)


prompt2 = PromptTemplate(
    template="Explain the following joke in 3 points - {text}",
    input_variables=["text"]
)


model = ChatOllama(
    model="llama3.2:1b"
)


parser = StrOutputParser()


chain = RunnableSequence(
    prompt1,
    model,
    parser,
    prompt2,
    model,
    parser
)


print(chain.invoke({"topic": "man"}))