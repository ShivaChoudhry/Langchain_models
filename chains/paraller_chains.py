from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


model1 = ChatOllama(
    model="llama3.2:1b"
)

model2 = ChatOllama(
    model="llama3.2:1b"
)


prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text: {text}",
    input_variables=["text"]
)


prompt2 = PromptTemplate(
    template="Generate 5 short questions and answers from the following text: {text}",
    input_variables=["text"]
)


prompt3 = PromptTemplate(
    template="""Merge the provided notes and questions into a single document.

Notes:
{notes}

Questions:
{questions}
""",
    input_variables=["notes", "questions"]
)


parser = StrOutputParser()


parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "questions": prompt2 | model2 | parser
})


merge_chain = prompt3 | model1 | parser


chain = parallel_chain | merge_chain


result = chain.invoke({
    "text": """The Indian IT sector has been facing a significant challenge in recent years due to rising unemployment rates. Several factors contribute to this issue, including automation, outsourcing, and the global economic slowdown. Many IT professionals are finding it difficult to secure jobs, leading to increased competition and job insecurity. The government and industry leaders are working on various initiatives to address this problem, such as skill development programs and incentives for startups. However, the situation remains complex and requires a multifaceted approach to ensure sustainable employment opportunities in the sector."""
})


print(result)