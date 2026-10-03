from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    task="text-generation",
    provider="hf-inference",
    max_new_tokens=256,
    temperature=0.7,
)
model = llm

chat_history = []

while True:
    user_input = input("Enter your prompt (or type 'exit' to quit): ")
    chat_history.append({"role": "user", "content": user_input})
    if user_input.lower() == 'exit':
        break

    result = model.invoke(chat_history)
    response = result.content if hasattr(result, "content") else result
    print("Model Response:", response)