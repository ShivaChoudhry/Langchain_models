from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv
import streamlit as st  

load_dotenv()

st.header("Hugging Face Chat Model")

user_input = st.text_input("Enter your question:", key="user_input")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
)
model = ChatHuggingFace(llm=llm)
if st.button("Summarize", key="submit_button"):
    result = model.invoke(user_input)
    st.write(result.content)




# result = model.invoke(user_input)
# st.write(result.content)