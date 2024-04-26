from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, respectful and honest assistant.Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."),
        ("user", "Question: {question}")
    ]
)

## streamlit framework

st.title('Chat with LLAMA2 or Gemma:2b')
input_text = st.text_input("Search the topic you want")

# Define LLAMA models
models = {
    "LLAMA2": "llama2",
    "Gemma:2b": "gemma:2b"
}

# Dropdown menu to select LLAMA model
selected_model = st.selectbox("Select Model", list(models.keys()))

# Choose the appropriate LLAMA model
llm = Ollama(model=models[selected_model])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
