import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import time # type: ignore

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    # st.session_state.embeddings=OllamaEmbeddings(model="nomic-embed-text")
    # st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.embeddings=OllamaEmbeddings(model="all-minilm")
    st.session_state.loader= PyPDFLoader("D:/Gen_AI_Tutorials/langchain/groqbot/pdfs/Lokender_Resume_New.pdf")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=Chroma.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input you prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")