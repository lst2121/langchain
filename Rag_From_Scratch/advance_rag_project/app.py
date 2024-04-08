import os
LANGCHAIN_API_KEY = os.getenv(key="LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv(key="LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv(key="LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv(key="LANGCHAIN_PROJECT")

# import modules
from bs4 import BeautifulSoup
from typing import Optional, List, Tuple # type: ignore
from IPython.display import display, Markdown
import streamlit as st
from langchain import hub
## Data Ingestion
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, DataFrameLoader, TextLoader, PyPDFDirectoryLoader
# Vector Embedding And Vector Store
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, ConfigurableField
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import bm25, EnsembleRetriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.chains.retrieval_qa.base import RetrievalQA


## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

# create the llm using phi Model
# model_kwargs={'maxTokens':512})(can be passed)
def phi_llm(temperature):
    llm = Ollama(model="phi", temperature=temperature, timeout=300)
    return llm

# create the llm using llama2 Model
def llama2_llm(temperature):
    llm = Ollama(model="phi", temperature=temperature, timeout=300)
    return llm

# create the llm using llama2 Model
def Gemma_llm(temperature):
    llm = Ollama(model="phi", temperature=temperature, timeout=300)
    return llm

# create the llm using phi Model
embed_llm = OllamaEmbeddings(model="nomic-embed-text")

## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        embed_llm
    )
    vectorstore_faiss.save_local("faiss_index")
  

# define prompt template
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
    )
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Ollama")

    user_question = st.text_input("Ask any Question from your PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vectore Store")

        if st.button("Vectors Update"):
            with st.spinner("Processing Your Files...."):
                docs = data_ingestion()
                get_vector_store(docs=docs)
                st.success("Done")
    
    if st.button("Use Gemma:2b"):
        with st.spinner("Processing with Gemma...."):
                faiss_index = FAISS.load_local("faiss_index", embed_llm())
                llm = Gemma_llm(0)    #provide temperature

                #faiss_index = get_vector_store(docs)
                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("Done")

    if st.button("Use Phi:2"):
        with st.spinner("Processing with Phi...."):
                faiss_index = FAISS.load_local("faiss_index", embed_llm())
                llm = phi_llm(0)    #provide temperature

                #faiss_index = get_vector_store(docs)
                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("Done")

if __name__ == "__main__":
    main()



