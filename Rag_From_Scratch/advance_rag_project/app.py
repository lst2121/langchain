import os
LANGCHAIN_API_KEY = os.getenv(key="LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv(key="LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv(key="LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv(key="LANGCHAIN_PROJECT")


import bs4
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.llms.ollama import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Data ingestion
def data_ingestion():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    # Split Data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents=docs)
    return splits


def phi_llm():
    llm = Ollama(model="phi", temperature=0, timeout=300)
    return llm


def gemma_llm():
    llm = Ollama(model="gemma", temperature=0, timeout=300)
    return llm


def embed_llm():
    llm = OllamaEmbeddings(model="nomic-embed-text")
    return llm

# persist_directory = "chroma_index"
# create a vector store
# def create_vector_store(doc):
#     vectordb = Chroma.from_documents(documents=doc, embedding=embed_llm(), persist_directory="chroma_index")
#     vectordb.persist()
#     return vectordb

# create a vector store
def create_vector_store(doc):
    vectordb = Chroma.from_documents(documents=doc, embedding=embed_llm(), persist_directory="chroma_index")
    vectordb.persist()
    return vectordb

# create retriever
# def create_retriever():
#     vectorstore = Chroma(persist_directory="chroma_index", embedding_function=embed_llm())
#     retriever = vectorstore.as_retriever()
#     return retriever

def create_retriever():
    vectorstore = Chroma(persist_directory="chroma_index", embedding_function=embed_llm())
    retriever = vectorstore.as_retriever()
    return retriever


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_llm_response(llm, retriever, question):
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    answer = rag_chain.invoke({"question": question})
    return answer



def main():
    st.set_page_config("RAG App")
    st.title("RAG App")

    persist_directory="chroma_index"

    # Load data and create vector store
    st.sidebar.subheader("Data Ingestion")
    if st.sidebar.button("Ingest Data"):
        docs = data_ingestion()
        vectordb = create_vector_store(docs)
        st.sidebar.success("Data ingested and vector store created.")

    # Select LLM
    st.sidebar.subheader("Select LLM")
    llm_option = st.sidebar.radio("Choose LLM:", ("Phi", "Gemma"))
    if llm_option == "Phi":
        llm = phi_llm()
    elif llm_option == "Gemma":
        llm = gemma_llm()
        

    # User input
    user_question = st.text_input("Ask any question:")
    if st.button("Get Answer"):
        retriever = create_retriever()
        answer = get_llm_response(llm, retriever, user_question)
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()

# def main():
#     st.set_page_config("RAG App")
#     st.title("RAG App")

#     vectorstore = None  # Initialize vectorstore variable outside the button block

#     # Load data and create vector store
#     st.sidebar.subheader("Data Ingestion")
#     if st.sidebar.button("Ingest Data"):
#         docs = data_ingestion()
#         vectorstore = create_vector_store(docs)
#         st.sidebar.success("Data ingested and vector store created.")

#     # Select LLM
#     st.sidebar.subheader("Select LLM")
#     llm_option = st.sidebar.radio("Choose LLM:", ("Phi", "Gemma"))
#     if llm_option == "Phi":
#         llm = phi_llm()
#     elif llm_option == "Gemma":
#         llm = gemma_llm()

#     # User input
#     user_question = st.text_input("Ask any question:")
#     if st.button("Get Answer"):
#         retriever = create_retriever(vectorstore)
#         answer = get_llm_response(llm, retriever, user_question)
#         st.write("Answer:", answer)


# if __name__ == "__main__":
#     main()




