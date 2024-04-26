import os
LANGCHAIN_API_KEY = os.getenv(key="LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv(key="LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv(key="LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv(key="LANGCHAIN_PROJECT")

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
from langchain.load import dumps, loads

def load_and_preprocess_documents_from_website(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    
    return splits

def create_vector_store(splits, embed_model):
    vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
    retriever =vectorstore.as_retriever()
    return retriever

# def generate_queries(llm, prompt_template, question):
#     prompt = ChatPromptTemplate.from_template(prompt_template)
#     generate_queries_chain = (
#         prompt
#         | llm
#         | StrOutputParser()
#         | (lambda x: x.split("\n"))
#     )
#     return generate_queries_chain.invoke({"question": question})

generate_queries = (
    prompt_perspectives 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def retrieve_documents(retriever, question):
    docs = retriever.invoke({"question": question})
    return [loads(doc) for doc in docs]

def rank_documents(documents):
    # Implement your ranking algorithm here
    # This could include sorting documents based on relevance scores, etc.
    pass

def main():
    # Main function to orchestrate the workflow
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    question = "What is task decomposition for LLM agents?"
    prompt_template = "You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}"
    
    splits = load_and_preprocess_documents_from_website(url)
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    retriever = create_vector_store(splits, embed_model)
    queries = generate_queries(Ollama(model="phi", temperature=0.1, timeout=300), prompt_template, question)
    retrieved_docs = retrieve_documents(retriever, question)
    ranked_docs = rank_documents(retrieved_docs)
    
    # Print or display the ranked documents
    for doc in ranked_docs:
        print(doc)

if __name__ == "__main__":
    main()
