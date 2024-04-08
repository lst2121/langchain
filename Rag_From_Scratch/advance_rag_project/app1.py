import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.chains.retrieval_qa.base import RetrievalQA
import streamlit as st


## Data ingestion
def data_ingestion():

    #  PDF Data Loader
    pdf_path = "D:/Gen_AI_Tutorials/langchain/Rag_From_Scratch/advance_rag_project/data/pdfs"
    loader=PyPDFDirectoryLoader(pdf_path)
    documents=loader.load()
    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
    splits=text_splitter.split_documents(documents)
    return splits

    # # Web Based Data Ingestion 
    # loader = WebBaseLoader(
    #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    #     bs_kwargs=dict(
    #     parse_only=bs4.SoupStrainer(
    #         class_=("post-content", "post-title", "post-header")
    #         )
    #     ),
    # )
    # docs = loader.load()
    # Split Data
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splits = text_splitter.split_documents(documents=docs)
    # return splits

# Define Your LLMs
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
def gemma_llm(temperature):
    llm = Ollama(model="phi", temperature=temperature, timeout=300)
    return llm

# create the llm using phi Model
embed_llm = OllamaEmbeddings(model="nomic-embed-text")


## Vector Embedding and vector store
def get_vector_store(splits):
    # Embed the data using any embedding model we are using "nomic-embed-text" and create a vectorstore
    # # Chroma DB
    # vectorstore_chroma = Chroma.from_documents(documents=splits, embedding=embed_llm)
    # vectorstore_chroma.save_local("chroma_index")

    # FAISS DB
    vectorstore_faiss=FAISS.from_documents(splits,embed_llm)
    vectorstore_faiss.save_local("faiss_index")


    # create a retriver
    # chroma_retriever = vectorstore_chroma.as_retriever()
    # faiss_retriever = vectorstore_faiss.as_retriever()


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
                get_vector_store(docs)
                st.success("Done")
    
    if st.button("Use Gemma:2b"):
        with st.spinner("Processing with Gemma...."):
                faiss_index = FAISS.load_local("faiss_index", embed_llm)
                llm = gemma_llm(0)    #provide temperature

                #faiss_index = get_vector_store(docs)
                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("Done")

    if st.button("Use Phi:2"):
        with st.spinner("Processing with Phi...."):
                faiss_index = FAISS.load_local("faiss_index", embed_llm)
                llm = phi_llm(0)    #provide temperature

                #faiss_index = get_vector_store(docs)
                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("Done")

if __name__ == "__main__":
    main()

