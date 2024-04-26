import os
LANGCHAIN_API_KEY = os.getenv(key="LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv(key="LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv(key="LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv(key="LANGCHAIN_PROJECT")

#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain import hub
import chromadb
import os
import argparse
import time

model = os.environ.get("MODEL", "Phi")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
# embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "nomic-embed-text")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    print(args)
    # callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # callbacks = [StreamingStdOutCallbackHandler()]
    # llm = Ollama(model=model, callbacks=callbacks)
    llm = Ollama(model=model)
    # make retriver
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OllamaEmbeddings(model="all-minilm")
    print(embeddings)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    print(retriever.invoke("what is we chat?"))
    print("-------******LLMResponse*******---------------")
    # llm.invoke("who are you?")
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
            | llm
    |StrOutputParser()
    )
    question = "what is this document all about?"
    answer = rag_chain.invoke({"question": question})
    print(answer)
    # answer = rag_chain.invoke({"question": question})
    # while True:
    #     query = input("\nEnter a query: ")
    #     if query == "exit":
    #         break
    #     if query.strip() == "":
    #         continue











# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == "__main__":
    main()