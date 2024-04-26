
import os
LANGCHAIN_API_KEY = os.getenv(key="LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv(key="LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv(key="LANGCHAIN_TRACING_V2")

import asyncio
import sys

if sys.platform:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain.chat_models.ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import requests
from bs4 import BeautifulSoup
import json

# DuckDuckGo Search
RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()

def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query,num_results)
    return [r["link"] for r in results]

# Promt
SUMMARY_TEMPLATE = """{text} 

-----------

Using the above text, answer in short the following question: 

> {question}
 
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

llm = ChatOllama(model="phi", temperature=0, timeout=300)

url = "https://blog.langchain.dev/announcing-langsmith"

def scrape_text(url:str):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            page_text = soup.get_text(separator=" ", strip=True)

            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

# Simple Chain
# page_content = scrape_text(url)[:10000]
# basic_chain = SUMMARY_PROMPT | llm | StrOutputParser()
# basic_chain.invoke(
#     {
#         "question": "what is langsmith?",
#         "text": page_content
#     }
# )

# Runnable pass through chain
scrape_and_summarize_chain = RunnablePassthrough.assign(
    text=lambda x: scrape_text(x["url"])[:10000]
    ) | SUMMARY_PROMPT | llm |StrOutputParser()
    
# scrape_and_summarize_chain.invoke(
#     {
#         "question": "what is langsmith?",
#         "url": url
#     }
# )

web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

web_search_chain.invoke(
    {
        "question": "what is langsmith?"
    }
)

SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

gemma_llm = ChatOllama(model="gemma:2b", temperature=0, timeout=300)
search_question_chain = SEARCH_PROMPT | llm | StrOutputParser()

# search_question_chain.invoke(
#     {
#         "question": "what is the difference between langchain and langsmith?"
#     }
# )

# chain = search_question_chain | (lambda x:[{"question": q} for q in x]) | web_search_chain.map()
chain = search_question_chain | (lambda x: [{"question": q.strip()} for q in x.split("\n") if q.strip()]) | web_search_chain.map()
import json

# chain = search_question_chain | (lambda x: [{"question": q} for q in json.loads(x)]) | web_search_chain.map()




# chain.invoke(
#     {
#         "question": "what is the difference between langchain and huggingface?"
#     }
# )